import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import yaml
from types import SimpleNamespace
import shutil
import csv

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.k_diffusion import image_transformer
from engine import OARFlowEngine
from utils import load_encoders

from dataset import CRDataset, ISPRSDataset
from diffusers.models import AutoencoderKL
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)
if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"])

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
SAT493M_DEFAULT_MEAN = (0.430, 0.411, 0.296)
SAT493M_DEFAULT_STD = (0.213, 0.156, 0.143)
CLIP_DEFAULT_MEAN = (0.481, 0.458, 0.408)
CLIP_DEFAULT_STD = (0.269, 0.261, 0.276)

DEFAULT_GCPA_MODELS = {
    "dinov3": "./vfm_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov2": "./vfm_weights/dinov2-large",
    "mae": "./vfm_weights/vit-mae-large",
    "clip": "./vfm_weights/clip-vit-large-patch14",
}

def get_cached_features(batch, feat_db, device='cuda'):
    feats = []
    for path in batch['filename']:
        file_name = os.path.basename(path)
        if file_name not in feat_db:
            raise KeyError(f"[ERROR] feature for '{file_name}' not found in feat_db")
        feat = feat_db[file_name]
        feats.append(torch.from_numpy(feat).to(device))

    # 堆叠成 [B, N, D]
    # feats = torch.stack(feats, dim=0).to(device)
    return feats

class GCPAFeatureExtractor(torch.nn.Module):
    def __init__(self, gcpa_config):
        super().__init__()
        self.vfm = getattr(gcpa_config, "vfm", "dinov3").lower()
        if self.vfm not in DEFAULT_GCPA_MODELS:
            raise ValueError(f"Unsupported GCPA VFM: {self.vfm}")

        self.model_path = getattr(
            gcpa_config,
            "model_path",
            getattr(gcpa_config, "hf_model_name", DEFAULT_GCPA_MODELS[self.vfm]),
        )

        if self.vfm == "dinov3":
            repo_path = getattr(gcpa_config, "repo_path", "facebookresearch/dinov3")
            source = getattr(gcpa_config, "source", "github")
            model_name = getattr(gcpa_config, "model_name", "dinov3_vitl16")
            weight_path = str(self.model_path).lower()
            mean = SAT493M_DEFAULT_MEAN if "sat" in weight_path else IMAGENET_DEFAULT_MEAN
            std = SAT493M_DEFAULT_STD if "sat" in weight_path else IMAGENET_DEFAULT_STD
            self.image_size = 256
            self.normalize = Normalize(mean=mean, std=std)
            self.model = torch.hub.load(
                repo_path,
                model_name,
                source=source,
                weights=self.model_path,
            )
        else:
            local_files_only = bool(getattr(gcpa_config, "local_files_only", True))
            self.image_size = 256 if self.vfm == "mae" else 224
            if self.vfm == "dinov2":
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(self.model_path, local_files_only=local_files_only)
                self.normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            elif self.vfm == "clip":
                from transformers import CLIPVisionModel
                self.model = CLIPVisionModel.from_pretrained(self.model_path, local_files_only=local_files_only)
                self.normalize = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)
            elif self.vfm == "mae":
                from transformers import ViTMAEModel
                self.model = ViTMAEModel.from_pretrained(self.model_path, local_files_only=local_files_only)
                self.model.config.mask_ratio = 0.0
                self.normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        self.model.eval()
        requires_grad(self.model, False)

    def preprocess(self, clear):
        x = (clear.float() + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        return self.normalize(x)

    @torch.no_grad()
    def forward(self, clear):
        x = self.preprocess(clear)
        if self.vfm == "dinov3":
            tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        elif self.vfm == "dinov2":
            tokens = self.model(pixel_values=x).last_hidden_state[:, 1:, :]
        else:
            tokens = self.model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:, :]

        if tokens.shape[1] != 256:
            raise ValueError(f"{self.vfm} produced {tokens.shape[1]} tokens; GCPA expects 256.")
        if tokens.shape[2] != 1024:
            raise ValueError(f"{self.vfm} produced dim {tokens.shape[2]}; GCPA projector expects 1024.")
        return [z.detach() for z in tokens]


def gcpa_is_offline(config):
    return bool(getattr(config.gcpa, "offline", True))


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def write_validation_metrics_csv(save_dir, step, metrics):
    csv_path = os.path.join(save_dir, "validation_metrics.csv")
    row = {"step": step}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        row[key] = value

    fieldnames = list(row.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def load_config(args=None):
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        else:
            return d

    config = dict_to_namespace(cfg_dict)
    if args.exp_name:
        config.logging.exp_name = args.exp_name
    elif hasattr(config.logging, "exp_name") and config.logging.exp_name:
        pass  # yml 中已设置
    else:
        from datetime import datetime
        config.logging.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return config


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    config = load_config(args)

    # set accelerator
    logging_dir = Path(config.logging.output_dir, config.logging.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.logging.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        mixed_precision=config.precision.mixed_precision,
        log_with=config.logging.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(config.logging.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(config.logging.output_dir, config.logging.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(save_dir, "config.yml"))
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if config.misc.seed is not None:
        set_seed(config.misc.seed + accelerator.process_index)

    def namespace_to_dict(obj):
        if isinstance(obj, dict):
            return {k: namespace_to_dict(v) for k, v in obj.items()}
        from types import SimpleNamespace
        if isinstance(obj, SimpleNamespace):
            return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(namespace_to_dict(v) for v in obj)
        return obj
    kwargs = namespace_to_dict(config.model)
    model = image_transformer.ImageTransformerDenoiserModelInterface(**kwargs)
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    if accelerator.is_main_process:
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if config.precision.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optimization.learning_rate),
        betas=(float(config.optimization.adam_beta1), float(config.optimization.adam_beta2)),
        weight_decay=float(config.optimization.adam_weight_decay),
        eps=float(config.optimization.adam_epsilon),
    )

    # Setup data:
    if 'changsha' in config.dataset.data_dir or 'guangzhou' in config.dataset.data_dir:
        train_dataset = CRDataset(config.dataset.data_dir, mode='train')
        val_dataset = CRDataset(config.dataset.data_dir, mode='test', val_size=config.dataset.val_size)

    # ISPRS data
    elif 'Potsdam' in config.dataset.data_dir or 'Vaihingen' in config.dataset.data_dir:
        train_dataset = ISPRSDataset(config.dataset, mode='train')
        val_dataset = ISPRSDataset(config.dataset, mode='test', val_size=config.dataset.val_size)
    else:
        raise ValueError(
            "Unknown dataset path."
            f"Got: {config.dataset.data_dir}"
        )

    local_batch_size = int(config.dataset.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=config.misc.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.misc.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({config.dataset.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    if config.logging.resume_step > 0:
        ckpt_name = str(config.logging.resume_step).zfill(7) + '.pt'
        ckpt = torch.load(
            f'{os.path.join(config.logging.output_dir, config.logging.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process and config.logging.report_to:
        tracker_config = vars(copy.deepcopy(config))
        accelerator.init_trackers(
            project_name="GACR",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{config.logging.exp_name}"}
            },
        )

    progress_bar = tqdm(
        range(0, config.optimization.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    engine = OARFlowEngine(
        model=model,
        prediction=config.loss.prediction,
        path_type=config.loss.path_type,
        accelerator=accelerator,
        weighting=config.loss.weighting,
        p=getattr(config.loss, "p", 3.0)
    )

    if not hasattr(config, "gcpa"):
        raise ValueError("Missing config.gcpa section.")

    use_offline_gcpa = gcpa_is_offline(config)
    if use_offline_gcpa:
        if not hasattr(config.gcpa, "feat_path") or not config.gcpa.feat_path:
            raise ValueError("config.gcpa.feat_path is required when config.gcpa.offline is true.")
        feats_db = np.load(config.gcpa.feat_path, allow_pickle=True).item()
        vfm = None
        if accelerator.is_main_process:
            logger.info(f"Using offline GCPA features: {config.gcpa.feat_path}")
    else:
        feats_db = None
        vfm = GCPAFeatureExtractor(config.gcpa).to(device)
        vfm.eval()
        if accelerator.is_main_process:
            logger.info(f"Using online GCPA VFM: {vfm.vfm} ({vfm.model_path})")
            logger.info(f"VFM Parameters: {sum(p.numel() for p in vfm.parameters()):,}")

    for epoch in range(config.optimization.epochs):
        engine.model.train()
        for batch in train_dataloader:
            if use_offline_gcpa:
                zs = get_cached_features(batch, feats_db, device=device)
            else:
                with torch.no_grad():
                    zs = vfm(batch["clear"])
            with accelerator.accumulate(engine.model):
                loss, proj_loss = engine(batch, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * config.loss.proj_coeff
                # loss = loss_mean

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = engine.model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, config.optimization.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, engine.model)  # change ema function

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % config.optimization.checkpointing_steps == 0 or global_step == 1:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(engine.model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "config": config,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # val step
            if (global_step == 1 or (global_step % config.logging.sampling_steps == 0 and global_step > 0)):
                engine.model.eval()
                engine.avg_metrics.reset()
                for j, batch_val in enumerate(tqdm(val_dataloader, desc='Val')):
                    engine.test_step(batch_val)
                metrics = engine.avg_metrics.value()
                accelerator.log(metrics, step=global_step)
                if accelerator.is_main_process:
                    write_validation_metrics_csv(save_dir, global_step, metrics)
                batch_log = {k: batch[k][:1] for k in batch}
                image_results = engine.log_images(batch_log, N=1, sample=True)
                if config.logging.report_to:
                    accelerator.log({"input": wandb.Image(array2grid(image_results['input'])),
                                     "mean": wandb.Image(array2grid(image_results['mean'])),
                                     "samples": wandb.Image(array2grid(image_results['samples'])),
                                     },
                                    step=global_step)
                engine.model.train()

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.optimization.max_train_steps:
                break
        if global_step >= config.optimization.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
