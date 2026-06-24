import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from dataset import CRDataset, ISPRSDataset
from engine import OARFlowEngine
from models.k_diffusion import image_transformer


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def namespace_to_dict(obj):
    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(namespace_to_dict(v) for v in obj)
    return obj


def load_config(config_path):
    with open(config_path, "r") as f:
        return dict_to_namespace(yaml.safe_load(f))


def prepare_batch(batch, device):
    if isinstance(batch, dict):
        return {k: prepare_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(prepare_batch(v, device) for v in batch)
    if isinstance(batch, np.ndarray):
        return torch.from_numpy(batch).to(device)
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def strip_prefix(state_dict, prefixes=("module.", "model.")):
    clean = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        clean[new_key] = value
    return clean


def find_checkpoint(config, args):
    if args.ckpt:
        return Path(args.ckpt)

    exp_dir = Path(config.logging.output_dir) / config.logging.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    if args.ckpt_step is not None:
        return ckpt_dir / f"{int(args.ckpt_step):07d}.pt"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found: {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")
    return ckpts[-1]


def build_test_dataset(config):
    data_dir = config.dataset.data_dir
    val_size = getattr(config.dataset, "val_size", None)
    if "changsha" in data_dir or "guangzhou" in data_dir:
        return CRDataset(data_dir, mode="test", val_size=val_size)
    if "Potsdam" in data_dir or "Vaihingen" in data_dir:
        return ISPRSDataset(config.dataset, mode="test", val_size=val_size)
    raise ValueError(f"Unknown dataset path: {data_dir}")


def tensor_to_uint8_image(x):
    x = torch.clamp(255.0 * x, 0, 255)
    x = x.permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
    return x


def save_prediction(sample, filename, pred_dir):
    output_name = Path(os.path.basename(filename)).with_suffix(".png").name
    Image.fromarray(tensor_to_uint8_image(sample)).save(pred_dir / output_name)


def save_metrics(test_dir, ckpt_name, metrics):
    metrics = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}
    json_path = test_dir / f"{ckpt_name}_metrics.json"
    csv_path = test_dir / f"{ckpt_name}_metrics.csv"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics, json_path, csv_path


def main(args):
    torch.set_grad_enabled(False)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(args.config)

    if getattr(config.misc, "seed", None) is not None:
        torch.manual_seed(config.misc.seed)

    model = image_transformer.ImageTransformerDenoiserModelInterface(
        **namespace_to_dict(config.model)
    ).to(device)

    ckpt_path = find_checkpoint(config, args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_key = args.state_key
    if state_key not in checkpoint:
        available = ", ".join(checkpoint.keys())
        raise KeyError(f"Checkpoint has no key '{state_key}'. Available keys: {available}")
    model.load_state_dict(strip_prefix(checkpoint[state_key]), strict=True)
    model.eval()

    test_dataset = build_test_dataset(config)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.misc.num_workers,
        pin_memory=True,
    )
    print(f"Dataset contains {len(test_dataset):,} images ({config.dataset.data_dir})")
    print(f"Loading checkpoint: {ckpt_path}")

    engine = OARFlowEngine(
        model=model,
        prediction=config.loss.prediction,
        path_type=config.loss.path_type,
        weighting=config.loss.weighting,
        p=getattr(config.loss, "p", 3.0),
    )
    engine.model.eval()
    engine.avg_metrics.reset()

    exp_dir = Path(config.logging.output_dir) / config.logging.exp_name
    test_dir = Path(args.output_dir) if args.output_dir else exp_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = test_dir / "preds"
    if args.save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test output directory: {test_dir}")

    for idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
        if args.max_samples is not None and idx >= args.max_samples:
            break
        batch = prepare_batch(batch, device)
        engine.test_step(batch, num_steps=args.num_steps)

        if args.save_pred:
            results = engine.log_images(batch, N=1, sample=True)
            sample = results["samples"][0]
            save_prediction(sample, batch["filename"][0], pred_dir)

    ckpt_name = ckpt_path.stem
    metrics, json_path, csv_path = save_metrics(test_dir, ckpt_name, engine.avg_metrics.value())
    print(f"Metrics: {metrics}")
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved metrics CSV: {csv_path}")
    if args.save_pred:
        print(f"Saved predictions: {pred_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test GACR hourglass Transformer cloud removal model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--ckpt-step", type=int, default=None, help="Checkpoint step under exps/<exp_name>/checkpoints.")
    parser.add_argument("--state-key", type=str, default="ema", choices=["ema", "model"])
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-pred", action="store_true", default=True)
    parser.add_argument("--no-save-pred", action="store_false", dest="save_pred")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
