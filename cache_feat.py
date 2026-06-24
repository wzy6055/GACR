import argparse
import os
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
from torchvision.transforms import Normalize
from tqdm.auto import tqdm


def preprocess_raw_image(x, weight_path):
    x = x.float() / 255.0
    if "sat" in str(weight_path).lower():
        return Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))(x)
    return Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(x)


def collect_images(data_path):
    image_root = Path(data_path)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [p for p in sorted(image_root.iterdir()) if p.suffix.lower() in suffixes]
    if not files:
        raise FileNotFoundError(f"No image files found in: {image_root}")
    return files


def main(args):
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    encoder = torch.hub.load(
        args.repo_path,
        args.model,
        source=args.source,
        weights=args.weight_path,
    )
    encoder = encoder.to(device)
    encoder.eval()

    feat_db = {}
    for image_path in tqdm(collect_images(args.data_path)):
        image = io.imread(image_path)[:, :, :3]
        image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
        image = preprocess_raw_image(image, args.weight_path).to(device)

        with torch.no_grad():
            features = encoder.forward_features(image)["x_norm_patchtokens"]
            feat_db[image_path.name] = features.squeeze(0).cpu().numpy()

    np.save(output_path, feat_db)
    print(f"Saved {len(feat_db)} cached features to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cache DINOv3 patch features for GCPA training.")
    parser.add_argument("--data-path", type=str, required=True, help="Directory that contains images to cache.")
    parser.add_argument("--output-path", type=str, required=True, help="Output .npy path.")
    parser.add_argument("--repo-path", type=str, default="facebookresearch/dinov3", help="Torch hub repo path.")
    parser.add_argument("--source", type=str, default="github", choices=["github", "local"])
    parser.add_argument("--model", type=str, default="dinov3_vitl16")
    parser.add_argument(
        "--weight-path",
        type=str,
        default="./vfm_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
