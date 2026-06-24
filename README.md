# Interpretation-Oriented Cloud Removal via Observation-Anchored Residual Flow with Geo-Contextual Alignment

Official implementation of **"Interpretation-Oriented Cloud Removal via Observation-Anchored Residual Flow with Geo-Contextual Alignment"**, accepted by **ECCV 2026**.

## Abstract

> [Paste the paper abstract here.]

## Overview

This repository provides the training and testing code for GACR, an interpretation-oriented cloud removal framework built around:

- **OAR-Flow**: Observation-Anchored Residual Flow for cloud-removal flow matching.
- **GCPA**: Geo-Contextual Prior Alignment using visual foundation model features.
- **Hourglass Transformer**: the cloud-removal backbone implemented in `models/k_diffusion/image_transformer.py`.

The release supports GCPA in two modes:

- **Offline GCPA**: precompute VFM patch features and load them from `gcpa.feat_path`.
- **Online GCPA**: instantiate the VFM during training and infer features on the fly.

Supported VFM choices are `dinov3`, `dinov2`, `mae`, and `clip`. The main paper path uses DINOv3.

## Environment

```bash
conda create -n gacr python=3.10 -y
conda activate gacr
pip install -r requirements.txt
```

The provided environment was tested with PyTorch 2.4.0 and CUDA 12.4. `natten` and `flash-attn` are CUDA/PyTorch sensitive; install wheels matching your local PyTorch and CUDA version if the pinned versions are not compatible with your machine.

## Repository Structure

```text
GACR/
  train.py                    # training entry
  test.py                     # evaluation and prediction export
  cache_feat.py               # offline DINOv3 feature cache script
  dataset.py                  # cloud-removal dataset loaders
  engine.py                   # OAR-Flow objective, sampler, metrics
  requirements.txt
  config/
  models/
    k_diffusion/              # hourglass Transformer backbone
    sgm/                      # sampler, denoiser, metrics utilities
  vfm_weights/                # VFM weights, e.g. DINOv3 .pth files
```

## Dataset Layout

For Changsha/Guangzhou-style datasets:

```text
DATA_ROOT/
  train.txt
  test.txt
  clear/
  cloudy/
```

For ISPRS Potsdam/Vaihingen-style datasets:

```text
DATA_ROOT/
  train.txt
  test.txt
  clear/
  thin/
  thick/
```

Images are loaded as RGB and normalized to `[-1, 1]` by `dataset.py`.

## VFM Weights

Place visual foundation model weights under `vfm_weights/` or update the paths in your config.

Example DINOv3 layout:

```text
vfm_weights/
  dinov3/
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

For DINOv2, MAE, and CLIP online GCPA, use Hugging Face model directories, for example:

```text
vfm_weights/
  dinov2-large/
  vit-mae-large/
  clip-vit-large-patch14/
```

## Offline DINOv3 Feature Cache

Offline GCPA expects a NumPy `.npy` dictionary:

```python
{
  "image_name.png": np.ndarray,  # shape [num_patch_tokens, feature_dim]
}
```

Generate DINOv3 features with:

```bash
cd /path/to/GACR

python cache_feat.py \
  --data-path /path/to/DATA_ROOT/clear \
  --output-path ./dataset_dino_v3/YOUR_DATASET/dinov3_lvd.npy \
  --source local \
  --repo-path /path/to/dinov3_repo \
  --weight-path ./vfm_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

`cache_feat.py` determines DINOv3 normalization from the weight path: if `sat` appears in `--weight-path`, SAT normalization is used; otherwise LVD/ImageNet normalization is used.

## Configuration

GCPA settings live under `gcpa`:

```yaml
gcpa:
  vfm: "dinov3"        # dinov3, dinov2, mae, clip
  offline: true        # true: load cached features; false: online VFM inference
  feat_path: "./dataset_dino_v3/YOUR_DATASET/dinov3_lvd.npy"
```

For online DINOv3:

```yaml
gcpa:
  vfm: "dinov3"
  offline: false
  model_path: "./vfm_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  source: "local"
  repo_path: "/path/to/dinov3_repo"
  model_name: "dinov3_vitl16"
```

For online Hugging Face VFMs:

```yaml
gcpa:
  vfm: "clip"          # or dinov2 / mae
  offline: false
  model_path: "./vfm_weights/clip-vit-large-patch14"
  local_files_only: true
```

## Training

Run training with Accelerate:

```bash
cd /path/to/GACR

accelerate launch train.py \
  --config config/YOUR_CONFIG.yml
```

Smoke-test configs are provided for quick checks:

```bash
accelerate launch train.py --config config/test_10step_dinov3_offline.yml
accelerate launch train.py --config config/test_10step_dinov3_online.yml
accelerate launch train.py --config config/test_10step_clip_online.yml
accelerate launch train.py --config config/test_10step_dinov2_online.yml
accelerate launch train.py --config config/test_10step_mae_online.yml
```

Training writes checkpoints to:

```text
exps/<exp_name>/checkpoints/
```

Validation metrics are appended locally, one row per validation step:

```text
exps/<exp_name>/validation_metrics.csv
```

## Testing and Prediction Export

Evaluate a checkpoint and export predictions:

```bash
cd /path/to/GACR

python test.py \
  --config config/YOUR_CONFIG.yml \
  --ckpt exps/YOUR_EXP/checkpoints/0200000.pt \
  --state-key ema \
  --num-steps 4 \
  --save-pred
```

For a quick subset:

```bash
python test.py \
  --config config/YOUR_CONFIG.yml \
  --ckpt exps/YOUR_EXP/checkpoints/0200000.pt \
  --max-samples 5 \
  --save-pred
```

Outputs are saved to:

```text
exps/<exp_name>/test/
  preds/
  <ckpt>_metrics.json
  <ckpt>_metrics.csv
```

You can also set a custom output directory:

```bash
python test.py \
  --config config/YOUR_CONFIG.yml \
  --ckpt /absolute/path/to/checkpoint.pt \
  --output-dir ./test_outputs/YOUR_RUN \
  --save-pred
```

## Citation

If you find this project useful, please cite our paper:

```bibtex
@inproceedings{gacr2026,
  title     = {Interpretation-Oriented Cloud Removal via Observation-Anchored Residual Flow with Geo-Contextual Alignment},
  author    = {TBD},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

This codebase uses PyTorch, Hugging Face Transformers/Diffusers, DINOv3, NATTEN, FlashAttention, and LPIPS. We thank the authors and maintainers of these projects.
