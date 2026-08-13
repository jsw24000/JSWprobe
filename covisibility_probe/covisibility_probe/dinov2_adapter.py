from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def add_repo_to_path(repo: str | Path) -> None:
    root = Path(repo).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def extract_official_dino_pool(
    image_paths: list[str | Path],
    *,
    dinov2_repo: str | Path,
    checkpoint_path: str | Path,
    lingbot_repo: str | Path,
    image_size: int = 518,
    patch_size: int = 14,
    batch_size: int = 4,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict[str, Any]]:
    repo = Path(dinov2_repo).expanduser().resolve()
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"DINOv2 repo not found: {repo}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {ckpt}")

    add_repo_to_path(repo)
    add_repo_to_path(lingbot_repo)
    from dinov2.hub.backbones import dinov2_vitl14_reg
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = dinov2_vitl14_reg(pretrained=True, weights=str(ckpt)).to(dev).eval().requires_grad_(False)
    images = load_and_preprocess_images(
        [str(p) for p in image_paths],
        mode="crop",
        image_size=int(image_size),
        patch_size=int(patch_size),
    )
    input_hw = [int(images.shape[-2]), int(images.shape[-1])]
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    pools: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)].to(dev)
            batch = (batch - mean) / std
            out = model.forward_features(batch)
            patch = out["x_norm_patchtokens"].detach().float().cpu()
            pools.append(patch.mean(dim=1))
    desc = torch.cat(pools, dim=0)
    meta = {
        "repo_path": str(repo),
        "checkpoint_path": str(ckpt),
        "model_name": "dinov2_vitl14_reg",
        "normalization": "ImageNet mean/std",
        "pooling": "mean over x_norm_patchtokens",
        "input_hw": input_hw,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return desc, meta

