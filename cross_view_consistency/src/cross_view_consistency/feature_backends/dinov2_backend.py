"""Official public DINOv2 feature backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from ..scannet_io import FrameRecord
from ..utils import torch_device
from .base import FeatureBackend, FeatureResult


class PublicDinoV2Backend(FeatureBackend):
    name = "public_dinov2_vitl14"

    def extract(self, frames: list[FrameRecord], scene, grid, cfg) -> list[FeatureResult]:
        frame_ids = [f.frame_id for f in frames]
        paths = cfg["paths"]
        repo = Path(paths.get("dinov2_repo", ""))
        ckpt = Path(paths.get("dinov2_checkpoint", ""))
        input_hw = grid.input_hw
        grid_hw = grid.grid_hw
        metadata = {
            "checkpoint_path": str(ckpt),
            "repo_path": str(repo),
            "normalization": "ImageNet mean/std",
            "feature_key": cfg.get("dinov2", {}).get("feature_key", "x_norm_patchtokens"),
        }
        if not repo.is_dir():
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=f"DINOv2 repo not found: {repo}", metadata=metadata)]
        if not ckpt.is_file():
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=f"DINOv2 checkpoint not found: {ckpt}", metadata=metadata)]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            from dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg

            dino_cfg = cfg.get("dinov2", {})
            use_register = bool(dino_cfg.get("use_register_model", False)) or "reg" in str(dino_cfg.get("model_name", ""))
            model_fn = dinov2_vitl14_reg if use_register else dinov2_vitl14
            model = model_fn(pretrained=True, weights=str(ckpt)).eval()
            device = torch_device(cfg.get("run", {}).get("device", "cuda"))
            model = model.to(device)
            batch_size = int(dino_cfg.get("batch_size", 4))
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            tensors = []
            for fr in frames:
                img = Image.open(fr.color_path).convert("RGB").resize((input_hw[1], input_hw[0]), Image.Resampling.BICUBIC)
                arr = np.asarray(img).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            feats = []
            with torch.inference_mode():
                for start in tqdm(range(0, len(tensors), batch_size), desc=self.name, unit="batch"):
                    batch = torch.stack(tensors[start : start + batch_size], dim=0).to(device)
                    batch = (batch - mean) / std
                    out = model.forward_features(batch)
                    patch = out[dino_cfg.get("feature_key", "x_norm_patchtokens")]
                    feats.append(patch.detach().float().cpu())
            patch_tokens = torch.cat(feats, dim=0)
            expected_tokens = grid_hw[0] * grid_hw[1]
            if patch_tokens.shape[1] != expected_tokens:
                raise RuntimeError(f"Expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}")
            features = patch_tokens.reshape(len(frames), grid_hw[0], grid_hw[1], patch_tokens.shape[-1])
            metadata.update(
                {
                    "model_name": "dinov2_vitl14_reg" if use_register else "dinov2_vitl14",
                    "use_register_model": use_register,
                    "feature_shape": list(features.shape),
                }
            )
            return [
                FeatureResult(
                    features=features,
                    frame_ids=frame_ids,
                    grid_hw=grid_hw,
                    input_hw=input_hw,
                    backend_name=self.name,
                    layer_or_stage="patch",
                    metadata=metadata,
                )
            ]
        except Exception as exc:
            return [
                FeatureResult.failed(
                    backend_name=self.name,
                    frame_ids=frame_ids,
                    grid_hw=grid_hw,
                    input_hw=input_hw,
                    reason=str(exc),
                    metadata=metadata,
                    status="failed_error",
                )
            ]

