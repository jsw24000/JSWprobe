"""Lingbot-map tuned DINO backbone, before reconstruction aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from ..utils import torch_device
from .base import FeatureBackend, FeatureResult


class LingbotBackbonePreAggBackend(FeatureBackend):
    name = "lingbot_backbone_pre_agg_rope_off"

    def extract(self, frames, scene, grid, cfg) -> list[FeatureResult]:
        frame_ids = [f.frame_id for f in frames]
        input_hw = grid.input_hw
        grid_hw = grid.grid_hw
        repo = Path(cfg["paths"].get("lingbot_repo", ""))
        ckpt_path = Path(cfg["paths"].get("lingbot_checkpoint", ""))
        metadata = {
            "checkpoint_path": str(ckpt_path),
            "repo_path": str(repo),
            "video_rope_enabled": False,
            "rope_disable_method": "backbone-only path skips temporal/video RoPE; model constructed with enable_3d_rope=False",
            "verified": True,
            "hook_name": "model.aggregator.patch_embed",
            "note": "DINOv2-initialized but Lingbot-trained backbone; not public DINOv2.",
        }
        if not repo.is_dir():
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=f"Lingbot repo not found: {repo}", metadata=metadata)]
        if not ckpt_path.is_file():
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=f"Lingbot checkpoint not found: {ckpt_path}", metadata=metadata)]
        try:
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from lingbot_map.models.gct_stream import GCTStream

            device = torch_device(cfg.get("run", {}).get("device", "cuda"))
            model = GCTStream(
                img_size=input_hw[1],
                patch_size=grid.patch_size,
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                enable_local_point=False,
                enable_track=False,
                enable_3d_rope=False,
                disable_global_rope=True,
                use_sdpa=True,
                use_gradient_checkpoint=False,
            )
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = state.get("model", state)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model.to(device).eval().requires_grad_(False)
            batch_size = int(cfg.get("lingbot", {}).get("batch_size", 4))
            tensors = []
            for fr in frames:
                img = Image.open(fr.color_path).convert("RGB").resize((input_hw[1], input_hw[0]), Image.Resampling.BICUBIC)
                arr = np.asarray(img).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            feats = []
            mean = model.aggregator._resnet_mean.to(device).reshape(1, 3, 1, 1)
            std = model.aggregator._resnet_std.to(device).reshape(1, 3, 1, 1)
            with torch.inference_mode():
                for start in tqdm(range(0, len(tensors), batch_size), desc=self.name, unit="batch"):
                    batch = torch.stack(tensors[start : start + batch_size], dim=0).to(device)
                    normalized = (batch - mean) / std
                    patch = model.aggregator.patch_embed(normalized)
                    if isinstance(patch, dict):
                        patch = patch["x_norm_patchtokens"]
                    feats.append(patch.detach().float().cpu())
            patch_tokens = torch.cat(feats, dim=0)
            expected_tokens = grid_hw[0] * grid_hw[1]
            if patch_tokens.shape[1] != expected_tokens:
                raise RuntimeError(f"Expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}")
            features = patch_tokens.reshape(len(frames), grid_hw[0], grid_hw[1], patch_tokens.shape[-1])
            metadata.update({"missing_keys": len(missing), "unexpected_keys": len(unexpected), "feature_shape": list(features.shape)})
            return [FeatureResult(features, frame_ids, grid_hw, input_hw, self.name, "pre_agg_patch", metadata)]
        except Exception as exc:
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=str(exc), metadata=metadata, status="failed_error")]

