"""Lingbot-map single-frame reconstruction token backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from ..utils import torch_device
from .base import FeatureBackend, FeatureResult


class LingbotSingleFrameReconBackend(FeatureBackend):
    name = "lingbot_single_frame_recon_rope_off"

    def extract(self, frames, scene, grid, cfg) -> list[FeatureResult]:
        frame_ids = [f.frame_id for f in frames]
        input_hw = grid.input_hw
        grid_hw = grid.grid_hw
        repo = Path(cfg["paths"].get("lingbot_repo", ""))
        ckpt_path = Path(cfg["paths"].get("lingbot_checkpoint", ""))
        stages = list(cfg.get("lingbot", {}).get("extract_stages", {}).get("single_frame_recon", ["stage1", "stage2"]))
        metadata_base = {
            "checkpoint_path": str(ckpt_path),
            "repo_path": str(repo),
            "video_rope_enabled": False,
            "rope_disable_method": "GCTStream(enable_3d_rope=False); clean_kv_cache before each one-frame forward",
            "verified": True,
            "hook_name": "model._aggregate_features(...), selected_idx=[4,11,17,23]",
            "single_frame_isolation": "Each frame is forwarded alone after KV cache reset.",
        }
        if not repo.is_dir():
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=f"Lingbot repo not found: {repo}", metadata=metadata_base)]
        if not ckpt_path.is_file():
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=f"Lingbot checkpoint not found: {ckpt_path}", metadata=metadata_base)]
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
            stage_to_idx = {"stage0": 0, "stage1": 1, "stage2": 2, "stage3": 3}
            wanted = {stage: stage_to_idx[stage] for stage in stages if stage in stage_to_idx}
            if not wanted:
                raise RuntimeError(f"No supported stages requested: {stages}")
            stage_tokens: dict[str, list[torch.Tensor]] = {stage: [] for stage in wanted}
            with torch.inference_mode():
                for fr in tqdm(frames, desc=self.name, unit="frame"):
                    img = Image.open(fr.color_path).convert("RGB").resize((input_hw[1], input_hw[0]), Image.Resampling.BICUBIC)
                    arr = np.asarray(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
                    model.clean_kv_cache()
                    outputs, patch_start_idx = model._aggregate_features(tensor, num_frame_for_scale=1, sliding_window_size=-1)
                    for stage, idx in wanted.items():
                        tokens = outputs[idx][0, 0, patch_start_idx:, :].detach().float().cpu()
                        stage_tokens[stage].append(tokens)
            results: list[FeatureResult] = []
            expected_tokens = grid_hw[0] * grid_hw[1]
            for stage, chunks in stage_tokens.items():
                patch_tokens = torch.stack(chunks, dim=0)
                if patch_tokens.shape[1] != expected_tokens:
                    raise RuntimeError(f"{stage}: expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}")
                features = patch_tokens.reshape(len(frames), grid_hw[0], grid_hw[1], patch_tokens.shape[-1])
                metadata = dict(metadata_base)
                metadata.update({"missing_keys": len(missing), "unexpected_keys": len(unexpected), "feature_shape": list(features.shape)})
                backend_name = f"{self.name}_{stage}"
                results.append(FeatureResult(features, frame_ids, grid_hw, input_hw, backend_name, stage, metadata))
            return results
        except Exception as exc:
            return [FeatureResult.failed(backend_name=self.name, frame_ids=frame_ids, grid_hw=grid_hw, input_hw=input_hw, reason=str(exc), metadata=metadata_base, status="failed_error")]

