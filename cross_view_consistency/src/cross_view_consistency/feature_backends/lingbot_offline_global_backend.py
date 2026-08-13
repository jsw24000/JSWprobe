"""Lingbot-map 32-frame all-scale multi-frame reconstruction backend.

This is a minimal wrapper around the local Lingbot-map streaming model. It does
not patch Lingbot source code. Instead, it forwards all selected frames in one
call and sets ``num_frame_for_scale`` and ``num_frame_per_block`` to the sequence
length. In the local SDPA path this makes the selected frames participate in one
multi-frame attention pass rather than frame-by-frame KV streaming.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..utils import torch_device
from .base import FeatureBackend, FeatureResult


class LingbotOfflineGlobalBackend(FeatureBackend):
    name = "lingbot_offline_global_32f_rope_off"

    def extract(self, frames, scene, grid, cfg) -> list[FeatureResult]:
        frame_ids = [f.frame_id for f in frames]
        input_hw = grid.input_hw
        grid_hw = grid.grid_hw
        seq_len = len(frames)
        repo = Path(cfg["paths"].get("lingbot_repo", ""))
        ckpt_path = Path(cfg["paths"].get("lingbot_checkpoint", ""))
        stages = list(cfg.get("lingbot", {}).get("extract_stages", {}).get("offline_global_32f", ["stage1", "stage2"]))
        metadata_base = {
            "checkpoint_path": str(ckpt_path),
            "repo_path": str(repo),
            "video_rope_enabled": False,
            "rope_disable_method": "GCTStream(enable_3d_rope=False); all selected frames forwarded together as scale frames",
            "verified": True,
            "hook_name": "GCTStream._aggregate_features",
            "mode": "all_scale_multiframe_phase1",
            "num_frame_for_scale": seq_len,
            "num_frame_per_block": seq_len,
            "sliding_window_size": -1,
            "note": (
                "Uses Lingbot streaming model's multi-frame scale pass as a temporary "
                "offline/global approximation; no Lingbot source patch and no separate "
                "offline model class."
            ),
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
                kv_cache_scale_frames=max(seq_len, int(cfg.get("lingbot", {}).get("num_scale_frames", seq_len))),
                kv_cache_sliding_window=max(64, seq_len),
                use_gradient_checkpoint=False,
            )
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = state.get("model", state)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model.to(device).eval().requires_grad_(False)

            tensors = []
            for fr in frames:
                img = Image.open(fr.color_path).convert("RGB").resize((input_hw[1], input_hw[0]), Image.Resampling.BICUBIC)
                arr = np.asarray(img).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            images = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

            stage_to_idx = {"stage0": 0, "stage1": 1, "stage2": 2, "stage3": 3}
            wanted = {stage: stage_to_idx[stage] for stage in stages if stage in stage_to_idx}
            if not wanted:
                raise RuntimeError(f"No supported offline stages requested: {stages}")

            with torch.inference_mode():
                model.clean_kv_cache()
                outputs, patch_start_idx = model._aggregate_features(
                    images,
                    num_frame_for_scale=seq_len,
                    num_frame_per_block=seq_len,
                    sliding_window_size=-1,
                )

            results: list[FeatureResult] = []
            expected_tokens = grid_hw[0] * grid_hw[1]
            for stage, idx in wanted.items():
                tokens = outputs[idx][0, :, patch_start_idx:, :].detach().float().cpu()
                if tokens.shape[1] != expected_tokens:
                    raise RuntimeError(f"{stage}: expected {expected_tokens} patch tokens, got {tokens.shape[1]}")
                if tokens.shape[-1] % 2 != 0:
                    raise RuntimeError(f"{stage}: expected even concat dim, got {tokens.shape[-1]}")

                half_dim = tokens.shape[-1] // 2
                branch_specs = [
                    ("concat", tokens, "frame_pre_global + global_post concat"),
                    ("frame_pre_global", tokens[..., :half_dim], "after frame attention, before same-stage global attention"),
                    ("global_post", tokens[..., half_dim:], "after same-stage global attention"),
                ]

                for branch_name, branch_tokens, branch_note in branch_specs:
                    features = branch_tokens.reshape(seq_len, grid_hw[0], grid_hw[1], branch_tokens.shape[-1])
                    metadata = dict(metadata_base)
                    metadata.update(
                        {
                            "missing_keys": len(missing),
                            "unexpected_keys": len(unexpected),
                            "feature_shape": list(features.shape),
                            "patch_start_idx": int(patch_start_idx),
                            "branch": branch_name,
                            "branch_note": branch_note,
                            "concat_dim": int(tokens.shape[-1]),
                            "branch_dim": int(branch_tokens.shape[-1]),
                        }
                    )
                    if branch_name == "concat":
                        backend_name = f"{self.name}_{stage}"
                        layer_or_stage = stage
                    else:
                        backend_name = f"{self.name}_{stage}_{branch_name}"
                        layer_or_stage = f"{stage}_{branch_name}"
                    results.append(FeatureResult(features, frame_ids, grid_hw, input_hw, backend_name, layer_or_stage, metadata))
            return results
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [
                FeatureResult.failed(
                    backend_name=self.name,
                    frame_ids=frame_ids,
                    grid_hw=grid_hw,
                    input_hw=input_hw,
                    reason=f"CUDA OOM during all-scale multi-frame pass: {exc}",
                    metadata=metadata_base,
                    status="failed_error",
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
                    metadata=metadata_base,
                    status="failed_error",
                )
            ]
