from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .dataset import SampleRecord
from .memory_hooks import DEFAULT_EXTRACT_LAYERS, SLOT_INDICES, SLOT_NAMES, MemoryTokenRecorder
from .utils import add_lingbot_to_path, ensure_dir, list_images, sha256_file, write_json


class LingbotRunner:
    def __init__(
        self,
        lingbot_root: str | Path,
        checkpoint_path: str | Path,
        *,
        device: Optional[str] = None,
        image_size: int = 518,
        patch_size: int = 14,
        num_scale_frames: int = 8,
        keyframe_interval: int = 1,
        kv_cache_sliding_window: int = 64,
        camera_num_iterations: int = 4,
        use_sdpa: bool = True,
        checkpoint_hash_bytes: Optional[int] = 64 * 1024 * 1024,
    ):
        self.lingbot_root = Path(lingbot_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_scale_frames = int(num_scale_frames)
        self.keyframe_interval = int(keyframe_interval)
        self.kv_cache_sliding_window = int(kv_cache_sliding_window)
        self.camera_num_iterations = int(camera_num_iterations)
        self.use_sdpa = bool(use_sdpa)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_hash_bytes = None if checkpoint_hash_bytes is not None and checkpoint_hash_bytes <= 0 else checkpoint_hash_bytes
        self.model = None
        self.lingbot_module_file: Optional[str] = None
        self.checkpoint_hash: Optional[str] = None

    def load_model(self) -> Any:
        if self.model is not None:
            return self.model

        add_lingbot_to_path(self.lingbot_root)
        import lingbot_map
        from lingbot_map.models.gct_stream import GCTStream

        self.lingbot_module_file = str(Path(lingbot_map.__file__).resolve())
        expected_prefix = str(self.lingbot_root)
        if not self.lingbot_module_file.startswith(expected_prefix):
            raise RuntimeError(
                f"lingbot_map imported from {self.lingbot_module_file}, expected it under {expected_prefix}. "
                "Pass --lingbot_root for the intended local source tree."
            )

        model = GCTStream(
            img_size=self.image_size,
            patch_size=self.patch_size,
            enable_3d_rope=True,
            max_frame_num=1024,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
            camera_num_iterations=self.camera_num_iterations,
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
            model.aggregator = model.aggregator.to(dtype=dtype)
        self.model = model
        self.model_load_info = {
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "lingbot_module_file": self.lingbot_module_file,
        }
        if self.checkpoint_hash is None:
            self.checkpoint_hash = sha256_file(self.checkpoint_path, self.checkpoint_hash_bytes)
        return self.model

    def _load_images(self, image_paths: Sequence[Path]) -> torch.Tensor:
        add_lingbot_to_path(self.lingbot_root)
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode="crop",
            image_size=self.image_size,
            patch_size=self.patch_size,
        )

    def _image_paths_for_sample(self, sample: SampleRecord, flush_frames: int) -> List[Path]:
        paths = list_images(sample.rgb_dir)
        required = sample.probe_frame + 1
        if len(paths) < required:
            raise RuntimeError(f"{sample.sample_id} has {len(paths)} RGB frames, need at least {required}")
        paths = paths[:required]
        if flush_frames > 0:
            paths.extend([paths[-1]] * int(flush_frames))
        return paths

    def run_sample(
        self,
        sample: SampleRecord,
        *,
        output_root: str | Path,
        save_reconstruction: str = "minimal",
        flush_frames: int = 0,
        extract_layers: Sequence[int] = DEFAULT_EXTRACT_LAYERS,
    ) -> Dict[str, Any]:
        if save_reconstruction not in {"none", "minimal", "full"}:
            raise ValueError("--save_reconstruction must be one of none|minimal|full")

        model = self.load_model()
        recon_dir = Path(output_root) / "reconstructions" / sample.sample_id
        ensure_dir(recon_dir)

        image_paths = self._image_paths_for_sample(sample, flush_frames)
        images = self._load_images(image_paths).to(self.device)
        input_frame_count = int(images.shape[0])
        dtype = torch.float32
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16

        output_device = torch.device("cpu") if save_reconstruction in {"minimal", "full"} else None
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast(self.device.type, dtype=dtype, enabled=self.device.type == "cuda"):
            with MemoryTokenRecorder(model, layer_indices=extract_layers) as recorder:
                predictions = model.inference_streaming(
                    images,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=self.keyframe_interval,
                    output_device=output_device,
                )
                memory_tokens = recorder.get_frame_tokens(sample.probe_frame)
                call_log = recorder.call_log

        elapsed = time.time() - t0
        if not memory_tokens:
            raise RuntimeError(f"No memory token record found for frame {sample.probe_frame} in {sample.sample_id}")
        for layer_name, tensor in memory_tokens.items():
            if list(tensor.shape)[0] != 6:
                raise RuntimeError(f"{sample.sample_id} {layer_name} has shape {tuple(tensor.shape)}, expected [6, D]")
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{sample.sample_id} {layer_name} contains non-finite values")

        reconstruction_files = self._save_reconstruction(
            sample=sample,
            predictions=predictions,
            images=images,
            recon_dir=recon_dir,
            mode=save_reconstruction,
        )

        run_metadata = {
            "sample_id": sample.sample_id,
            "base_scene_id": sample.base_scene_id,
            "variant_id": sample.variant_id,
            "input_frame_count": input_frame_count,
            "probe_frame": sample.probe_frame,
            "model_config": self.model_config(),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_hash,
            "checkpoint_hash_bytes": self.checkpoint_hash_bytes,
            "local_window_size": self.kv_cache_sliding_window,
            "flush_frames_used": int(flush_frames),
            "success": True,
            "elapsed_sec": elapsed,
            "memory_token_layers": sorted(memory_tokens.keys()),
            "memory_token_shapes": {k: list(v.shape) for k, v in memory_tokens.items()},
            "slot_names": SLOT_NAMES,
            "slot_indices": SLOT_INDICES,
            "recorder_call_log": call_log,
            "reconstruction_files": reconstruction_files,
        }
        write_json(recon_dir / "run_metadata.json", run_metadata)

        return {
            "sample": sample,
            "memory_tokens": memory_tokens,
            "run_metadata": run_metadata,
            "predictions": predictions if save_reconstruction == "full" else None,
        }

    def _save_reconstruction(
        self,
        *,
        sample: SampleRecord,
        predictions: Mapping[str, Any],
        images: torch.Tensor,
        recon_dir: Path,
        mode: str,
    ) -> Dict[str, str]:
        if mode == "none":
            write_json(
                recon_dir / "run_metadata.json",
                {"sample_id": sample.sample_id, "save_reconstruction": "none"},
            )
            return {}

        files: Dict[str, str] = {}
        probe = sample.probe_frame
        if "pose_enc" in predictions:
            try:
                add_lingbot_to_path(self.lingbot_root)
                from lingbot_map.utils.geometry import closed_form_inverse_se3_general
                from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

                pose_enc = predictions["pose_enc"].to("cpu")
                extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                extrinsic_4x4 = torch.zeros((*extrinsic_w2c.shape[:-2], 4, 4), dtype=extrinsic_w2c.dtype)
                extrinsic_4x4[..., :3, :4] = extrinsic_w2c.cpu()
                extrinsic_4x4[..., 3, 3] = 1.0
                c2w = closed_form_inverse_se3_general(extrinsic_4x4)[..., :3, :4]
                path = recon_dir / "probe_frame_camera.npy"
                np.save(path, c2w[0, probe].detach().cpu().numpy())
                files["probe_frame_camera"] = str(path)
                path_i = recon_dir / "probe_frame_intrinsic.npy"
                np.save(path_i, intrinsic[0, probe].detach().cpu().numpy())
                files["probe_frame_intrinsic"] = str(path_i)
            except Exception as exc:
                files["probe_frame_camera_error"] = repr(exc)

        if "depth" in predictions:
            path = recon_dir / "probe_frame_depth.npy"
            np.save(path, predictions["depth"][0, probe].detach().cpu().numpy())
            files["probe_frame_depth"] = str(path)
        if "world_points" in predictions:
            path = recon_dir / "probe_frame_pointmap.npy"
            np.save(path, predictions["world_points"][0, probe].detach().cpu().numpy())
            files["probe_frame_pointmap"] = str(path)
        return files

    def model_config(self) -> Dict[str, Any]:
        return {
            "lingbot_root": str(self.lingbot_root),
            "lingbot_module_file": self.lingbot_module_file,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_scale_frames": self.num_scale_frames,
            "keyframe_interval": self.keyframe_interval,
            "kv_cache_sliding_window": self.kv_cache_sliding_window,
            "camera_num_iterations": self.camera_num_iterations,
            "use_sdpa": self.use_sdpa,
            "device": str(self.device),
            **getattr(self, "model_load_info", {}),
        }
