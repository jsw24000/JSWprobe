"""Lingbot-map adapter for official reconstruction/token artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .base_adapter import BaseModelAdapter
from .token_schema import TokenBundle


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HEAD_LAYERS = [4, 11, 17, 23]


class LingbotAdapter(BaseModelAdapter):
    """Adapter that keeps downstream experiments independent of Lingbot internals.

    The heavy model run is delegated to ``scripts/run_lingbot_official_tokens.py``.
    This class then adapts the saved ``metadata.json`` and token ``.npy`` files
    into the shared :class:`TokenBundle` schema used by controlled experiments.
    """

    def _config_path(self, key: str, default: str | Path) -> Path:
        return self._resolve_path(self.config.get(key, default)) or Path(default)

    @property
    def repo_path(self) -> Path:
        return self._config_path("repo_path", PROJECT_DIR / "third_party" / "lingbot-map")

    @property
    def checkpoint_path(self) -> Path:
        return self._config_path(
            "checkpoint_path",
            PROJECT_DIR / "third_party" / "lingbot-map" / "checkpoints" / "lingbot-map.pt",
        )

    @property
    def token_output_root(self) -> Path:
        return self._config_path("token_output_root", PROJECT_DIR / "outputs" / "tokens")

    def check_environment(self) -> dict[str, Any]:
        repo_exists = self.repo_path.exists()
        checkpoint_exists = self.checkpoint_path.is_file()
        token_script = PROJECT_DIR / "scripts" / "run_lingbot_official_tokens.py"

        return {
            "adapter": self.__class__.__name__,
            "model_name": self.config.get("model_name", "lingbot-map"),
            "repo_path": str(self.repo_path),
            "repo_exists": repo_exists,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": checkpoint_exists,
            "token_script": str(token_script),
            "token_script_exists": token_script.is_file(),
            "head_layers": DEFAULT_HEAD_LAYERS,
            "status": "ok" if repo_exists and checkpoint_exists and token_script.is_file() else "missing_dependency",
        }

    def load_model(self) -> Any:
        """No-op implementation for the common adapter interface.

        Lingbot-map is loaded inside the official extraction script so that the
        model path, preprocessing, streaming cache, and camera/depth heads stay
        aligned with the upstream demo behavior.
        """

        return None

    def build_token_command(
        self,
        image_folder: str | Path,
        output_dir: str | Path,
        *,
        layers: list[int] | None = None,
        token_slice: str = "patch",
        dtype: str = "float16",
        device: str = "cuda",
        use_sdpa: bool = True,
        num_scale_frames: int = 8,
        camera_num_iterations: int = 4,
        keyframe_interval: int = 1,
        kv_cache_sliding_window: int = 64,
        max_frame_num: int = 1024,
        python_executable: str | Path | None = None,
    ) -> list[str]:
        """Return the command used to run official Lingbot token extraction."""

        layer_arg = "all" if layers is None else ",".join(str(layer) for layer in layers)
        command = [
            str(python_executable or sys.executable),
            str(PROJECT_DIR / "scripts" / "run_lingbot_official_tokens.py"),
            "--image-folder",
            str(Path(image_folder).resolve()),
            "--output-dir",
            str(Path(output_dir).resolve()),
            "--lingbot-repo",
            str(self.repo_path.resolve()),
            "--model-path",
            str(self.checkpoint_path.resolve()),
            "--device",
            device,
            "--layers",
            layer_arg,
            "--token-slice",
            token_slice,
            "--dtype",
            dtype,
            "--num-scale-frames",
            str(num_scale_frames),
            "--camera-num-iterations",
            str(camera_num_iterations),
            "--keyframe-interval",
            str(keyframe_interval),
            "--kv-cache-sliding-window",
            str(kv_cache_sliding_window),
            "--max-frame-num",
            str(max_frame_num),
        ]
        command.append("--use-sdpa" if use_sdpa else "--no-use-sdpa")
        return command

    def extract_tokens(
        self,
        input_sequence: Any,
        condition_id: str,
        **kwargs: Any,
    ) -> TokenBundle:
        """Run official token extraction and return a path-based TokenBundle.

        ``input_sequence`` can be an image-folder path or a mapping with an
        ``image_folder`` key. Tokens are stored on disk; the returned bundle
        points to those files instead of loading multi-GB arrays into memory.
        """

        image_folder = self._coerce_image_folder(input_sequence)
        sequence_id = kwargs.pop("sequence_id", self._infer_sequence_id(image_folder))
        output_dir = Path(
            kwargs.pop(
                "output_dir",
                self.token_output_root / sequence_id / condition_id,
            )
        )

        command = self.build_token_command(
            image_folder,
            output_dir,
            layers=kwargs.pop("layers", DEFAULT_HEAD_LAYERS),
            token_slice=kwargs.pop("token_slice", "patch"),
            dtype=kwargs.pop("dtype", "float16"),
            device=kwargs.pop("device", "cuda"),
            use_sdpa=kwargs.pop("use_sdpa", True),
            num_scale_frames=kwargs.pop("num_scale_frames", 8),
            camera_num_iterations=kwargs.pop("camera_num_iterations", 4),
            keyframe_interval=kwargs.pop("keyframe_interval", 1),
            kv_cache_sliding_window=kwargs.pop("kv_cache_sliding_window", 64),
            max_frame_num=kwargs.pop("max_frame_num", 1024),
            python_executable=kwargs.pop("python_executable", None),
        )

        run = bool(kwargs.pop("run", True))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown LingbotAdapter.extract_tokens kwargs: {unknown}")

        if run:
            subprocess.run(command, check=True)
            return self.load_token_bundle(output_dir, condition_id=condition_id, sequence_id=sequence_id)

        return TokenBundle(
            model_name="lingbot-map",
            sequence_id=sequence_id,
            condition_id=condition_id,
            frame_paths=[],
            layer_tokens={},
            metadata={
                "status": "dry_run",
                "command": command,
                "output_dir": str(output_dir.resolve()),
            },
        )

    def load_token_bundle(
        self,
        output_dir: str | Path,
        *,
        condition_id: str = "default",
        sequence_id: str | None = None,
    ) -> TokenBundle:
        """Adapt a saved Lingbot token extraction directory to TokenBundle."""

        output_path = Path(output_dir)
        metadata_path = output_path if output_path.name == "metadata.json" else output_path / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Lingbot token metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        params = metadata.get("params", {})
        layout = metadata.get("token_layout", {})
        token_slice = params.get("token_slice", "patch")
        shape_by_layer = layout.get("shape_per_file", {})
        token_paths = metadata.get("outputs", {}).get("tokens", {})

        layer_tokens: dict[str, dict[str, Any]] = {}
        for label, token_path in token_paths.items():
            token_key = f"{token_slice}_tokens"
            layer_tokens[label] = {
                token_key: token_path,
                "shape": shape_by_layer.get(label),
                "patch_start_idx": layout.get("patch_start_idx"),
                "extra": {
                    "layer_label": label,
                    "layer_index": self._parse_layer_index(label),
                    "token_slice": token_slice,
                    "source": metadata.get("source"),
                },
            }

        inferred_sequence_id = sequence_id or self._sequence_id_from_metadata(metadata)
        return TokenBundle(
            model_name=metadata.get("model", "lingbot-map"),
            sequence_id=inferred_sequence_id,
            condition_id=condition_id,
            frame_paths=list(metadata.get("frame_paths", [])),
            layer_tokens=layer_tokens,
            output_predictions=None,
            metadata={
                "source_metadata": str(metadata_path.resolve()),
                "image_folder": metadata.get("image_folder"),
                "frame_count": metadata.get("frame_count"),
                "params": params,
                "token_layout": layout,
                "outputs": metadata.get("outputs", {}),
            },
        )

    def _coerce_image_folder(self, input_sequence: Any) -> Path:
        if isinstance(input_sequence, (str, Path)):
            return Path(input_sequence).resolve()
        if isinstance(input_sequence, dict):
            for key in ("image_folder", "images", "path"):
                value = input_sequence.get(key)
                if value:
                    return Path(value).resolve()
        raise TypeError("input_sequence must be an image-folder path or a dict with image_folder")

    def _infer_sequence_id(self, image_folder: Path) -> str:
        return image_folder.parent.name if image_folder.name == "images" else image_folder.name

    def _sequence_id_from_metadata(self, metadata: dict[str, Any]) -> str:
        image_folder = metadata.get("image_folder")
        if image_folder:
            return self._infer_sequence_id(Path(image_folder))
        return "unknown_sequence"

    def _parse_layer_index(self, label: str) -> int | None:
        prefix = "layer_"
        if label.startswith(prefix):
            try:
                return int(label[len(prefix) :])
            except ValueError:
                return None
        return None
