#!/usr/bin/env python3
"""Run model adapters over controlled ScanNet manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from controlled_common import (
    PROJECT_DIR,
    add_common_filters,
    filtered_manifest_records,
    load_config,
    load_json,
    output_root,
    project_name,
    record_matches_filters,
    resolve_index_path,
    resolve_manifest_path,
    save_json,
    save_resolved_config,
    setup_logging,
)

from adapters.lingbot_adapter import LingbotAdapter
from adapters.token_schema import TokenBundle, save_token_bundle
from adapters.vggt_adapter import VGGTAdapter


def parse_layers(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    import re

    layers: list[int] = []
    for value in values:
        layers.extend(int(part) for part in re.split(r"[, ]+", value) if part)
    return layers or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract model tokens for controlled ScanNet manifests.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "controlled_scannet_v1.yaml")
    parser.add_argument("--manifest-index", type=Path, default=PROJECT_DIR / "outputs" / "manifests" / "controlled_scannet_v1" / "index.json")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--layers", nargs="+", default=None, help="Override extraction layers, e.g. --layers 17 or --layers 4,11,17,23")
    parser.add_argument("--force", action="store_true", help="Re-run extraction even if a matching token bundle exists.")
    parser.add_argument("--dry-run", action="store_true")
    add_common_filters(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config, "run_controlled_extraction")
    project = project_name(config)
    index_path = resolve_index_path(args.manifest_index, config)
    index = load_json(index_path)
    records = filtered_manifest_records(index, args)

    extraction_cfg = dict(config.get("extraction", {}))
    cli_layers = parse_layers(args.layers)
    if cli_layers is not None:
        extraction_cfg["layers"] = cli_layers
    models = args.models or list(extraction_cfg.get("models", ["lingbot-map"]))
    device = args.device or extraction_cfg.get("device", "cuda")
    skip_existing = bool(extraction_cfg.get("skip_existing", True)) and not args.force

    logger.info("Selected %d manifests from %s", len(records), index_path)
    logger.info("Models: %s", ", ".join(models))

    summaries: list[dict[str, Any]] = []
    for record in records:
        manifest_path = resolve_manifest_path(record["path"], index_path)
        manifest = load_json(manifest_path)
        for model_name in models:
            summary = process_manifest_model(
                manifest,
                manifest_path=manifest_path,
                model_name=model_name,
                config=config,
                extraction_cfg=extraction_cfg,
                device=device,
                project=project,
                dry_run=args.dry_run,
                skip_existing=skip_existing,
            )
            summaries.append(summary)
            logger.info(
                "%s | %s | %s | %s",
                summary.get("status"),
                model_name,
                manifest.get("scene_id"),
                manifest.get("condition_id"),
            )

    if not args.dry_run:
        run_dir = output_root(config) / "tokens" / project
        save_resolved_config(config, run_dir)
        save_json(run_dir / "last_extraction_summary.json", {"runs": summaries})
    else:
        print(json.dumps({"dry_run": True, "runs": summaries}, ensure_ascii=False, indent=2))
    return 0


def process_manifest_model(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    model_name: str,
    config: dict[str, Any],
    extraction_cfg: dict[str, Any],
    device: str,
    project: str,
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    token_dir = (
        output_root(config)
        / "tokens"
        / project
        / model_name
        / manifest["scene_id"]
        / manifest["setting_name"]
        / manifest["condition_id"]
    )
    bundle_json = token_dir / "token_bundle.json"
    if skip_existing and bundle_json.is_file() and not dry_run:
        skip_status = existing_bundle_status(
            token_dir,
            required_layers=[int(value) for value in extraction_cfg.get("layers", [])],
        )
        if skip_status == "ok":
            return {
                "status": "skipped_existing",
                "model_name": model_name,
                "token_bundle": str(bundle_json),
                "condition_id": manifest["condition_id"],
            }

    if model_name == "lingbot-map":
        return run_lingbot_extraction(
            manifest,
            manifest_path=manifest_path,
            token_dir=token_dir,
            config=config,
            extraction_cfg=extraction_cfg,
            device=device,
            dry_run=dry_run,
        )
    if model_name == "vggt":
        return save_skipped_bundle(
            manifest,
            token_dir,
            model_name=model_name,
            reason="VGGT adapter is interface-only in this repository state.",
            dry_run=dry_run,
        )
    return save_skipped_bundle(
        manifest,
        token_dir,
        model_name=model_name,
        reason=f"Unknown model adapter: {model_name}",
        dry_run=dry_run,
    )


def run_lingbot_extraction(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    token_dir: Path,
    config: dict[str, Any],
    extraction_cfg: dict[str, Any],
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    sequence_dir = (
        output_root(config)
        / "controlled_sequences"
        / project_name(config)
        / manifest["scene_id"]
        / manifest["setting_name"]
        / manifest["condition_id"]
    )
    image_folder = sequence_dir / "images"

    if not dry_run:
        materialize_image_sequence(manifest, image_folder, relative_base=PROJECT_DIR)
        save_json(sequence_dir / "sequence_manifest.json", manifest)

    adapter = LingbotAdapter(config)
    kwargs = {
        "sequence_id": controlled_sequence_id(manifest),
        "output_dir": token_dir,
        "layers": [int(value) for value in extraction_cfg.get("layers", [4, 11, 17, 23])],
        "token_slice": extraction_cfg.get("token_slice", "patch"),
        "dtype": extraction_cfg.get("save_tokens_dtype", "float16"),
        "device": device,
        "use_sdpa": bool(extraction_cfg.get("use_sdpa", True)),
        "num_scale_frames": int(extraction_cfg.get("num_scale_frames", 8)),
        "camera_num_iterations": int(extraction_cfg.get("camera_num_iterations", 4)),
        "keyframe_interval": int(extraction_cfg.get("keyframe_interval", 1)),
        "kv_cache_sliding_window": int(extraction_cfg.get("kv_cache_sliding_window", 64)),
        "max_frame_num": int(extraction_cfg.get("max_frame_num", 1024)),
        "python_executable": extraction_cfg.get("python_executable"),
        "run": not dry_run,
    }

    try:
        bundle = adapter.extract_tokens(image_folder, condition_id=manifest["condition_id"], **kwargs)
    except Exception as exc:
        if dry_run:
            command = adapter.build_token_command(
                image_folder,
                token_dir,
                layers=kwargs["layers"],
                token_slice=kwargs["token_slice"],
                dtype=kwargs["dtype"],
                device=kwargs["device"],
                use_sdpa=kwargs["use_sdpa"],
                num_scale_frames=kwargs["num_scale_frames"],
                camera_num_iterations=kwargs["camera_num_iterations"],
                keyframe_interval=kwargs["keyframe_interval"],
                kv_cache_sliding_window=kwargs["kv_cache_sliding_window"],
                max_frame_num=kwargs["max_frame_num"],
                python_executable=kwargs["python_executable"],
            )
            return {
                "status": "dry_run",
                "model_name": "lingbot-map",
                "condition_id": manifest["condition_id"],
                "image_folder": str(image_folder),
                "output_dir": str(token_dir),
                "command": command,
            }
        return save_skipped_bundle(manifest, token_dir, model_name="lingbot-map", reason=str(exc), dry_run=False)

    bundle = enrich_bundle_from_manifest(bundle, manifest, manifest_path=manifest_path)
    if dry_run:
        result = bundle.to_dict()
        result["status"] = "dry_run"
        return result

    token_dir.mkdir(parents=True, exist_ok=True)
    bundle_json = save_token_bundle(bundle, token_dir / "token_bundle.json")
    bundle_pt = save_token_bundle(bundle, token_dir / "token_bundle.pt")
    summary = token_summary(bundle, bundle_json=bundle_json, bundle_pt=bundle_pt)
    save_json(token_dir / "token_summary.json", summary)
    (token_dir / "extraction_log.txt").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def materialize_image_sequence(manifest: dict[str, Any], image_folder: Path, *, relative_base: Path) -> None:
    """Create a symlink-only image folder ordered by manifest time index."""

    image_folder.mkdir(parents=True, exist_ok=True)
    for frame in manifest.get("frames", []):
        src = resolve_data_path(frame["rgb_path"], relative_base=relative_base)
        suffix = src.suffix or ".jpg"
        dst = image_folder / f"{int(frame['t']):06d}{suffix}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def resolve_data_path(path_value: str, *, relative_base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (relative_base / path).resolve()


def existing_bundle_status(token_dir: Path, *, required_layers: list[int] | None = None) -> str | None:
    """Return status for an existing extraction, treating failures as rerunnable."""

    summary_path = token_dir / "token_summary.json"
    if summary_path.is_file():
        try:
            summary = load_json(summary_path)
        except Exception:
            return None
        status = summary.get("status")
        if status in {None, "ok"} and not has_required_layers(summary.get("layers", {}), required_layers):
            return None
        return "ok" if status in {None, "ok"} else str(status)

    bundle_path = token_dir / "token_bundle.json"
    if bundle_path.is_file():
        try:
            bundle = load_json(bundle_path)
        except Exception:
            return None
        metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
        status = metadata.get("status", bundle.get("status") if isinstance(bundle, dict) else None)
        if status in {None, "ok"} and not has_required_layers(bundle.get("layer_tokens", {}), required_layers):
            return None
        return "ok" if status in {None, "ok"} else str(status)

    return None


def has_required_layers(layers: dict[str, Any], required_layers: list[int] | None) -> bool:
    if not required_layers:
        return True
    available = set(layers)
    required = {f"layer_{int(layer):02d}" for layer in required_layers}
    return required.issubset(available)


def enrich_bundle_from_manifest(bundle: TokenBundle, manifest: dict[str, Any], *, manifest_path: Path) -> TokenBundle:
    frames = manifest.get("frames", [])
    bundle.scene_id = manifest.get("scene_id")
    bundle.setting_name = manifest.get("setting_name")
    bundle.condition_id = manifest.get("condition_id", bundle.condition_id)
    bundle.sequence_id = controlled_sequence_id(manifest)
    bundle.seq_len = manifest.get("seq_len")
    bundle.frame_paths = [frame.get("rgb_path") for frame in frames]
    bundle.frame_ids = [frame.get("source_frame_id") for frame in frames]
    bundle.metadata.setdefault("manifest_path", str(manifest_path))
    bundle.metadata.setdefault("manifest_metadata", manifest.get("metadata", {}))
    bundle.metadata.setdefault("missing_fields", missing_fields(bundle))
    return bundle


def controlled_sequence_id(manifest: dict[str, Any]) -> str:
    return "/".join(
        [
            manifest.get("scene_id", "unknown_scene"),
            manifest.get("setting_name", "unknown_setting"),
            manifest.get("condition_id", "unknown_condition"),
        ]
    )


def missing_fields(bundle: TokenBundle) -> dict[str, Any]:
    expected_layer_tokens = ["patch_tokens", "camera_tokens", "register_tokens", "memory_tokens"]
    missing_by_layer: dict[str, list[str]] = {}
    for layer_name, layer_values in bundle.layer_tokens.items():
        if not isinstance(layer_values, dict):
            missing_by_layer[layer_name] = expected_layer_tokens
            continue
        missing = [name for name in expected_layer_tokens if name not in layer_values]
        if missing:
            missing_by_layer[layer_name] = missing
    return {
        "layer_tokens": missing_by_layer,
        "attention": not bool(bundle.attention),
        "geometry_outputs": not bool(bundle.geometry_outputs or bundle.output_predictions),
    }


def token_summary(bundle: TokenBundle, *, bundle_json: Path, bundle_pt: Path | None) -> dict[str, Any]:
    return {
        "status": bundle.metadata.get("status", "ok"),
        "model_name": bundle.model_name,
        "scene_id": bundle.scene_id,
        "setting_name": bundle.setting_name,
        "condition_id": bundle.condition_id,
        "seq_len": bundle.seq_len,
        "frame_count": len(bundle.frame_paths),
        "token_bundle": str(bundle_json),
        "token_bundle_pt": str(bundle_pt) if bundle_pt else None,
        "layers": {
            layer: {
                "available_token_fields": sorted(
                    key for key in values if isinstance(key, str) and key.endswith("_tokens")
                )
                if isinstance(values, dict)
                else [],
                "shape": values.get("shape") if isinstance(values, dict) else None,
            }
            for layer, values in bundle.layer_tokens.items()
        },
        "missing_fields": missing_fields(bundle),
    }


def save_skipped_bundle(
    manifest: dict[str, Any],
    token_dir: Path,
    *,
    model_name: str,
    reason: str,
    dry_run: bool,
) -> dict[str, Any]:
    summary = {
        "status": "dry_run" if dry_run else "skipped",
        "model_name": model_name,
        "scene_id": manifest.get("scene_id"),
        "setting_name": manifest.get("setting_name"),
        "condition_id": manifest.get("condition_id"),
        "seq_len": manifest.get("seq_len"),
        "reason": reason,
        "output_dir": str(token_dir),
    }
    if dry_run:
        return summary
    token_dir.mkdir(parents=True, exist_ok=True)
    bundle = TokenBundle(
        model_name=model_name,
        sequence_id=controlled_sequence_id(manifest),
        scene_id=manifest.get("scene_id"),
        setting_name=manifest.get("setting_name"),
        condition_id=manifest.get("condition_id"),
        seq_len=manifest.get("seq_len"),
        frame_paths=[frame.get("rgb_path") for frame in manifest.get("frames", [])],
        frame_ids=[frame.get("source_frame_id") for frame in manifest.get("frames", [])],
        metadata={"status": "skipped", "reason": reason, "manifest_metadata": manifest.get("metadata", {})},
    )
    bundle_path = save_token_bundle(bundle, token_dir / "token_bundle.json")
    summary["token_bundle"] = str(bundle_path)
    save_json(token_dir / "token_summary.json", summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
