#!/usr/bin/env python3
"""Validate controlled_16_staircase_exp_v4 manifests and optional tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from controlled_common import (
    PROJECT_DIR,
    load_config,
    load_json,
    output_root,
    project_name,
    resolve_index_path,
    resolve_manifest_path,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate staircase v4 manifests and token endpoints.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "controlled_16_staircase_exp_v4.yaml")
    parser.add_argument(
        "--manifest-index",
        type=Path,
        default=PROJECT_DIR / "outputs" / "manifests" / "controlled_16_staircase_exp_v4" / "index.json",
    )
    parser.add_argument("--model", default="lingbot-map")
    parser.add_argument("--setting", default="controlled_16_staircase")
    parser.add_argument("--check-tokens", action="store_true")
    parser.add_argument("--require-tokens", action="store_true")
    parser.add_argument("--write-summary", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    index_path = resolve_index_path(args.manifest_index, config)
    index = load_json(index_path)
    manifests = [
        load_json(resolve_manifest_path(record["path"], index_path))
        for record in index.get("manifests", [])
        if record.get("setting_name") == args.setting
    ]
    validation = validate_manifests(config, manifests, setting_name=args.setting)
    if args.check_tokens or args.require_tokens:
        validation["token_checks"] = validate_tokens(
            config,
            manifests,
            model=args.model,
            require_tokens=args.require_tokens,
            setting_name=args.setting,
        )
    out_dir = output_root(config) / "manifests" / project_name(config)
    if args.write_summary:
        prefix = "staircase" if args.setting == "controlled_16_staircase" else args.setting
        save_json(out_dir / f"{prefix}_manifest_summary.json", build_total_manifest_summary(manifests, setting_name=args.setting))
        save_json(out_dir / f"{prefix}_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation.get("status") == "ok" else 1


def validate_manifests(config: dict[str, Any], manifests: list[dict[str, Any]], *, setting_name: str) -> dict[str, Any]:
    errors: list[str] = []
    target = int(config.get("target_frame_id", 600))
    total = int(config.get("sequence_total_length", 16))
    prefix = int(config.get("prefix_length", total - 1))
    refs = config.get("reference_sequences", {})
    reference_names = reference_names_for_setting(config, setting_name)
    unrelated_ids = [int(value) for value in refs[reference_names["unrelated"]]["frame_ids"]]
    relevant_ids = [int(value) for value in refs[reference_names["relevant"]]["frame_ids"]]
    replacement_order = staircase_replacement_order(config, setting_name)
    expected_conditions = [f"staircase_k{k:02d}" for k in range(prefix + 1)]
    by_condition = {manifest.get("condition_id"): manifest for manifest in manifests}

    if len(manifests) != prefix + 1:
        errors.append(f"expected {prefix + 1} manifests, found {len(manifests)}")
    missing = [condition for condition in expected_conditions if condition not in by_condition]
    if missing:
        errors.append(f"missing conditions: {missing}")

    target_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for k, condition in enumerate(expected_conditions):
        manifest = by_condition.get(condition)
        if not manifest:
            continue
        frames = manifest.get("frames", [])
        frame_ids = [int(frame.get("original_order_index")) for frame in frames]
        expected_prefix = expected_staircase_prefix(
            k=k,
            prefix_length=prefix,
            unrelated_ids=unrelated_ids,
            relevant_ids=relevant_ids,
            replacement_order=replacement_order,
        )
        expected = expected_prefix + [target]
        if len(frames) != total:
            errors.append(f"{condition}: expected {total} frames, found {len(frames)}")
        if frame_ids != expected:
            errors.append(f"{condition}: frame ids {frame_ids} != expected {expected}")
        if frame_ids and frame_ids[-1] != target:
            errors.append(f"{condition}: last frame is {frame_ids[-1]}, expected {target}")
        target_paths.add(str(frames[-1].get("rgb_path")) if frames else "")
        relevant_positions = [idx for idx, frame in enumerate(frames[:-1]) if frame.get("is_relevant_history")]
        expected_positions = expected_relevant_positions(k=k, prefix_length=prefix, replacement_order=replacement_order)
        if relevant_positions != expected_positions:
            errors.append(f"{condition}: relevant positions {relevant_positions} != expected {expected_positions}")
        if sum(1 for frame in frames[:-1] if frame.get("is_relevant_history")) != k:
            errors.append(f"{condition}: relevant count mismatch")
        rows.append(
            {
                "condition_id": condition,
                "k": k,
                "frame_ids": frame_ids,
                "relevant_positions": relevant_positions,
                "target_rgb_path": frames[-1].get("rgb_path") if frames else None,
            }
        )

    if len(target_paths - {""}) != 1:
        errors.append(f"target frame paths are not identical: {sorted(target_paths)}")
    if (
        by_condition.get("staircase_k00")
        and [int(frame["original_order_index"]) for frame in by_condition["staircase_k00"]["frames"]] != unrelated_ids
    ):
        errors.append(f"staircase_k00 does not exactly match v3 {reference_names['unrelated']}")
    if (
        by_condition.get("staircase_k15")
        and [int(frame["original_order_index"]) for frame in by_condition["staircase_k15"]["frames"]] != relevant_ids
    ):
        errors.append(f"staircase_k15 does not exactly match v3 {reference_names['relevant']}")

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "setting_name": setting_name,
        "replacement_order": replacement_order,
        "reference_names": reference_names,
        "manifest_count": len(manifests),
        "conditions": rows,
    }


def validate_tokens(
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
    *,
    model: str,
    require_tokens: bool,
    setting_name: str,
) -> dict[str, Any]:
    root = output_root(config)
    project = project_name(config)
    scene_id = manifests[0]["scene_id"] if manifests else "scene0002_00"
    layers = [int(value) for value in config.get("layers", [4, 11, 17, 23])]
    reference_names = reference_names_for_setting(config, setting_name)
    missing: list[str] = []
    bundle_paths: dict[str, Path] = {}
    for k in range(16):
        condition = f"staircase_k{k:02d}"
        path = root / "tokens" / project / model / scene_id / setting_name / condition / "token_bundle.json"
        if path.is_file():
            bundle_paths[condition] = path
        else:
            missing.append(str(path))
    baseline_paths = {
        "relevant": baseline_bundle_path(root, model, scene_id, reference_names["relevant"]),
        "unrelated": baseline_bundle_path(root, model, scene_id, reference_names["unrelated"]),
    }
    for path in baseline_paths.values():
        if not path.is_file():
            missing.append(str(path))
    if missing:
        return {
            "status": "failed" if require_tokens else "skipped",
            "reason": "missing token bundles",
            "missing": missing,
        }

    shape_checks: dict[str, Any] = {}
    endpoint_checks: dict[str, Any] = {}
    bundles = {condition: load_json(path) for condition, path in bundle_paths.items()}
    baselines = {name: load_json(path) for name, path in baseline_paths.items()}
    for layer in layers:
        layer_label = f"layer_{layer:02d}"
        shapes = {
            condition: tuple(bundle.get("layer_tokens", {}).get(layer_label, {}).get("shape", []))
            for condition, bundle in bundles.items()
        }
        shape_checks[layer_label] = {
            "status": "ok" if len(set(shapes.values())) == 1 else "failed",
            "shapes": {condition: list(shape) for condition, shape in sorted(shapes.items())},
        }
        k00_ref = bundles["staircase_k00"]["layer_tokens"][layer_label]["patch_tokens"]
        k15_ref = bundles["staircase_k15"]["layer_tokens"][layer_label]["patch_tokens"]
        unrelated_ref = baselines["unrelated"]["layer_tokens"][layer_label]["patch_tokens"]
        relevant_ref = baselines["relevant"]["layer_tokens"][layer_label]["patch_tokens"]
        endpoint_checks[layer_label] = {
            "k00_vs_v3_unrelated": final_frame_diff(k00_ref, unrelated_ref),
            "k15_vs_v3_relevant": final_frame_diff(k15_ref, relevant_ref),
        }
    status = "ok"
    if any(check["status"] != "ok" for check in shape_checks.values()):
        status = "failed"
    return {
        "status": status,
        "reference_names": reference_names,
        "shape_checks": shape_checks,
        "endpoint_checks": endpoint_checks,
    }


def baseline_bundle_path(root: Path, model: str, scene_id: str, condition_id: str) -> Path:
    return (
        root
        / "tokens"
        / "controlled_16_exp_v3"
        / model
        / scene_id
        / "prefix_induced_modulation"
        / condition_id
        / "token_bundle.json"
    )


def final_frame_diff(a_path: str, b_path: str) -> dict[str, Any]:
    a = np.load(a_path, mmap_mode="r")[-1].astype(np.float32)
    b = np.load(b_path, mmap_mode="r")[-1].astype(np.float32)
    diff = a - b
    return {
        "shape": list(diff.shape),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "l2": float(np.linalg.norm(diff.reshape(-1))),
    }


def build_total_manifest_summary(manifests: list[dict[str, Any]], *, setting_name: str) -> dict[str, Any]:
    sequences: dict[str, Any] = {}
    for manifest in sorted(manifests, key=lambda item: item.get("condition_id", "")):
        condition = str(manifest.get("condition_id"))
        frames = []
        for frame in manifest.get("frames", []):
            frames.append(
                {
                    "t": frame.get("t"),
                    "source_frame_id": frame.get("source_frame_id"),
                    "frame_id": frame.get("original_order_index"),
                    "rgb_path": frame.get("rgb_path"),
                    "source_sequence": frame.get("source_sequence"),
                    "source_reference": frame.get("source_reference"),
                    "is_relevant_history": frame.get("is_relevant_history"),
                    "is_target_frame": frame.get("is_target_frame"),
                }
            )
        sequences[condition] = {
            "k": manifest.get("metadata", {}).get("staircase_k"),
            "num_relevant_history_frames": manifest.get("metadata", {}).get("num_relevant_history_frames"),
            "frames": frames,
        }
    return {
        "experiment_name": "controlled_16_staircase_exp_v4",
        "setting_name": setting_name,
        "sequence_definition": sequence_definition_for_setting(config=None, setting_name=setting_name),
        "sequences": sequences,
    }


def reference_names_for_setting(config: dict[str, Any], setting_name: str) -> dict[str, str]:
    setting_cfg = config.get("sequence", {}).get("settings", {}).get(setting_name, {})
    return {
        "relevant": str(setting_cfg.get("relevant_sequence_name", "B1_local_relevant")),
        "unrelated": str(setting_cfg.get("unrelated_sequence_name", "C")),
        "x": str(setting_cfg.get("x_sequence_name", "X")),
    }


def staircase_replacement_order(config: dict[str, Any], setting_name: str) -> str:
    setting_cfg = config.get("sequence", {}).get("settings", {}).get(setting_name, {})
    default = "prefix" if setting_name.endswith(("_reverse", "_reversed")) else "suffix"
    value = str(setting_cfg.get("replacement_order", default)).lower()
    if value not in {"prefix", "suffix"}:
        raise ValueError(f"Unsupported replacement_order for {setting_name}: {value}")
    return value


def expected_staircase_prefix(
    *,
    k: int,
    prefix_length: int,
    unrelated_ids: list[int],
    relevant_ids: list[int],
    replacement_order: str,
) -> list[int]:
    if replacement_order == "prefix":
        return relevant_ids[:k] + unrelated_ids[k:prefix_length]
    return unrelated_ids[: prefix_length - k] + relevant_ids[prefix_length - k : prefix_length]


def expected_relevant_positions(*, k: int, prefix_length: int, replacement_order: str) -> list[int]:
    if replacement_order == "prefix":
        return list(range(k))
    return list(range(prefix_length - k, prefix_length))


def sequence_definition_for_setting(config: dict[str, Any] | None, setting_name: str) -> str:
    unrelated_prefix = "X" if "x_to_B1" in setting_name else "U"
    if setting_name.endswith(("_reverse", "_reversed")):
        return f"S_k = R01..Rk, {unrelated_prefix}(k+1)..{unrelated_prefix}15, target_0600"
    return f"S_k = {unrelated_prefix}01..{unrelated_prefix}(15-k), R(16-k)..R15, target_0600"


if __name__ == "__main__":
    raise SystemExit(main())
