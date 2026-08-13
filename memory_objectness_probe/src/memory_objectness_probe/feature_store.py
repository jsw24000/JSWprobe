from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .dataset import SampleRecord
from .memory_hooks import SLOT_INDICES, SLOT_NAMES
from .utils import (
    EXPECTED_VARIANTS,
    ensure_dir,
    finite_stats,
    stable_sample_id,
    torch_load_cpu,
    write_csv,
    write_json,
    write_jsonl,
)


def make_feature_payload(
    *,
    sample: SampleRecord,
    memory_tokens: Mapping[str, torch.Tensor],
    run_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    tokens = {k: v.detach().cpu().float().contiguous() for k, v in memory_tokens.items()}
    return {
        "sample_id": sample.sample_id,
        "base_scene_id": sample.base_scene_id,
        "variant_id": sample.variant_id,
        "probe_frame": int(sample.probe_frame),
        "target_present": int(sample.target_present),
        "target_position_id": int(sample.target_position_id),
        "target_world_position": sample.target_world_position,
        "probe_mask_area": sample.frame15_target_mask_area,
        "probe_visible_ratio": sample.frame15_target_visible_ratio,
        "memory_tokens": tokens,
        "slot_names": SLOT_NAMES,
        "slot_indices": SLOT_INDICES,
        "frame_id": int(sample.probe_frame),
        "local_window_size": int(run_metadata.get("local_window_size", -1)),
        "flush_frames_used": int(run_metadata.get("flush_frames_used", 0)),
        "checkpoint": str(run_metadata.get("checkpoint_path")),
        "checkpoint_sha256": run_metadata.get("checkpoint_sha256"),
        "model_config": dict(run_metadata.get("model_config", {})),
        "run_metadata": dict(run_metadata),
    }


def save_feature(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(dict(payload), path)


def validate_feature_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"valid": False, "error": "missing_file"}
    try:
        payload = torch_load_cpu(path)
        tokens = payload.get("memory_tokens", {})
        if not tokens:
            return {"valid": False, "error": "missing_memory_tokens"}
        shapes = {}
        finite = True
        for layer, tensor in tokens.items():
            shapes[layer] = list(tensor.shape)
            if tensor.ndim != 2 or tensor.shape[0] != 6:
                return {"valid": False, "error": f"bad_shape_{layer}_{tuple(tensor.shape)}"}
            finite = finite and bool(torch.isfinite(tensor).all())
        if not finite:
            return {"valid": False, "error": "non_finite_tokens"}
        return {"valid": True, "available_layers": sorted(tokens.keys()), "layer_shapes": shapes}
    except Exception as exc:
        return {"valid": False, "error": repr(exc)}


def feature_manifest_row(
    *,
    sample: SampleRecord,
    feature_path: str | Path,
    status: str,
    error: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "sample_id": sample.sample_id,
        "base_scene_id": sample.base_scene_id,
        "variant_id": sample.variant_id,
        "feature_path": str(Path(feature_path).resolve()),
        "target_present": int(sample.target_present),
        "target_position_id": int(sample.target_position_id),
        "probe_frame": int(sample.probe_frame),
        "extraction_status": status,
        "error": error or "",
    }
    if payload:
        tokens = payload.get("memory_tokens", {})
        row.update(
            {
                "available_layers": sorted(tokens.keys()),
                "layer_shapes": {k: list(v.shape) for k, v in tokens.items()},
                "slot_names": payload.get("slot_names", []),
                "flush_frames": payload.get("flush_frames_used", 0),
            }
        )
    return row


def write_feature_manifests(feature_root: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    feature_root = Path(feature_root)
    write_jsonl(feature_root / "feature_manifest.jsonl", rows)
    write_csv(feature_root / "feature_manifest.csv", rows)


def summarize_feature_manifest(feature_root: str | Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    feature_root = Path(feature_root)
    successful = [row for row in rows if row.get("extraction_status") == "success"]
    failed = [row for row in rows if row.get("extraction_status") != "success"]
    variant_counts: Dict[str, int] = {v: 0 for v in EXPECTED_VARIANTS}
    layer_shapes: Dict[str, List[int]] = {}
    norms: List[float] = []
    finite_all = True
    flush_counts: Dict[str, int] = {}
    probe_frame_errors: List[str] = []
    duplicate_paths: Dict[str, int] = {}
    seen_paths: Dict[str, int] = {}
    payloads_by_sample: Dict[str, Dict[str, Any]] = {}

    for row in successful:
        variant_counts[row["variant_id"]] = variant_counts.get(row["variant_id"], 0) + 1
        flush_counts[str(row.get("flush_frames", 0))] = flush_counts.get(str(row.get("flush_frames", 0)), 0) + 1
        path = str(row["feature_path"])
        seen_paths[path] = seen_paths.get(path, 0) + 1
        try:
            payload = torch_load_cpu(path)
            payloads_by_sample[str(payload["sample_id"])] = payload
            if int(payload.get("frame_id", -1)) != int(payload.get("probe_frame", -2)):
                probe_frame_errors.append(str(payload.get("sample_id")))
            for layer, tensor in payload.get("memory_tokens", {}).items():
                layer_shapes.setdefault(layer, list(tensor.shape))
                finite_all = finite_all and bool(torch.isfinite(tensor).all())
                norms.append(float(torch.linalg.vector_norm(tensor.float()).item()))
        except Exception:
            finite_all = False

    duplicate_paths = {k: v for k, v in seen_paths.items() if v > 1}

    by_base: Dict[str, List[str]] = {}
    for row in successful:
        by_base.setdefault(str(row["base_scene_id"]), []).append(str(row["variant_id"]))
    complete_bases = [base for base, variants in by_base.items() if sorted(variants) == sorted(EXPECTED_VARIANTS)]

    pairwise_diffs: List[Dict[str, Any]] = []
    for base_scene_id, variants in by_base.items():
        absent_id = stable_sample_id(base_scene_id, "absent")
        if absent_id not in payloads_by_sample:
            continue
        absent = payloads_by_sample[absent_id]
        for variant in ("pos1", "pos2", "pos3", "pos4"):
            sample_id = stable_sample_id(base_scene_id, variant)
            if sample_id not in payloads_by_sample:
                continue
            present = payloads_by_sample[sample_id]
            for layer, absent_tensor in absent["memory_tokens"].items():
                if layer not in present["memory_tokens"]:
                    continue
                a = absent_tensor.flatten().float()
                b = present["memory_tokens"][layer].flatten().float()
                l2 = float(torch.linalg.vector_norm(a - b).item())
                denom = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
                cosine = float(torch.dot(a, b).item() / denom) if denom > 0 else float("nan")
                pairwise_diffs.append(
                    {
                        "base_scene_id": base_scene_id,
                        "variant_id": variant,
                        "layer": layer,
                        "l2": l2,
                        "cosine": cosine,
                    }
                )

    summary = {
        "total_rows": len(rows),
        "success_count": len(successful),
        "failure_count": len(failed),
        "variant_counts_success": variant_counts,
        "layer_shapes": layer_shapes,
        "all_features_finite": finite_all,
        "feature_norm_stats": finite_stats(norms),
        "flush_frame_usage": flush_counts,
        "complete_base_scenes": sorted(complete_bases),
        "num_complete_base_scenes": len(complete_bases),
        "probe_frame_errors": probe_frame_errors,
        "duplicate_feature_paths": duplicate_paths,
        "failed_samples": failed,
        "absent_vs_present_diff_stats": {
            "count": len(pairwise_diffs),
            "l2": finite_stats([r["l2"] for r in pairwise_diffs]),
            "cosine": finite_stats([r["cosine"] for r in pairwise_diffs if math.isfinite(r["cosine"])]),
        },
        "absent_vs_present_diffs": pairwise_diffs,
    }
    write_json(feature_root / "extraction_summary.json", summary)
    return summary
