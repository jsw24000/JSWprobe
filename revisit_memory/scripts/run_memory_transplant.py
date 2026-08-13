#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.hooks.feature_hooks import AggregateFeatureCapture  # noqa: E402
from revisit_memory.scripts.run_reconstruction import (  # noqa: E402
    append_predictions,
    build_model,
    choose_dtype,
    consume_features,
    load_images_tensor,
    postprocess_pose_arrays,
    save_feature_store,
    set_seed,
)
from revisit_memory.transplant.cache_state import (  # noqa: E402
    SPECIAL_TOKEN_NAMES,
    clone_model_stream_state,
    clone_stream_state,
    local_cache_frame_ids_before_target,
    replace_all_trajectory_special,
    replace_selected_trajectory_records,
    restore_model_stream_state,
    special_frame_ids_before_target,
    state_layout_summary,
)
from revisit_memory.utils.io import (  # noqa: E402
    condition_dir,
    find_target_presence_mask,
    load_any_target_mask,
    load_config,
    output_root,
    project_path,
    target_object,
    write_csv,
    write_json,
    write_text,
)


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("run_memory_transplant.py must be run in the lingbot-map environment with torch.") from exc


def baseline_root(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["project"]["output_root"]) / "runs" / cfg["transplant"]["baseline_run_id"]


def condition_metadata(cfg: dict[str, Any], condition: str) -> dict[str, Any]:
    path = condition_dir(cfg, condition) / "metadata.json"
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_config_used(cfg: dict[str, Any]) -> None:
    path = output_root(cfg) / "config_used.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_baseline_predictions(cfg: dict[str, Any], condition: str) -> tuple[list[int], dict[str, np.ndarray]]:
    root = baseline_root(cfg)
    with np.load(root / "reconstruction" / condition / "inputs.npz", allow_pickle=True) as inp:
        frame_ids = inp["frame_ids"].astype(int).tolist()
    with np.load(root / "reconstruction" / condition / "predictions.npz", allow_pickle=True) as pred:
        arrays = {key: pred[key] for key in pred.files}
    return frame_ids, arrays


def load_baseline_features(cfg: dict[str, Any], condition: str) -> dict[str, Any]:
    torch = require_torch()
    path = baseline_root(cfg) / "features" / condition / "features.pt"
    return torch.load(path, map_location="cpu")


def source_visible_frames(cfg: dict[str, Any]) -> list[int]:
    cond = cfg["transplant"].get("source_visibility_condition", "seen_then_removed")
    metadata_visible = condition_metadata(cfg, cond).get("target", {}).get("actual_first_loop_visible_frames")
    if metadata_visible:
        return [int(x) for x in metadata_visible]
    first_start, first_end = [int(x) for x in cfg["frames"]["first_loop_visible"]]
    min_pixels = int(cfg["transplant"].get("target_mask_min_pixels", 1))
    target_id = int(target_object(cfg)["object_id"])
    visible = []
    for frame_id in range(first_start, first_end + 1):
        path = find_target_presence_mask(cfg, cond, frame_id, target_id)
        if path is None:
            continue
        mask = load_any_target_mask(path, target_id)
        if int(mask.sum()) >= min_pixels:
            visible.append(frame_id)
    if visible:
        return visible
    return list(range(first_start, first_end + 1))


def mask_area_for_frame(cfg: dict[str, Any], condition: str, frame_id: int) -> int:
    actual_counts = condition_metadata(cfg, condition).get("target", {}).get("actual_pixel_counts_by_frame", {})
    if str(frame_id) in actual_counts:
        return int(actual_counts[str(frame_id)])
    target_id = int(target_object(cfg)["object_id"])
    path = find_target_presence_mask(cfg, condition, frame_id, target_id)
    if path is None:
        return 0
    return int(load_any_target_mask(path, target_id).sum())


def choose_negative_control_frames(cfg: dict[str, Any], visible: list[int], special_frames: list[int]) -> list[int]:
    if not visible:
        return []
    cond = cfg["transplant"].get("source_visibility_condition", "seen_then_removed")
    special = set(int(x) for x in special_frames)
    visible_set = set(int(x) for x in visible)
    candidates = []
    for frame_id in special_frames:
        if frame_id not in special or frame_id in visible_set:
            continue
        if mask_area_for_frame(cfg, cond, frame_id) > 0:
            continue
        candidates.append(frame_id)
    after = sorted(fid for fid in candidates if fid > max(visible))
    before = sorted((fid for fid in candidates if fid < min(visible)), reverse=True)
    ordered = after + before
    return ordered[: len(visible)]


def store_output(
    cfg: dict[str, Any],
    variant: str,
    frame_id: int,
    output: dict[str, Any],
    capture: AggregateFeatureCapture,
    stores: dict[str, dict[str, Any]],
) -> None:
    append_predictions(stores[variant]["pred"], output, [frame_id])
    consume_features(capture, [frame_id], stores[variant]["features"], stores[variant]["memory"], {frame_id})
    if "images" in stores[variant]["pred"] and not bool(cfg.get("exports", {}).get("save_model_input_images", False)):
        stores[variant]["pred"].pop("images", None)


def finalize_variant_outputs(cfg: dict[str, Any], stores: dict[str, dict[str, Any]], image_hw: tuple[int, int]) -> None:
    torch = require_torch()
    root = output_root(cfg)
    for variant, payload in stores.items():
        recon_dir = root / "reconstruction" / variant
        feature_dir = root / "features" / variant
        recon_dir.mkdir(parents=True, exist_ok=True)
        pred = {key: np.stack(values, axis=0) for key, values in payload["pred"].items() if values}
        if "pose_enc" in pred:
            postprocess_pose_arrays(cfg, pred, image_hw)
        np.savez_compressed(recon_dir / "predictions.npz", frame_ids=np.asarray(payload["frame_ids"], dtype=np.int32), **pred)
        save_feature_store(feature_dir / "features.pt", payload["features"])
        torch.save(
            {
                "source_frame_ids": payload["memory"].get("source_frame_ids", []),
                "selected_layers": payload["memory"].get("selected_layers", []),
                "patch_start_idx": payload["memory"].get("patch_start_idx"),
                "token_grid_hw": payload["memory"].get("token_grid_hw"),
                "frame_special_tokens": payload["memory"].get("frame_special_tokens", {}),
                "global_special_tokens": payload["memory"].get("global_special_tokens", {}),
            },
            feature_dir / "replay_special_tokens_raw.pt",
        )


def save_variant_inputs(cfg: dict[str, Any], variants: list[str], target_frames: list[int]) -> None:
    root = output_root(cfg)
    base = baseline_root(cfg)
    query_condition = {
        "AA": "always_absent",
        "AR": "always_absent",
        "AS": "always_absent",
        "AN": "always_absent",
        "RR": "seen_then_removed",
        "RA": "seen_then_removed",
    }
    for variant in variants:
        cond = query_condition.get(variant, "always_absent")
        with np.load(base / "reconstruction" / cond / "inputs.npz", allow_pickle=True) as data:
            frame_ids = data["frame_ids"].astype(int).tolist()
            indices = [frame_ids.index(frame_id) for frame_id in target_frames]
            arrays = {"frame_ids": np.asarray(target_frames, dtype=np.int32), "query_condition": np.asarray(cond)}
            for key in data.files:
                arr = data[key]
                if key == "frame_ids":
                    continue
                if getattr(arr, "shape", ()) and arr.shape[0] == len(frame_ids):
                    arrays[key] = arr[indices]
                else:
                    arrays[key] = arr
        out = root / "reconstruction" / variant / "inputs.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)


def tensor_error(a: Any, b: Any, eps: float) -> dict[str, float]:
    torch = require_torch()
    if not torch.is_tensor(a):
        a = torch.as_tensor(a)
    if not torch.is_tensor(b):
        b = torch.as_tensor(b)
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    diff = a - b
    denom = ((torch.linalg.norm(a.reshape(-1)) + torch.linalg.norm(b.reshape(-1))) / 2.0).clamp_min(eps)
    return {
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "relative_l2": float(torch.linalg.norm(diff.reshape(-1)).item() / float(denom.item())),
    }


def write_replay_consistency(cfg: dict[str, Any], target_frames: list[int], variants: list[str]) -> None:
    torch = require_torch()
    root = output_root(cfg)
    rows: list[dict[str, Any]] = []
    eps = float(cfg["analysis"]["eps"])
    pairs = [("AA", "always_absent"), ("RR", "seen_then_removed")]
    for variant, condition in pairs:
        if variant not in variants:
            continue
        base_frame_ids, base_pred = load_baseline_predictions(cfg, condition)
        with np.load(root / "reconstruction" / variant / "predictions.npz", allow_pickle=True) as pred:
            replay_pred = {key: pred[key] for key in pred.files}
        target_indices = [base_frame_ids.index(frame_id) for frame_id in target_frames]
        for key in ["depth", "depth_conf", "pose_enc", "pose_enc_iterations", "pred_cam2world", "pred_intrinsic"]:
            if key not in base_pred or key not in replay_pred:
                continue
            err = tensor_error(replay_pred[key], base_pred[key][target_indices], eps)
            rows.append({"variant": variant, "baseline_condition": condition, "quantity": key, **err})

        base_feat = load_baseline_features(cfg, condition)
        replay_feat = torch.load(root / "features" / variant / "features.pt", map_location="cpu")
        base_feature_indices = [list(base_feat["frame_ids"]).index(frame_id) for frame_id in target_frames]
        for group in ["frame_block", "global_block", "frame_special_tokens", "global_special_tokens"]:
            for layer in replay_feat[group]:
                err = tensor_error(replay_feat[group][layer], base_feat[group][layer][base_feature_indices], eps)
                rows.append({"variant": variant, "baseline_condition": condition, "quantity": f"{group}_layer_{layer}", **err})
    write_csv(root / "metrics" / "replay_consistency.csv", rows)


def write_cache_docs(
    cfg: dict[str, Any],
    first_layout: dict[str, Any],
    transplant_events: list[dict[str, Any]],
    source_frames: list[int],
    negative_frames_by_target: dict[int, list[int]],
) -> None:
    root = output_root(cfg)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    doc_layout = first_layout["A"] if "A" in first_layout else first_layout
    write_json(
        metadata / "memory_layout.json",
        {
            "layout_example": first_layout,
            "token_names": SPECIAL_TOKEN_NAMES,
            "source_target_visible_frames": source_frames,
            "negative_control_frames_by_target": {str(k): v for k, v in negative_frames_by_target.items()},
        },
    )
    lines = [
        "# Memory Cache Structure",
        "",
        "This experiment used LingBot-Map's SDPA streaming cache (`model.aggregator.kv_cache`).",
        "No LingBot-Map source file was modified.",
        "",
        "## Aggregator Cache",
        "",
        "- Each global block has `k_i` and `v_i` tensors for anchor/scale frames plus the active local window.",
        "- Each global block also has `k_i_special` and `v_i_special` tensors for evicted historical special-token records.",
        "- `k_i_special/v_i_special` are the transplanted trajectory-memory tensors.",
        "- Patch/image-token local cache and the first scale/anchor frames remain in `k_i/v_i` and are not changed by AR/RA/AS/AN.",
        "- Cached K tensors already include 2D/3D RoPE application; there is no separate RoPE tensor per memory slot.",
        "- The current frame's 3D RoPE time is controlled by `aggregator.total_frames_processed`, which is cloned/restored with the state.",
        "",
        "## Token Order",
        "",
        "`camera`, `register_0`, `register_1`, `register_2`, `register_3`, `scale_or_anchor`.",
        "",
        "The user-facing phrase `anchor token` maps to LingBot-Map's causal `scale_token` in this code path.",
        "",
        "## Example Target",
        "",
        f"- target frame: `{doc_layout['target_frame']}`",
        f"- trajectory source frames before target: `{doc_layout['special_source_frame_ids_before_target']}`",
        f"- local/anchor cache source frames before target: `{doc_layout['local_anchor_source_frame_ids_before_target']}`",
        f"- total frames processed before target: `{doc_layout['aggregator_total_frames_processed_before_target']}`",
        "",
        "## Replacement Scope",
        "",
        "- AR: query/current/local/anchor/camera state from A; all aggregator trajectory special K/V from R.",
        "- RA: query/current/local/anchor/camera state from R; all aggregator trajectory special K/V from A.",
        "- AS: query/current/local/anchor/camera state from A; only target-visible source-frame special K/V records from R.",
        "- AN: optional matched non-target record replacement from R.",
    ]
    write_text(root / "cache_structure.md", "\n".join(lines) + "\n")
    write_json(root / "logs" / "transplant_events.json", transplant_events)
    write_text(
        root / "source_changes.md",
        "\n".join(
            [
                "# Source Changes",
                "",
                "No files under `lingbot-map/` were modified.",
                "",
                "Experiment-side additions:",
                "",
                "- `revisit_memory/configs/memory_transplant_v1.yaml`",
                "- `revisit_memory/transplant/cache_state.py`",
                "- `revisit_memory/scripts/run_memory_transplant.py`",
                "- `revisit_memory/analysis/analyze_memory_transplant.py`",
                "",
                "The transplant is implemented by cloning/restoring public Python attributes on the constructed model "
                "and editing only `aggregator.kv_cache['k_i_special']` / `aggregator.kv_cache['v_i_special']` in replay branches.",
            ]
        )
        + "\n",
    )


def save_snapshot_manifest(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    root = output_root(cfg)
    path = root / "snapshots" / "snapshot_manifest.csv"
    write_csv(path, rows)
    write_json(
        root / "snapshots" / "snapshot_policy.json",
        {
            "save_full_cache_snapshots": bool(cfg["transplant"].get("save_full_cache_snapshots", False)),
            "note": (
                "Full K/V cache tensors are cloned in memory for replay isolation. They are not persisted by default "
                "because each target-frame patch cache is large; metadata and replacement logs are saved instead."
            ),
        },
    )


def run(cfg: dict[str, Any], *, run_analysis: bool = True) -> None:
    os.environ.setdefault("REVISIT_MEMORY_RUN_ID", cfg["project"].get("run_id", "memory_transplant_v1"))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch = require_torch()
    set_seed(int(cfg["project"]["random_seed"]))
    random.seed(int(cfg["project"]["random_seed"]))
    np.random.seed(int(cfg["project"]["random_seed"]))

    root = output_root(cfg)
    for subdir in ["metadata", "snapshots", "features", "reconstruction", "metrics", "visualizations", "logs"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    write_config_used(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(cfg, device)
    print(f"Device: {device}; dtype: {dtype}; output: {root}", flush=True)

    target_frames = [int(x) for x in cfg["transplant"]["target_frames"]]
    max_target = max(target_frames)
    scale_frames = int(cfg["model"]["anchor_frame_count"])
    local_window = int(cfg["model"]["local_window_size"])
    selected_layers = [int(x) for x in cfg["features"]["selected_layers"]]
    save_dtype = torch.float16 if cfg["features"].get("save_dtype") == "float16" else torch.float32
    variants = list(cfg["transplant"].get("variants", ["AA", "RR", "AR", "RA", "AS"]))
    if not bool(cfg["transplant"].get("include_negative_control", False)):
        variants = [variant for variant in variants if variant != "AN"]

    visible_source_frames_all = source_visible_frames(cfg)
    print(f"Target-visible source frames for AS: {visible_source_frames_all}", flush=True)

    images_a_cpu, _ = load_images_tensor(cfg, "always_absent", list(range(max_target + 1)))
    images_r_cpu, _ = load_images_tensor(cfg, "seen_then_removed", list(range(max_target + 1)))
    image_hw = (int(images_a_cpu.shape[-2]), int(images_a_cpu.shape[-1]))
    images_a = images_a_cpu.to(device)
    images_r = images_r_cpu.to(device)

    set_seed(int(cfg["project"]["random_seed"]))
    model_a, load_info_a = build_model(cfg, device)
    set_seed(int(cfg["project"]["random_seed"]))
    model_r, load_info_r = build_model(cfg, device)
    if dtype != torch.float32:
        model_a.aggregator = model_a.aggregator.to(dtype=dtype)
        model_r.aggregator = model_r.aggregator.to(dtype=dtype)

    write_json(root / "metadata" / "model_load_info.json", {"A": load_info_a, "R": load_info_r})

    stores: dict[str, dict[str, Any]] = {
        variant: {"pred": {}, "features": {}, "memory": {}, "frame_ids": []} for variant in variants
    }
    transplant_events: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    source_frame_rows: list[dict[str, Any]] = []
    negative_frames_by_target: dict[int, list[int]] = {}
    first_layout: dict[str, Any] | None = None

    def forward_one(model, images, frame_index: int):
        return model.forward(
            images[frame_index : frame_index + 1].unsqueeze(0),
            num_frame_for_scale=scale_frames,
            num_frame_per_block=1,
            causal_inference=True,
        )

    def forward_scale(model, images):
        return model.forward(
            images[:scale_frames].unsqueeze(0),
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )

    def run_variant(
        *,
        variant: str,
        model,
        capture: AggregateFeatureCapture,
        base_state,
        frame_id: int,
        images,
        replace_from=None,
        selected_source_frames: list[int] | None = None,
        special_frames: list[int] | None = None,
    ):
        branch_state = clone_stream_state(base_state)
        events: list[dict[str, Any]] = []
        replaced_frames: list[int] = []
        missing_frames: list[int] = []
        if replace_from is not None and selected_source_frames is None:
            events = replace_all_trajectory_special(branch_state, replace_from)
        elif replace_from is not None and selected_source_frames is not None:
            events, replaced_frames, missing_frames = replace_selected_trajectory_records(
                branch_state,
                replace_from,
                selected_source_frames=selected_source_frames,
                special_frame_ids=special_frames or [],
            )
        restore_model_stream_state(model, branch_state)
        out = forward_one(model, images, frame_id)
        store_output(cfg, variant, frame_id, out, capture, stores)
        stores[variant]["frame_ids"].append(frame_id)
        transplant_events.append(
            {
                "variant": variant,
                "target_frame": frame_id,
                "query_state": "A" if variant in {"AA", "AR", "AS", "AN"} else "R",
                "trajectory_memory_source": {
                    "AA": "A",
                    "RR": "R",
                    "AR": "R_all_records",
                    "RA": "A_all_records",
                    "AS": "R_target_visible_records",
                    "AN": "R_non_target_control_records",
                }.get(variant, "unknown"),
                "selected_source_frames": selected_source_frames or [],
                "replaced_source_frames": replaced_frames,
                "missing_source_frames": missing_frames,
                "local_window_unchanged_by_code_path": True,
                "anchor_context_unchanged_by_code_path": True,
                "camera_head_cache_unchanged_by_transplant": True,
                "replacement_events": events,
            }
        )
        del out

    model_a.clean_kv_cache()
    model_r.clean_kv_cache()
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    target_set = set(target_frames)
    with AggregateFeatureCapture(model_a, selected_layers, save_dtype=save_dtype) as capture_a, AggregateFeatureCapture(
        model_r, selected_layers, save_dtype=save_dtype
    ) as capture_r:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            out_a = forward_scale(model_a, images_a)
            out_r = forward_scale(model_r, images_r)
            del out_a, out_r
            for frame_id in range(scale_frames, max_target + 1):
                if frame_id not in target_set:
                    out_a = forward_one(model_a, images_a, frame_id)
                    out_r = forward_one(model_r, images_r, frame_id)
                    del out_a, out_r
                    continue

                print(f"Replay target frame {frame_id}", flush=True)
                a_before = clone_model_stream_state(model_a)
                r_before = clone_model_stream_state(model_r)
                special_frames = special_frame_ids_before_target(frame_id, scale_frames, local_window)
                local_frames = local_cache_frame_ids_before_target(frame_id, scale_frames, local_window)
                visible_for_target = [fid for fid in visible_source_frames_all if fid in set(special_frames)]
                negative_for_target = choose_negative_control_frames(cfg, visible_for_target, special_frames)
                negative_frames_by_target[frame_id] = negative_for_target

                layout_a = state_layout_summary(
                    a_before,
                    source_sequence="always_absent",
                    target_frame=frame_id,
                    scale_frames=scale_frames,
                    sliding_window=local_window,
                )
                layout_r = state_layout_summary(
                    r_before,
                    source_sequence="seen_then_removed",
                    target_frame=frame_id,
                    scale_frames=scale_frames,
                    sliding_window=local_window,
                )
                if first_layout is None:
                    first_layout = {"A": layout_a, "R": layout_r}
                for seq_name, layout in [("A", layout_a), ("R", layout_r)]:
                    snapshot_rows.append(
                        {
                            "target_frame": frame_id,
                            "sequence": seq_name,
                            "total_frames_processed": layout["aggregator_total_frames_processed_before_target"],
                            "local_anchor_source_frame_ids": ",".join(str(x) for x in local_frames),
                            "trajectory_special_source_frame_ids": ",".join(str(x) for x in special_frames),
                        }
                    )
                for fid in visible_source_frames_all:
                    source_frame_rows.append(
                        {
                            "target_frame": frame_id,
                            "source_frame": fid,
                            "target_mask_area": mask_area_for_frame(cfg, "seen_then_removed", fid),
                            "in_trajectory_memory": fid in set(special_frames),
                            "memory_slot": special_frames.index(fid) if fid in set(special_frames) else "",
                            "token_slots": ",".join(f"{idx}:{name}" for idx, name in enumerate(SPECIAL_TOKEN_NAMES)),
                            "time_distance": frame_id - fid,
                        }
                    )

                if "AA" in variants:
                    run_variant(variant="AA", model=model_a, capture=capture_a, base_state=a_before, frame_id=frame_id, images=images_a)
                    a_after = clone_model_stream_state(model_a)
                else:
                    restore_model_stream_state(model_a, a_before)
                    out_a = forward_one(model_a, images_a, frame_id)
                    a_after = clone_model_stream_state(model_a)
                    del out_a
                if "RR" in variants:
                    run_variant(variant="RR", model=model_r, capture=capture_r, base_state=r_before, frame_id=frame_id, images=images_r)
                    r_after = clone_model_stream_state(model_r)
                else:
                    restore_model_stream_state(model_r, r_before)
                    out_r = forward_one(model_r, images_r, frame_id)
                    r_after = clone_model_stream_state(model_r)
                    del out_r

                if "AR" in variants:
                    run_variant(
                        variant="AR",
                        model=model_a,
                        capture=capture_a,
                        base_state=a_before,
                        frame_id=frame_id,
                        images=images_a,
                        replace_from=r_before,
                    )
                if "RA" in variants:
                    run_variant(
                        variant="RA",
                        model=model_r,
                        capture=capture_r,
                        base_state=r_before,
                        frame_id=frame_id,
                        images=images_r,
                        replace_from=a_before,
                    )
                if "AS" in variants:
                    run_variant(
                        variant="AS",
                        model=model_a,
                        capture=capture_a,
                        base_state=a_before,
                        frame_id=frame_id,
                        images=images_a,
                        replace_from=r_before,
                        selected_source_frames=visible_for_target,
                        special_frames=special_frames,
                    )
                if "AN" in variants:
                    run_variant(
                        variant="AN",
                        model=model_a,
                        capture=capture_a,
                        base_state=a_before,
                        frame_id=frame_id,
                        images=images_a,
                        replace_from=r_before,
                        selected_source_frames=negative_for_target,
                        special_frames=special_frames,
                    )

                restore_model_stream_state(model_a, a_after)
                restore_model_stream_state(model_r, r_after)
                del a_before, r_before, a_after, r_after
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finalize_variant_outputs(cfg, stores, image_hw)
    save_variant_inputs(cfg, variants, target_frames)
    write_replay_consistency(cfg, target_frames, variants)
    write_csv(root / "metadata" / "source_frame_records.csv", source_frame_rows)
    save_snapshot_manifest(cfg, snapshot_rows)
    write_json(
        root / "metadata" / "target_frames.json",
        {
            "target_frames": target_frames,
            "source_target_visible_frames_all": visible_source_frames_all,
            "negative_control_frames_by_target": {str(k): v for k, v in negative_frames_by_target.items()},
        },
    )
    if first_layout is None:
        raise RuntimeError("No target frame was processed; cannot write cache docs.")
    write_cache_docs(cfg, first_layout, transplant_events, visible_source_frames_all, negative_frames_by_target)
    if run_analysis:
        from revisit_memory.analysis.analyze_memory_transplant import run as run_transplant_analysis

        run_transplant_analysis(cfg)
    print(f"Wrote memory transplant run to {root}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run trajectory-memory transplant replays.")
    parser.add_argument("--config", default="revisit_memory/configs/memory_transplant_v1.yaml")
    parser.add_argument("--run-id", default=None, help="Override project.run_id / REVISIT_MEMORY_RUN_ID for this invocation.")
    parser.add_argument("--target-frames", default=None, help="Comma-separated target frames overriding transplant.target_frames.")
    parser.add_argument("--skip-analysis", action="store_true", help="Only run replay/transplant generation.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.run_id:
        cfg["project"]["run_id"] = args.run_id
        os.environ["REVISIT_MEMORY_RUN_ID"] = args.run_id
    if args.target_frames:
        cfg["transplant"]["target_frames"] = [int(part.strip()) for part in args.target_frames.split(",") if part.strip()]
        cfg["transplant"]["max_report_heatmap_frames"] = list(cfg["transplant"]["target_frames"])
    run(cfg, run_analysis=not args.skip_analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
