#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, write_json  # noqa: E402
from long_seq_memory.memory_index import preprocessed_patch_grid  # noqa: E402


SPECIAL_QUERY_TOKENS = 6


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_one(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Expected one file for {pattern}, found {len(matches)}")
    return matches[0]


def required_records(cfg: dict[str, Any], args: argparse.Namespace) -> list[tuple[int, int]]:
    extraction = cfg.get("extraction", {})
    frames = parse_int_list(args.frames) or [int(v) for v in extraction.get("raw_qkv_frames", [])]
    layers = parse_int_list(args.layers) or [int(v) for v in extraction.get("representative_layers", [])]
    return [(frame, layer) for frame in frames for layer in layers]


def resolve_dirs(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    interaction_dir = Path(args.interaction_dir or cfg["interaction_output_dir"]).expanduser()
    default_output = (
        Path(cfg.get("analysis_output_dir", interaction_dir.parent / "analysis"))
        .expanduser()
        .parent
        / "patch_memory_specificity"
    )
    return {
        "qkv_dir": Path(args.qkv_dir or interaction_dir / "raw_qkv_debug" / "qkv").expanduser(),
        "metadata_dir": Path(args.metadata_dir or interaction_dir / "raw_qkv_debug" / "token_metadata").expanduser(),
        "output_dir": Path(args.output_dir or default_output).expanduser(),
    }


def validate_inputs(
    records: list[tuple[int, int]],
    qkv_dir: Path,
    metadata_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = []
    missing = []
    for frame, layer in records:
        qkv_path = find_one(f"qkv_frame{frame:06d}_layer{layer:02d}_*.npz", qkv_dir)
        metadata_path = metadata_dir / f"frame{frame:06d}_layer{layer:02d}_token_metadata.json"
        item = {
            "frame_id": int(frame),
            "layer_id": int(layer),
            "qkv_path": str(qkv_path) if qkv_path else None,
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        }
        if qkv_path is None or not metadata_path.exists():
            missing.append(item)
        else:
            available.append(item)
    return available, missing


def infer_grid(cfg: dict[str, Any], frame_id: int, patch_count: int) -> tuple[int, int]:
    dataset_root = Path(cfg.get("dataset_root", "")).expanduser()
    image_path = dataset_root / "images" / f"{frame_id:06d}.png"
    defaults = cfg.get("model_defaults", {})
    image_size = int(defaults.get("image_size", cfg.get("inference", {}).get("image_size", 518)))
    patch_size = int(defaults.get("patch_size", 14))
    if image_path.exists():
        try:
            from PIL import Image

            image = Image.open(image_path)
            grid = preprocessed_patch_grid(image.size[0], image.size[1], image_size=image_size, patch_size=patch_size)
            if grid[0] * grid[1] == patch_count:
                return grid
        except Exception:
            pass
    factors = [(w, patch_count // w) for w in range(1, patch_count + 1) if patch_count % w == 0]
    target_w = image_size // patch_size
    return min(factors, key=lambda wh: (abs(wh[0] - target_w), abs(wh[0] - wh[1])))


def summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "cv": float(arr.std() / max(abs(float(arr.mean())), 1e-12)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def cosine_to_mean(distributions: np.ndarray) -> np.ndarray:
    mean_distribution = distributions.mean(axis=0, keepdims=True)
    a = l2_normalize(distributions, axis=1)
    b = l2_normalize(mean_distribution, axis=1)
    return (a * b).sum(axis=1)


def jensen_shannon_to_mean(distributions: np.ndarray) -> np.ndarray:
    p = np.asarray(distributions, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    q = p.mean(axis=0, keepdims=True)
    q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(m, 1e-12)))).sum(axis=1)
    kl_qm = (q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(m, 1e-12)))).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm)


def neighbor_pairs(width: int, height: int) -> np.ndarray:
    pairs = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x + 1 < width:
                pairs.append((idx, idx + 1))
            if y + 1 < height:
                pairs.append((idx, idx + width))
    return np.asarray(pairs, dtype=np.int64)


def pair_cosines(vectors: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.asarray([], dtype=np.float64)
    normalized = l2_normalize(vectors, axis=1)
    return (normalized[pairs[:, 0]] * normalized[pairs[:, 1]]).sum(axis=1)


def random_pairs(count: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, count, size=samples)
    right = rng.integers(0, count, size=samples)
    mask = left != right
    return np.stack([left[mask], right[mask]], axis=1).astype(np.int64)


def effective_count(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr / max(float(arr.sum()), 1e-12)
    entropy = -(arr * np.log(np.maximum(arr, 1e-12))).sum()
    return float(math.exp(entropy))


def classify_record(metrics: dict[str, Any]) -> str:
    distribution_cos = metrics["source_distribution_cosine_to_frame_mean"]["mean"]
    output_cos = metrics["memory_output_cosine_to_frame_mean"]["mean"]
    loop_cv = metrics.get("loop_history_conditional_mass", {}).get("cv", 0.0)
    full_mass_cv = metrics.get("old_memory_full_context_mass", {}).get("cv", 0.0)
    top1_effective = metrics["top1_source_frame_effective_count"]
    if full_mass_cv > 0.30:
        return "patch_specific"
    if distribution_cos > 0.985 and output_cos > 0.985 and loop_cv < 0.15 and top1_effective < 5:
        return "mostly_full_frame_uniform"
    if distribution_cos < 0.95 or output_cos < 0.95 or loop_cv > 0.30 or top1_effective > 20:
        return "patch_specific"
    return "mixed_or_weakly_patch_specific"


def compute_full_context_old_mass(
    q_chunk: Any,
    k_all: Any,
    logsumexp_old: Any,
    scale: float,
    key_chunk_size: int,
) -> Any:
    import torch

    logsumexp_all = None
    for start in range(0, k_all.shape[1], key_chunk_size):
        end = min(start + key_chunk_size, k_all.shape[1])
        scores = torch.matmul(q_chunk, k_all[:, start:end, :].transpose(-1, -2)) * scale
        chunk_lse = torch.logsumexp(scores, dim=-1)
        logsumexp_all = chunk_lse if logsumexp_all is None else torch.logaddexp(logsumexp_all, chunk_lse)
        del scores, chunk_lse
    return torch.exp(logsumexp_old - logsumexp_all)


def analyze_record(
    cfg: dict[str, Any],
    item: dict[str, Any],
    output_dir: Path,
    positive_range: tuple[int, int],
    device_name: str,
    patch_chunk_size: int,
    key_chunk_size: int,
    include_full_context_mass: bool,
    make_figure: bool,
) -> dict[str, Any]:
    import torch

    qkv_path = Path(item["qkv_path"])
    metadata = load_json(Path(item["metadata_path"]))
    frame_id = int(item["frame_id"])
    layer_id = int(item["layer_id"])
    old_frame_count = len(metadata["old_memory_frames"])
    old_token_count = int(metadata["old_memory_token_count"])
    source_frames = np.asarray(metadata["old_memory_frames"], dtype=np.int64)
    target_mask = (source_frames >= positive_range[0]) & (source_frames <= positive_range[1])

    z = np.load(qkv_path)
    q_np = z["q"]
    k_np = z["k"]
    v_np = z["v"]
    if q_np.ndim != 4 or k_np.ndim != 4 or v_np.ndim != 4:
        raise ValueError(f"Expected q/k/v [B,H,T,D], got {q_np.shape}, {k_np.shape}, {v_np.shape}")
    patch_count = q_np.shape[2] - SPECIAL_QUERY_TOKENS
    grid_w, grid_h = infer_grid(cfg, frame_id, patch_count)
    if grid_w * grid_h != patch_count:
        raise ValueError(f"Patch grid {grid_w}x{grid_h} does not match patch count {patch_count}")

    device = torch.device(device_name)
    q = torch.from_numpy(q_np[0, :, SPECIAL_QUERY_TOKENS:, :]).to(device=device, dtype=torch.float32)
    k_old = torch.from_numpy(k_np[0, :, :old_token_count, :]).to(device=device, dtype=torch.float32)
    v_old = torch.from_numpy(v_np[0, :, :old_token_count, :]).to(device=device, dtype=torch.float32)
    k_all = None
    if include_full_context_mass:
        k_all = torch.from_numpy(k_np[0]).to(device=device, dtype=torch.float32)
    del z, q_np, k_np, v_np

    heads, patches, dim = q.shape
    if old_token_count != old_frame_count * SPECIAL_QUERY_TOKENS:
        raise ValueError(f"old_token_count={old_token_count} but old frames imply {old_frame_count * SPECIAL_QUERY_TOKENS}")

    scale = 1.0 / math.sqrt(dim)
    source_distribution = np.zeros((patches, old_frame_count), dtype=np.float32)
    memory_output = np.zeros((patches, heads * dim), dtype=np.float32)
    old_memory_mass_full = np.zeros((patches,), dtype=np.float32) if include_full_context_mass else None

    for start in range(0, patches, patch_chunk_size):
        end = min(start + patch_chunk_size, patches)
        q_chunk = q[:, start:end, :]
        logits_old = torch.matmul(q_chunk, k_old.transpose(-1, -2)) * scale
        attn_old = torch.softmax(logits_old, dim=-1)
        frame_dist = attn_old.reshape(heads, end - start, old_frame_count, SPECIAL_QUERY_TOKENS).sum(dim=-1)
        source_distribution[start:end] = frame_dist.mean(dim=0).detach().cpu().numpy()
        old_out = torch.matmul(attn_old, v_old)
        memory_output[start:end] = old_out.permute(1, 0, 2).reshape(end - start, heads * dim).detach().cpu().numpy()
        if include_full_context_mass and k_all is not None:
            logsumexp_old = torch.logsumexp(logits_old, dim=-1)
            full_mass = compute_full_context_old_mass(q_chunk, k_all, logsumexp_old, scale, key_chunk_size)
            old_memory_mass_full[start:end] = full_mass.mean(dim=0).detach().cpu().numpy()
            del logsumexp_old, full_mass
        del logits_old, attn_old, frame_dist, old_out

    del q, k_old, v_old, k_all
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source_distribution = source_distribution / np.maximum(source_distribution.sum(axis=1, keepdims=True), 1e-12)
    source_cos = cosine_to_mean(source_distribution)
    source_js = jensen_shannon_to_mean(source_distribution)
    source_entropy = -(
        source_distribution * np.log(np.maximum(source_distribution, 1e-12))
    ).sum(axis=1) / max(math.log(max(old_frame_count, 2)), 1e-12)

    output_cos = cosine_to_mean(memory_output)
    output_norm = np.linalg.norm(memory_output, axis=1)
    top_source_idx = np.argmax(source_distribution, axis=1)
    top_source_frames = source_frames[top_source_idx]
    top_counts = Counter(int(frame) for frame in top_source_frames.tolist())
    neighbor = neighbor_pairs(grid_w, grid_h)
    random = random_pairs(patches, samples=min(20000, patches * 20), seed=frame_id * 100 + layer_id)
    source_neighbor_cos = pair_cosines(source_distribution, neighbor)
    source_random_cos = pair_cosines(source_distribution, random)
    output_neighbor_cos = pair_cosines(memory_output, neighbor)
    output_random_cos = pair_cosines(memory_output, random)

    maps: dict[str, np.ndarray] = {
        "source_entropy": source_entropy.reshape(grid_h, grid_w),
        "source_cosine_to_frame_mean": source_cos.reshape(grid_h, grid_w),
        "memory_output_norm": output_norm.reshape(grid_h, grid_w),
        "memory_output_cosine_to_frame_mean": output_cos.reshape(grid_h, grid_w),
        "top_source_frame": top_source_frames.reshape(grid_h, grid_w),
        "top_source_age": (frame_id - top_source_frames).reshape(grid_h, grid_w),
    }
    metrics: dict[str, Any] = {
        "frame_id": frame_id,
        "layer_id": layer_id,
        "qkv_path": str(qkv_path),
        "metadata_path": str(item["metadata_path"]),
        "patch_grid_wh": [grid_w, grid_h],
        "patch_count": int(patches),
        "heads": int(heads),
        "old_memory_frame_count": int(old_frame_count),
        "old_memory_token_count": int(old_token_count),
        "source_frame_range": [int(source_frames.min()), int(source_frames.max())],
        "target_history_range": list(positive_range),
        "target_history_present": bool(target_mask.any()),
        "source_distribution_cosine_to_frame_mean": summary_stats(source_cos),
        "source_distribution_js_to_frame_mean": summary_stats(source_js),
        "source_distribution_normalized_entropy": summary_stats(source_entropy),
        "source_neighbor_cosine": summary_stats(source_neighbor_cos),
        "source_random_pair_cosine": summary_stats(source_random_cos),
        "source_spatial_structure_delta": float(source_neighbor_cos.mean() - source_random_cos.mean()),
        "memory_output_cosine_to_frame_mean": summary_stats(output_cos),
        "memory_output_norm": summary_stats(output_norm),
        "memory_output_neighbor_cosine": summary_stats(output_neighbor_cos),
        "memory_output_random_pair_cosine": summary_stats(output_random_cos),
        "memory_output_spatial_structure_delta": float(output_neighbor_cos.mean() - output_random_cos.mean()),
        "top1_source_frame_unique_count": int(len(top_counts)),
        "top1_source_frame_effective_count": effective_count(np.asarray(list(top_counts.values()), dtype=np.float64)),
        "top1_source_frame_mode_fraction": float(max(top_counts.values()) / patches),
        "top1_source_frame_most_common": [
            {"source_frame_id": int(frame), "count": int(count), "fraction": float(count / patches)}
            for frame, count in top_counts.most_common(20)
        ],
    }
    if target_mask.any():
        loop_mass = source_distribution[:, target_mask].sum(axis=1)
        maps["loop_history_conditional_mass"] = loop_mass.reshape(grid_h, grid_w)
        metrics["loop_history_conditional_mass"] = summary_stats(loop_mass)
        metrics["top1_source_in_target_history_fraction"] = float(target_mask[top_source_idx].mean())
    if old_memory_mass_full is not None:
        maps["old_memory_full_context_mass"] = old_memory_mass_full.reshape(grid_h, grid_w)
        metrics["old_memory_full_context_mass"] = summary_stats(old_memory_mass_full)

    metrics["classification"] = classify_record(metrics)

    maps_dir = ensure_dir(output_dir / "maps")
    np.savez_compressed(
        maps_dir / f"patch_memory_frame{frame_id:06d}_layer{layer_id:02d}.npz",
        source_frames=source_frames,
        source_distribution_mean_heads=source_distribution.astype(np.float32),
        **{key: value.astype(np.float32) if np.issubdtype(value.dtype, np.floating) else value for key, value in maps.items()},
    )
    if make_figure:
        render_record_figure(maps, metrics, output_dir / "figures")
    return metrics


def render_record_figure(maps: dict[str, np.ndarray], metrics: dict[str, Any], figures_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dir(figures_dir)
    panels = [
        ("source_entropy", "Source Entropy"),
        ("source_cosine_to_frame_mean", "Source Cosine To Mean"),
        ("memory_output_cosine_to_frame_mean", "Output Cosine To Mean"),
        ("top_source_age", "Top Source Age"),
    ]
    if "loop_history_conditional_mass" in maps:
        panels[0] = ("loop_history_conditional_mass", "Loop Conditional Mass")
    if "old_memory_full_context_mass" in maps:
        panels[1] = ("old_memory_full_context_mass", "Old Memory Full Mass")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (key, title) in zip(axes.flat, panels):
        im = ax.imshow(maps[key], aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"frame {metrics['frame_id']} layer {metrics['layer_id']} | {metrics['classification']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / f"patch_memory_frame{metrics['frame_id']:06d}_layer{metrics['layer_id']:02d}.png", dpi=150)
    plt.close(fig)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = Counter(record["classification"] for record in records)

    def collect(path: list[str]) -> np.ndarray:
        values = []
        for record in records:
            cur: Any = record
            for key in path:
                cur = cur[key]
            values.append(float(cur))
        return np.asarray(values, dtype=np.float64)

    summary = {
        "num_records": len(records),
        "classification_counts": dict(classifications),
        "mean_source_cosine_to_frame_mean": summary_stats(collect(["source_distribution_cosine_to_frame_mean", "mean"])),
        "mean_source_js_to_frame_mean": summary_stats(collect(["source_distribution_js_to_frame_mean", "mean"])),
        "mean_memory_output_cosine_to_frame_mean": summary_stats(collect(["memory_output_cosine_to_frame_mean", "mean"])),
        "mean_source_spatial_structure_delta": summary_stats(collect(["source_spatial_structure_delta"])),
        "mean_memory_output_spatial_structure_delta": summary_stats(collect(["memory_output_spatial_structure_delta"])),
        "top1_source_frame_effective_count": summary_stats(collect(["top1_source_frame_effective_count"])),
    }
    loop_records = [r for r in records if "loop_history_conditional_mass" in r]
    if loop_records:
        values = np.asarray([r["loop_history_conditional_mass"]["cv"] for r in loop_records], dtype=np.float64)
        summary["loop_history_conditional_mass_cv"] = summary_stats(values)
    full_records = [r for r in records if "old_memory_full_context_mass" in r]
    if full_records:
        values = np.asarray([r["old_memory_full_context_mass"]["cv"] for r in full_records], dtype=np.float64)
        summary["old_memory_full_context_mass_cv"] = summary_stats(values)

    patch_specific = classifications.get("patch_specific", 0)
    uniform = classifications.get("mostly_full_frame_uniform", 0)
    if patch_specific > uniform:
        conclusion = "patch_specific"
    elif uniform > patch_specific:
        conclusion = "mostly_full_frame_uniform"
    else:
        conclusion = "mixed_or_weakly_patch_specific"
    summary["conclusion"] = conclusion
    return summary


def write_report(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        "# Patch Memory Specificity",
        "",
        f"- Status: `{summary['status']}`",
        f"- Conclusion: `{summary['aggregate']['conclusion']}`",
        f"- Records analyzed: {summary['aggregate']['num_records']}",
        f"- Classification counts: {summary['aggregate']['classification_counts']}",
        "",
        "## Interpretation",
        "",
        summary["current_answer"],
        "",
        "## Key Metrics",
        "",
        "- Source-distribution cosine to frame mean: "
        f"{summary['aggregate']['mean_source_cosine_to_frame_mean']['mean']:.6f}",
        "- Memory-output cosine to frame mean: "
        f"{summary['aggregate']['mean_memory_output_cosine_to_frame_mean']['mean']:.6f}",
        "- Source spatial neighbor-minus-random cosine: "
        f"{summary['aggregate']['mean_source_spatial_structure_delta']['mean']:.6f}",
        "- Memory-output neighbor-minus-random cosine: "
        f"{summary['aggregate']['mean_memory_output_spatial_structure_delta']['mean']:.6f}",
        "- Top-1 source effective count: "
        f"{summary['aggregate']['top1_source_frame_effective_count']['mean']:.3f}",
    ]
    if "old_memory_full_context_mass_cv" in summary["aggregate"]:
        lines.append(
            "- Old-memory full-context mass CV: "
            f"{summary['aggregate']['old_memory_full_context_mass_cv']['mean']:.6f}"
        )
    lines.extend(["", "## Per Record", ""])
    for record in records:
        full_mass_note = ""
        if "old_memory_full_context_mass" in record:
            full_mass_note = f", full_mass_cv={record['old_memory_full_context_mass']['cv']:.3f}"
        lines.append(
            f"- frame {record['frame_id']} layer {record['layer_id']}: "
            f"{record['classification']}, "
            f"source_cos={record['source_distribution_cosine_to_frame_mean']['mean']:.4f}, "
            f"output_cos={record['memory_output_cosine_to_frame_mean']['mean']:.4f}, "
            f"top1_eff={record['top1_source_frame_effective_count']:.1f}"
            f"{full_mass_note}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This analysis uses old-memory-conditional attention by default; full local-window competition is included only when old_memory_full_context_mass is present.",
            "- It analyzes attention and value-proxy memory output from stored Q/K/V, not downstream rendered depth or final map quality.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--interaction-dir", default=None)
    parser.add_argument("--qkv-dir", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--frames", default=None, help="Comma-separated frame ids. Defaults to extraction.raw_qkv_frames.")
    parser.add_argument("--layers", default=None, help="Comma-separated layer ids. Defaults to extraction.representative_layers.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-chunk-size", type=int, default=64)
    parser.add_argument("--key-chunk-size", type=int, default=8192)
    parser.add_argument("--include-full-context-mass", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dirs = resolve_dirs(cfg, args)
    output_dir = ensure_dir(dirs["output_dir"])
    records = required_records(cfg, args)
    available, missing = validate_inputs(records, dirs["qkv_dir"], dirs["metadata_dir"])
    if missing:
        summary = {
            "status": "missing_qkv_or_metadata",
            "missing": missing,
            "available_count": len(available),
            "required_count": len(records),
            "current_answer": "Patch-level QKV analysis was not run because at least one required raw QKV or token metadata file is missing.",
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        raise SystemExit(2)

    import torch

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    extraction = cfg.get("extraction", {})
    positive_range = tuple(int(v) for v in extraction.get("target_history_range", [3370, 3440]))
    per_record = []
    for idx, item in enumerate(available, start=1):
        print(
            f"[{idx}/{len(available)}] frame {item['frame_id']} layer {item['layer_id']} "
            f"on {device_name}",
            flush=True,
        )
        per_record.append(
            analyze_record(
                cfg=cfg,
                item=item,
                output_dir=output_dir,
                positive_range=positive_range,
                device_name=device_name,
                patch_chunk_size=args.patch_chunk_size,
                key_chunk_size=args.key_chunk_size,
                include_full_context_mass=args.include_full_context_mass,
                make_figure=not args.no_figures,
            )
        )

    aggregate = aggregate_records(per_record)
    current_answer = (
        "Stored Level-3 QKV supports patch-level analysis. The aggregate evidence is "
        f"`{aggregate['conclusion']}`: patch queries do not all receive identical old-memory source distributions or "
        "identical memory-only output vectors."
    )
    if aggregate["conclusion"] == "mostly_full_frame_uniform":
        current_answer = (
            "Stored Level-3 QKV supports patch-level analysis. The aggregate evidence is mostly full-frame uniform: "
            "patch queries have very similar old-memory source distributions and memory-only output vectors."
        )
    elif aggregate["conclusion"] == "mixed_or_weakly_patch_specific":
        current_answer = (
            "Stored Level-3 QKV supports patch-level analysis. The aggregate evidence is mixed: memory is not perfectly "
            "uniform over patches, but patch-specific structure is weak or layer-dependent."
        )
    if args.include_full_context_mass and aggregate.get("old_memory_full_context_mass_cv", {}).get("p90", 0.0) > 0.30:
        current_answer += (
            " Full-context old-memory mass shows a layer-dependent patch-specific gate: some records vary strongly across "
            "patches after local-window competition is included."
        )
    summary = {
        "status": "complete",
        "config": str(Path(args.config).expanduser()),
        "qkv_dir": str(dirs["qkv_dir"]),
        "metadata_dir": str(dirs["metadata_dir"]),
        "output_dir": str(output_dir),
        "device": device_name,
        "include_full_context_mass": bool(args.include_full_context_mass),
        "aggregate": aggregate,
        "records": per_record,
        "current_answer": current_answer,
    }
    write_json(output_dir / "per_record_metrics.json", per_record)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, per_record)
    print(f"Wrote patch memory specificity summary: {output_dir / 'summary.json'}")
    print(current_answer)


if __name__ == "__main__":
    main()
