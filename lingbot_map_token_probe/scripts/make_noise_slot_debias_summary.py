#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from noise_slot_utils import append_run_log, load_metadata, resolve_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final summary for noise-slot SVD debias experiment.")
    parser.add_argument("--config", default="configs/noise_slot_svd_debias.yaml")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fmt_list(values) -> str:
    return ", ".join(str(v) for v in values)


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    output_root = Path(cfg["output_root"])
    summary_dir = output_root / "06_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    append_run_log(cfg, "Starting make_noise_slot_debias_summary.py")

    input_root = Path(cfg["input_root"])
    metadata = load_metadata(input_root)
    metrics_rows = read_csv_rows(output_root / "05_metrics" / "metrics_summary_by_stage_k.csv")

    stage_token_lines = []
    noise_token_lines = []
    evr_lines = []
    basis_lines = []
    real_token_shapes = {}

    for stage_info in metadata.get("token_stages", []):
        real_token_shapes[stage_info.get("stage_name")] = stage_info.get("tokens_shape")

    for stage in cfg["stages_resolved"]:
        real_shape = real_token_shapes.get(stage, "unknown")
        stage_token_lines.append(f"- `{stage}` real tokens: `{real_shape}`")

        noise_path = output_root / "02_noise_slot_tokens" / stage / f"noise_slot_tokens_{stage}.pt"
        if noise_path.is_file():
            noise_payload = torch.load(noise_path, map_location="cpu", weights_only=False)
            noise_token_lines.append(f"- `{stage}` noise-slot tokens: `{list(noise_payload['tokens'].shape)}`")
        else:
            noise_token_lines.append(f"- `{stage}` noise-slot tokens: missing")

        for k in cfg.get("k_list", []):
            basis_path = output_root / "03_position_subspaces" / stage / f"noise_slot_svd_k{int(k):02d}.pt"
            if not basis_path.is_file():
                evr_lines.append(f"- `{stage}` K={k}: missing")
                continue
            payload = torch.load(basis_path, map_location="cpu", weights_only=False)
            evr_lines.append(
                f"- `{stage}` K={int(k)}: explained variance ratio top-K = "
                f"`{float(payload.get('explained_variance_ratio_topk', 0.0)):.6f}`"
            )

        maps_dir = output_root / "03_position_subspaces" / stage / "basis_score_maps"
        mean_maps = [maps_dir / f"basis_{i:02d}_mean_noise_slot.png" for i in range(3)]
        real_sheets = [maps_dir / f"basis_{i:02d}_real_frames_contact_sheet.png" for i in range(3)]
        basis_lines.append(
            f"- `{stage}` basis score maps: mean maps `{sum(p.is_file() for p in mean_maps)}/3`, "
            f"real-frame sheets `{sum(p.is_file() for p in real_sheets)}/3`"
        )

    metric_lines = [
        "| stage | K | raw R2 | debiased R2 | delta R2 | raw peakiness | debiased peakiness | raw entropy | debiased entropy | argmax shift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best = None
    for row in metrics_rows:
        raw_r2 = float(row["mean_raw_positional_r2"])
        deb_r2 = float(row["mean_debiased_positional_r2"])
        delta = raw_r2 - deb_r2
        raw_peak = float(row["mean_raw_peakiness"])
        deb_peak = float(row["mean_debiased_peakiness"])
        if best is None or delta > best["delta"]:
            best = {
                "stage": row["stage"],
                "k": row["k"],
                "delta": delta,
                "raw_peak": raw_peak,
                "deb_peak": deb_peak,
                "raw_entropy": float(row["mean_raw_entropy"]),
                "deb_entropy": float(row["mean_debiased_entropy"]),
            }
        metric_lines.append(
            f"| {row['stage']} | {row['k']} | {raw_r2:.4f} | {deb_r2:.4f} | {delta:.4f} | "
            f"{raw_peak:.4f} | {deb_peak:.4f} | {float(row['mean_raw_entropy']):.3f} | "
            f"{float(row['mean_debiased_entropy']):.3f} | {float(row['mean_argmax_shift']):.3f} |"
        )
    if len(metric_lines) == 2:
        metric_lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    if best is None:
        recommendation = "No recommendation yet because metrics were not generated."
        interpretation = (
            "The pipeline did not produce metrics, so no claim can be made about whether the "
            "estimated subspace reduces coordinate-driven correspondence structure."
        )
    else:
        recommendation = f"Start inspection with `{best['stage']}` K={best['k']} because it has the largest mean positional R2 drop ({best['delta']:.4f})."
        if best["delta"] > 0.05 and best["deb_peak"] >= 0.5 * max(best["raw_peak"], 1e-8):
            interpretation = (
                "noise_slot_svd debias reduces coordinate-explainable heatmap structure while preserving "
                "visually plausible correspondence for some queries. In Chinese: 去 bias 后，heatmap 中由二维坐标解释的大范围梯度下降，"
                "同时部分 query 仍可能保持视觉上合理的局部响应，说明原始 correspondence 中确实存在二维位置驱动的成分，而正交补空间中仍保留了一部分内容/几何对应信息。"
            )
        elif best["delta"] > 0.05:
            interpretation = (
                "The removed positional subspace may contain not only harmful 2D coordinate bias but also "
                "reconstruction-useful geometry/layout information. In Chinese: 被移除的子空间可能不只是有害的二维坐标 bias，"
                "也包含 Lingbot-map 重建任务中有用的 camera-relative geometry 或 layout 信息。"
            )
        else:
            interpretation = (
                "The estimated noise-slot subspace does not fully explain the observed correspondence bias. "
                "The bias may be scene-specific, temporal-memory-related, or encoded outside a low-dimensional linear subspace. "
                "In Chinese: noise-slot 估计出的二维位置子空间不足以解释 raw correspondence 中的大范围位置激活，"
                "bias 可能来自 scene-specific layout、temporal memory，或者不是低维线性子空间能简单移除的。"
            )

    lines = [
        "# Noise-Slot SVD Debias Summary",
        "",
        "## Experiment Purpose",
        "",
        "This experiment estimates and removes a low-dimensional feature subspace associated with 2D image position bias, then compares raw and debiased token cosine correspondence.",
        "",
        "## Data",
        "",
        f"- Scene: `{cfg['scene']}`",
        f"- Input root: `{cfg['input_root']}`",
        f"- Output root: `{cfg['output_root']}`",
        f"- Real frame indices: `{cfg.get('frame_indices', [])}`",
        f"- Source frame for correspondence: sampled index `{cfg['correspondence']['src_frame']}`",
        f"- Target frame for correspondence: sampled index `{cfg['correspondence']['tgt_frame']}`",
        "",
        "## Noise-Slot Construction",
        "",
        "For each selected slot, exactly one RGB frame is replaced with a low-semantic image while all other frames remain real. Lingbot-map is then run from a clean KV-cache state, and only the replaced slot's tokens are saved.",
        "",
        f"- Slot indices used: `{cfg.get('slot_indices_resolved', [])}`",
        f"- Noise types: `{fmt_list(cfg.get('noise_types', []))}`",
        f"- Noise seeds: `{fmt_list(cfg.get('noise_seeds', []))}`",
        f"- Number of replacement sequences: `{len(cfg.get('sample_specs', []))}`",
        "",
        "## Token Shapes",
        "",
        *stage_token_lines,
        *noise_token_lines,
        "",
        "## SVD Subspaces",
        "",
        *evr_lines,
        "",
        "## Basis Visualizations",
        "",
        *basis_lines,
        "",
        "Inspect `03_position_subspaces/<stage>/basis_score_maps/` for horizontal, vertical, center-edge, corner, or global gradient patterns.",
        "",
        "## Metrics",
        "",
        *metric_lines,
        "",
        "## Recommended Stage / K",
        "",
        recommendation,
        "",
        "## Cautious Interpretation",
        "",
        interpretation,
        "",
        "This experiment estimates a low-dimensional positional subspace using real-context noise-slot replacement. By replacing only one frame with a low-semantic image while keeping the remaining sequence real, the estimated subspace is intended to capture coordinate-driven feature components under a more in-distribution Lingbot-map inference setting than an all-noise video. If the orthogonal-complement features reduce large-scale positional gradients in correspondence heatmaps while preserving visually plausible matches, this suggests that part of the raw correspondence response is driven by 2D position bias. However, the removed subspace may also contain camera-relative layout or reconstruction-useful geometric information, so the result should not be interpreted as pure semantic correspondence.",
        "",
        "## Next Steps",
        "",
        "- Inspect raw-vs-debiased contact sheets before reading metrics as performance.",
        "- Compare K=1/2/4/8/16 within each stage rather than assuming larger K is better.",
        "- Add depth/pose-based or annotation-based evaluation before making quantitative claims about correspondence quality.",
        "- Try more scenes to separate scene-specific layout from model-wide coordinate bias.",
    ]
    out_path = summary_dir / "debias_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    append_run_log(cfg, f"Wrote summary: {out_path}")
    append_run_log(cfg, "Finished make_noise_slot_debias_summary.py")


if __name__ == "__main__":
    main()
