#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.io import load_config, output_root, write_text  # noqa: E402


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def csv_head(path: Path, n: int = 5) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))[:n]


def metric_file_list(root: Path) -> list[str]:
    metrics = root / "metrics"
    if not metrics.exists():
        return []
    return [str(path.relative_to(root)) for path in sorted(metrics.glob("*.csv"))]


def cache_highlight(root: Path) -> str:
    path = root / "reconstruction" / "seen_then_removed" / "cache_log.json"
    data = load_json_if_exists(path)
    if not data:
        return "Cache log not found yet."
    record = next((row for row in data if row.get("current_frame") == 56), None)
    if not record:
        return "Frame 56 cache record not found."
    return (
        f"At frame 56, history window before current is "
        f"{record['local_window_history_source_frame_ids_before_current']}; "
        f"target-visible first-loop frames in that local history: "
        f"{record['target_visible_frames_in_local_history_before_current']}; "
        f"target-visible frames retained as special tokens: "
        f"{record['target_visible_frames_as_special_tokens_after_step']}."
    )


def build_report(cfg: dict[str, Any]) -> str:
    root = output_root(cfg)
    image_token_pca_dir = root / "analysis" / "image_token_pca" / "layer_11_frames_0068_0083"
    validation = load_json_if_exists(root / "validation" / "input_validation.json")
    sanity = load_json_if_exists(root / "validation" / "sanity_checks.json")
    modules = load_json_if_exists(root / "validation" / "lingbot_module_inspection.json")
    depth_summary = load_json_if_exists(root / "analysis" / "depth" / "summary.json")
    point_summary = load_json_if_exists(root / "analysis" / "pointcloud" / "summary.json")
    camera_summary = load_json_if_exists(root / "analysis" / "camera" / "summary.json")
    image_summary = load_json_if_exists(root / "analysis" / "image_token_l2" / "summary.json")
    memory_summary = load_json_if_exists(root / "analysis" / "memory_token_simple" / "summary.json")

    lines = [
        "# Revisit Memory Summary",
        "",
        "## Actual LingBot-Map Configuration",
        "",
        f"- mode: `{cfg['model']['mode']}`",
        f"- anchor/scale frames: `{cfg['model']['anchor_frame_count']}`",
        f"- local window: `{cfg['model']['local_window_size']}`",
        f"- keyframe interval: `{cfg['model']['keyframe_interval']}`",
        f"- Video RoPE / 3D RoPE enabled: `{cfg['model']['enable_3d_rope']}`",
        f"- checkpoint: `{cfg['paths']['model_path']}`",
        f"- DINOv2 preinit loaded separately: `{cfg['model'].get('load_dinov2_preinit', False)}`",
        f"- data root: `{cfg['paths']['data_root']}`",
        "",
        "## Hooked Modules And Token Layout",
        "",
    ]
    build = (modules or {}).get("build_inspection") if modules else None
    if build:
        lines.extend(
            [
                f"- build status: `{build.get('built')}`",
                f"- aggregator: `{build.get('aggregator_class')}`",
                f"- output heads enabled: `{build.get('output_heads_enabled')}`",
                f"- frame block count: `{build.get('frame_block_count')}`",
                f"- global block count: `{build.get('global_block_count')}`",
                f"- selected layers: `{build.get('selected_layers_available') or build.get('selected_layers_requested')}`",
                f"- token layout: `{build.get('token_layout')}`",
            ]
        )
    else:
        lines.append("- Module inspection has not been run yet.")

    lines.extend(
        [
            "",
            "## Input And Cache Checks",
            "",
            f"- input validation status: `{(validation or {}).get('status', 'not_run')}`",
            f"- validation warnings: `{(validation or {}).get('warnings', [])}`",
            f"- feature/memory sanity status: `{(sanity or {}).get('status', 'not_run')}`",
            f"- sanity caveats: `{(sanity or {}).get('caveats', [])}`",
            f"- local-window verification: {cache_highlight(root)}",
            "",
            "## Derived Point Cloud",
            "",
            f"- summary: `{root / 'analysis' / 'pointcloud' / 'summary.json'}`",
            f"- ghost occupancy delta: `{(point_summary or {}).get('ghost_occupancy_delta', 'not_available')}`",
            "",
            "## Depth",
            "",
            f"- summary: `{root / 'analysis' / 'depth' / 'summary.json'}`",
            f"- shared scale: `{root / 'analysis' / 'depth' / 'shared_depth_scale.json'}`",
            f"- same-scale condition differences: `{root / 'metrics' / 'depth_condition_differences.csv'}`",
            f"- scale/equivalence checks: `{root / 'metrics' / 'depth_scale_equivalence.csv'}`",
            f"- weighted summaries: `{depth_summary or 'not_available'}`",
            "",
            "## Camera",
            "",
            f"- summary: `{root / 'analysis' / 'camera' / 'summary.json'}`",
            f"- pose comparison note: full-sequence Sim(3) alignment only; no local 56-83-only alignment.",
            "",
            "## Image Tokens",
            "",
            f"- summary: `{image_summary or 'not_available'}`",
            f"- metrics: `{root / 'metrics' / 'image_token_metrics.csv'}` and `{root / 'metrics' / 'object_direction_metrics.csv'}`",
            f"- patch samples: `{root / 'metrics' / 'image_token_patch_samples.csv'}`",
            f"- PCA RGB maps: `{image_token_pca_dir}`",
            f"- PCA summary: `{image_token_pca_dir / 'pca_summary.csv'}`",
            "",
            "## Memory Tokens",
            "",
            f"- summary: `{memory_summary or 'not_available'}`",
            f"- band summary: `{root / 'metrics' / 'memory_token_band_summary.csv'}`",
            f"- sampled source frames: `{root / 'metrics' / 'memory_token_sampled_metrics.csv'}`",
            "- memory analysis uses saved special/context tokens, not raw K/V.",
            "",
            "## Output Files",
            "",
        ]
    )
    for rel in metric_file_list(root):
        lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Geometry ghost requires pointcloud occupancy and/or signed depth residual evidence in 56-83, not just memory-token differences.",
            "- Positive object-direction cosine means representation shifts toward the real-object direction, but it is not by itself proof that the model fully remembered the object.",
            "- Point-cloud occupancy here is derived from depth-head output plus GT camera poses; confirm any occupancy signal against signed depth residuals.",
        ]
    )
    if camera_summary:
        lines.extend(["", "<!-- camera_summary_json", json.dumps(camera_summary, indent=2), "-->"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate revisit-memory summary report.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report_path = output_root(cfg) / "report" / "summary.md"
    write_text(report_path, build_report(cfg))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
