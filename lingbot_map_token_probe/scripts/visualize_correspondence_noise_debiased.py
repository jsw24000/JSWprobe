#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from noise_slot_utils import (
    append_run_log,
    colorize_map,
    compose_row,
    cosine_heatmap,
    draw_marker,
    label_image,
    load_real_tokens,
    make_contact_sheet,
    overlay_heatmap,
    parse_query_points,
    project_to_orthogonal_complement,
    query_grid,
    resolve_config,
    token_to_pixel,
    topk_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw vs noise-slot-SVD debiased correspondence.")
    parser.add_argument("--config", default="configs/noise_slot_svd_debias.yaml")
    parser.add_argument("--query-points", default=None, help='Optional "h,w;h,w" query list override.')
    return parser.parse_args()


def argmax_hw(sim: torch.Tensor) -> tuple[int, int]:
    idx = int(sim.reshape(-1).argmax().item())
    h, w = sim.shape
    return divmod(idx, w)


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    output_root = Path(cfg["output_root"])
    corr_cfg = cfg.get("correspondence", {})
    vis_cfg = cfg.get("visualization", {})
    proj_cfg = cfg.get("projection", {})
    cmap = vis_cfg.get("heatmap_colormap", "magma")
    alpha = float(vis_cfg.get("overlay_alpha", 0.55))
    save_raw_npy = bool(vis_cfg.get("save_raw_heatmap_npy", True))
    save_deb_npy = bool(vis_cfg.get("save_debiased_heatmap_npy", True))
    append_run_log(cfg, "Starting visualize_correspondence_noise_debiased.py")

    frame_paths = [Path(p) for p in cfg["frame_paths"]]
    src_frame = int(corr_cfg.get("src_frame", 0))
    tgt_frame = int(corr_cfg.get("tgt_frame", 5))
    with Image.open(frame_paths[src_frame]) as img:
        src_img = img.convert("RGB")
    with Image.open(frame_paths[tgt_frame]) as img:
        tgt_img = img.convert("RGB")

    manual_queries = parse_query_points(args.query_points) or parse_query_points(corr_cfg.get("query_points"))

    for stage in cfg["stages_resolved"]:
        real_tokens, _ = load_real_tokens(cfg["input_root"], stage)
        t, h, w, c = real_tokens.shape
        if src_frame >= t or tgt_frame >= t:
            append_run_log(cfg, f"Warning: skipping {stage}; src/tgt frame outside T={t}")
            continue
        queries = manual_queries or query_grid(h, w, int(corr_cfg.get("query_grid", 4)))
        basis_root = output_root / "03_position_subspaces" / stage

        for k in cfg.get("k_list", [1, 2, 4, 8, 16]):
            k = int(k)
            basis_path = basis_root / f"noise_slot_svd_k{k:02d}.pt"
            if not basis_path.is_file():
                append_run_log(cfg, f"Warning: missing basis {basis_path}; skipping {stage} k={k}")
                continue
            basis_payload = torch.load(basis_path, map_location="cpu", weights_only=False)
            basis = basis_payload["basis"].float()
            mean = basis_payload["mean"].float()
            debiased = project_to_orthogonal_complement(
                real_tokens,
                mean=mean,
                basis=basis,
                normalize_before=bool(proj_cfg.get("normalize_before", True)),
                center_mode=proj_cfg.get("center_mode", "subspace_mean"),
                normalize_after=bool(proj_cfg.get("normalize_after", True)),
            )

            out_dir = output_root / "04_debiased_correspondence" / stage / f"k{k:02d}"
            raw_dir = out_dir / "heatmaps_raw"
            deb_dir = out_dir / "heatmaps_debiased"
            raw_dir.mkdir(parents=True, exist_ok=True)
            deb_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            query_records = []

            for query_id, (qh, qw) in enumerate(queries):
                if not (0 <= qh < h and 0 <= qw < w):
                    append_run_log(cfg, f"Warning: skipping invalid query {(qh, qw)} for {stage}")
                    continue
                raw_sim = cosine_heatmap(real_tokens[src_frame, qh, qw], real_tokens[tgt_frame])
                deb_sim = cosine_heatmap(debiased[src_frame, qh, qw], debiased[tgt_frame])
                raw_ah, raw_aw = argmax_hw(raw_sim)
                deb_ah, deb_aw = argmax_hw(deb_sim)

                if save_raw_npy:
                    np.save(raw_dir / f"query_{query_id:02d}.npy", raw_sim.numpy())
                if save_deb_npy:
                    np.save(deb_dir / f"query_{query_id:02d}.npy", deb_sim.numpy())

                qx, qy = token_to_pixel(qh, qw, h, w, src_img.size)
                raw_px = token_to_pixel(raw_ah, raw_aw, h, w, tgt_img.size)
                deb_px = token_to_pixel(deb_ah, deb_aw, h, w, tgt_img.size)

                source_panel = draw_marker(src_img, (qx, qy), color=(255, 45, 45))
                target_panel = tgt_img
                raw_panel = overlay_heatmap(tgt_img, raw_sim.numpy(), cmap, alpha)
                raw_panel = draw_marker(raw_panel, raw_px, color=(255, 230, 40))
                deb_panel = overlay_heatmap(tgt_img, deb_sim.numpy(), cmap, alpha)
                deb_panel = draw_marker(deb_panel, deb_px, color=(40, 240, 255))
                diff_panel = colorize_map((raw_sim - deb_sim).numpy(), "coolwarm").resize(tgt_img.size, Image.Resampling.BILINEAR)

                panels = [
                    label_image(source_panel, f"source q{query_id:02d} ({qh},{qw})"),
                    label_image(target_panel, "target RGB"),
                    label_image(raw_panel, f"raw max ({raw_ah},{raw_aw})"),
                    label_image(deb_panel, f"debiased k{k:02d} max ({deb_ah},{deb_aw})"),
                    label_image(diff_panel, "raw - debiased"),
                ]
                out_path = out_dir / f"query_{query_id:02d}_raw_vs_debiased.png"
                compose_row(panels, out_path)
                saved.append(out_path)

                query_records.append(
                    {
                        "query_id": int(query_id),
                        "qh": int(qh),
                        "qw": int(qw),
                        "query_pixel_x": int(qx),
                        "query_pixel_y": int(qy),
                        "raw_argmax_h": int(raw_ah),
                        "raw_argmax_w": int(raw_aw),
                        "debiased_argmax_h": int(deb_ah),
                        "debiased_argmax_w": int(deb_aw),
                        "raw_top5": topk_entries(raw_sim, 5),
                        "debiased_top5": topk_entries(deb_sim, 5),
                    }
                )

            make_contact_sheet(saved, out_dir / "contact_sheet.png", cols=2, thumb_width=480)
            metadata = {
                "stage": stage,
                "k": k,
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "projection": proj_cfg,
                "queries": query_records,
            }
            (out_dir / "query_metadata.json").write_text(json.dumps(metadata, indent=2))
            append_run_log(cfg, f"Saved raw-vs-debiased correspondence for {stage} k={k}: {len(saved)} queries")

    append_run_log(cfg, "Finished visualize_correspondence_noise_debiased.py")


if __name__ == "__main__":
    main()
