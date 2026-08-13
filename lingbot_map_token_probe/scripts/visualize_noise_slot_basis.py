#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from noise_slot_utils import append_run_log, colorize_map, load_real_tokens, make_contact_sheet, resolve_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize noise-slot SVD basis score maps.")
    parser.add_argument("--config", default="configs/noise_slot_svd_debias.yaml")
    return parser.parse_args()


def choose_basis_file(stage_dir: Path) -> Path | None:
    files = sorted(stage_dir.glob("noise_slot_svd_k*.pt"))
    if not files:
        return None
    with_rank3 = []
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("k", 0)) >= 3:
            with_rank3.append(path)
    return with_rank3[-1] if with_rank3 else files[-1]


def save_score_map(score: torch.Tensor, out_path: Path, size: tuple[int, int], cmap: str) -> Path:
    img = colorize_map(score.detach().cpu().numpy(), cmap).resize(size, Image.Resampling.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    output_root = Path(cfg["output_root"])
    cmap = cfg.get("visualization", {}).get("heatmap_colormap", "magma")
    append_run_log(cfg, "Starting visualize_noise_slot_basis.py")

    frame_paths = [Path(p) for p in cfg["frame_paths"]]
    with Image.open(frame_paths[0]) as img:
        frame_size = img.size

    for stage in cfg["stages_resolved"]:
        stage_dir = output_root / "03_position_subspaces" / stage
        basis_file = choose_basis_file(stage_dir)
        if basis_file is None:
            append_run_log(cfg, f"Warning: no basis file for {stage}; skipping basis visualization")
            continue
        payload = torch.load(basis_file, map_location="cpu", weights_only=False)
        basis = payload["basis"].float()
        mean = payload["mean"].float()
        p = payload["mean_noise_slot_field"].float()
        num_basis = min(3, basis.shape[1])
        map_dir = stage_dir / "basis_score_maps"
        map_dir.mkdir(parents=True, exist_ok=True)

        centered = p - mean
        for i in range(num_basis):
            score = torch.einsum("hwc,c->hw", centered, basis[:, i])
            save_score_map(score, map_dir / f"basis_{i:02d}_mean_noise_slot.png", frame_size, cmap)

        real_tokens, _ = load_real_tokens(cfg["input_root"], stage)
        real_norm = F.normalize(real_tokens.float(), dim=-1, eps=1e-8) - mean
        for i in range(num_basis):
            basis_i_dir = map_dir / f"basis_{i:02d}_real_frames"
            basis_i_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            score = torch.einsum("thwc,c->thw", real_norm, basis[:, i])
            for t in range(score.shape[0]):
                out_path = basis_i_dir / f"frame_{t:03d}_basis_{i:02d}.png"
                saved.append(save_score_map(score[t], out_path, frame_size, cmap))
            make_contact_sheet(
                saved,
                map_dir / f"basis_{i:02d}_real_frames_contact_sheet.png",
                cols=4,
                thumb_width=320,
            )

        append_run_log(cfg, f"Saved basis score maps for {stage} using {basis_file.name}")

    append_run_log(cfg, "Finished visualize_noise_slot_basis.py")


if __name__ == "__main__":
    main()
