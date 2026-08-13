#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from probe_utils import (
    load_metadata,
    log,
    make_contact_sheet,
    natural_key,
    save_metadata,
    token_files,
    write_summary_md,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cross-frame token cosine correspondence.")
    parser.add_argument("--input", required=True, help="Scene output directory.")
    parser.add_argument("--stage", default="auto", help="stage_XX or auto.")
    parser.add_argument("--src-frame", type=int, default=0, help="0-based sampled frame index.")
    parser.add_argument("--tgt-frame", type=int, default=5, help="0-based sampled frame index.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--grid", type=int, default=4, help="Query grid size, e.g. 4 gives 4x4 points.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def choose_stage(scene_dir: Path, requested: str) -> Path:
    files = token_files(scene_dir)
    if not files:
        raise FileNotFoundError(f"No token files found under {scene_dir / 'tokens'}")
    if requested != "auto":
        path = scene_dir / "tokens" / f"{requested}_tokens.pt"
        if not path.is_file():
            available = ", ".join(p.stem.replace("_tokens", "") for p in files)
            raise FileNotFoundError(f"Stage {requested} not found. Available: {available}")
        return path
    chosen = files[len(files) // 2]
    log(f"Auto-selected correspondence stage: {chosen.stem.replace('_tokens', '')}")
    return chosen


def load_tokens(stage_file: Path) -> tuple[torch.Tensor, str]:
    payload = torch.load(stage_file, map_location="cpu", weights_only=False)
    tokens = payload["tokens"].float()
    if tokens.ndim == 3:
        hw = payload.get("token_hw")
        if not hw:
            raise ValueError(f"{stage_file} has [T,N,C] tokens but no token_hw")
        tokens = tokens.reshape(tokens.shape[0], int(hw[0]), int(hw[1]), tokens.shape[-1])
    if tokens.ndim != 4:
        raise ValueError(f"Unsupported token shape {tuple(tokens.shape)}")
    stage = payload.get("stage_name", stage_file.stem.replace("_tokens", ""))
    return tokens.contiguous(), stage


def frame_paths(scene_dir: Path, metadata: dict) -> List[Path]:
    paths = [Path(p) for p in metadata.get("frame_paths", [])]
    if paths and all(p.is_file() for p in paths):
        return paths
    return sorted((scene_dir / "frames").glob("frame_*.*"), key=lambda p: natural_key(p.name))


def query_grid(h: int, w: int, grid: int) -> List[Tuple[int, int]]:
    grid = max(1, grid)
    if h > 2:
        ys = np.linspace(1, h - 2, grid).round().astype(int)
    else:
        ys = np.linspace(0, h - 1, min(grid, h)).round().astype(int)
    if w > 2:
        xs = np.linspace(1, w - 2, grid).round().astype(int)
    else:
        xs = np.linspace(0, w - 1, min(grid, w)).round().astype(int)
    points = []
    for y in ys:
        for x in xs:
            point = (int(y), int(x))
            if point not in points:
                points.append(point)
    return points


def token_to_pixel(y: int, x: int, token_h: int, token_w: int, image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    px = int(round((x + 0.5) / token_w * width))
    py = int(round((y + 0.5) / token_h * height))
    return px, py


def draw_marker(img: Image.Image, xy: tuple[int, int], color=(255, 40, 40), radius: int = 8) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    draw.line((x - radius - 3, y, x + radius + 3, y), fill=color, width=3)
    draw.line((x, y - radius - 3, x, y + radius + 3), fill=color, width=3)
    return out


def simple_jet(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def normalize_heatmap(sim: torch.Tensor) -> np.ndarray:
    arr = sim.numpy()
    lo, hi = np.percentile(arr, [1, 99])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def label_image(img: Image.Image, text: str) -> Image.Image:
    font = ImageFont.load_default()
    strip_h = 20
    out = Image.new("RGB", (img.width, img.height + strip_h), (245, 245, 245))
    out.paste(img.convert("RGB"), (0, strip_h))
    draw = ImageDraw.Draw(out)
    draw.text((5, 4), text, fill=(20, 20, 20), font=font)
    return out


def compose_row(panels: Sequence[Image.Image], output_path: Path) -> None:
    height = max(p.height for p in panels)
    padded = []
    for panel in panels:
        if panel.height != height:
            canvas = Image.new("RGB", (panel.width, height), (245, 245, 245))
            canvas.paste(panel, (0, 0))
            panel = canvas
        padded.append(panel)
    row = Image.new("RGB", (sum(p.width for p in padded), height), (245, 245, 245))
    x = 0
    for panel in padded:
        row.paste(panel, (x, 0))
        x += panel.width
    row.save(output_path)


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(scene_dir)
    frames = frame_paths(scene_dir, metadata)
    if not frames:
        raise FileNotFoundError(f"No RGB frames found for {scene_dir}")

    stage_file = choose_stage(scene_dir, args.stage)
    tokens, stage = load_tokens(stage_file)
    t, h, w, c = tokens.shape
    src_id = max(0, min(args.src_frame, t - 1))
    tgt_id = max(0, min(args.tgt_frame, t - 1))
    if src_id != args.src_frame or tgt_id != args.tgt_frame:
        log(f"Frame ids clamped to src={src_id}, tgt={tgt_id} for T={t}")

    src = F.normalize(tokens[src_id].reshape(h * w, c), dim=-1, eps=1e-6).reshape(h, w, c)
    tgt = F.normalize(tokens[tgt_id].reshape(h * w, c), dim=-1, eps=1e-6).reshape(h, w, c)

    with Image.open(frames[src_id]) as img:
        src_img = img.convert("RGB")
    with Image.open(frames[tgt_id]) as img:
        tgt_img = img.convert("RGB")

    saved = []
    for query_id, (qy, qx) in enumerate(query_grid(h, w, args.grid)):
        q = src[qy, qx]
        sim = torch.einsum("hwc,c->hw", tgt, q)
        best_flat = int(sim.argmax().item())
        by, bx = divmod(best_flat, w)

        src_xy = token_to_pixel(qy, qx, h, w, src_img.size)
        tgt_xy = token_to_pixel(by, bx, h, w, tgt_img.size)
        src_marked = draw_marker(src_img, src_xy, color=(255, 45, 45))
        tgt_marked = draw_marker(tgt_img, tgt_xy, color=(255, 210, 40))

        heat = simple_jet(normalize_heatmap(sim))
        heat_img = Image.fromarray(heat).resize(tgt_img.size, Image.Resampling.BILINEAR)

        panels = [
            label_image(src_marked, f"source q{query_id:02d} ({qy},{qx})"),
            label_image(tgt_marked, f"target max ({by},{bx})"),
            label_image(heat_img, "target cosine heatmap"),
        ]
        out_path = output_dir / f"{stage}_src{src_id:03d}_tgt{tgt_id:03d}_query_{query_id:02d}.png"
        compose_row(panels, out_path)
        saved.append(out_path)

    make_contact_sheet(
        saved,
        output_dir / f"{stage}_src{src_id:03d}_tgt{tgt_id:03d}_contact_sheet.png",
        cols=2,
        thumb_width=420,
    )

    latest = load_metadata(scene_dir) or metadata
    latest.setdefault("correspondence", {})
    run_record = {
        "success": True,
        "stage": stage,
        "src_frame": src_id,
        "tgt_frame": tgt_id,
        "query_count": len(saved),
        "output": str(output_dir),
    }
    latest["correspondence"].update(run_record)
    runs = latest["correspondence"].setdefault("runs", [])
    run_key = (stage, src_id, tgt_id)
    runs = [
        run
        for run in runs
        if (run.get("stage"), run.get("src_frame"), run.get("tgt_frame")) != run_key
    ]
    runs.append(run_record)
    latest["correspondence"]["runs"] = sorted(
        runs,
        key=lambda run: (natural_key(run.get("stage", "")), run.get("src_frame", 0), run.get("tgt_frame", 0)),
    )
    save_metadata(scene_dir, latest)
    write_summary_md(scene_dir)
    log(f"Correspondence visualization written to {output_dir}")


if __name__ == "__main__":
    main()
