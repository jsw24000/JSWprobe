from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

from probe_utils import load_metadata, log, make_contact_sheet, natural_key, token_files


DEFAULT_CONFIG: Dict[str, Any] = {
    "scene": "scene0000_00",
    "input_root": "outputs/scene0000_00",
    "lingbot_root": "../lingbot-map",
    "output_root": "outputs/scene0000_00/05_noise_slot_svd_debias",
    "stages": "all",
    "slot_indices": "all",
    "max_slots": 6,
    "max_noise_samples": None,
    "noise_types": ["gaussian", "smooth_noise", "gray", "hgrad", "vgrad"],
    "noise_seeds": [0, 1],
    "storage_dtype": "float16",
    "k_list": [1, 2, 4, 8, 16],
    "normalize_before_svd": "l2",
    "center_before_svd": True,
    "projection": {
        "normalize_before": True,
        "center_mode": "subspace_mean",
        "normalize_after": True,
    },
    "correspondence": {
        "src_frame": 0,
        "tgt_frame": 5,
        "query_grid": 4,
        "query_points": None,
    },
    "visualization": {
        "save_raw_heatmap_npy": True,
        "save_debiased_heatmap_npy": True,
        "heatmap_colormap": "magma",
        "overlay_alpha": 0.55,
    },
    "metrics": {
        "compute_positional_r2": True,
        "compute_same_position_distance": True,
        "compute_peakiness": True,
        "compute_entropy": True,
        "compute_top1_top2_gap": True,
        "softmax_temperature": 0.07,
    },
    "model": {
        "image_size": 518,
        "patch_size": 14,
        "selected_idx": [4, 11, 17, 23],
        "num_scale_frames": 8,
        "max_frame_num": 1024,
        "kv_cache_sliding_window": 64,
        "camera_num_iterations": 1,
        "use_sdpa": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def project_root_from_config(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().resolve().parents[1]


def resolve_existing_lingbot_root(path: Path) -> Path:
    if (path / "lingbot_map").is_dir():
        return path
    parent = path.parent
    candidates = [
        parent / "lingbot-map",
        parent / "Lingbot-map",
        parent / "lingbot_map",
        Path("/home/3dsm/Desktop/JSWprobe/lingbot-map"),
    ]
    for candidate in candidates:
        if (candidate / "lingbot_map").is_dir():
            return candidate.resolve()
    return path


def rel_or_abs(path: str | Path, root: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def parse_stage_list(stages: Any, input_root: Path) -> List[str]:
    available = [p.stem.replace("_tokens", "") for p in token_files(input_root)]
    available = sorted(available, key=natural_key)
    if stages in (None, "all"):
        return available
    if isinstance(stages, str):
        values = [s.strip() for s in stages.split(",") if s.strip()]
    else:
        values = list(stages)
    missing = [s for s in values if s not in available]
    if missing:
        raise FileNotFoundError(f"Requested stages not found: {missing}. Available: {available}")
    return values


def parse_int_list(value: Any, count: int, max_items: Optional[int] = None) -> List[int]:
    if value in (None, "all"):
        values = list(range(count))
    elif isinstance(value, str):
        values = [int(v.strip()) for v in value.split(",") if v.strip()]
    else:
        values = [int(v) for v in value]
    values = [v for v in values if 0 <= v < count]
    values = sorted(dict.fromkeys(values))
    if max_items is not None and max_items > 0 and len(values) > max_items:
        if max_items == 1:
            values = [values[0]]
        else:
            picks = np.linspace(0, len(values) - 1, max_items).round().astype(int)
            values = [values[i] for i in picks]
    return values


def capped_sample_specs(
    slots: Sequence[int],
    noise_types: Sequence[str],
    seeds: Sequence[int],
    max_noise_samples: Optional[int],
) -> List[Dict[str, Any]]:
    specs = []
    sample_id = 0
    for slot in slots:
        for noise_type in noise_types:
            for seed in seeds:
                specs.append(
                    {
                        "sample_id": sample_id,
                        "slot_index_in_sampled_sequence": int(slot),
                        "noise_type": str(noise_type),
                        "seed": int(seed),
                    }
                )
                sample_id += 1
    if max_noise_samples is not None and max_noise_samples > 0 and len(specs) > max_noise_samples:
        picks = np.linspace(0, len(specs) - 1, max_noise_samples).round().astype(int)
        specs = [specs[i] for i in picks]
        for new_id, spec in enumerate(specs):
            spec["sample_id"] = new_id
    return specs


def resolve_config(config_path: str | Path, save: bool = True) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    project_root = config_path.parents[1]
    user_cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    cfg["config_path"] = str(config_path)
    cfg["project_root"] = str(project_root)

    input_root = rel_or_abs(cfg["input_root"], project_root)
    output_root = rel_or_abs(cfg["output_root"], project_root)
    lingbot_root = resolve_existing_lingbot_root(rel_or_abs(cfg["lingbot_root"], project_root))
    cfg["input_root"] = str(input_root)
    cfg["output_root"] = str(output_root)
    cfg["lingbot_root"] = str(lingbot_root)

    metadata = load_metadata(input_root)
    if not metadata:
        raise FileNotFoundError(f"Could not read metadata.json under {input_root}")
    frame_paths = resolve_frame_paths(input_root, metadata)
    cfg["scene"] = cfg.get("scene") or metadata.get("scene") or input_root.name
    cfg["frame_indices"] = metadata.get("frame_indices", list(range(len(frame_paths))))
    cfg["frame_paths"] = [str(p) for p in frame_paths]
    cfg["original_sizes"] = metadata.get("original_sizes", [])
    cfg["model_input_hw"] = metadata.get("model_input_hw")
    cfg["model_path"] = str(rel_or_abs(metadata.get("model_path", lingbot_root / "checkpoints/lingbot-map.pt"), project_root))
    if not Path(cfg["model_path"]).is_file():
        fallback = lingbot_root / "checkpoints" / "lingbot-map.pt"
        cfg["model_path"] = str(fallback)

    cfg["stages_resolved"] = parse_stage_list(cfg.get("stages"), input_root)
    cfg["slot_indices_resolved"] = parse_int_list(
        cfg.get("slot_indices", "all"),
        count=len(frame_paths),
        max_items=cfg.get("max_slots"),
    )
    cfg["sample_specs"] = capped_sample_specs(
        cfg["slot_indices_resolved"],
        cfg.get("noise_types", []),
        [int(s) for s in cfg.get("noise_seeds", [])],
        cfg.get("max_noise_samples"),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "00_config").mkdir(parents=True, exist_ok=True)
    if save:
        write_yaml(output_root / "00_config" / "resolved_config.yaml", cfg)
    return cfg


def write_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def append_run_log(cfg: Dict[str, Any], message: str) -> None:
    out = Path(cfg["output_root"]) / "00_config" / "run_log.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(f"[noise-slot] {message}", flush=True)
    with out.open("a") as handle:
        handle.write(line + "\n")


def resolve_frame_paths(input_root: Path, metadata: Dict[str, Any]) -> List[Path]:
    paths = [Path(p) for p in metadata.get("frame_paths", [])]
    resolved = []
    for path in paths:
        if path.is_file():
            resolved.append(path)
        elif (input_root / path).is_file():
            resolved.append(input_root / path)
        elif (input_root.parent / path).is_file():
            resolved.append(input_root.parent / path)
        else:
            resolved.append(path)
    if resolved and all(p.is_file() for p in resolved):
        return resolved
    frames = sorted((input_root / "frames").glob("frame_*.*"), key=lambda p: natural_key(p.name))
    if frames:
        return frames
    raise FileNotFoundError(f"No frames found under {input_root}")


def load_real_tokens(input_root: str | Path, stage: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    path = Path(input_root) / "tokens" / f"{stage}_tokens.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tokens = payload["tokens"].float()
    if tokens.ndim == 3:
        hw = payload.get("token_hw")
        if not hw:
            raise ValueError(f"{path} has [T,N,C] tokens but no token_hw")
        tokens = tokens.reshape(tokens.shape[0], int(hw[0]), int(hw[1]), tokens.shape[-1])
    if tokens.ndim != 4:
        raise ValueError(f"{path} has unsupported token shape {tuple(tokens.shape)}")
    return tokens.contiguous(), payload


def add_lingbot_to_path(lingbot_root: str | Path) -> None:
    root = Path(lingbot_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_lingbot_aggregator_model(cfg: Dict[str, Any]):
    add_lingbot_to_path(cfg["lingbot_root"])
    from lingbot_map.models.gct_stream import GCTStream

    model_cfg = cfg.get("model", {})
    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = GCTStream(
        img_size=int(model_cfg.get("image_size", 518)),
        patch_size=int(model_cfg.get("patch_size", 14)),
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_local_point=False,
        enable_track=False,
        enable_3d_rope=True,
        max_frame_num=int(model_cfg.get("max_frame_num", 1024)),
        kv_cache_sliding_window=int(model_cfg.get("kv_cache_sliding_window", 64)),
        kv_cache_scale_frames=int(model_cfg.get("num_scale_frames", 8)),
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=bool(model_cfg.get("use_sdpa", True)),
        camera_num_iterations=int(model_cfg.get("camera_num_iterations", 1)),
        use_gradient_checkpoint=False,
    )
    ckpt = torch.load(cfg["model_path"], map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    append_run_log(cfg, f"Loaded Lingbot checkpoint. Missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device).eval()
    model.requires_grad_(False)

    if device.type == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        except Exception:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype=dtype)
    return model, device, dtype


def preprocess_sequence(cfg: Dict[str, Any], paths: Sequence[str | Path]) -> torch.Tensor:
    add_lingbot_to_path(cfg["lingbot_root"])
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    model_cfg = cfg.get("model", {})
    return load_and_preprocess_images(
        [str(p) for p in paths],
        mode="crop",
        image_size=int(model_cfg.get("image_size", 518)),
        patch_size=int(model_cfg.get("patch_size", 14)),
    )


def run_aggregator_tokens(
    cfg: Dict[str, Any],
    model,
    device: torch.device,
    dtype: torch.dtype,
    image_paths: Sequence[str | Path],
) -> Dict[str, torch.Tensor]:
    images = preprocess_sequence(cfg, image_paths).unsqueeze(0).to(device)
    if hasattr(model.aggregator, "clean_kv_cache"):
        model.aggregator.clean_kv_cache()
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()
    with torch.no_grad(), autocast_ctx:
        aggregated, patch_start_idx = model._aggregate_features(
            images,
            num_frame_for_scale=min(int(cfg.get("model", {}).get("num_scale_frames", 8)), images.shape[1]),
            num_frame_per_block=images.shape[1],
        )
    if hasattr(model.aggregator, "clean_kv_cache"):
        model.aggregator.clean_kv_cache()

    out = {}
    model_input_hw = [int(images.shape[-2]), int(images.shape[-1])]
    patch_size = int(cfg.get("model", {}).get("patch_size", 14))
    for i, raw in enumerate(aggregated):
        stage = f"stage_{i:02d}"
        spatial = raw[0, :, int(patch_start_idx) :, :].detach().float().cpu()
        h = model_input_hw[0] // patch_size
        w = model_input_hw[1] // patch_size
        if h * w != spatial.shape[1]:
            root = int(math.sqrt(spatial.shape[1]))
            if root * root == spatial.shape[1]:
                h = w = root
            else:
                raise ValueError(f"Cannot reshape {stage} tokens with N={spatial.shape[1]}")
        out[stage] = spatial.reshape(spatial.shape[0], h, w, spatial.shape[-1]).contiguous()
    return out


def generate_noise_image(
    noise_type: str,
    seed: int,
    size: Tuple[int, int],
    gray_value: int = 127,
) -> Image.Image:
    width, height = size
    rng = np.random.default_rng(seed)
    if noise_type == "gaussian":
        arr = rng.normal(loc=127.5, scale=52.0, size=(height, width, 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif noise_type == "smooth_noise":
        low_h = max(1, height // 16)
        low_w = max(1, width // 16)
        low = rng.random((low_h, low_w, 3), dtype=np.float32)
        img = Image.fromarray((low * 255).astype(np.uint8), mode="RGB")
        img = img.resize((width, height), Image.Resampling.BILINEAR)
        arr = np.asarray(img).astype(np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
        arr = (arr * 255).astype(np.uint8)
    elif noise_type == "gray":
        arr = np.full((height, width, 3), gray_value, dtype=np.uint8)
    elif noise_type == "hgrad":
        grad = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
        arr = np.repeat(np.repeat(grad, height, axis=0), 3, axis=2)
    elif noise_type == "vgrad":
        grad = np.linspace(0, 255, height, dtype=np.uint8)[:, None, None]
        arr = np.repeat(np.repeat(grad, width, axis=1), 3, axis=2)
    elif noise_type == "black":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
    elif noise_type == "white":
        arr = np.full((height, width, 3), 255, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return Image.fromarray(arr, mode="RGB")


def save_noise_examples(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["output_root"]) / "01_noise_inputs"
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(cfg["frame_paths"][0]) as img:
        size = img.size
    saved = []
    for noise_type in cfg.get("noise_types", []):
        seeds = cfg.get("noise_seeds", [0])
        seed_values = seeds[:2] if noise_type in {"gaussian", "smooth_noise"} else [seeds[0] if seeds else 0]
        for seed in seed_values:
            image = generate_noise_image(noise_type, int(seed), size)
            name = f"{noise_type}_seed{int(seed):03d}.png" if noise_type in {"gaussian", "smooth_noise"} else f"{noise_type}.png"
            path = ex_dir / name
            image.save(path)
            saved.append(path)
    make_contact_sheet(saved, out_dir / "contact_sheet_noise_types.png", cols=3, thumb_width=320)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def orthonormalize_basis(basis: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
    if basis.numel() == 0:
        return basis
    eye = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    err = torch.linalg.norm(basis.T @ basis - eye).item()
    if err > tol:
        q, _ = torch.linalg.qr(basis, mode="reduced")
        return q[:, : basis.shape[1]]
    return basis


def project_to_orthogonal_complement(
    tokens: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    normalize_before: bool = True,
    center_mode: str = "subspace_mean",
    normalize_after: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = tokens.float()
    if normalize_before:
        x = l2_normalize(x, eps=eps)
    mean = mean.to(dtype=x.dtype, device=x.device)
    basis = orthonormalize_basis(basis.to(dtype=x.dtype, device=x.device))
    if center_mode == "subspace_mean":
        x = x - mean
    elif center_mode in (None, "none"):
        pass
    else:
        raise ValueError(f"Unsupported center_mode: {center_mode}")
    x_flat = x.reshape(-1, x.shape[-1])
    x_pos = (x_flat @ basis) @ basis.T
    x_perp = (x_flat - x_pos).reshape_as(x)
    if normalize_after:
        x_perp = l2_normalize(x_perp, eps=eps)
    return x_perp


def parse_query_points(value: Any) -> Optional[List[Tuple[int, int]]]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, str):
        points = []
        for item in value.split(";"):
            if not item.strip():
                continue
            y, x = item.split(",")
            points.append((int(y), int(x)))
        return points
    return [(int(p[0]), int(p[1])) for p in value]


def query_grid(h: int, w: int, grid: int) -> List[Tuple[int, int]]:
    grid = max(1, int(grid))
    ys = np.linspace(1 if h > 2 else 0, h - 2 if h > 2 else h - 1, min(grid, h)).round().astype(int)
    xs = np.linspace(1 if w > 2 else 0, w - 2 if w > 2 else w - 1, min(grid, w)).round().astype(int)
    out = []
    for y in ys:
        for x in xs:
            p = (int(y), int(x))
            if p not in out:
                out.append(p)
    return out


def token_to_pixel(qh: int, qw: int, h: int, w: int, image_size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = image_size
    return int(round((qw + 0.5) / w * width)), int(round((qh + 0.5) / h * height))


def draw_marker(img: Image.Image, xy: Tuple[int, int], color=(255, 45, 45), radius: int = 8) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    draw.line((x - radius - 3, y, x + radius + 3, y), fill=color, width=3)
    draw.line((x, y - radius - 3, x, y + radius + 3), fill=color, width=3)
    return out


def robust_normalize(arr: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def colorize_map(values: np.ndarray, cmap_name: str = "magma") -> Image.Image:
    values = robust_normalize(values)
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
        rgb = (cmap(values)[..., :3] * 255).astype(np.uint8)
    except Exception:
        v = values
        rgb = np.stack([v, np.sqrt(v), 1.0 - v], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(base: Image.Image, heatmap: np.ndarray, cmap_name: str = "magma", alpha: float = 0.55) -> Image.Image:
    color = colorize_map(heatmap, cmap_name).resize(base.size, Image.Resampling.BILINEAR)
    return Image.blend(base.convert("RGB"), color, alpha)


def label_image(img: Image.Image, text: str) -> Image.Image:
    font = ImageFont.load_default()
    strip_h = 22
    out = Image.new("RGB", (img.width, img.height + strip_h), (245, 245, 245))
    out.paste(img.convert("RGB"), (0, strip_h))
    draw = ImageDraw.Draw(out)
    draw.text((5, 5), text, fill=(20, 20, 20), font=font)
    return out


def compose_row(panels: Sequence[Image.Image], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def cosine_heatmap(src_vec: torch.Tensor, tgt_map: torch.Tensor) -> torch.Tensor:
    src = F.normalize(src_vec.float(), dim=-1, eps=1e-8)
    tgt = F.normalize(tgt_map.float(), dim=-1, eps=1e-8)
    return torch.einsum("hwc,c->hw", tgt, src)


def topk_entries(sim: torch.Tensor, k: int = 5) -> List[List[float]]:
    flat = sim.reshape(-1)
    values, indices = torch.topk(flat, k=min(k, flat.numel()))
    h, w = sim.shape
    out = []
    for value, index in zip(values.tolist(), indices.tolist()):
        y, x = divmod(int(index), w)
        out.append([int(y), int(x), float(value)])
    return out


def positional_r2(sim: np.ndarray) -> float:
    h, w = sim.shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h),
        np.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    x = np.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
            (xx * xx).reshape(-1),
            (yy * yy).reshape(-1),
            (xx * yy).reshape(-1),
            np.ones(h * w),
        ],
        axis=1,
    )
    y = sim.reshape(-1).astype(np.float64)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def heatmap_entropy(sim: np.ndarray, temperature: float = 0.07) -> float:
    flat = sim.reshape(-1).astype(np.float64) / max(temperature, 1e-8)
    flat = flat - flat.max()
    p = np.exp(flat)
    p = p / max(float(p.sum()), 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def top1_top2_gap(sim: np.ndarray) -> float:
    flat = np.sort(sim.reshape(-1))
    if flat.size < 2:
        return 0.0
    return float(flat[-1] - flat[-2])


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_subprocess(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)
