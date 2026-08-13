from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return project_root().parent


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return expand_env_vars(data)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        expr = match.group(1)
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(expr, "")

    return re.sub(r"\$\{([^}]+)\}", repl, value)


def deep_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(data: dict[str, Any], path: str, value: Any) -> None:
    cur = data
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def scene_output_name(scene_id: str) -> str:
    return re.sub(r"_\d\d$", "", scene_id)


def safe_layer_name(layer: str, token_type: str = "patch_tokens") -> str:
    layer = str(layer)
    if "/" in layer:
        return layer.replace("/", "_")
    if layer.endswith(f"_{token_type}"):
        return layer
    return f"{layer}_{token_type}"


def layer_key(layer: str, token_type: str = "patch_tokens") -> str:
    layer = str(layer)
    if "/" in layer:
        return layer
    if layer.endswith(f"_{token_type}"):
        return layer[: -len(f"_{token_type}")] + f"/{token_type}"
    return f"{layer}/{token_type}"


def discover_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    controlled_root = resolve_path(deep_get(cfg, "paths.controlled_root"), root)
    if controlled_root is None:
        controlled_root = workspace_root() / "controlled_streaming_repr"

    experiment = deep_get(cfg, "controlled.experiment_name", "controlled_128_exp_v1")
    model = deep_get(cfg, "controlled.model_name", "lingbot-map")
    scene = deep_get(cfg, "controlled.scene_id", "scene0002_00")
    setting = deep_get(cfg, "controlled.setting_name", "order_perturbation")
    condition = deep_get(cfg, "controlled.condition_id", "order_normal_seq128_seg000")

    manifest = resolve_path(deep_get(cfg, "paths.manifest_path"), root)
    if manifest is None:
        candidate = controlled_root / "outputs" / "manifests" / experiment / scene / setting / f"{condition}.json"
        manifest = candidate if candidate.exists() else first_existing(
            sorted((controlled_root / "outputs" / "manifests" / experiment / scene).glob(f"**/{condition}.json"))
        )
    if manifest is None:
        manifest = controlled_root / "outputs" / "manifests" / experiment / scene / setting / f"{condition}.json"

    token_dir = resolve_path(deep_get(cfg, "paths.token_dir"), root)
    if token_dir is None:
        token_dir = controlled_root / "outputs" / "tokens" / experiment / model / scene / setting / condition

    pca_figures_dir = resolve_path(deep_get(cfg, "paths.pca_figures_dir"), root)
    if pca_figures_dir is None:
        pca_figures_dir = controlled_root / "outputs" / "figures" / experiment / model / scene / setting / condition

    pca_metrics = resolve_path(deep_get(cfg, "paths.pca_metrics_path"), root)
    if pca_metrics is None:
        pca_metrics = controlled_root / "outputs" / "metrics" / experiment / model / scene / setting / condition / "shared_pca_patchmap.json"

    out_root_cfg = deep_get(cfg, "project.output_root", "outputs")
    output_root = resolve_path(out_root_cfg, root) or (root / "outputs")
    scene_output = output_root / scene_output_name(scene)

    return {
        "project_root": root,
        "workspace_root": workspace_root(),
        "controlled_root": controlled_root,
        "manifest": manifest,
        "token_dir": token_dir,
        "pca_figures_dir": pca_figures_dir,
        "pca_metrics": pca_metrics,
        "output_root": output_root,
        "scene_output": scene_output,
    }


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=False)
        f.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_or_metadata(paths: dict[str, Path], cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_path = paths["manifest"]
    if manifest_path.exists():
        return load_json(manifest_path)

    token_bundle = paths["token_dir"] / "token_bundle.json"
    metadata = paths["token_dir"] / "metadata.json"
    source = token_bundle if token_bundle.exists() else metadata
    if source.exists():
        data = load_json(source)
        frame_paths = data.get("frame_paths", [])
        frames = []
        for t, rgb in enumerate(frame_paths):
            rgb_path = Path(rgb)
            frame_id = frame_id_from_path(rgb_path, fallback=f"{t:04d}")
            frames.append(
                {
                    "t": t,
                    "source_frame_id": frame_id,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(default_depth_path(rgb_path)),
                    "intrinsic_path": str(default_intrinsic_path(rgb_path)),
                }
            )
        return {
            "manifest_version": "metadata_fallback",
            "scene_id": deep_get(cfg, "controlled.scene_id", data.get("scene_id", "")),
            "setting_name": deep_get(cfg, "controlled.setting_name", data.get("setting_name", "")),
            "condition_id": deep_get(cfg, "controlled.condition_id", data.get("condition_id", "")),
            "frames": frames,
        }

    scene = deep_get(cfg, "controlled.scene_id", "scene0002_00")
    scannet_root = resolve_path(deep_get(cfg, "paths.scannet_root"), project_root())
    if scannet_root is None:
        raise FileNotFoundError(f"No manifest, token metadata, or scannet_root available for {scene}")
    scene_dir = scannet_root / scene
    color_dir = scene_dir / "color"
    frames = []
    for t, rgb_path in enumerate(sorted(color_dir.glob("*.jpg"), key=natural_key)):
        frame_id = frame_id_from_path(rgb_path, fallback=f"{t:04d}")
        frames.append(
            {
                "t": t,
                "source_frame_id": frame_id,
                "rgb_path": str(rgb_path),
                "depth_path": str(scene_dir / "depth" / f"{int(frame_id)}.png"),
                "intrinsic_path": str(scene_dir / "intrinsic" / "intrinsic_color.txt"),
            }
        )
    return {
        "manifest_version": "scannet_root_fallback",
        "scene_id": scene,
        "setting_name": deep_get(cfg, "controlled.setting_name", "raw_stream"),
        "condition_id": deep_get(cfg, "controlled.condition_id", "raw_discovered"),
        "frames": frames,
    }


def natural_key(path: Path) -> list[Any]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def frame_id_from_path(path: Path, fallback: str) -> str:
    stem = path.stem
    match = re.search(r"(\d+)$", stem)
    if match:
        return f"{int(match.group(1)):04d}"
    return fallback


def default_depth_path(rgb_path: Path) -> Path:
    parts = list(rgb_path.parts)
    if "color" in parts:
        parts[parts.index("color")] = "depth"
        return Path(*parts).with_suffix(".png")
    return rgb_path.parent.parent / "depth" / f"{int(frame_id_from_path(rgb_path, '0'))}.png"


def default_intrinsic_path(rgb_path: Path) -> Path:
    parts = list(rgb_path.parts)
    if "color" in parts:
        return Path(*parts[: parts.index("color")]) / "intrinsic" / "intrinsic_color.txt"
    return rgb_path.parent.parent / "intrinsic" / "intrinsic_color.txt"


def select_frames(frames: list[dict[str, Any]], max_frames: int | None = None) -> list[dict[str, Any]]:
    if max_frames is None or max_frames <= 0:
        return frames
    return frames[:max_frames]


def read_intrinsic(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.loadtxt(p, dtype=np.float64)
    except Exception:
        return None


def token_file_for_layer(token_dir: Path, layer: str, token_type: str) -> Path | None:
    safe = safe_layer_name(layer, token_type)
    candidates = [
        token_dir / f"{safe}.npy",
        token_dir / f"{str(layer).replace('/', '_')}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(token_dir.glob(f"*{safe}*.npy"))
    if matches:
        return matches[0]
    matches = sorted(token_dir.glob(f"*{token_type}.npy"))
    return matches[0] if matches else None


def infer_patch_grid(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    frames: list[dict[str, Any]],
) -> tuple[int, int]:
    configured = deep_get(cfg, "tokens.patch_grid")
    if isinstance(configured, (list, tuple)) and len(configured) == 2:
        return int(configured[0]), int(configured[1])

    token_type = deep_get(cfg, "tokens.token_type", "patch_tokens")
    layer = deep_get(cfg, "tokens.preferred_layer", "layer_11")
    token_path = token_file_for_layer(paths["token_dir"], layer, token_type)
    n_patches = None
    if token_path is not None and token_path.exists():
        arr = np.load(token_path, mmap_mode="r")
        if arr.ndim >= 2:
            n_patches = int(arr.shape[1])

    if n_patches is None:
        raise ValueError(
            "Could not infer patch grid because no token npy file was found. "
            "Set tokens.patch_grid in configs/default.yaml."
        )

    aspect = None
    for frame in frames:
        rgb_path = Path(frame.get("rgb_path", ""))
        if rgb_path.exists():
            with Image.open(rgb_path) as image:
                width, height = image.size
            if height > 0:
                aspect = width / height
                break
    return infer_factor_grid(n_patches, aspect)


def infer_factor_grid(n_patches: int, aspect: float | None = None) -> tuple[int, int]:
    pairs = []
    for h in range(1, int(math.sqrt(n_patches)) + 1):
        if n_patches % h == 0:
            w = n_patches // h
            pairs.append((h, w))
            if h != w:
                pairs.append((w, h))
    if not pairs:
        raise ValueError(f"Cannot factor n_patches={n_patches}")
    if aspect is None:
        return min(pairs, key=lambda hw: abs(hw[0] - hw[1]))
    return min(pairs, key=lambda hw: abs((hw[1] / hw[0]) - aspect))


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_depth(path: str | Path, depth_scale: float) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image)
    depth = arr.astype(np.float32)
    if depth_scale and depth_scale > 0:
        depth = depth / float(depth_scale)
    return depth


def resize_float_2d(image: np.ndarray, grid: tuple[int, int], resample: int = Image.Resampling.BOX) -> np.ndarray:
    h, w = grid
    pil = Image.fromarray(np.asarray(image, dtype=np.float32), mode="F")
    return np.asarray(pil.resize((w, h), resample=resample), dtype=np.float32)


def resize_rgb(image: np.ndarray, grid: tuple[int, int], resample: int = Image.Resampling.BOX) -> np.ndarray:
    h, w = grid
    arr = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    return np.asarray(pil.resize((w, h), resample=resample), dtype=np.float32) / 255.0


def pool_valid_average(values: np.ndarray, valid: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    valid_f = np.asarray(valid, dtype=np.float32)
    numerator = resize_float_2d(values * valid_f, grid)
    denom = resize_float_2d(valid_f, grid)
    out = np.full(grid, np.nan, dtype=np.float32)
    np.divide(numerator, np.maximum(denom, 1e-8), out=out, where=denom > 1e-8)
    return out, denom


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    p = np.pad(gray, 1, mode="edge")
    gx = (
        -p[:-2, :-2]
        + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2]
        + 2.0 * p[1:-1, 2:]
        - p[2:, :-2]
        + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2]
        - 2.0 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2.0 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy) / 8.0


def robust_normalize(values: np.ndarray, valid: np.ndarray | None = None, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(arr)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(mask):
        return out
    low, high = np.nanpercentile(arr[mask], [lo, hi])
    if not np.isfinite(high - low) or high <= low:
        high = float(np.nanmax(arr[mask]))
        low = float(np.nanmin(arr[mask]))
    if high <= low:
        out[mask] = 0.0
        return out
    out[mask] = np.clip((arr[mask] - low) / (high - low), 0.0, 1.0)
    return out


def build_patch_features(
    frame: dict[str, Any],
    grid: tuple[int, int],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    rgb = read_rgb(frame["rgb_path"])
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    rgb_patch = resize_rgb(rgb, grid, Image.Resampling.BOX)
    edge_patch = resize_float_2d(sobel_magnitude(gray), grid)
    edge_patch = robust_normalize(edge_patch)

    cues = []
    names = []
    h, w = grid
    yy, xx = np.meshgrid(
        (np.arange(h, dtype=np.float32) + 0.5) / h,
        (np.arange(w, dtype=np.float32) + 0.5) / w,
        indexing="ij",
    )
    if deep_get(cfg, "cues.coordinate_range", "zero_one") == "minus_one_one":
        xx = xx * 2.0 - 1.0
        yy = yy * 2.0 - 1.0
    cues.extend([xx, yy])
    names.extend(["x", "y"])
    for idx, name in enumerate(("rgb_r", "rgb_g", "rgb_b")):
        cues.append(rgb_patch[..., idx])
        names.append(name)
    cues.append(edge_patch)
    names.append("sobel_edge")

    rgb_grad_patch = None
    if bool(deep_get(cfg, "cues.include_rgb_gradient", True)):
        grads = [sobel_magnitude(rgb[..., c]) for c in range(3)]
        rgb_grad = np.mean(np.stack(grads, axis=0), axis=0)
        rgb_grad_patch = robust_normalize(resize_float_2d(rgb_grad, grid))
        cues.append(rgb_grad_patch)
        names.append("rgb_gradient")

    depth_scale = float(deep_get(cfg, "depth.depth_scale", 1000.0))
    depth = read_depth(frame["depth_path"], depth_scale)
    valid = np.isfinite(depth)
    valid &= depth > float(deep_get(cfg, "depth.valid_min_m", 0.05))
    valid &= depth < float(deep_get(cfg, "depth.valid_max_m", 20.0))
    depth_patch, valid_ratio = pool_valid_average(depth, valid, grid)
    valid_patch = valid_ratio >= float(deep_get(cfg, "depth.valid_patch_ratio_min", 0.2))
    valid_patch &= np.isfinite(depth_patch)

    feature = np.stack(cues, axis=-1).reshape(-1, len(cues)).astype(np.float32)
    return {
        "rgb": rgb,
        "rgb_patch": rgb_patch,
        "depth_patch": depth_patch.astype(np.float32),
        "valid_patch": valid_patch,
        "valid_ratio": valid_ratio.astype(np.float32),
        "feature": feature,
        "feature_names": names,
        "edge_patch": edge_patch.astype(np.float32),
        "rgb_grad_patch": None if rgb_grad_patch is None else rgb_grad_patch.astype(np.float32),
    }


def fit_ridge_model(X: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    alphas = [float(a) for a in deep_get(cfg, "ridge.alphas", [1.0])]
    use_sklearn = bool(deep_get(cfg, "ridge.use_sklearn_if_available", True))
    if use_sklearn:
        try:
            from sklearn.linear_model import RidgeCV

            model = RidgeCV(alphas=alphas)
            model.fit(X, y)
            return {"backend": "sklearn.RidgeCV", "model": model, "alpha": float(model.alpha_)}
        except Exception:
            pass

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    folds = int(deep_get(cfg, "ridge.cv_folds", 5))
    folds = max(2, min(folds, len(y)))
    rng = np.random.default_rng(int(deep_get(cfg, "project.random_seed", 2026)))
    order = rng.permutation(len(y))
    split = np.array_split(order, folds)
    best_alpha = alphas[0]
    best_mse = math.inf
    if len(y) >= folds * 2 and len(alphas) > 1:
        for alpha in alphas:
            mses = []
            for val_idx in split:
                train_mask = np.ones(len(y), dtype=bool)
                train_mask[val_idx] = False
                fitted = _fit_ridge_closed_form(X[train_mask], y[train_mask], alpha)
                pred = predict_ridge(fitted, X[val_idx])
                mses.append(float(np.mean((pred - y[val_idx]) ** 2)))
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
    fitted = _fit_ridge_closed_form(X, y, best_alpha)
    fitted.update({"backend": "numpy.RidgeCV", "alpha": float(best_alpha), "cv_mse": best_mse})
    return fitted


def _fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    x_mean = np.nanmean(X, axis=0)
    x_std = np.nanstd(X, axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = float(np.nanmean(y))
    Xs = (X - x_mean) / x_std
    yc = y - y_mean
    xtx = Xs.T @ Xs
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    beta = np.linalg.solve(xtx + reg, Xs.T @ yc)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "beta": beta}


def predict_ridge(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if "model" in model:
        return np.asarray(model["model"].predict(X), dtype=np.float32)
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - model["x_mean"]) / model["x_std"]
    return (Xs @ model["beta"] + model["y_mean"]).astype(np.float32)


def top_percent_mask(score: np.ndarray, valid: np.ndarray, top_percent: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(score)
    mask = np.zeros_like(valid, dtype=bool)
    if not np.any(valid):
        return mask
    k = int(math.ceil(np.count_nonzero(valid) * float(top_percent) / 100.0))
    k = max(1, min(k, np.count_nonzero(valid)))
    values = score[valid]
    threshold = np.partition(values, -k)[-k]
    mask[valid] = score[valid] >= threshold
    if np.count_nonzero(mask) > k:
        coords = np.flatnonzero(mask.reshape(-1))
        vals = score.reshape(-1)[coords]
        keep = coords[np.argsort(vals)[-k:]]
        mask[...] = False
        mask.reshape(-1)[keep] = True
    return mask


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image), mode="RGB").save(path)


def score_to_heatmap(score: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0))
    return rgba[..., :3].astype(np.float32)


def resize_grid_to_shape(grid_image: np.ndarray, target_hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    target_h, target_w = target_hw
    arr = np.asarray(grid_image)
    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        return np.asarray(pil.resize((target_w, target_h), resample=resample), dtype=np.float32)
    pil = Image.fromarray(to_uint8(arr), mode="RGB")
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(pil.resize((target_w, target_h), resample=resample), dtype=np.float32) / 255.0


def overlay_heatmap(rgb: np.ndarray, score: np.ndarray, alpha: float = 0.45, cmap_name: str = "magma") -> np.ndarray:
    heat = score_to_heatmap(score, cmap_name)
    heat = resize_grid_to_shape(heat, rgb.shape[:2], nearest=False)
    return np.clip((1.0 - alpha) * rgb + alpha * heat, 0.0, 1.0)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: list[int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.copy()
    up = resize_grid_to_shape(mask.astype(np.float32), rgb.shape[:2], nearest=True) > 0.5
    color_arr = np.asarray(color, dtype=np.float32) / 255.0
    out[up] = (1.0 - alpha) * out[up] + alpha * color_arr
    return np.clip(out, 0.0, 1.0)


def compare_overlay(
    rgb: np.ndarray,
    geometry_mask: np.ndarray,
    noise_mask: np.ndarray,
    geometry_color: list[int],
    noise_color: list[int],
    overlap_color: list[int],
) -> np.ndarray:
    out = rgb.copy()
    g = resize_grid_to_shape(geometry_mask.astype(np.float32), rgb.shape[:2], nearest=True) > 0.5
    n = resize_grid_to_shape(noise_mask.astype(np.float32), rgb.shape[:2], nearest=True) > 0.5
    colors = [
        (g & ~n, geometry_color),
        (n & ~g, noise_color),
        (g & n, overlap_color),
    ]
    for mask, color in colors:
        color_arr = np.asarray(color, dtype=np.float32) / 255.0
        out[mask] = 0.35 * out[mask] + 0.65 * color_arr
    return np.clip(out, 0.0, 1.0)


def label_image(image: Image.Image, label: str, height: int = 24) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + height), (255, 255, 255))
    out.paste(image, (0, height))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), label, fill=(20, 20, 20), font=font)
    return out


def make_panel(items: list[tuple[str, np.ndarray]], path: Path, max_width: int = 360, columns: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles: list[Image.Image] = []
    for label, arr in items:
        img = Image.fromarray(to_uint8(arr), mode="RGB")
        if img.width > max_width:
            scale = max_width / img.width
            img = img.resize((max_width, max(1, round(img.height * scale))), Image.Resampling.BILINEAR)
        tiles.append(label_image(img, label))
    if not tiles:
        return
    columns = max(1, min(columns, len(tiles)))
    rows = int(math.ceil(len(tiles) / columns))
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (255, 255, 255))
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, columns)
        sheet.paste(tile, (c * tile_w, r * tile_h))
    sheet.save(path)


def pooled_pca_rgb(path: Path, grid: tuple[int, int]) -> np.ndarray:
    return resize_rgb(read_rgb(path), grid, Image.Resampling.BOX)


def local_color_anomaly(patch_rgb: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = patch_rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        for x in range(w):
            x0, x1 = max(0, x - radius), min(w, x + radius + 1)
            window = patch_rgb[y0:y1, x0:x1].reshape(-1, 3)
            center_index = (y - y0) * (x1 - x0) + (x - x0)
            if len(window) > 1:
                window = np.delete(window, center_index, axis=0)
            med = np.median(window, axis=0)
            out[y, x] = float(np.linalg.norm(patch_rgb[y, x] - med))
    return robust_normalize(out)


def saturation_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    return sat.astype(np.float32), mx.astype(np.float32)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes: list[int] = []
    current = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            q: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = current
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        q.append((ny, nx))
            sizes.append(size)
    return labels, sizes


def small_component_mask(mask: np.ndarray, max_area: int) -> np.ndarray:
    labels, sizes = connected_components(mask)
    out = np.zeros_like(mask, dtype=bool)
    for label, size in enumerate(sizes, start=1):
        if size <= max_area:
            out |= labels == label
    return out


def pearsonr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return math.nan
    a = a[mask]
    b = b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-12:
        return math.nan
    return float(np.sum(a * b) / denom)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearmanr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return math.nan
    return pearsonr_np(rankdata_average(a[mask]), rankdata_average(b[mask]))


def nearest_patch_distance(source_mask: np.ndarray, target_mask: np.ndarray) -> float:
    src = np.argwhere(source_mask)
    tgt = np.argwhere(target_mask)
    if len(src) == 0 or len(tgt) == 0:
        return math.nan
    dists = []
    tgt_f = tgt.astype(np.float32)
    for coord in src.astype(np.float32):
        diff = tgt_f - coord[None, :]
        dists.append(float(np.sqrt(np.min(np.sum(diff * diff, axis=1)))))
    return float(np.mean(dists))


def frame_output_name(frame: dict[str, Any]) -> str:
    t = int(frame.get("t", 0))
    frame_id = str(frame.get("source_frame_id", f"{t:04d}"))
    return f"t{t:03d}_frame_{frame_id}"
