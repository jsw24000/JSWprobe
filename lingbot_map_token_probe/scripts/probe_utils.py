from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")

DEFAULT_DATA_ROOTS = [
    "/disk1/3dsm/S2VGGT",
    "/home/data1/3dsm/S2VGGT",
    "/disk1/scannet",
    "/disk1/ScanNet",
    "/home/data1/ScanNet",
    "/data1",
    "/home/data1",
]


@dataclass
class SceneInfo:
    scene_name: str
    scene_dir: str
    rgb_dir: Optional[str]
    image_count: int
    has_sens: bool
    sens_path: Optional[str]
    notes: List[str]


def log(message: str) -> None:
    print(f"[probe] {message}", flush=True)


def natural_key(path_or_name) -> list:
    name = Path(path_or_name).name
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        p = Path(path).expanduser()
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def path_aliases(path: str | Path) -> List[Path]:
    p = Path(path).expanduser()
    aliases = [p]
    s = str(p)
    for src, dst in (("/disk1", "/home/data1"), ("/data1", "/home/data1")):
        if s == src:
            aliases.append(Path(dst))
        elif s.startswith(src + "/"):
            aliases.append(Path(dst) / s[len(src) + 1 :])
    if s == "/home/data1":
        aliases.extend([Path("/data1"), Path("/disk1")])
    elif s.startswith("/home/data1/"):
        rest = s[len("/home/data1/") :]
        aliases.extend([Path("/data1") / rest, Path("/disk1") / rest])
    return unique_paths(aliases)


def candidate_data_roots(data_root: str | Path | None = None) -> List[Path]:
    roots: List[Path] = []
    if data_root:
        for alias in path_aliases(data_root):
            roots.append(alias)
    for default in DEFAULT_DATA_ROOTS:
        for alias in path_aliases(default):
            roots.append(alias)
    return unique_paths(roots)


def list_images(directory: str | Path | None) -> List[Path]:
    if directory is None:
        return []
    d = Path(directory)
    if not d.is_dir():
        return []
    images = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images, key=natural_key)


def find_scene_dirs(root: str | Path, max_depth: int = 7) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if root.is_dir() and SCENE_RE.match(root.name):
        return [root]

    scenes: List[Path] = []
    base_parts = len(root.resolve().parts)
    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        depth = len(current.resolve().parts) - base_parts
        if depth > max_depth:
            dirnames[:] = []
            continue

        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in {"__pycache__", "node_modules", ".git"}
        ]

        if SCENE_RE.match(current.name):
            scenes.append(current)
            dirnames[:] = []
    return sorted(set(scenes), key=lambda p: natural_key(p.name))


def find_sens_file(scene_dir: str | Path) -> Optional[Path]:
    scene = Path(scene_dir)
    direct = scene / f"{scene.name}.sens"
    if direct.is_file():
        return direct
    sens = sorted(scene.glob("*.sens"), key=natural_key)
    return sens[0] if sens else None


def find_rgb_dir(scene_dir: str | Path, max_depth: int = 5) -> Optional[Path]:
    scene = Path(scene_dir)
    candidates = [
        "color",
        "rgb",
        "image",
        "images",
        "frames/color",
        "frames/rgb",
        "frames/image",
        "frames/images",
        "sensor_data/color",
        "sensor_data/rgb",
        "posed_images",
    ]
    scored: List[tuple[int, int, Path]] = []
    for rel in candidates:
        d = scene / rel
        count = len(list_images(d))
        if count:
            scored.append((1000 + count, count, d))

    skip_names = {
        "depth",
        "pose",
        "label",
        "labels",
        "instance",
        "instances",
        "intrinsic",
        "intrinsics",
        "semantic",
        "semantics",
    }
    base_parts = len(scene.resolve().parts)
    for dirpath, dirnames, _ in os.walk(scene):
        current = Path(dirpath)
        depth = len(current.resolve().parts) - base_parts
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d.lower() not in skip_names and not d.startswith(".")]
        count = len(list_images(current))
        if not count:
            continue
        name = current.name.lower()
        score = count
        if any(k in name for k in ("color", "rgb", "image", "frame")):
            score += 500
        scored.append((score, count, current))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], natural_key(item[2])))
    return scored[0][2]


def inspect_scene(scene_dir: str | Path) -> SceneInfo:
    scene = Path(scene_dir)
    rgb_dir = find_rgb_dir(scene)
    sens = find_sens_file(scene)
    notes: List[str] = []
    count = len(list_images(rgb_dir)) if rgb_dir else 0
    if rgb_dir is None:
        notes.append("no extracted RGB directory found")
    if sens is not None:
        notes.append("ScanNet .sens file available")
        if count == 0:
            try:
                info = read_sens_header(sens)
                count = int(info.get("num_frames", 0))
                notes.append("RGB can be extracted from .sens on demand")
            except Exception as exc:
                notes.append(f"failed to read .sens header: {exc}")
    return SceneInfo(
        scene_name=scene.name,
        scene_dir=str(scene),
        rgb_dir=str(rgb_dir) if rgb_dir else None,
        image_count=count,
        has_sens=sens is not None,
        sens_path=str(sens) if sens else None,
        notes=notes,
    )


def discover_scenes(data_root: str | Path | None = None, max_depth: int = 7) -> tuple[List[Path], List[SceneInfo]]:
    roots = candidate_data_roots(data_root)
    scene_dirs: List[Path] = []
    for root in roots:
        if root.exists():
            scene_dirs.extend(find_scene_dirs(root, max_depth=max_depth))
    unique = unique_paths(scene_dirs)
    infos = [inspect_scene(scene) for scene in unique]
    infos.sort(key=lambda info: natural_key(info.scene_name))
    return roots, infos


def find_scene(data_root: str | Path | None, scene_name: str, max_depth: int = 7) -> SceneInfo:
    _, infos = discover_scenes(data_root, max_depth=max_depth)
    matches = [info for info in infos if info.scene_name == scene_name]
    if not matches:
        roots = "\n  ".join(str(p) for p in candidate_data_roots(data_root))
        raise FileNotFoundError(
            f"Could not find scene {scene_name}. Searched candidate roots:\n  {roots}"
        )
    with_rgb = [info for info in matches if info.rgb_dir or info.has_sens]
    return with_rgb[0] if with_rgb else matches[0]


def sample_indices(count: int, num_frames: int, stride: int) -> List[int]:
    if count <= 0:
        return []
    num_frames = max(1, min(num_frames, count))
    if stride > 0 and count >= (num_frames - 1) * stride + 1:
        return [i * stride for i in range(num_frames)]
    if num_frames == 1:
        return [0]
    return sorted({round(i * (count - 1) / (num_frames - 1)) for i in range(num_frames)})


def numeric_stem(path: str | Path) -> Optional[int]:
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else None


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def save_rgb_copy(src: str | Path, dst: str | Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst, quality=95)


_COLOR_COMPRESSION = {
    0: "raw",
    1: "png",
    2: "jpeg",
}


def _read_exact(handle, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


def _read_u32(handle) -> int:
    return struct.unpack("I", _read_exact(handle, 4))[0]


def _read_i32(handle) -> int:
    return struct.unpack("i", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("Q", _read_exact(handle, 8))[0]


def _read_f32(handle) -> float:
    return struct.unpack("f", _read_exact(handle, 4))[0]


def _read_matrix4(handle) -> list:
    return list(struct.unpack("f" * 16, _read_exact(handle, 16 * 4)))


def read_sens_header(sens_path: str | Path) -> dict:
    with open(sens_path, "rb") as handle:
        version = _read_u32(handle)
        name_len = _read_u64(handle)
        sensor_name = _read_exact(handle, name_len).decode("utf-8", errors="replace")
        _read_matrix4(handle)  # intrinsic color
        _read_matrix4(handle)  # extrinsic color
        _read_matrix4(handle)  # intrinsic depth
        _read_matrix4(handle)  # extrinsic depth
        color_compression = _COLOR_COMPRESSION.get(_read_i32(handle), "unknown")
        depth_compression = _read_i32(handle)
        color_width = _read_u32(handle)
        color_height = _read_u32(handle)
        depth_width = _read_u32(handle)
        depth_height = _read_u32(handle)
        depth_shift = _read_f32(handle)
        num_frames = _read_u64(handle)
    return {
        "version": version,
        "sensor_name": sensor_name,
        "color_compression": color_compression,
        "depth_compression": depth_compression,
        "color_width": color_width,
        "color_height": color_height,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "depth_shift": depth_shift,
        "num_frames": num_frames,
    }


def _skip_sens_header(handle) -> dict:
    version = _read_u32(handle)
    name_len = _read_u64(handle)
    sensor_name = _read_exact(handle, name_len).decode("utf-8", errors="replace")
    _read_matrix4(handle)
    _read_matrix4(handle)
    _read_matrix4(handle)
    _read_matrix4(handle)
    color_compression = _COLOR_COMPRESSION.get(_read_i32(handle), "unknown")
    depth_compression = _read_i32(handle)
    color_width = _read_u32(handle)
    color_height = _read_u32(handle)
    depth_width = _read_u32(handle)
    depth_height = _read_u32(handle)
    depth_shift = _read_f32(handle)
    num_frames = _read_u64(handle)
    return {
        "version": version,
        "sensor_name": sensor_name,
        "color_compression": color_compression,
        "depth_compression": depth_compression,
        "color_width": color_width,
        "color_height": color_height,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "depth_shift": depth_shift,
        "num_frames": num_frames,
    }


def extract_sens_color_frames(
    sens_path: str | Path,
    selected_indices: Sequence[int],
    output_dir: str | Path,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(int(i) for i in selected_indices)
    saved: dict[int, Path] = {}

    with open(sens_path, "rb") as handle:
        header = _skip_sens_header(handle)
        compression = header["color_compression"]
        ext = ".jpg" if compression == "jpeg" else ".png"
        for frame_idx in range(int(header["num_frames"])):
            _read_exact(handle, 16 * 4)  # camera_to_world
            _read_exact(handle, 8)  # timestamp color
            _read_exact(handle, 8)  # timestamp depth
            color_size = _read_u64(handle)
            depth_size = _read_u64(handle)
            color_data = _read_exact(handle, color_size)
            handle.seek(depth_size, os.SEEK_CUR)

            if frame_idx not in selected:
                continue

            out_path = output_dir / f"frame_{frame_idx:06d}{ext}"
            try:
                with Image.open(io.BytesIO(color_data)) as img:
                    img.convert("RGB").save(out_path, quality=95)
            except Exception:
                out_path.write_bytes(color_data)
            saved[frame_idx] = out_path

    missing = [idx for idx in selected_indices if idx not in saved]
    if missing:
        raise RuntimeError(f"Failed to extract selected .sens frames: {missing}")
    return [saved[int(idx)] for idx in selected_indices]


def prepare_selected_frames(
    scene: SceneInfo,
    num_frames: int,
    stride: int,
    output_frame_dir: str | Path,
) -> tuple[List[int], List[str], List[str], List[list[int]], str]:
    output_frame_dir = Path(output_frame_dir)
    if output_frame_dir.exists():
        shutil.rmtree(output_frame_dir)
    output_frame_dir.mkdir(parents=True, exist_ok=True)

    if scene.rgb_dir:
        images = list_images(scene.rgb_dir)
        indices = sample_indices(len(images), num_frames, stride)
        if not indices:
            raise FileNotFoundError(f"No RGB images found in {scene.rgb_dir}")
        source_paths = [images[i] for i in indices]
        frame_paths: List[str] = []
        sizes: List[list[int]] = []
        for idx, src in zip(indices, source_paths):
            dst = output_frame_dir / f"frame_{idx:06d}.jpg"
            save_rgb_copy(src, dst)
            frame_paths.append(str(dst))
            sizes.append(list(image_size(dst)))
        return indices, [str(p) for p in source_paths], frame_paths, sizes, "rgb_dir"

    if scene.has_sens and scene.sens_path:
        header = read_sens_header(scene.sens_path)
        indices = sample_indices(int(header["num_frames"]), num_frames, stride)
        if not indices:
            raise RuntimeError(f"No frames available in {scene.sens_path}")
        frame_paths = [str(p) for p in extract_sens_color_frames(scene.sens_path, indices, output_frame_dir)]
        sizes = [list(image_size(p)) for p in frame_paths]
        source_paths = [f"{scene.sens_path}#{idx}" for idx in indices]
        return indices, source_paths, frame_paths, sizes, "sens"

    raise FileNotFoundError(
        f"Scene {scene.scene_name} has no RGB directory and no readable .sens file. "
        f"Scene dir: {scene.scene_dir}"
    )


def infer_token_hw(num_tokens: int, image_hw: Sequence[int] | None, patch_size: int) -> tuple[int, int]:
    if image_hw and len(image_hw) == 2:
        h = int(image_hw[0]) // int(patch_size)
        w = int(image_hw[1]) // int(patch_size)
        if h * w == num_tokens:
            return h, w
    root = int(math.sqrt(num_tokens))
    if root * root == num_tokens:
        return root, root
    best = None
    for h in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % h == 0:
            w = num_tokens // h
            score = abs(w - h)
            if best is None or score < best[0]:
                best = (score, h, w)
    if best is not None:
        return best[1], best[2]
    raise ValueError(f"Cannot infer token grid for {num_tokens} tokens")


def load_metadata(output_dir: str | Path) -> dict:
    path = Path(output_dir) / "metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_metadata(output_dir: str | Path, metadata: dict) -> None:
    path = Path(output_dir) / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def stage_sort_key(path_or_name) -> list:
    name = Path(path_or_name).stem.replace("_tokens", "")
    if name == "backbone":
        return [0]
    return [1, *natural_key(name)]


def token_files(output_dir: str | Path, include_backbone: bool = False) -> List[Path]:
    token_dir = Path(output_dir) / "tokens"
    files = list(token_dir.glob("stage_*_tokens.pt"))
    if include_backbone:
        backbone = token_dir / "backbone_tokens.pt"
        if backbone.is_file():
            files.append(backbone)
    return sorted(files, key=stage_sort_key)


def make_contact_sheet(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    cols: int = 4,
    thumb_width: int = 320,
    label: bool = True,
    bg=(245, 245, 245),
) -> Optional[Path]:
    paths = [Path(p) for p in image_paths if Path(p).is_file()]
    if not paths:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    label_h = 18 if label else 0
    thumbs = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            scale = thumb_width / max(1, img.width)
            thumb_h = max(1, int(round(img.height * scale)))
            thumb = img.resize((thumb_width, thumb_h), Image.Resampling.BILINEAR)
            thumbs.append((path, thumb))

    cols = max(1, cols)
    rows = math.ceil(len(thumbs) / cols)
    cell_h = max(t.height for _, t in thumbs) + label_h
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), bg)
    draw = ImageDraw.Draw(sheet)
    for i, (path, thumb) in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y + label_h))
        if label:
            draw.text((x + 4, y + 2), path.stem, fill=(20, 20, 20), font=font)
    sheet.save(output_path)
    return output_path


def write_summary_md(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    metadata = load_metadata(output_dir)
    if not metadata:
        return

    stages = metadata.get("token_stages", [])
    pca = metadata.get("pca", {})
    controls = metadata.get("controls", {})
    corr = metadata.get("correspondence", {})

    lines = [
        f"# Probe summary: {metadata.get('scene', output_dir.name)}",
        "",
        f"- Lingbot-map path: `{metadata.get('lingbot_root', 'unknown')}`",
        f"- Requested data root: `{metadata.get('data_root_request', 'unknown')}`",
        f"- Resolved scene dir: `{metadata.get('resolved_scene_dir', 'unknown')}`",
        f"- RGB source: `{metadata.get('rgb_source_type', 'unknown')}`",
        f"- Frame indices: `{metadata.get('frame_indices', [])}`",
        f"- Model input size HW: `{metadata.get('model_input_hw', 'unknown')}`",
        "",
        "## Token stages",
        "",
    ]
    if stages:
        for stage in stages:
            lines.append(
                f"- `{stage.get('stage_name')}`: shape `{stage.get('tokens_shape')}`, "
                f"token_hw `{stage.get('token_hw')}`, file `{stage.get('file')}`"
            )
    else:
        lines.append("- No token stage recorded.")

    lines.extend(["", "## Visualization status", ""])
    if pca:
        for stage_name, ok in sorted(pca.items(), key=lambda kv: natural_key(kv[0])):
            lines.append(f"- PCA RGB `{stage_name}`: `{ok}`")
    else:
        lines.append("- PCA RGB: not run")

    if controls:
        for key, value in sorted(controls.items()):
            lines.append(f"- Control `{key}`: `{value}`")
    else:
        lines.append("- Controls: not run")

    if corr:
        lines.append(f"- Correspondence: `{corr.get('success', False)}`")
        if corr.get("stage"):
            lines.append(f"- Correspondence stage: `{corr.get('stage')}`")
        if corr.get("output"):
            lines.append(f"- Correspondence output: `{corr.get('output')}`")
    else:
        lines.append("- Correspondence: not run")

    lines.extend(
        [
            "",
            "## Main outputs",
            "",
            f"- Frames: `{output_dir / 'frames'}`",
            f"- Tokens: `{output_dir / 'tokens'}`",
            f"- PCA visualization: `{output_dir / 'pca_vis'}`",
            f"- PCA controls: `{output_dir / 'pca_vis_controls'}`",
            f"- Correspondence: `{output_dir / 'correspondence_vis'}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def scene_info_to_dict(info: SceneInfo) -> dict:
    return asdict(info)
