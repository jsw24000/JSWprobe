import blenderproc as bproc

"""Generate paired CLEVR-style scenes for memory-write probing.

This generator reuses the scene/render/mask plumbing from
``generate_memory_scene.py`` but changes the experiment logic: each random
base scene is cloned into five variants that differ only by the red metal
sphere target state (absent or placed at one of four fixed positions).
"""

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import bpy
import numpy as np
from mathutils import Matrix

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import memory_scene_blender.config as scene_config  # noqa: E402
import memory_scene_blender.generate_memory_scene as base_gen  # noqa: E402
import memory_scene_blender.utils as scene_utils  # noqa: E402
from memory_scene_blender.config import (  # noqa: E402
    CAMERA as DEFAULT_CAMERA,
    CATEGORY_IDS,
    RENDER as DEFAULT_RENDER,
    CameraConfig,
    FrameConfig,
    MaterialSpec,
    ObjectSpec,
    RenderConfig,
)
from memory_scene_blender.utils import (  # noqa: E402
    camera_intrinsics_matrix,
    ensure_dir,
    look_at_cam2world,
    make_contact_sheet,
    np_to_list,
    write_json,
)


EXPERIMENT_NAME = "memory_write_probe_v1"
TARGET_NAME = "memory_write_target_red_metal_sphere"
TARGET_OBJECT_ID = 20
TARGET_RADIUS = 0.22
TARGET_DIAMETER = TARGET_RADIUS * 2.0

# Fixed world-space target center positions.  The z coordinate places the
# sphere on the floor/tabletop plane whose top surface is z=0.
TARGET_POSITIONS: Tuple[Tuple[float, float, float], ...] = (
    (-1.15, -0.78, TARGET_RADIUS),
    (1.15, -0.78, TARGET_RADIUS),
    (-1.15, 0.95, TARGET_RADIUS),
    (1.15, 0.95, TARGET_RADIUS),
)

VARIANTS: Tuple[Tuple[str, int], ...] = (
    ("absent", 0),
    ("pos1", 1),
    ("pos2", 2),
    ("pos3", 3),
    ("pos4", 4),
)

CAMERA_TRAJECTORY_ID = "fixed_front_oblique_small_slide_v1"
LIGHTING_PROFILE = "simplified"
SENSOR_WIDTH_MM = 36.0

BACKGROUND_OBJECT_ID_START = 100
DEFAULT_MIN_MASK_AREA_RATIO = 0.0015
DEFAULT_MIN_VISIBLE_RATIO = 0.55
DEFAULT_MIN_GEOMETRIC_VISIBLE_FRACTION = 0.55
TARGET_CLEARANCE_RADIUS = 0.72

# Keep the center of the table clear from the camera viewpoint so the four
# reserved positions remain easy probe locations rather than occlusion tests.
CLEAR_VIEW_RECT = (-1.72, 1.72, -1.55, 1.35)  # xmin, xmax, ymin, ymax
PLACEMENT_BOUNDS = (-2.65, 2.65, -2.15, 2.45)  # xmin, xmax, ymin, ymax


MATERIALS: Dict[str, MaterialSpec] = {
    "probe_floor_gray": MaterialSpec(
        "probe_floor_gray",
        (0.49, 0.50, 0.47, 1.0),
        roughness=0.78,
        procedural_noise=True,
        noise_scale=18.0,
        noise_strength=0.04,
    ),
    "probe_wall_light": MaterialSpec(
        "probe_wall_light",
        (0.67, 0.69, 0.67, 1.0),
        roughness=0.82,
        procedural_noise=True,
        noise_scale=12.0,
        noise_strength=0.03,
    ),
    "target_red_metal": MaterialSpec("target_red_metal", (0.90, 0.03, 0.025, 1.0), roughness=0.28, metallic=1.0),
    "matte_red": MaterialSpec("matte_red", (0.78, 0.08, 0.06, 1.0), roughness=0.68),
    "matte_blue": MaterialSpec("matte_blue", (0.07, 0.22, 0.78, 1.0), roughness=0.63),
    "matte_green": MaterialSpec("matte_green", (0.12, 0.58, 0.24, 1.0), roughness=0.66),
    "matte_yellow": MaterialSpec("matte_yellow", (0.86, 0.66, 0.12, 1.0), roughness=0.62),
    "matte_purple": MaterialSpec("matte_purple", (0.42, 0.20, 0.66, 1.0), roughness=0.69),
    "matte_white": MaterialSpec("matte_white", (0.82, 0.82, 0.76, 1.0), roughness=0.60),
    "metal_blue": MaterialSpec("metal_blue", (0.04, 0.16, 0.70, 1.0), roughness=0.32, metallic=1.0),
    "metal_green": MaterialSpec("metal_green", (0.05, 0.45, 0.30, 1.0), roughness=0.34, metallic=1.0),
    "metal_gold": MaterialSpec("metal_gold", (0.95, 0.62, 0.13, 1.0), roughness=0.30, metallic=1.0),
    "metal_silver": MaterialSpec("metal_silver", (0.68, 0.70, 0.72, 1.0), roughness=0.25, metallic=1.0),
}

MATERIAL_METADATA: Dict[str, Dict[str, Any]] = {
    "probe_floor_gray": {"color": "gray", "material_type": "matte"},
    "probe_wall_light": {"color": "light_gray", "material_type": "matte"},
    "target_red_metal": {"color": "red", "material_type": "metal"},
    "matte_red": {"color": "red", "material_type": "matte"},
    "matte_blue": {"color": "blue", "material_type": "matte"},
    "matte_green": {"color": "green", "material_type": "matte"},
    "matte_yellow": {"color": "yellow", "material_type": "matte"},
    "matte_purple": {"color": "purple", "material_type": "matte"},
    "matte_white": {"color": "white", "material_type": "matte"},
    "metal_blue": {"color": "blue", "material_type": "metal"},
    "metal_green": {"color": "green", "material_type": "metal"},
    "metal_gold": {"color": "gold", "material_type": "metal"},
    "metal_silver": {"color": "silver", "material_type": "metal"},
}

BACKGROUND_MATERIAL_NAMES: Tuple[str, ...] = tuple(
    name for name in MATERIALS if name not in {"probe_floor_gray", "probe_wall_light", "target_red_metal"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate memory-write probe scene variants with BlenderProc.")
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        type=Path,
        default=Path("outputs") / EXPERIMENT_NAME,
        help="Experiment directory. Relative outputs/... paths are resolved inside memory_scene_blender/.",
    )
    parser.add_argument("--num-base-scenes", "--num_base_scenes", dest="num_base_scenes", type=int, default=30)
    parser.add_argument("--num-frames", "--num_frames", dest="num_frames", type=int, default=16)
    parser.add_argument("--probe-frame", "--probe_frame", dest="probe_frame", type=int, default=15)
    parser.add_argument("--tail-frames", "--tail_frames", dest="tail_frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(DEFAULT_CAMERA.width, DEFAULT_CAMERA.height),
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_RENDER.samples)
    parser.add_argument("--min-background-objects", "--min_background_objects", dest="min_background_objects", type=int, default=5)
    parser.add_argument("--max-background-objects", "--max_background_objects", dest="max_background_objects", type=int, default=9)
    parser.add_argument("--max-resample-attempts", "--max_resample_attempts", dest="max_resample_attempts", type=int, default=600)
    parser.add_argument("--min-mask-area", "--min_mask_area", dest="min_mask_area", type=int, default=0)
    parser.add_argument(
        "--min-mask-area-ratio",
        "--min_mask_area_ratio",
        dest="min_mask_area_ratio",
        type=float,
        default=DEFAULT_MIN_MASK_AREA_RATIO,
    )
    parser.add_argument(
        "--min-visible-ratio",
        "--min_visible_ratio",
        dest="min_visible_ratio",
        type=float,
        default=DEFAULT_MIN_VISIBLE_RATIO,
    )
    parser.add_argument(
        "--min-geometric-visible-fraction",
        "--min_geometric_visible_fraction",
        dest="min_geometric_visible_fraction",
        type=float,
        default=DEFAULT_MIN_GEOMETRIC_VISIBLE_FRACTION,
    )
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Sample and check scenes without rendering the dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output experiment directory.")
    return parser.parse_args()


def normalize_output_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return SCRIPT_DIR / path
    return Path.cwd() / path


def validate_args(args: argparse.Namespace) -> None:
    if args.num_base_scenes <= 0:
        raise ValueError("--num-base-scenes must be positive")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.tail_frames < 0:
        raise ValueError("--tail-frames must be >= 0")
    if not 0 <= args.probe_frame < args.num_frames + args.tail_frames:
        raise ValueError("--probe-frame must be within the rendered frame range")
    if args.probe_frame >= args.num_frames:
        raise ValueError("--probe-frame should be inside the core num_frames window, before tail frames")
    if args.min_background_objects < 0 or args.max_background_objects < args.min_background_objects:
        raise ValueError("Invalid background object count range")
    if args.max_resample_attempts < args.num_base_scenes:
        raise ValueError("--max-resample-attempts should be at least --num-base-scenes")


def make_frame_config(total_frames: int) -> FrameConfig:
    return FrameConfig(
        total_frames=total_frames,
        anchor_count=0,
        first_loop_start=0,
        first_loop_end=max(0, total_frames - 1),
        second_loop_start=0,
        second_loop_end=max(0, total_frames - 1),
        local_window=16,
        loop_frames=total_frames,
    )


def make_camera_config(width: int, height: int) -> CameraConfig:
    return CameraConfig(
        width=width,
        height=height,
        lens_mm=22.0,
        clip_start=0.05,
        clip_end=50.0,
        look_at=(0.0, 0.05, 0.18),
    )


def make_render_config(samples: int, seed: int, output_root: Path) -> RenderConfig:
    return RenderConfig(
        samples=samples,
        resolution_percentage=100,
        exposure=0.0,
        gamma=1.0,
        visibility_pixel_threshold=1,
        visibility_area_ratio=0.0,
        random_seed=seed,
        output_root=str(output_root),
    )


def install_scene_globals(
    object_specs: Sequence[ObjectSpec],
    frame_cfg: FrameConfig,
    camera_cfg: CameraConfig,
    render_cfg: RenderConfig,
) -> None:
    """Point the reused base generator helpers at this experiment's state."""
    base_gen.EXPERIMENT_NAME = EXPERIMENT_NAME
    base_gen.TARGET_NAME = TARGET_NAME
    base_gen.FRAME = frame_cfg
    base_gen.CAMERA = camera_cfg
    base_gen.RENDER = render_cfg
    base_gen.MATERIALS = MATERIALS
    base_gen.ALL_OBJECTS = tuple(object_specs)
    base_gen.LIGHTING_PROFILE = LIGHTING_PROFILE

    scene_config.EXPERIMENT_NAME = EXPERIMENT_NAME
    scene_config.TARGET_NAME = TARGET_NAME
    scene_config.FRAME = frame_cfg
    scene_config.CAMERA = camera_cfg
    scene_config.RENDER = render_cfg
    scene_config.MATERIALS = MATERIALS
    scene_config.ALL_OBJECTS = tuple(object_specs)
    scene_config.LIGHTING_PROFILE = LIGHTING_PROFILE

    scene_utils.FRAME = frame_cfg
    scene_utils.CAMERA = camera_cfg
    scene_utils.RENDER = render_cfg


def reserved_positions_payload() -> List[Dict[str, Any]]:
    return [
        {"target_position_id": idx, "name": f"pos{idx}", "world_position": list(position)}
        for idx, position in enumerate(TARGET_POSITIONS, start=1)
    ]


def target_object_spec_payload() -> Dict[str, Any]:
    return {
        "name": TARGET_NAME,
        "object_id": TARGET_OBJECT_ID,
        "shape": "sphere",
        "color": "red",
        "material": "metal",
        "material_name": "target_red_metal",
        "radius": TARGET_RADIUS,
        "dimensions": [TARGET_DIAMETER, TARGET_DIAMETER, TARGET_DIAMETER],
        "rotation_euler": [0.0, 0.0, 0.0],
    }


def make_room_specs() -> List[ObjectSpec]:
    return [
        ObjectSpec(1, "floor", "room", "cube", (0.0, 0.0, -0.025), (0.0, 0.0, 0.0), (6.4, 6.4, 0.05), "probe_floor_gray"),
        ObjectSpec(2, "wall_north", "room", "cube", (0.0, 3.2, 1.25), (0.0, 0.0, 0.0), (6.4, 0.08, 2.5), "probe_wall_light"),
        ObjectSpec(3, "wall_south", "room", "cube", (0.0, -3.2, 1.25), (0.0, 0.0, 0.0), (6.4, 0.08, 2.5), "probe_wall_light"),
        ObjectSpec(4, "wall_east", "room", "cube", (3.2, 0.0, 1.25), (0.0, 0.0, 0.0), (0.08, 6.4, 2.5), "probe_wall_light"),
        ObjectSpec(5, "wall_west", "room", "cube", (-3.2, 0.0, 1.25), (0.0, 0.0, 0.0), (0.08, 6.4, 2.5), "probe_wall_light"),
    ]


def make_target_spec(position: Tuple[float, float, float]) -> ObjectSpec:
    return ObjectSpec(
        TARGET_OBJECT_ID,
        TARGET_NAME,
        "target",
        "sphere",
        position,
        (0.0, 0.0, 0.0),
        (TARGET_DIAMETER, TARGET_DIAMETER, TARGET_DIAMETER),
        "target_red_metal",
        True,
    )


def object_xy_radius(dimensions: Sequence[float]) -> float:
    return 0.5 * float(max(dimensions[0], dimensions[1]))


def position_in_clear_view_rect(x: float, y: float, radius: float) -> bool:
    xmin, xmax, ymin, ymax = CLEAR_VIEW_RECT
    return (xmin - radius) <= x <= (xmax + radius) and (ymin - radius) <= y <= (ymax + radius)


def too_close_to_reserved_positions(x: float, y: float, radius: float) -> bool:
    for px, py, _pz in TARGET_POSITIONS:
        if math.hypot(x - px, y - py) < TARGET_CLEARANCE_RADIUS + radius:
            return True
    return False


def too_close_to_existing(x: float, y: float, radius: float, placed: Sequence[Tuple[float, float, float]]) -> bool:
    for ox, oy, other_radius in placed:
        if math.hypot(x - ox, y - oy) < radius + other_radius + 0.12:
            return True
    return False


def sample_background_object(
    rng: np.random.Generator,
    object_id: int,
    placed: Sequence[Tuple[float, float, float]],
) -> Optional[ObjectSpec]:
    primitive = str(rng.choice(["cube", "sphere", "cylinder"], p=[0.42, 0.34, 0.24]))
    material = str(rng.choice(BACKGROUND_MATERIAL_NAMES))
    rotation_z = float(rng.uniform(-math.pi, math.pi))

    if primitive == "sphere":
        diameter = float(rng.uniform(0.24, 0.48))
        dimensions = (diameter, diameter, diameter)
        category = "sphere"
    elif primitive == "cylinder":
        diameter = float(rng.uniform(0.24, 0.46))
        height = float(rng.uniform(0.20, 0.58))
        dimensions = (diameter, diameter, height)
        category = "cylinder"
    else:
        dimensions = (
            float(rng.uniform(0.24, 0.62)),
            float(rng.uniform(0.24, 0.62)),
            float(rng.uniform(0.18, 0.58)),
        )
        category = "block"

    radius = object_xy_radius(dimensions)
    xmin, xmax, ymin, ymax = PLACEMENT_BOUNDS
    for _attempt in range(240):
        x = float(rng.uniform(xmin + radius, xmax - radius))
        y = float(rng.uniform(ymin + radius, ymax - radius))
        if position_in_clear_view_rect(x, y, radius):
            continue
        if too_close_to_reserved_positions(x, y, radius):
            continue
        if too_close_to_existing(x, y, radius, placed):
            continue

        z = dimensions[2] * 0.5
        return ObjectSpec(
            object_id,
            f"background_{object_id:03d}",
            category,
            primitive,
            (x, y, z),
            (0.0, 0.0, rotation_z if primitive == "cube" else 0.0),
            dimensions,
            material,
            False,
        )

    return None


def sample_background_specs(
    rng: np.random.Generator,
    min_objects: int,
    max_objects: int,
) -> List[ObjectSpec]:
    desired = int(rng.integers(min_objects, max_objects + 1))
    specs: List[ObjectSpec] = []
    placed: List[Tuple[float, float, float]] = []

    next_id = BACKGROUND_OBJECT_ID_START
    while len(specs) < desired and next_id < BACKGROUND_OBJECT_ID_START + 80:
        spec = sample_background_object(rng, next_id, placed)
        next_id += 1
        if spec is None:
            continue
        specs.append(spec)
        placed.append((spec.location[0], spec.location[1], object_xy_radius(spec.dimensions)))

    if len(specs) < min_objects:
        raise RuntimeError(f"Could only place {len(specs)} background objects; requested at least {min_objects}")
    return specs


def make_scene_specs(background_specs: Sequence[ObjectSpec], target_position: Tuple[float, float, float]) -> List[ObjectSpec]:
    return [*make_room_specs(), *background_specs, make_target_spec(target_position)]


def generate_probe_camera_poses(
    total_frames: int,
    probe_frame: int,
    camera_cfg: CameraConfig,
) -> List[np.ndarray]:
    settle_frame = max(0, probe_frame - 3)
    start_location = np.array([-0.10, -2.88, 2.08], dtype=np.float64)
    end_location = np.array([0.06, -2.95, 2.05], dtype=np.float64)
    start_look_at = np.array([-0.02, 0.03, 0.18], dtype=np.float64)
    end_look_at = np.array(camera_cfg.look_at, dtype=np.float64)

    poses: List[np.ndarray] = []
    for frame in range(total_frames):
        if settle_frame <= 0:
            t = 1.0
        elif frame >= settle_frame:
            t = 1.0
        else:
            raw = frame / settle_frame
            t = raw * raw * (3.0 - 2.0 * raw)
        location = (1.0 - t) * start_location + t * end_location
        look_at = (1.0 - t) * start_look_at + t * end_look_at
        poses.append(look_at_cam2world(location, look_at))
    return poses


def build_probe_scene(
    object_specs: Sequence[ObjectSpec],
    poses: Sequence[np.ndarray],
    frame_cfg: FrameConfig,
    camera_cfg: CameraConfig,
    render_cfg: RenderConfig,
    width: int,
    height: int,
) -> Dict[str, bpy.types.Object]:
    install_scene_globals(object_specs, frame_cfg, camera_cfg, render_cfg)
    base_gen.reset_scene()
    base_gen.configure_render(width, height, render_cfg.samples)
    materials = base_gen.create_materials()
    objects = {spec.name: base_gen.create_object(spec, materials) for spec in object_specs}
    base_gen.add_lights()

    for frame, pose in enumerate(poses):
        bproc.camera.add_camera_pose(pose, frame=frame)
    base_gen.set_keyframe_interpolation(bpy.context.scene.camera, "LINEAR")
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    return objects


def apply_target_state(target: bpy.types.Object, present: bool, total_frames: int) -> None:
    target.animation_data_clear()
    for frame in range(total_frames):
        target.hide_render = not present
        target.hide_viewport = not present
        target.keyframe_insert(data_path="hide_render", frame=frame)
        target.keyframe_insert(data_path="hide_viewport", frame=frame)
    base_gen.set_keyframe_interpolation(target, "CONSTANT")
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()


def set_target_position(target: bpy.types.Object, position: Tuple[float, float, float]) -> None:
    target.location = position
    bpy.context.view_layer.update()


def object_summary(spec: ObjectSpec) -> Dict[str, Any]:
    material_meta = MATERIAL_METADATA.get(spec.material, {})
    return {
        "object_id": int(spec.object_id),
        "name": spec.name,
        "category": spec.category,
        "category_id": CATEGORY_IDS[spec.category],
        "shape": spec.primitive,
        "color": material_meta.get("color"),
        "material_type": material_meta.get("material_type"),
        "material_name": spec.material,
        "location": list(spec.location),
        "rotation_euler": list(spec.rotation_euler),
        "dimensions": list(spec.dimensions),
        "is_target": bool(spec.is_target),
    }


def background_summary(background_specs: Sequence[ObjectSpec]) -> List[Dict[str, Any]]:
    return [object_summary(spec) for spec in background_specs]


def non_target_layout_signature(background_specs: Sequence[ObjectSpec]) -> List[Any]:
    return [
        (
            spec.object_id,
            spec.name,
            spec.category,
            tuple(round(float(value), 6) for value in spec.location),
            tuple(round(float(value), 6) for value in spec.rotation_euler),
            tuple(round(float(value), 6) for value in spec.dimensions),
            spec.material,
        )
        for spec in [*make_room_specs(), *background_specs]
    ]


def camera_intrinsics_payload(width: int, height: int, camera_cfg: CameraConfig) -> Dict[str, Any]:
    intrinsic = camera_intrinsics_matrix(width, height, camera_cfg.lens_mm, SENSOR_WIDTH_MM)
    return {
        "width": width,
        "height": height,
        "lens_mm": camera_cfg.lens_mm,
        "sensor_width_mm": SENSOR_WIDTH_MM,
        "K": np_to_list(intrinsic),
        "clip_start": camera_cfg.clip_start,
        "clip_end": camera_cfg.clip_end,
    }


def camera_frames_payload(poses: Sequence[np.ndarray], probe_frame: int, num_frames: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    settle_start = max(0, probe_frame - 3)
    for frame, pose in enumerate(poses):
        if frame < settle_start:
            segment = "small_slide"
        elif frame < num_frames:
            segment = "probe_settled"
        else:
            segment = "tail_static"
        rows.append(
            {
                "frame": int(frame),
                "segment": segment,
                "cam2world": np_to_list(pose),
                "world2cam": np_to_list(np.linalg.inv(pose)),
            }
        )
    return rows


def camera_payload(
    width: int,
    height: int,
    camera_cfg: CameraConfig,
    poses: Sequence[np.ndarray],
    probe_frame: int,
    num_frames: int,
) -> Dict[str, Any]:
    return {
        "trajectory_id": CAMERA_TRAJECTORY_ID,
        "intrinsic": camera_intrinsics_payload(width, height, camera_cfg),
        "frames": camera_frames_payload(poses, probe_frame, num_frames),
    }


def save_camera_payload(
    variant_dir: Path,
    width: int,
    height: int,
    camera_cfg: CameraConfig,
    poses: Sequence[np.ndarray],
    probe_frame: int,
    num_frames: int,
) -> Dict[str, Any]:
    payload = camera_payload(width, height, camera_cfg, poses, probe_frame, num_frames)
    write_json(variant_dir / "cameras.json", payload)
    return payload


def target_sample_visibility(target: bpy.types.Object, frame: int, pose: np.ndarray) -> Dict[str, Any]:
    scene = bpy.context.scene
    target.hide_render = False
    target.hide_viewport = False
    scene.frame_set(frame)
    scene.camera.matrix_world = Matrix(pose)
    bpy.context.view_layer.update()

    points = base_gen.target_sample_points(target)
    inside = 0
    visible = 0
    for point in points:
        if base_gen.point_inside_camera(scene, scene.camera, point):
            inside += 1
            if base_gen.ray_reaches_target(scene, target, point):
                visible += 1
    return {
        "inside_points": int(inside),
        "visible_points": int(visible),
        "visible_fraction": float(visible / inside) if inside else 0.0,
    }


def apply_segmentation_materials_from_scene_objects() -> Dict[int, Tuple[int, int, int]]:
    """Assign deterministic ID colors using object custom properties.

    The base generator applies segmentation materials by looking objects up by
    name from its static specs.  This experiment creates many dynamic scenes,
    so the custom ``object_id`` property set during object creation is the more
    reliable source of truth.
    """
    palette: Dict[int, Tuple[int, int, int]] = {0: base_gen.object_id_to_expected_srgb(0)}
    materials_by_id: Dict[int, bpy.types.Material] = {}

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or "object_id" not in obj:
            continue
        object_id = int(obj["object_id"])
        palette[object_id] = base_gen.object_id_to_expected_srgb(object_id)
        if object_id not in materials_by_id:
            color = base_gen.object_id_to_linear_color(object_id)
            materials_by_id[object_id] = base_gen.create_emission_material(f"seg_id_{object_id:03d}_{obj.name}", color)
        obj.data.materials.clear()
        obj.data.materials.append(materials_by_id[object_id])

    return palette


def render_segmentation_maps(
    condition_dir: Path,
    prefix: str,
    save_full_maps: bool = True,
    save_binary_dir_name: str = "target_binary",
) -> Dict[str, Any]:
    masks_dir = ensure_dir(condition_dir / "masks")
    color_dir_name = "instance_color" if save_full_maps else f"{save_binary_dir_name}_color"
    color_dir = ensure_dir(masks_dir / color_dir_name)
    palette = apply_segmentation_materials_from_scene_objects()
    base_gen.configure_segmentation_render(color_dir, prefix)
    bpy.ops.render.render(animation=True)

    out_dirs = {
        "instance": ensure_dir(masks_dir / "instance"),
        "object_id": ensure_dir(masks_dir / "object_id"),
        "object_id_png": ensure_dir(masks_dir / "object_id_png"),
        "category_id": ensure_dir(masks_dir / "category_id"),
        "category_id_png": ensure_dir(masks_dir / "category_id_png"),
        save_binary_dir_name: ensure_dir(masks_dir / save_binary_dir_name),
    }

    threshold = base_gen.visibility_pixel_threshold(bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
    pixel_counts: Dict[int, int] = {}
    for frame in range(base_gen.FRAME.total_frames):
        color_path = color_dir / f"{prefix}_color_{frame:04d}.png"
        object_id_map = base_gen.decode_color_mask(color_path, palette)
        instance_map = object_id_map.copy()
        category_id_map = base_gen.category_map_from_object_map(object_id_map)
        target_mask = object_id_map == TARGET_OBJECT_ID
        pixel_counts[frame] = int(target_mask.sum())

        if save_full_maps:
            np.save(out_dirs["instance"] / f"frame_{frame:04d}.npy", instance_map)
            np.save(out_dirs["object_id"] / f"frame_{frame:04d}.npy", object_id_map)
            np.save(out_dirs["category_id"] / f"frame_{frame:04d}.npy", category_id_map)
            base_gen.save_uint16_png(object_id_map, out_dirs["object_id_png"] / f"frame_{frame:04d}.png")
            base_gen.save_uint16_png(category_id_map, out_dirs["category_id_png"] / f"frame_{frame:04d}.png")

        base_gen.save_binary_mask(target_mask, out_dirs[save_binary_dir_name] / f"{save_binary_dir_name}_{frame:04d}.png")

    return {
        "target_pixel_counts": pixel_counts,
        "visibility_pixel_threshold": threshold,
        "visible_frames": base_gen.visible_frames_from_counts(pixel_counts, threshold),
    }


def geometric_candidate_check(
    background_specs: Sequence[ObjectSpec],
    poses: Sequence[np.ndarray],
    frame_cfg: FrameConfig,
    camera_cfg: CameraConfig,
    render_cfg: RenderConfig,
    width: int,
    height: int,
    probe_frame: int,
    min_visible_fraction: float,
) -> Tuple[bool, Dict[str, Any]]:
    object_specs = make_scene_specs(background_specs, TARGET_POSITIONS[0])
    objects = build_probe_scene(object_specs, poses, frame_cfg, camera_cfg, render_cfg, width, height)
    target = objects[TARGET_NAME]

    per_position: Dict[str, Any] = {}
    ok = True
    for pos_id, position in enumerate(TARGET_POSITIONS, start=1):
        set_target_position(target, position)
        stats = target_sample_visibility(target, probe_frame, poses[probe_frame])
        per_position[f"pos{pos_id}"] = stats
        if stats["inside_points"] <= 0 or stats["visible_fraction"] < min_visible_fraction:
            ok = False

    return ok, {"per_position": per_position, "min_visible_fraction": min_visible_fraction}


def render_single_frame_target_mask(
    output_dir: Path,
    prefix: str,
    frame: int,
) -> Tuple[np.ndarray, Path]:
    masks_dir = ensure_dir(output_dir / "masks")
    color_dir = ensure_dir(masks_dir / f"{prefix}_color")
    palette = apply_segmentation_materials_from_scene_objects()
    base_gen.configure_segmentation_render(color_dir, prefix)
    scene = bpy.context.scene
    scene.frame_start = frame
    scene.frame_end = frame
    scene.render.filepath = str(color_dir / f"{prefix}_color_")
    bpy.ops.render.render(animation=True)

    color_path = color_dir / f"{prefix}_color_{frame:04d}.png"
    object_id_map = base_gen.decode_color_mask(color_path, palette)
    mask = object_id_map == TARGET_OBJECT_ID
    mask_path = masks_dir / f"target_mask_frame_{frame:04d}.png"
    base_gen.save_binary_mask(mask, mask_path)
    return mask, mask_path


def compute_reference_target_areas(
    output_root: Path,
    poses: Sequence[np.ndarray],
    frame_cfg: FrameConfig,
    camera_cfg: CameraConfig,
    render_cfg: RenderConfig,
    width: int,
    height: int,
    probe_frame: int,
) -> Dict[int, int]:
    reference_root = ensure_dir(output_root / "reference_visibility")
    areas: Dict[int, int] = {}
    for pos_id, position in enumerate(TARGET_POSITIONS, start=1):
        object_specs = [*make_room_specs(), make_target_spec(position)]
        objects = build_probe_scene(object_specs, poses, frame_cfg, camera_cfg, render_cfg, width, height)
        apply_target_state(objects[TARGET_NAME], present=True, total_frames=frame_cfg.total_frames)
        mask, _mask_path = render_single_frame_target_mask(reference_root / f"pos{pos_id}", "reference", probe_frame)
        areas[pos_id] = int(mask.sum())
    write_json(reference_root / "reference_target_areas.json", {str(key): int(value) for key, value in areas.items()})
    return areas


def load_binary_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 0


def mask_stats(mask_path: Path, visible_reference_area: int) -> Dict[str, Any]:
    mask = load_binary_mask(mask_path)
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    if area == 0:
        return {
            "bbox": None,
            "mask_area": 0,
            "visible_ratio": 0.0,
            "center_2d": None,
        }

    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    center_2d = [float(xs.mean()), float(ys.mean())]
    ratio = float(area / visible_reference_area) if visible_reference_area > 0 else 0.0
    return {
        "bbox": bbox,
        "mask_area": area,
        "visible_ratio": ratio,
        "center_2d": center_2d,
    }


def target_mask_path(variant_dir: Path, probe_frame: int) -> Path:
    return variant_dir / "masks" / "target_binary" / f"target_binary_{probe_frame:04d}.png"


def copy_probe_mask(variant_dir: Path, probe_frame: int) -> None:
    src = target_mask_path(variant_dir, probe_frame)
    if src.exists():
        shutil.copyfile(src, variant_dir / f"target_mask_frame_{probe_frame:04d}.png")


def write_variant_summary(path: Path, metadata: Mapping[str, Any]) -> None:
    lines = [
        f"Experiment: {metadata['experiment_name']}",
        f"Base scene: {metadata['base_scene_id']}",
        f"Variant: {metadata['variant_id']}",
        f"Target present: {metadata['target_present']}",
        f"Target position id: {metadata['target_position_id']}",
        f"Probe frame: {metadata['probe_frame']}",
        f"Frame {metadata['probe_frame']} mask area: {metadata['frame15_target_mask_area']}",
        f"Frame {metadata['probe_frame']} visible ratio: {metadata['frame15_target_visible_ratio']}",
        f"Frame {metadata['probe_frame']} bbox: {metadata['frame15_target_bbox']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_variant_contact_sheet(variant_dir: Path, total_frames: int, probe_frame: int) -> None:
    candidate_frames = sorted(set([0, probe_frame // 2, max(0, probe_frame - 3), probe_frame, total_frames - 1]))
    image_paths = [variant_dir / "rgb" / f"frame_{frame:04d}.png" for frame in candidate_frames]
    image_paths = [path for path in image_paths if path.exists()]
    labels = [f"f{int(path.stem.split('_')[-1]):04d}" for path in image_paths]
    make_contact_sheet(image_paths, labels, variant_dir / "contact_sheet.png", thumb_width=180)


def render_variant(
    base_scene_id: str,
    variant_id: str,
    position_id: int,
    base_dir: Path,
    background_specs: Sequence[ObjectSpec],
    poses: Sequence[np.ndarray],
    frame_cfg: FrameConfig,
    camera_cfg: CameraConfig,
    render_cfg: RenderConfig,
    width: int,
    height: int,
    num_frames: int,
    tail_frames: int,
    probe_frame: int,
    reference_target_areas: Mapping[int, int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    present = position_id > 0
    target_position = TARGET_POSITIONS[position_id - 1] if present else TARGET_POSITIONS[0]
    variant_dir = ensure_dir(base_dir / variant_id)
    object_specs = make_scene_specs(background_specs, target_position)
    objects = build_probe_scene(object_specs, poses, frame_cfg, camera_cfg, render_cfg, width, height)
    apply_target_state(objects[TARGET_NAME], present=present, total_frames=frame_cfg.total_frames)

    print(f"[{base_scene_id}/{variant_id}] Rendering RGB + depth...")
    base_gen.render_rgb_and_depth(variant_dir)

    print(f"[{base_scene_id}/{variant_id}] Rendering instance/object masks...")
    render_segmentation_maps(variant_dir, prefix="actual", save_full_maps=True, save_binary_dir_name="target_binary")
    copy_probe_mask(variant_dir, probe_frame)

    visible_reference_area = int(reference_target_areas.get(position_id, 0)) if present else 0
    stats = mask_stats(target_mask_path(variant_dir, probe_frame), visible_reference_area)
    camera_data = save_camera_payload(variant_dir, width, height, camera_cfg, poses, probe_frame, num_frames)

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "base_scene_id": base_scene_id,
        "variant_id": variant_id,
        "target_present": int(present),
        "target_position_id": int(position_id),
        "target_world_position": list(target_position) if present else None,
        "target_object_spec": target_object_spec_payload(),
        "probe_frame": int(probe_frame),
        "num_frames": int(num_frames),
        "tail_frames": int(tail_frames),
        "total_render_frames": int(frame_cfg.total_frames),
        "camera_trajectory_id": CAMERA_TRAJECTORY_ID,
        "camera_intrinsics": camera_data["intrinsic"],
        "camera_poses": camera_data["frames"],
        "frame15_target_bbox": stats["bbox"],
        "frame15_target_mask_area": int(stats["mask_area"]),
        "frame15_target_visible_ratio": float(stats["visible_ratio"]),
        "frame15_target_center_2d": stats["center_2d"],
        "frame15_target_center_3d": list(target_position) if present else None,
        "reserved_positions": reserved_positions_payload(),
        "background_object_summary": background_summary(background_specs),
        "non_target_layout_signature": non_target_layout_signature(background_specs),
    }
    write_json(variant_dir / "metadata.json", metadata)
    write_variant_summary(variant_dir / "summary.txt", metadata)
    write_variant_contact_sheet(variant_dir, frame_cfg.total_frames, probe_frame)

    manifest_row = {
        "base_scene_id": base_scene_id,
        "variant_id": variant_id,
        "scene_path": str(variant_dir),
        "target_present": int(present),
        "target_position_id": int(position_id),
        "probe_frame": int(probe_frame),
        "frame15_target_mask_area": int(stats["mask_area"]),
        "frame15_target_visible_ratio": float(stats["visible_ratio"]),
    }
    return metadata, manifest_row


def post_render_base_valid(
    metadatas: Sequence[Mapping[str, Any]],
    min_mask_area: int,
    min_visible_ratio: float,
) -> Tuple[bool, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    for metadata in metadatas:
        if int(metadata["target_present"]) == 0:
            if int(metadata["frame15_target_mask_area"]) != 0:
                failures.append(
                    {
                        "variant_id": metadata["variant_id"],
                        "reason": "absent_variant_has_target_pixels",
                        "mask_area": int(metadata["frame15_target_mask_area"]),
                    }
                )
            continue

        mask_area = int(metadata["frame15_target_mask_area"])
        visible_ratio = float(metadata["frame15_target_visible_ratio"])
        if metadata["frame15_target_bbox"] is None or mask_area < min_mask_area or visible_ratio < min_visible_ratio:
            failures.append(
                {
                    "variant_id": metadata["variant_id"],
                    "reason": "target_low_visibility_or_offscreen",
                    "mask_area": mask_area,
                    "visible_ratio": visible_ratio,
                    "min_mask_area": min_mask_area,
                    "min_visible_ratio": min_visible_ratio,
                }
            )
    return not failures, failures


def write_base_scene_metadata(
    base_dir: Path,
    base_scene_id: str,
    base_seed: int,
    background_specs: Sequence[ObjectSpec],
    geometric_check: Mapping[str, Any],
    variant_ids: Sequence[str],
) -> None:
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "base_scene_id": base_scene_id,
        "base_seed": int(base_seed),
        "variant_ids": list(variant_ids),
        "reserved_positions": reserved_positions_payload(),
        "target_object_spec": target_object_spec_payload(),
        "background_object_summary": background_summary(background_specs),
        "non_target_layout_signature": non_target_layout_signature(background_specs),
        "geometric_probe_frame_check": geometric_check,
    }
    write_json(base_dir / "base_scene_metadata.json", payload)


def render_base_scene(
    base_scene_index: int,
    base_seed: int,
    output_root: Path,
    background_specs: Sequence[ObjectSpec],
    geometric_check: Mapping[str, Any],
    poses: Sequence[np.ndarray],
    frame_cfg: FrameConfig,
    camera_cfg: CameraConfig,
    render_cfg: RenderConfig,
    width: int,
    height: int,
    num_frames: int,
    tail_frames: int,
    probe_frame: int,
    reference_target_areas: Mapping[int, int],
    min_mask_area: int,
    min_visible_ratio: float,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    base_scene_id = f"base_scene_{base_scene_index:03d}"
    base_dir = output_root / base_scene_id
    if base_dir.exists():
        shutil.rmtree(base_dir)
    ensure_dir(base_dir)

    variant_metadatas: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for variant_id, position_id in VARIANTS:
        metadata, row = render_variant(
            base_scene_id,
            variant_id,
            position_id,
            base_dir,
            background_specs,
            poses,
            frame_cfg,
            camera_cfg,
            render_cfg,
            width,
            height,
            num_frames,
            tail_frames,
            probe_frame,
            reference_target_areas,
        )
        variant_metadatas.append(metadata)
        manifest_rows.append(row)

    ok, failures = post_render_base_valid(variant_metadatas, min_mask_area, min_visible_ratio)
    base_status = {
        "base_scene_id": base_scene_id,
        "accepted": bool(ok),
        "failures": failures,
    }
    if not ok:
        shutil.rmtree(base_dir)
        return False, [], base_status

    write_base_scene_metadata(
        base_dir,
        base_scene_id,
        base_seed,
        background_specs,
        geometric_check,
        [variant_id for variant_id, _position_id in VARIANTS],
    )
    return True, manifest_rows, base_status


def write_manifest(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    jsonl_path = output_root / "manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = output_root / "manifest.csv"
    fieldnames = [
        "base_scene_id",
        "variant_id",
        "scene_path",
        "target_present",
        "target_position_id",
        "probe_frame",
        "frame15_target_mask_area",
        "frame15_target_visible_ratio",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def summarize_dataset(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    requested_base_scenes: int,
    rejected_resamples: int,
    post_render_rejections: int,
    reference_target_areas: Mapping[int, int],
    min_mask_area: int,
    min_visible_ratio: float,
    base_statuses: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_position: Dict[str, Dict[str, Any]] = {}
    severe_samples: List[Dict[str, Any]] = []

    for pos_id in range(1, 5):
        pos_rows = [row for row in rows if int(row["target_position_id"]) == pos_id]
        mask_areas = [float(row["frame15_target_mask_area"]) for row in pos_rows]
        visible_ratios = [float(row["frame15_target_visible_ratio"]) for row in pos_rows]
        key = f"pos{pos_id}"
        by_position[key] = {
            "count": len(pos_rows),
            "mean_mask_area": float(np.mean(mask_areas)) if mask_areas else None,
            "mean_visible_ratio": float(np.mean(visible_ratios)) if visible_ratios else None,
            "min_mask_area": float(np.min(mask_areas)) if mask_areas else None,
            "min_visible_ratio": float(np.min(visible_ratios)) if visible_ratios else None,
            "reference_unoccluded_mask_area": int(reference_target_areas.get(pos_id, 0)),
        }
        for row in pos_rows:
            if (
                int(row["frame15_target_mask_area"]) < min_mask_area
                or float(row["frame15_target_visible_ratio"]) < min_visible_ratio
            ):
                severe_samples.append(dict(row))

    base_scene_ids = sorted({str(row["base_scene_id"]) for row in rows})
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "requested_base_scenes": int(requested_base_scenes),
        "successful_base_scenes": len(base_scene_ids),
        "variant_count": len(rows),
        "rejected_resampled_scenes": int(rejected_resamples),
        "post_render_rejections": int(post_render_rejections),
        "position_stats": by_position,
        "min_mask_area_threshold": int(min_mask_area),
        "min_visible_ratio_threshold": float(min_visible_ratio),
        "has_offscreen_or_severely_occluded_samples": bool(severe_samples),
        "offscreen_or_severely_occluded_samples": severe_samples,
        "base_statuses": list(base_statuses),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def print_summary(summary: Mapping[str, Any], output_root: Path) -> None:
    print("\n=== memory_write_probe_v1 summary ===")
    print(f"Output root: {output_root}")
    print(f"Successful base scenes: {summary['successful_base_scenes']}")
    print(f"Rejected/resampled scenes: {summary['rejected_resampled_scenes']}")
    print(f"Post-render rejections: {summary['post_render_rejections']}")
    for pos_id in range(1, 5):
        stats = summary["position_stats"][f"pos{pos_id}"]
        print(
            f"pos{pos_id}: mean_mask_area={stats['mean_mask_area']}, "
            f"mean_visible_ratio={stats['mean_visible_ratio']}, "
            f"min_mask_area={stats['min_mask_area']}, "
            f"min_visible_ratio={stats['min_visible_ratio']}"
        )
    print(f"Offscreen/severe occlusion samples: {summary['has_offscreen_or_severely_occluded_samples']}")


def write_experiment_config(
    output_root: Path,
    args: argparse.Namespace,
    width: int,
    height: int,
    frame_cfg: FrameConfig,
    camera_cfg: CameraConfig,
    render_cfg: RenderConfig,
    poses: Sequence[np.ndarray],
) -> None:
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "target_object_spec": target_object_spec_payload(),
        "reserved_positions": reserved_positions_payload(),
        "variants": [{"variant_id": variant_id, "target_position_id": position_id} for variant_id, position_id in VARIANTS],
        "frame_config": {
            "num_frames": int(args.num_frames),
            "tail_frames": int(args.tail_frames),
            "total_render_frames": int(frame_cfg.total_frames),
            "probe_frame": int(args.probe_frame),
        },
        "camera": camera_payload(width, height, camera_cfg, poses, args.probe_frame, args.num_frames),
        "render_config": {
            "samples": int(render_cfg.samples),
            "resolution_percentage": int(render_cfg.resolution_percentage),
            "exposure": float(render_cfg.exposure),
            "gamma": float(render_cfg.gamma),
        },
        "category_ids": CATEGORY_IDS,
        "materials": {
            name: {
                "base_color": list(spec.base_color),
                "roughness": spec.roughness,
                "metallic": spec.metallic,
                **MATERIAL_METADATA.get(name, {}),
            }
            for name, spec in MATERIALS.items()
        },
    }
    write_json(output_root / "experiment_config.json", payload)


def prepare_output_root(output_root: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        ensure_dir(output_root)
        return
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Re-run with --overwrite or choose another output root.")
        shutil.rmtree(output_root)
    ensure_dir(output_root)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_root = normalize_output_root(args.output_root)
    width, height = args.resolution
    total_frames = args.num_frames + args.tail_frames
    min_mask_area = max(args.min_mask_area, int(round(width * height * args.min_mask_area_ratio)))

    frame_cfg = make_frame_config(total_frames)
    camera_cfg = make_camera_config(width, height)
    render_cfg = make_render_config(args.samples, args.seed, output_root)
    poses = generate_probe_camera_poses(total_frames, args.probe_frame, camera_cfg)

    prepare_output_root(output_root, args.overwrite, args.dry_run)

    bproc.init()
    write_experiment_config(output_root, args, width, height, frame_cfg, camera_cfg, render_cfg, poses)

    rng = np.random.default_rng(args.seed)
    accepted = 0
    attempts = 0
    rejected_resamples = 0
    post_render_rejections = 0
    manifest_rows: List[Dict[str, Any]] = []
    base_statuses: List[Dict[str, Any]] = []

    reference_target_areas: Dict[int, int]
    if args.dry_run:
        reference_target_areas = {pos_id: 0 for pos_id in range(1, 5)}
    else:
        print("Computing unoccluded frame-15 target reference areas...")
        reference_target_areas = compute_reference_target_areas(
            output_root, poses, frame_cfg, camera_cfg, render_cfg, width, height, args.probe_frame
        )

    while accepted < args.num_base_scenes and attempts < args.max_resample_attempts:
        attempts += 1
        base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        base_rng = np.random.default_rng(base_seed)
        base_scene_id = f"base_scene_{accepted:03d}"
        print(f"\n=== Sampling {base_scene_id} (attempt {attempts}, seed {base_seed}) ===")

        try:
            background_specs = sample_background_specs(base_rng, args.min_background_objects, args.max_background_objects)
        except RuntimeError as exc:
            rejected_resamples += 1
            print(f"[{base_scene_id}] Rejected during placement: {exc}")
            continue

        geometric_ok, geometric_check = geometric_candidate_check(
            background_specs,
            poses,
            frame_cfg,
            camera_cfg,
            render_cfg,
            width,
            height,
            args.probe_frame,
            args.min_geometric_visible_fraction,
        )
        if not geometric_ok:
            rejected_resamples += 1
            print(f"[{base_scene_id}] Rejected by geometric frame-{args.probe_frame} visibility check.")
            continue

        if args.dry_run:
            accepted += 1
            base_statuses.append({"base_scene_id": base_scene_id, "accepted": True, "dry_run": True})
            continue

        ok, rows, base_status = render_base_scene(
            accepted,
            base_seed,
            output_root,
            background_specs,
            geometric_check,
            poses,
            frame_cfg,
            camera_cfg,
            render_cfg,
            width,
            height,
            args.num_frames,
            args.tail_frames,
            args.probe_frame,
            reference_target_areas,
            min_mask_area,
            args.min_visible_ratio,
        )
        base_statuses.append(base_status)
        if not ok:
            rejected_resamples += 1
            post_render_rejections += 1
            print(f"[{base_scene_id}] Rejected after rendered mask check: {base_status['failures']}")
            continue

        manifest_rows.extend(rows)
        accepted += 1

    if accepted < args.num_base_scenes:
        raise RuntimeError(
            f"Generated {accepted}/{args.num_base_scenes} base scenes after {attempts} attempts. "
            "Try relaxing visibility thresholds or background density."
        )

    if args.dry_run:
        summary = {
            "experiment_name": EXPERIMENT_NAME,
            "dry_run": True,
            "successful_base_scenes": accepted,
            "rejected_resampled_scenes": rejected_resamples,
            "attempts": attempts,
            "note": "No RGB/depth/masks were rendered in dry-run mode.",
        }
        write_json(output_root / "summary.json", summary)
        print_summary(
            {
                **summary,
                "position_stats": {
                    f"pos{idx}": {
                        "mean_mask_area": None,
                        "mean_visible_ratio": None,
                        "min_mask_area": None,
                        "min_visible_ratio": None,
                    }
                    for idx in range(1, 5)
                },
                "post_render_rejections": 0,
                "has_offscreen_or_severely_occluded_samples": False,
            },
            output_root,
        )
        return

    write_manifest(output_root, manifest_rows)
    summary = summarize_dataset(
        output_root,
        manifest_rows,
        args.num_base_scenes,
        rejected_resamples,
        post_render_rejections,
        reference_target_areas,
        min_mask_area,
        args.min_visible_ratio,
        base_statuses,
    )
    print_summary(summary, output_root)


if __name__ == "__main__":
    main()
