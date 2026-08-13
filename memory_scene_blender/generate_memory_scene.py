import blenderproc as bproc

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import memory_scene_blender.config as scene_config  # noqa: E402
import memory_scene_blender.utils as scene_utils  # noqa: E402
from memory_scene_blender.config import (  # noqa: E402
    ALL_OBJECTS,
    CAMERA,
    CATEGORY_IDS,
    CONDITIONS,
    DEFAULT_SCENE_PROFILE,
    EXPERIMENT_NAME,
    FRAME,
    LIGHTING_PROFILE,
    MATERIALS,
    RENDER,
    SCENE_PROFILES,
    TARGET_NAME,
    MaterialSpec,
    ObjectSpec,
    non_target_layout_signature,
    target_present_for_condition,
)
from memory_scene_blender.utils import (  # noqa: E402
    assert_paired_camera_poses,
    camera_intrinsics_matrix,
    camera_loop_metadata,
    compare_rgb_dirs,
    compute_memory_gap,
    ensure_dir,
    generate_camera_poses,
    make_contact_sheet,
    np_to_list,
    summarize_frame_list,
    visibility_pixel_threshold,
    visible_frames_from_counts,
    write_json,
)

SCENE_PROFILE_KEY = DEFAULT_SCENE_PROFILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired BlenderProc memory-retention probe sequences.")
    parser.add_argument("--scene-profile", choices=tuple(SCENE_PROFILES.keys()), default=DEFAULT_SCENE_PROFILE)
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--output-root", type=Path, default=Path(RENDER.output_root))
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(CAMERA.width, CAMERA.height))
    parser.add_argument("--samples", type=int, default=RENDER.samples)
    parser.add_argument("--dry-run", action="store_true", help="Build the scene and run geometric checks without rendering.")
    parser.add_argument("--save-blend", type=Path, default=None, help="Save a debug .blend scene and exit without rendering.")
    parser.add_argument("--preview-frame", type=int, default=28, help="Frame to show when saving a debug .blend scene.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing condition output directories.")
    parser.add_argument("--no-rgb-compare", action="store_true", help="Skip cross-condition RGB equality checks.")
    return parser.parse_args()


def apply_scene_profile(profile_key: str) -> None:
    global ALL_OBJECTS, CAMERA, EXPERIMENT_NAME, FRAME, LIGHTING_PROFILE, MATERIALS, RENDER, SCENE_PROFILE_KEY, TARGET_NAME

    profile = SCENE_PROFILES[profile_key]
    SCENE_PROFILE_KEY = profile.key
    EXPERIMENT_NAME = profile.experiment_name
    TARGET_NAME = profile.target_name
    FRAME = profile.frame
    CAMERA = profile.camera
    RENDER = profile.render
    MATERIALS = profile.materials
    ALL_OBJECTS = profile.objects
    LIGHTING_PROFILE = profile.lighting

    scene_config.EXPERIMENT_NAME = profile.experiment_name
    scene_config.TARGET_NAME = profile.target_name
    scene_config.FRAME = profile.frame
    scene_config.CAMERA = profile.camera
    scene_config.RENDER = profile.render
    scene_config.MATERIALS = profile.materials
    scene_config.ALL_OBJECTS = profile.objects
    scene_config.LIGHTING_PROFILE = profile.lighting

    scene_utils.FRAME = profile.frame
    scene_utils.CAMERA = profile.camera
    scene_utils.RENDER = profile.render


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)

    camera_data = bpy.data.cameras.new("MemoryProbeCamera")
    camera_obj = bpy.data.objects.new("MemoryProbeCamera", camera_data)
    bpy.context.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj


def configure_render(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = max(1, samples)
    scene.eevee.taa_samples = max(1, min(samples, 16))
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = RENDER.resolution_percentage
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None" if LIGHTING_PROFILE == "simplified" else "Medium High Contrast"
    scene.view_settings.exposure = RENDER.exposure
    scene.view_settings.gamma = RENDER.gamma
    scene.frame_start = 0
    scene.frame_end = FRAME.total_frames - 1

    scene.camera.data.lens = CAMERA.lens_mm
    scene.camera.data.sensor_width = 36.0
    scene.camera.data.clip_start = CAMERA.clip_start
    scene.camera.data.clip_end = CAMERA.clip_end


def create_material(spec: MaterialSpec) -> bpy.types.Material:
    mat = bpy.data.materials.new(spec.name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = spec.base_color
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = spec.roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = spec.metallic

    if spec.procedural_noise:
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = spec.noise_scale
        noise.inputs["Detail"].default_value = 7.0
        noise.inputs["Roughness"].default_value = 0.55

        ramp = nodes.new(type="ShaderNodeValToRGB")
        low = tuple(max(0.0, c - spec.noise_strength) for c in spec.base_color[:3]) + (1.0,)
        high = tuple(min(1.0, c + spec.noise_strength) for c in spec.base_color[:3]) + (1.0,)
        ramp.color_ramp.elements[0].position = 0.18
        ramp.color_ramp.elements[0].color = low
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = high

        mat.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        mat.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    return mat


def create_materials() -> Dict[str, bpy.types.Material]:
    return {name: create_material(spec) for name, spec in MATERIALS.items()}


def set_custom_properties(obj: bpy.types.Object, spec: ObjectSpec) -> None:
    obj["object_id"] = int(spec.object_id)
    obj["category_id"] = int(CATEGORY_IDS[spec.category])
    obj["category_name"] = spec.category
    obj["is_target"] = bool(spec.is_target)
    obj["object_name"] = spec.name
    obj.pass_index = int(spec.object_id)


def add_bevel_if_needed(obj: bpy.types.Object, spec: ObjectSpec) -> None:
    if spec.category == "room":
        return
    bevel_width = 0.012 if spec.category == "occluder" else 0.018
    bevel = obj.modifiers.new("small_bevel", "BEVEL")
    bevel.width = bevel_width
    bevel.segments = 2
    bevel.affect = "EDGES"
    normals = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    normals.keep_sharp = True


def create_object(spec: ObjectSpec, materials: Mapping[str, bpy.types.Material]) -> bpy.types.Object:
    loc = spec.location
    rot = spec.rotation_euler

    if spec.primitive == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    elif spec.primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.5, depth=1.0, location=loc, rotation=rot)
    elif spec.primitive == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.5, location=loc, rotation=rot)
    else:
        raise ValueError(f"Unsupported primitive {spec.primitive!r} for {spec.name}")

    obj = bpy.context.object
    obj.name = spec.name
    obj.data.name = f"{spec.name}_mesh"
    obj.dimensions = spec.dimensions
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)

    obj.data.materials.append(materials[spec.material])
    if spec.primitive in {"cylinder", "sphere"}:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    add_bevel_if_needed(obj, spec)
    set_custom_properties(obj, spec)
    return obj


def add_lights() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world

    if LIGHTING_PROFILE == "simplified":
        world.color = (0.62, 0.64, 0.65)

        bpy.ops.object.light_add(type="AREA", location=(-0.30, -0.40, 2.65))
        key = bpy.context.object
        key.name = "simplified_large_softbox"
        key.data.energy = 260.0
        key.data.size = 5.2
        return

    world.color = (0.80, 0.82, 0.84)

    bpy.ops.object.light_add(type="AREA", location=(-0.35, -0.55, 2.45))
    key = bpy.context.object
    key.name = "large_softbox_area_light"
    key.data.energy = 430.0
    key.data.size = 4.2

    bpy.ops.object.light_add(type="POINT", location=(-2.25, 2.15, 1.85))
    fill = bpy.context.object
    fill.name = "low_fill_point_light"
    fill.data.energy = 55.0
    fill.data.shadow_soft_size = 3.0


def build_scene(width: int, height: int, samples: int) -> Tuple[Dict[str, bpy.types.Object], List[np.ndarray]]:
    reset_scene()
    configure_render(width, height, samples)
    materials = create_materials()
    objects = {spec.name: create_object(spec, materials) for spec in ALL_OBJECTS}
    add_lights()

    poses = generate_camera_poses()
    for frame, pose in enumerate(poses):
        bproc.camera.add_camera_pose(pose, frame=frame)
    set_keyframe_interpolation(bpy.context.scene.camera, "LINEAR")
    assert_paired_camera_poses(poses)
    return objects, poses


def set_keyframe_interpolation(obj: bpy.types.Object, interpolation: str) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    for fcurve in obj.animation_data.action.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = interpolation


def apply_target_visibility(condition: str, target: bpy.types.Object) -> None:
    target.animation_data_clear()
    for frame in range(FRAME.total_frames):
        hidden = not target_present_for_condition(condition, frame)
        target.hide_render = hidden
        target.hide_viewport = hidden
        target.keyframe_insert(data_path="hide_render", frame=frame)
        target.keyframe_insert(data_path="hide_viewport", frame=frame)
    set_keyframe_interpolation(target, "CONSTANT")
    bpy.context.scene.frame_set(0)


def force_target_visible(target: bpy.types.Object) -> None:
    target.animation_data_clear()
    target.hide_render = False
    target.hide_viewport = False
    bpy.context.view_layer.update()


def setup_depth_compositor(depth_dir: Path) -> None:
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.use_nodes = True
    scene.view_layers[0].use_pass_z = True

    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    composite = tree.nodes.new(type="CompositorNodeComposite")
    depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
    combine = tree.nodes.new(type="CompositorNodeCombineColor")
    combine.mode = "HSV"

    depth_output.base_path = str(depth_dir)
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    depth_output.file_slots[0].path = "depth_"

    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    tree.links.new(render_layers.outputs["Depth"], combine.inputs[2])
    tree.links.new(combine.outputs["Image"], depth_output.inputs["Image"])


def render_rgb_and_depth(condition_dir: Path, frames: Optional[Iterable[int]] = None) -> None:
    if frames is not None:
        raise NotImplementedError("Partial frame rendering is not supported for paired animation output.")
    rgb_dir = ensure_dir(condition_dir / "rgb")
    depth_dir = ensure_dir(condition_dir / "depth")
    setup_depth_compositor(depth_dir)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.frame_start = 0
    scene.frame_end = FRAME.total_frames - 1
    scene.render.filepath = str(rgb_dir / "frame_")

    bpy.ops.render.render(animation=True)


def save_uint16_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    ensure_dir(path.parent)
    clipped = np.asarray(array, dtype=np.uint32)
    if clipped.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError(f"Cannot save {path} as uint16 PNG; max value is {clipped.max()}")
    Image.fromarray(clipped.astype(np.uint16), mode="I;16").save(path)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    ensure_dir(path.parent)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def linear_to_srgb_channel(value: float) -> int:
    value = min(1.0, max(0.0, value))
    if value <= 0.0031308:
        srgb = 12.92 * value
    else:
        srgb = 1.055 * (value ** (1.0 / 2.4)) - 0.055
    return int(round(min(1.0, max(0.0, srgb)) * 255.0))


def object_id_to_linear_color(object_id: int) -> Tuple[float, float, float]:
    """Return a deterministic high-saturation linear RGB color for an object id."""
    if object_id == 0:
        return (0.05, 0.05, 0.05)
    hue = ((object_id * 0.61803398875) % 1.0)
    sector = int(hue * 6.0)
    frac = hue * 6.0 - sector
    value = 1.0
    saturation = 0.88
    p = value * (1.0 - saturation)
    q = value * (1.0 - frac * saturation)
    t = value * (1.0 - (1.0 - frac) * saturation)
    sector %= 6
    if sector == 0:
        return (value, t, p)
    if sector == 1:
        return (q, value, p)
    if sector == 2:
        return (p, value, t)
    if sector == 3:
        return (p, q, value)
    if sector == 4:
        return (t, p, value)
    return (value, p, q)


def object_id_to_expected_srgb(object_id: int) -> Tuple[int, int, int]:
    return tuple(linear_to_srgb_channel(channel) for channel in object_id_to_linear_color(object_id))


def create_emission_material(name: str, color: Tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def apply_segmentation_materials() -> Dict[int, Tuple[int, int, int]]:
    palette: Dict[int, Tuple[int, int, int]] = {0: object_id_to_expected_srgb(0)}
    for spec in ALL_OBJECTS:
        color = object_id_to_linear_color(spec.object_id)
        palette[spec.object_id] = object_id_to_expected_srgb(spec.object_id)
        mat = create_emission_material(f"seg_id_{spec.object_id:03d}_{spec.name}", color)
        obj = bpy.data.objects.get(spec.name)
        if obj is None:
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    return palette


def configure_segmentation_render(color_dir: Path, prefix: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 1
    scene.eevee.taa_samples = 1
    scene.world.color = tuple(channel / 255.0 for channel in object_id_to_expected_srgb(0))
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(color_dir / f"{prefix}_color_")
    scene.frame_start = 0
    scene.frame_end = FRAME.total_frames - 1
    scene.use_nodes = False
    scene.render.use_compositing = False


def decode_color_mask(image_path: Path, palette: Mapping[int, Tuple[int, int, int]]) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int32)
    ids = np.asarray(list(palette.keys()), dtype=np.int32)
    colors = np.asarray([palette[int(object_id)] for object_id in ids], dtype=np.int32)
    flat = arr.reshape(-1, 3)
    # Squared nearest-color decode is robust to tiny edge/color-management variations.
    dists = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
    decoded = ids[np.argmin(dists, axis=1)]
    return decoded.reshape(arr.shape[:2]).astype(np.uint16)


def category_map_from_object_map(object_id_map: np.ndarray) -> np.ndarray:
    max_id = int(max(object_id_map.max(initial=0), max(spec.object_id for spec in ALL_OBJECTS)))
    lut = np.zeros(max_id + 1, dtype=np.uint16)
    for spec in ALL_OBJECTS:
        lut[spec.object_id] = CATEGORY_IDS[spec.category]
    clipped = np.clip(object_id_map.astype(np.int64), 0, max_id)
    return lut[clipped]


def render_segmentation_maps(
    condition_dir: Path,
    prefix: str,
    save_full_maps: bool = True,
    save_binary_dir_name: str = "target_binary",
) -> Dict[str, Any]:
    masks_dir = ensure_dir(condition_dir / "masks")
    color_dir_name = "instance_color" if save_full_maps else f"{save_binary_dir_name}_color"
    color_dir = ensure_dir(masks_dir / color_dir_name)
    palette = apply_segmentation_materials()
    configure_segmentation_render(color_dir, prefix)
    bpy.ops.render.render(animation=True)

    out_dirs = {
        "instance": ensure_dir(masks_dir / "instance"),
        "object_id": ensure_dir(masks_dir / "object_id"),
        "object_id_png": ensure_dir(masks_dir / "object_id_png"),
        "category_id": ensure_dir(masks_dir / "category_id"),
        "category_id_png": ensure_dir(masks_dir / "category_id_png"),
        save_binary_dir_name: ensure_dir(masks_dir / save_binary_dir_name),
    }

    threshold = visibility_pixel_threshold(bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
    pixel_counts: Dict[int, int] = {}
    for frame in range(FRAME.total_frames):
        color_path = color_dir / f"{prefix}_color_{frame:04d}.png"
        object_id_map = decode_color_mask(color_path, palette)
        instance_map = object_id_map.copy()
        category_id_map = category_map_from_object_map(object_id_map)
        target_mask = object_id_map == object_id_for_target()
        pixel_counts[frame] = int(target_mask.sum())

        if save_full_maps:
            np.save(out_dirs["instance"] / f"frame_{frame:04d}.npy", instance_map)
            np.save(out_dirs["object_id"] / f"frame_{frame:04d}.npy", object_id_map)
            np.save(out_dirs["category_id"] / f"frame_{frame:04d}.npy", category_id_map)
            save_uint16_png(object_id_map, out_dirs["object_id_png"] / f"frame_{frame:04d}.png")
            save_uint16_png(category_id_map, out_dirs["category_id_png"] / f"frame_{frame:04d}.png")

        save_binary_mask(target_mask, out_dirs[save_binary_dir_name] / f"{save_binary_dir_name}_{frame:04d}.png")

    return {
        "target_pixel_counts": pixel_counts,
        "visibility_pixel_threshold": threshold,
        "visible_frames": visible_frames_from_counts(pixel_counts, threshold),
    }


def object_id_for_target() -> int:
    for spec in ALL_OBJECTS:
        if spec.name == TARGET_NAME:
            return spec.object_id
    raise RuntimeError(f"Missing target spec {TARGET_NAME}")


def make_objects_metadata(condition: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in ALL_OBJECTS:
        rows.append(
            {
                "object_id": spec.object_id,
                "name": spec.name,
                "category": spec.category,
                "category_id": CATEGORY_IDS[spec.category],
                "primitive": spec.primitive,
                "location": list(spec.location),
                "rotation_euler": list(spec.rotation_euler),
                "dimensions": list(spec.dimensions),
                "material": spec.material,
                "is_target": spec.is_target,
                "present_by_frame": [
                    bool(target_present_for_condition(condition, frame)) if spec.is_target else True
                    for frame in range(FRAME.total_frames)
                ],
            }
        )
    return rows


def camera_payload(width: int, height: int, poses: Sequence[np.ndarray]) -> Dict[str, Any]:
    intrinsic = camera_intrinsics_matrix(width, height, CAMERA.lens_mm)
    rows = []
    loop_rows = camera_loop_metadata()
    for frame, pose in enumerate(poses):
        rows.append(
            {
                **loop_rows[frame],
                "cam2world": np_to_list(pose),
                "world2cam": np_to_list(np.linalg.inv(pose)),
            }
        )
    return {
        "intrinsic": {
            "width": width,
            "height": height,
            "lens_mm": CAMERA.lens_mm,
            "sensor_width_mm": 36.0,
            "K": np_to_list(intrinsic),
            "clip_start": CAMERA.clip_start,
            "clip_end": CAMERA.clip_end,
        },
        "frames": rows,
    }


def write_condition_metadata(
    condition_dir: Path,
    condition: str,
    poses: Sequence[np.ndarray],
    width: int,
    height: int,
    actual_seg: Dict[str, Any],
    counterfactual_seg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    actual_visible = actual_seg["visible_frames"]
    actual_first_visible = [frame for frame in actual_visible if FRAME.first_loop_start <= frame <= FRAME.first_loop_end]
    actual_second_visible = [frame for frame in actual_visible if FRAME.second_loop_start <= frame <= FRAME.second_loop_end]

    if counterfactual_seg is not None:
        revisit_visible = [
            frame
            for frame in counterfactual_seg["visible_frames"]
            if FRAME.second_loop_start <= frame <= FRAME.second_loop_end
        ]
    else:
        revisit_visible = actual_second_visible

    gap = compute_memory_gap(actual_first_visible, revisit_visible)
    anchor_visible = [frame for frame in actual_visible if frame in FRAME.anchor_frames]

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "scene_profile": SCENE_PROFILE_KEY,
        "condition": condition,
        "frame_config": {
            "total_frames": FRAME.total_frames,
            "anchor_frames": [FRAME.anchor_frames.start, FRAME.anchor_count - 1],
            "first_loop": [FRAME.first_loop_start, FRAME.first_loop_end],
            "second_loop": [FRAME.second_loop_start, FRAME.second_loop_end],
            "loop_frames": FRAME.loop_frames,
            "local_window": FRAME.local_window,
        },
        "target": {
            "name": TARGET_NAME,
            "object_id": object_id_for_target(),
            "visibility_pixel_threshold": actual_seg["visibility_pixel_threshold"],
            "visibility_area_ratio": RENDER.visibility_area_ratio,
            "actual_pixel_counts_by_frame": {
                str(frame): int(count) for frame, count in actual_seg["target_pixel_counts"].items()
            },
            "actual_visible_frames": actual_visible,
            "actual_first_loop_visible_frames": actual_first_visible,
            "actual_second_loop_visible_frames": actual_second_visible,
            "anchor_visible_frames": anchor_visible,
            "anchor_contains_target": bool(anchor_visible),
            "present_by_frame": [target_present_for_condition(condition, frame) for frame in range(FRAME.total_frames)],
            "counterfactual_second_loop_visible_frames": revisit_visible,
            **gap,
        },
        "objects": make_objects_metadata(condition),
        "non_target_layout_signature": non_target_layout_signature(),
    }

    if counterfactual_seg is not None:
        metadata["target"]["counterfactual_pixel_counts_by_frame"] = {
            str(frame): int(count) for frame, count in counterfactual_seg["target_pixel_counts"].items()
        }

    write_json(condition_dir / "metadata.json", metadata)
    write_json(condition_dir / "cameras.json", camera_payload(width, height, poses))
    write_summary(condition_dir / "summary.txt", metadata)
    write_contact_sheet(condition_dir, metadata)
    return metadata


def write_summary(path: Path, metadata: Mapping[str, Any]) -> None:
    target = metadata["target"]
    lines = [
        f"Experiment: {metadata['experiment']}",
        f"Condition: {metadata['condition']}",
        f"Total frames: {FRAME.total_frames}",
        f"Anchor frames: 0-{FRAME.anchor_count - 1}",
        f"First loop: {FRAME.first_loop_start}-{FRAME.first_loop_end}",
        f"Second loop: {FRAME.second_loop_start}-{FRAME.second_loop_end}",
        f"Local window: {FRAME.local_window}",
        "",
        f"Target visible frames: {summarize_frame_list(target['actual_visible_frames'])}",
        f"First-loop target visible frames: {summarize_frame_list(target['actual_first_loop_visible_frames'])}",
        f"Second-loop actual target visible frames: {summarize_frame_list(target['actual_second_loop_visible_frames'])}",
        f"Counterfactual second-loop target frames: {summarize_frame_list(target['counterfactual_second_loop_visible_frames'])}",
        f"Anchor contains target: {target['anchor_contains_target']}",
        f"Last first-loop target frame: {target['last_first_loop_visible_frame']}",
        f"First second-loop revisit frame: {target['first_second_loop_revisit_frame']}",
        f"Gap frames: {target['gap_frames']}",
        f"Gap > {FRAME.local_window}: {target['gap_ok']}",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_contact_frames(metadata: Mapping[str, Any]) -> List[int]:
    target = metadata["target"]
    frames: List[int] = [0, 4, 7]
    first_visible = target["actual_first_loop_visible_frames"]
    if first_visible:
        frames.extend([first_visible[0], first_visible[len(first_visible) // 2], first_visible[-1]])
        frames.extend([min(frame + FRAME.loop_frames, FRAME.second_loop_end) for frame in [first_visible[0], first_visible[-1]]])
    else:
        counter = target["counterfactual_second_loop_visible_frames"]
        if counter:
            frames.extend([counter[0] - FRAME.loop_frames, counter[len(counter) // 2] - FRAME.loop_frames, counter[0], counter[-1]])
    frames.extend([52, 74, 95])
    return sorted(set(frame for frame in frames if 0 <= frame < FRAME.total_frames))


def write_contact_sheet(condition_dir: Path, metadata: Mapping[str, Any]) -> None:
    frames = selected_contact_frames(metadata)
    image_paths = [condition_dir / "rgb" / f"frame_{frame:04d}.png" for frame in frames]
    image_paths = [path for path in image_paths if path.exists()]
    labels = [f"f{int(path.stem.split('_')[-1]):04d}" for path in image_paths]
    make_contact_sheet(image_paths, labels, condition_dir / "contact_sheet.png")


def run_condition(
    condition: str,
    run_root: Path,
    width: int,
    height: int,
    samples: int,
    overwrite: bool,
) -> Dict[str, Any]:
    condition_dir = run_root / condition
    if condition_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{condition_dir} already exists. Re-run with --overwrite or choose another output root.")
        shutil.rmtree(condition_dir)
    ensure_dir(condition_dir)

    print(f"\n=== Building condition: {condition} ===")
    objects, poses = build_scene(width, height, samples)
    target = objects[TARGET_NAME]
    apply_target_visibility(condition, target)

    print(f"[{condition}] Rendering RGB + depth...")
    render_rgb_and_depth(condition_dir)

    print(f"[{condition}] Rendering actual instance/object masks...")
    actual_seg = render_segmentation_maps(condition_dir, prefix="actual", save_full_maps=True, save_binary_dir_name="target_binary")

    counterfactual_seg = None
    if condition in {"always_absent", "seen_then_removed"}:
        print(f"[{condition}] Rendering counterfactual target masks...")
        force_target_visible(target)
        counterfactual_seg = render_segmentation_maps(
            condition_dir,
            prefix="counterfactual",
            save_full_maps=False,
            save_binary_dir_name="counterfactual_target",
        )

    metadata = write_condition_metadata(condition_dir, condition, poses, width, height, actual_seg, counterfactual_seg)
    sanity_check_condition(metadata)
    print_condition_gap(condition, metadata)
    return metadata


def sanity_check_condition(metadata: Mapping[str, Any]) -> None:
    target = metadata["target"]
    if target["anchor_contains_target"]:
        raise AssertionError(f"{metadata['condition']}: target appears in anchor frames {target['anchor_visible_frames']}")
    if metadata["condition"] in {"always_present", "seen_then_removed"} and not target["actual_first_loop_visible_frames"]:
        raise AssertionError(f"{metadata['condition']}: target is never visible in first loop")
    if metadata["condition"] == "seen_then_removed" and target["actual_second_loop_visible_frames"]:
        raise AssertionError("seen_then_removed: target should not be visible after removal")
    if metadata["condition"] in {"always_absent", "seen_then_removed"} and not target["counterfactual_second_loop_visible_frames"]:
        raise AssertionError(f"{metadata['condition']}: counterfactual target is not visible in second loop")
    if metadata["condition"] in {"always_present", "seen_then_removed"} and not target["gap_ok"]:
        raise AssertionError(
            f"{metadata['condition']}: first-to-second target revisit gap {target['gap_frames']} is not > {FRAME.local_window}"
        )


def print_condition_gap(condition: str, metadata: Mapping[str, Any]) -> None:
    target = metadata["target"]
    print(
        f"[{condition}] first-loop visible={summarize_frame_list(target['actual_first_loop_visible_frames'])}; "
        f"second-loop revisit={summarize_frame_list(target['counterfactual_second_loop_visible_frames'])}; "
        f"gap={target['gap_frames']} (> {FRAME.local_window}: {target['gap_ok']})"
    )


def write_shared_metadata(run_root: Path, width: int, height: int, poses: Sequence[np.ndarray]) -> None:
    payload = {
        "experiment": EXPERIMENT_NAME,
        "scene_profile": SCENE_PROFILE_KEY,
        "lighting_profile": LIGHTING_PROFILE,
        "conditions": list(CONDITIONS),
        "target_name": TARGET_NAME,
        "target_object_id": object_id_for_target(),
        "category_ids": CATEGORY_IDS,
        "frame_config": {
            "total_frames": FRAME.total_frames,
            "anchor_frames": [0, FRAME.anchor_count - 1],
            "first_loop": [FRAME.first_loop_start, FRAME.first_loop_end],
            "second_loop": [FRAME.second_loop_start, FRAME.second_loop_end],
            "loop_frames": FRAME.loop_frames,
            "local_window": FRAME.local_window,
        },
        "camera": camera_payload(width, height, poses),
        "objects": make_objects_metadata("always_present"),
        "non_target_layout_signature": non_target_layout_signature(),
    }
    write_json(run_root / "shared_scene_metadata.json", payload)


def run_cross_condition_checks(run_root: Path, metadatas: Mapping[str, Mapping[str, Any]], do_rgb_compare: bool) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "conditions_checked": sorted(metadatas.keys()),
        "camera_trajectories_identical": None,
        "non_target_layouts_identical": None,
        "seen_then_removed_first_loop_matches_always_present": None,
        "seen_then_removed_second_loop_matches_always_absent": None,
    }

    if len(metadatas) >= 2:
        camera_payloads = {
            condition: (run_root / condition / "cameras.json").read_text(encoding="utf-8")
            for condition in metadatas
            if (run_root / condition / "cameras.json").exists()
        }
        checks["camera_trajectories_identical"] = len(set(camera_payloads.values())) == 1
        checks["non_target_layouts_identical"] = len(
            {repr(metadata["non_target_layout_signature"]) for metadata in metadatas.values()}
        ) == 1

    if do_rgb_compare and {"always_present", "always_absent", "seen_then_removed"}.issubset(metadatas):
        seen_rgb = run_root / "seen_then_removed" / "rgb"
        present_rgb = run_root / "always_present" / "rgb"
        absent_rgb = run_root / "always_absent" / "rgb"
        checks["seen_then_removed_first_loop_matches_always_present"] = compare_rgb_dirs(
            seen_rgb, present_rgb, FRAME.first_loop_frames
        )
        checks["seen_then_removed_second_loop_matches_always_absent"] = compare_rgb_dirs(
            seen_rgb, absent_rgb, FRAME.second_loop_frames
        )

    write_json(run_root / "sanity_checks.json", checks)
    return checks


def print_cross_condition_checks(checks: Mapping[str, Any]) -> None:
    print("\n=== Cross-condition sanity checks ===")
    print(f"Camera trajectories identical: {checks['camera_trajectories_identical']}")
    print(f"Non-target layouts identical: {checks['non_target_layouts_identical']}")

    for key in (
        "seen_then_removed_first_loop_matches_always_present",
        "seen_then_removed_second_loop_matches_always_absent",
    ):
        value = checks.get(key)
        if isinstance(value, Mapping):
            print(
                f"{key}: frames={value['frames_compared']}, "
                f"max_abs={value['max_abs']}, mean_abs={value['mean_abs']}"
            )


def target_sample_points(obj: bpy.types.Object) -> List[Vector]:
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector((0.0, 0.0, 0.0))) / len(bbox)
    points = [center, *bbox]
    z_mid = center.z
    x_min, x_max = min(p.x for p in bbox), max(p.x for p in bbox)
    y_min, y_max = min(p.y for p in bbox), max(p.y for p in bbox)
    points.extend(
        [
            Vector((x_min, center.y, z_mid)),
            Vector((x_max, center.y, z_mid)),
            Vector((center.x, y_min, z_mid)),
            Vector((center.x, y_max, z_mid)),
        ]
    )
    points.extend(obj.matrix_world @ vertex.co for vertex in obj.data.vertices)
    for polygon in obj.data.polygons:
        poly_center = sum((obj.data.vertices[index].co for index in polygon.vertices), Vector((0.0, 0.0, 0.0)))
        poly_center /= len(polygon.vertices)
        points.append(obj.matrix_world @ poly_center)

    # The target is a cylinder; dense side samples make the quick geometric
    # dry-run agree better with the rendered pixel mask for partial silhouettes.
    local_bbox = [Vector(corner) for corner in obj.bound_box]
    x_min, x_max = min(p.x for p in local_bbox), max(p.x for p in local_bbox)
    y_min, y_max = min(p.y for p in local_bbox), max(p.y for p in local_bbox)
    z_min, z_max = min(p.z for p in local_bbox), max(p.z for p in local_bbox)
    radius_x = (x_max - x_min) * 0.5
    radius_y = (y_max - y_min) * 0.5
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    for z_alpha in np.linspace(0.12, 0.88, 5):
        z = z_min + (z_max - z_min) * float(z_alpha)
        for theta in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False):
            local = Vector((center_x + radius_x * np.cos(theta), center_y + radius_y * np.sin(theta), z))
            points.append(obj.matrix_world @ local)
    return points


def point_inside_camera(scene: bpy.types.Scene, camera: bpy.types.Object, point: Vector) -> bool:
    from bpy_extras.object_utils import world_to_camera_view

    ndc = world_to_camera_view(scene, camera, point)
    return 0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0 and ndc.z > 0.0


def ray_reaches_target(scene: bpy.types.Scene, target: bpy.types.Object, point: Vector) -> bool:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = scene.camera.location
    direction = point - origin
    distance = direction.length
    if distance <= 1e-6:
        return False
    direction.normalize()
    hit, _loc, _normal, _face_index, hit_obj, _matrix = scene.ray_cast(depsgraph, origin, direction, distance=distance + 0.03)
    return bool(hit and hit_obj == target)


def geometric_visible_frames(target: bpy.types.Object, poses: Sequence[np.ndarray]) -> List[int]:
    scene = bpy.context.scene
    force_target_visible(target)
    points = target_sample_points(target)
    frames: List[int] = []
    for frame, pose in enumerate(poses):
        scene.frame_set(frame)
        scene.camera.matrix_world = Matrix(pose)
        bpy.context.view_layer.update()
        visible_votes = 0
        for point in points:
            if point_inside_camera(scene, scene.camera, point) and ray_reaches_target(scene, target, point):
                visible_votes += 1
        if visible_votes >= 1:
            frames.append(frame)
    return frames


def run_dry_check(width: int, height: int, samples: int) -> None:
    print("=== Dry-run: build scene and check geometry ===")
    objects, poses = build_scene(width, height, samples)
    target = objects[TARGET_NAME]
    visible = geometric_visible_frames(target, poses)
    anchor_visible = [frame for frame in visible if frame in FRAME.anchor_frames]
    first_visible = [frame for frame in visible if FRAME.first_loop_start <= frame <= FRAME.first_loop_end]
    second_visible = [frame for frame in visible if FRAME.second_loop_start <= frame <= FRAME.second_loop_end]
    gap = compute_memory_gap(first_visible, second_visible)

    print(f"Geometric target visible frames: {summarize_frame_list(visible)}")
    print(f"Anchor target visible frames: {summarize_frame_list(anchor_visible)}")
    print(f"First-loop visible frames: {summarize_frame_list(first_visible)}")
    print(f"Second-loop revisit frames: {summarize_frame_list(second_visible)}")
    print(f"Gap frames: {gap['gap_frames']} (> {FRAME.local_window}: {gap['gap_ok']})")

    if anchor_visible:
        raise AssertionError(f"Dry-run failed: target visible in anchor frames {anchor_visible}")
    if not first_visible or not second_visible or not gap["gap_ok"]:
        print(
            "Geometry-only dry-run is conservative for partial silhouettes; "
            "rendered mask sanity checks are authoritative for visibility and gap."
        )


def save_debug_blend(path: Path, condition: str, width: int, height: int, samples: int, preview_frame: int) -> None:
    objects, _poses = build_scene(width, height, samples)
    apply_target_visibility(condition, objects[TARGET_NAME])

    scene = bpy.context.scene
    frame = max(0, min(preview_frame, FRAME.total_frames - 1))
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    ensure_dir(path.parent)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    print(f"Saved debug Blender scene for condition '{condition}' at frame {frame}: {path}")


def main() -> None:
    args = parse_args()
    apply_scene_profile(args.scene_profile)
    width, height = args.resolution
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    run_root = ensure_dir(args.output_root / EXPERIMENT_NAME)

    bproc.init()

    if args.save_blend is not None:
        condition = conditions[0]
        if args.condition == "all":
            print("Debug .blend save requested with --condition all; using always_present.")
        save_debug_blend(args.save_blend, condition, width, height, args.samples, args.preview_frame)
        return

    if args.dry_run:
        run_dry_check(width, height, args.samples)
        print("Dry-run completed.")
        return

    reference_poses = generate_camera_poses()
    write_shared_metadata(run_root, width, height, reference_poses)

    metadatas: Dict[str, Mapping[str, Any]] = {}
    for condition in conditions:
        metadatas[condition] = run_condition(condition, run_root, width, height, args.samples, args.overwrite)

    checks = run_cross_condition_checks(run_root, metadatas, do_rgb_compare=not args.no_rgb_compare)
    print_cross_condition_checks(checks)
    print(f"\nWrote experiment output to: {run_root}")


if __name__ == "__main__":
    main()
