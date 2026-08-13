"""Blender scene construction and rendering for object translation."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .assets import (
    CATEGORY_IDS,
    FurnitureAsset,
    asset_world_bbox,
    create_materials,
    create_room_shell,
    create_static_reference_assets,
    create_target_asset,
    logical_asset_metadata,
)
from .camera_builder import generate_cameras
from .label_utils import corners_inside_room, validate_no_collision
from .manifest_utils import ensure_dir, write_json


TARGET_OBJECT_ID = 20


@dataclass
class SceneBundle:
    scene_id: str
    seed: int
    layout: Mapping[str, Any]
    materials: Mapping[str, Any]
    room_assets: List[FurnitureAsset]
    static_assets: List[FurnitureAsset]
    target_asset: FurnitureAsset
    cameras: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    static_metadata: List[Dict[str, Any]]


def reset_scene() -> None:
    import bpy  # type: ignore

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)

    camera_data = bpy.data.cameras.new("ObjectTranslationCamera")
    camera_obj = bpy.data.objects.new("ObjectTranslationCamera", camera_data)
    bpy.context.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj


def _render_engine_name(preferred: str) -> str:
    import bpy  # type: ignore

    engines = {item.identifier for item in bpy.context.scene.render.bl_rna.properties["engine"].enum_items}
    if preferred in engines:
        return preferred
    if "BLENDER_EEVEE" in engines:
        return "BLENDER_EEVEE"
    return next(iter(engines))


def configure_render(render_cfg: Mapping[str, Any], mode_settings: Mapping[str, Any]) -> None:
    import bpy  # type: ignore

    scene = bpy.context.scene
    scene.render.engine = _render_engine_name(str(render_cfg.get("engine", "BLENDER_EEVEE")))
    samples = int(mode_settings["samples"])
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, samples)
        scene.eevee.taa_samples = max(1, min(samples, 16))
    width, height = [int(v) for v in mode_settings["resolution"]]
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.frame_start = 1
    scene.frame_end = 1


def configure_camera(camera_payload: Mapping[str, Any]) -> None:
    import bpy  # type: ignore
    from mathutils import Matrix  # type: ignore

    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("Scene has no camera")
    camera.matrix_world = Matrix(camera_payload["blender_camera_to_world"])
    camera.data.angle = math.radians(float(camera_payload["fov_degrees"]))
    camera.data.clip_start = float(camera_payload["clip_start"])
    camera.data.clip_end = float(camera_payload["clip_end"])
    bpy.context.view_layer.update()


def setup_lighting(preset: Mapping[str, Any]) -> None:
    import bpy  # type: ignore

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = tuple(float(v) for v in preset["world_color"])

    bpy.ops.object.light_add(type="AREA", location=tuple(float(v) for v in preset["key_location"]))
    key = bpy.context.object
    key.name = "large_soft_key_light"
    key.data.energy = float(preset["key_energy"])
    key.data.size = float(preset["key_size"])

    bpy.ops.object.light_add(type="AREA", location=tuple(float(v) for v in preset["fill_location"]))
    fill = bpy.context.object
    fill.name = "soft_fill_light"
    fill.data.energy = float(preset["fill_energy"])
    fill.data.size = float(preset["fill_size"])


def set_target_position(target_asset: FurnitureAsset, xy_position: Sequence[float]) -> None:
    import bpy  # type: ignore

    target_asset.root.location = (float(xy_position[0]), float(xy_position[1]), 0.0)
    bpy.context.view_layer.update()


def target_corners_at_position(target_asset: FurnitureAsset, xy_position: Sequence[float]) -> List[List[float]]:
    current = tuple(float(v) for v in target_asset.root.location)
    set_target_position(target_asset, xy_position)
    corners = asset_world_bbox(target_asset)
    target_asset.root.location = current
    import bpy  # type: ignore

    bpy.context.view_layer.update()
    return corners


def _position_grid_candidate(
    rows: int,
    cols: int,
    center_xy: Sequence[float],
    spacing: float,
) -> List[Dict[str, Any]]:
    x0 = float(center_xy[0]) - (cols - 1) * spacing * 0.5
    y0 = float(center_xy[1]) - (rows - 1) * spacing * 0.5
    positions: List[Dict[str, Any]] = []
    state_idx = 0
    for row in range(rows):
        for col in range(cols):
            positions.append(
                {
                    "state_id": f"state_{state_idx:03d}",
                    "position_index": [row, col],
                    "xy": [float(x0 + col * spacing), float(y0 + row * spacing)],
                    "grid_spacing_m": float(spacing),
                }
            )
            state_idx += 1
    return positions


def _position_grid_is_valid(
    candidate: Sequence[Mapping[str, Any]],
    target_asset: FurnitureAsset,
    room_dimensions: Sequence[float],
    wall_margin: float,
    static_metadata: Sequence[Mapping[str, Any]],
    static_clearance: float,
) -> bool:
    for pos in candidate:
        corners = target_corners_at_position(target_asset, pos["xy"])
        if not corners_inside_room(corners, room_dimensions, wall_margin):
            return False
        if not validate_no_collision(corners, static_metadata, static_clearance):
            return False
    return True


def _candidate_grid_centers(
    room_dimensions: Sequence[float],
    rows: int,
    cols: int,
    spacing: float,
    target_width: float,
    target_depth: float,
    wall_margin: float,
    preferred_centers: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    room_w, room_d, _room_h = [float(v) for v in room_dimensions]
    half_span_x = (cols - 1) * spacing * 0.5 + target_width * 0.5
    half_span_y = (rows - 1) * spacing * 0.5 + target_depth * 0.5
    x_min = -room_w * 0.5 + wall_margin + half_span_x
    x_max = room_w * 0.5 - wall_margin - half_span_x
    y_min = -room_d * 0.5 + wall_margin + half_span_y
    y_max = room_d * 0.5 - wall_margin - half_span_y
    if x_min > x_max or y_min > y_max:
        return list(preferred_centers)

    scan_x = np.linspace(x_min, x_max, 9)
    scan_y = np.linspace(y_min, y_max, 9)
    scan = [np.array([float(x), float(y)], dtype=np.float64) for y in scan_y for x in scan_x]
    preferred = preferred_centers[0] if preferred_centers else np.array([0.0, 0.0], dtype=np.float64)
    # The tiny deterministic noise prevents ties from always selecting the same
    # side of symmetric rooms while preserving reproducibility through scene_seed.
    scored = [
        (
            float(np.linalg.norm(center - preferred)) + float(rng.uniform(0.0, 1.0e-5)),
            center,
        )
        for center in scan
    ]
    scored.sort(key=lambda item: item[0])

    centers: List[np.ndarray] = []
    seen = set()
    for center in [*preferred_centers, *(center for _score, center in scored)]:
        clipped = np.array(
            [
                float(np.clip(center[0], x_min, x_max)),
                float(np.clip(center[1], y_min, y_max)),
            ],
            dtype=np.float64,
        )
        key = (round(float(clipped[0]), 4), round(float(clipped[1]), 4))
        if key in seen:
            continue
        seen.add(key)
        centers.append(clipped)
    return centers


def generate_positions(
    config: Mapping[str, Any],
    mode_settings: Mapping[str, Any],
    layout: Mapping[str, Any],
    target_asset: FurnitureAsset,
    static_metadata: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    rows = int(mode_settings["grid_rows"])
    cols = int(mode_settings["grid_cols"])
    pos_cfg = config["positions"]
    room_dimensions = layout["dimensions"]
    min_spacing = float(pos_cfg["grid_spacing_min_m"])
    max_spacing = float(pos_cfg["grid_spacing_max_m"])
    room_w, room_d, _room_h = [float(v) for v in room_dimensions]
    local_corners = target_corners_at_position(target_asset, (0.0, 0.0))
    arr = np.asarray(local_corners, dtype=np.float64)
    target_w = float(arr[:, 0].max() - arr[:, 0].min())
    target_d = float(arr[:, 1].max() - arr[:, 1].min())
    usable_x = room_w - 2.0 * float(pos_cfg["wall_margin_m"]) - target_w
    usable_y = room_d - 2.0 * float(pos_cfg["wall_margin_m"]) - target_d
    spacing = min(max_spacing, max(min_spacing, 0.85 * min(usable_x / max(1, cols - 1), usable_y / max(1, rows - 1))))

    global_jitter = float(pos_cfg.get("global_jitter_m", 0.0))
    jitter = rng.uniform(-global_jitter, global_jitter, size=2)
    preferred_centers = [
        np.asarray(layout.get("grid_center", pos_cfg.get("preferred_center", [0.0, 0.0])), dtype=np.float64) + jitter,
        np.asarray(pos_cfg.get("preferred_center", [0.0, 0.0]), dtype=np.float64) + jitter,
        np.array([0.35, -0.40], dtype=np.float64) + jitter,
        np.array([0.15, -0.55], dtype=np.float64) + jitter,
        np.array([-0.10, -0.65], dtype=np.float64) + jitter,
        np.array([-0.35, -0.35], dtype=np.float64) + jitter,
    ]

    spacing_values = list(np.linspace(spacing, min_spacing, 5))
    for current_spacing in spacing_values:
        centers = _candidate_grid_centers(
            room_dimensions,
            rows,
            cols,
            float(current_spacing),
            target_w,
            target_d,
            float(pos_cfg["wall_margin_m"]),
            preferred_centers,
            rng,
        )
        for center in centers:
            candidate = _position_grid_candidate(rows, cols, center, float(current_spacing))
            if _position_grid_is_valid(
                candidate,
                target_asset,
                room_dimensions,
                float(pos_cfg["wall_margin_m"]),
                static_metadata,
                float(pos_cfg["static_clearance_m"]),
            ):
                return candidate
    raise RuntimeError(f"Could not place a valid {rows}x{cols} target grid for layout {layout.get('layout_id')}")


def build_scene_bundle(
    scene_index: int,
    config: Mapping[str, Any],
    mode_settings: Mapping[str, Any],
) -> SceneBundle:
    import bpy  # type: ignore

    scene_id = f"scene_{scene_index:03d}"
    scene_seed = int(config["seed"]) + scene_index * 104729
    rng = np.random.default_rng(scene_seed)
    layouts = config["scene"]["room_layouts"]
    layout = layouts[scene_index % len(layouts)]
    target_categories = config["scene"]["target_categories"]
    target_category = str(target_categories[scene_index % len(target_categories)])
    variant_count = int(config["scene"]["target_variants_per_category"])
    target_variant = (scene_index // len(target_categories)) % variant_count

    reset_scene()
    configure_render(config["render"], mode_settings)
    materials = create_materials()
    room_assets = create_room_shell(layout, materials)
    static_layout_id = int(layout.get("static_layout_id", scene_index))
    static_assets = create_static_reference_assets(static_layout_id, layout["dimensions"], materials)
    target_asset = create_target_asset(target_category, target_variant, materials, object_id=TARGET_OBJECT_ID)
    setup_lighting(config["scene"]["lighting_presets"][layout["lighting_preset"]])
    bpy.context.view_layer.update()

    static_metadata = [logical_asset_metadata(asset, is_room_shell=True) for asset in room_assets]
    static_metadata.extend(logical_asset_metadata(asset, is_room_shell=False) for asset in static_assets)
    positions = generate_positions(config, mode_settings, layout, target_asset, static_metadata, rng)
    cameras = generate_cameras(scene_id, config["camera"], mode_settings)
    return SceneBundle(
        scene_id=scene_id,
        seed=scene_seed,
        layout=layout,
        materials=materials,
        room_assets=room_assets,
        static_assets=static_assets,
        target_asset=target_asset,
        cameras=cameras,
        positions=positions,
        static_metadata=static_metadata,
    )


def all_mesh_objects() -> List[Any]:
    import bpy  # type: ignore

    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def capture_material_slots() -> Dict[str, List[Any]]:
    return {obj.name: [slot for slot in obj.data.materials] for obj in all_mesh_objects()}


def restore_material_slots(slots_by_object: Mapping[str, Sequence[Any]]) -> None:
    import bpy  # type: ignore

    for obj in all_mesh_objects():
        original = slots_by_object.get(obj.name)
        if original is None:
            continue
        obj.data.materials.clear()
        for mat in original:
            obj.data.materials.append(mat)
    bpy.context.view_layer.update()


def create_emission_material(name: str, color: Sequence[float]) -> Any:
    import bpy  # type: ignore

    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(float(v) for v in color[:4])
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = tuple(float(v) for v in color[:4])
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def create_normal_material() -> Any:
    import bpy  # type: ignore

    existing = bpy.data.materials.get("pass_world_normal_rgb")
    if existing is not None:
        return existing
    mat = bpy.data.materials.new("pass_world_normal_rgb")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    geometry = nodes.new(type="ShaderNodeNewGeometry")
    multiply = nodes.new(type="ShaderNodeVectorMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
    add = nodes.new(type="ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(geometry.outputs["Normal"], multiply.inputs[0])
    mat.node_tree.links.new(multiply.outputs["Vector"], add.inputs[0])
    mat.node_tree.links.new(add.outputs["Vector"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def object_id_to_color(object_id: int) -> Tuple[float, float, float, float]:
    if object_id == 0:
        return (0.0, 0.0, 0.0, 1.0)
    hue = (object_id * 0.61803398875) % 1.0
    sector = int(hue * 6.0)
    frac = hue * 6.0 - sector
    value = 1.0
    saturation = 0.84
    p = value * (1.0 - saturation)
    q = value * (1.0 - frac * saturation)
    t = value * (1.0 - (1.0 - frac) * saturation)
    sector %= 6
    if sector == 0:
        rgb = (value, t, p)
    elif sector == 1:
        rgb = (q, value, p)
    elif sector == 2:
        rgb = (p, value, t)
    elif sector == 3:
        rgb = (p, q, value)
    elif sector == 4:
        rgb = (t, p, value)
    else:
        rgb = (value, p, q)
    return (*rgb, 1.0)


def category_color(category_id: int) -> Tuple[float, float, float, float]:
    return object_id_to_color(1000 + int(category_id))


def apply_single_material_to_meshes(material: Any) -> None:
    import bpy  # type: ignore

    for obj in all_mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)
    bpy.context.view_layer.update()


def apply_target_mask_materials(target_parts: Sequence[Any]) -> None:
    import bpy  # type: ignore

    black = create_emission_material("pass_mask_black", (0.0, 0.0, 0.0, 1.0))
    white = create_emission_material("pass_mask_white", (1.0, 1.0, 1.0, 1.0))
    target_names = {obj.name for obj in target_parts}
    for obj in all_mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(white if obj.name in target_names else black)
    bpy.context.view_layer.update()


def apply_instance_materials() -> Dict[str, Any]:
    import bpy  # type: ignore

    palette: Dict[str, Any] = {}
    for obj in all_mesh_objects():
        object_id = int(obj.get("object_id", 0))
        color = object_id_to_color(object_id)
        mat = create_emission_material(f"pass_instance_{object_id:04d}", color)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        palette[str(object_id)] = {"linear_rgba": [float(v) for v in color], "category": obj.get("category_name", "")}
    bpy.context.view_layer.update()
    return palette


def apply_semantic_materials() -> Dict[str, Any]:
    import bpy  # type: ignore

    palette: Dict[str, Any] = {}
    for obj in all_mesh_objects():
        category_id = int(obj.get("category_id", 0))
        category = str(obj.get("category_name", "unknown"))
        color = category_color(category_id)
        mat = create_emission_material(f"pass_semantic_{category_id:04d}", color)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        palette[str(category_id)] = {"linear_rgba": [float(v) for v in color], "category": category}
    bpy.context.view_layer.update()
    return palette


def apply_albedo_materials(original_slots: Mapping[str, Sequence[Any]]) -> None:
    import bpy  # type: ignore

    for obj in all_mesh_objects():
        slots = original_slots.get(obj.name)
        if not slots:
            continue
        source = slots[0]
        color = tuple(float(v) for v in getattr(source, "diffuse_color", (0.8, 0.8, 0.8, 1.0)))
        mat = create_emission_material(f"pass_albedo_{source.name}", color)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    bpy.context.view_layer.update()


def configure_png_output() -> None:
    import bpy  # type: ignore

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def render_png(path: Path, samples: int) -> None:
    import bpy  # type: ignore

    configure_png_output()
    scene = bpy.context.scene
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, samples)
        scene.eevee.taa_samples = max(1, min(samples, 16))
    scene.frame_set(1)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def setup_depth_compositor(frame_dir: Path) -> None:
    import bpy  # type: ignore

    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.use_nodes = True
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    if hasattr(scene, "node_tree"):
        tree = scene.node_tree
    else:
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("ObjectTranslationDepthCompositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
    tree.nodes.clear()
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
    try:
        depth_output.format.file_format = "OPEN_EXR"
    except TypeError:
        depth_output.format.file_format = "OPEN_EXR_MULTILAYER"
    depth_output.format.color_depth = "32"
    if hasattr(depth_output, "base_path"):
        depth_output.base_path = str(frame_dir)
        depth_output.file_slots[0].path = "depth_"
        depth_input = depth_output.inputs[0]
    else:
        depth_output.directory = str(frame_dir)
        depth_output.file_name = "depth"
        depth_output.use_file_extension = True
        depth_output.file_output_items.clear()
        item = depth_output.file_output_items.new("FLOAT", "Depth")
        item.override_node_format = False
        depth_input = depth_output.inputs["Depth"]
    tree.links.new(render_layers.outputs["Depth"], depth_input)


def render_depth(frame_dir: Path, samples: int) -> Path:
    import bpy  # type: ignore

    setup_depth_compositor(frame_dir)
    scene = bpy.context.scene
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, samples)
        scene.eevee.taa_samples = max(1, min(samples, 16))
    temp_main = frame_dir / "_depth_main.png"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(temp_main)
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)
    target = frame_dir / "depth.exr"
    if not target.exists():
        for path in list(frame_dir.glob("depth_*.exr")) + list(frame_dir.glob("depth*.exr")):
            if path == target:
                continue
            if target.exists():
                target.unlink()
            path.rename(target)
            break
    if temp_main.exists():
        temp_main.unlink()
    scene.use_nodes = False
    scene.render.use_compositing = False
    if not target.exists():
        raise FileNotFoundError(f"Depth compositor did not create {target}")
    return target


def save_depth_npy(depth_path: Path, npy_path: Path) -> None:
    import bpy  # type: ignore

    image = bpy.data.images.load(str(depth_path), check_existing=False)
    try:
        width, height = [int(value) for value in image.size]
        if image.channels > 0 and width > 0 and height > 0:
            pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, image.channels)
            depth = np.flipud(pixels[:, :, 0])
        else:
            raise ValueError("Blender image API returned an empty EXR buffer")
    except Exception:
        import OpenImageIO as oiio  # type: ignore

        inp = oiio.ImageInput.open(str(depth_path))
        if inp is None:
            raise RuntimeError(f"OpenImageIO could not open {depth_path}")
        try:
            spec = inp.spec()
            pixels = inp.read_image(format=oiio.FLOAT)
            if pixels is None:
                raise RuntimeError(f"OpenImageIO could not read pixels from {depth_path}")
            arr = np.asarray(pixels, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
            depth = arr[:, :, 0]
        finally:
            inp.close()
    finally:
        if image.name in bpy.data.images:
            bpy.data.images.remove(image)
    np.save(npy_path, depth)


def required_frame_files(frame_dir: Path) -> List[Path]:
    return [
        frame_dir / "rgb.png",
        frame_dir / "depth.exr",
        frame_dir / "depth.npy",
        frame_dir / "normal.png",
        frame_dir / "target_mask.png",
        frame_dir / "instance.png",
        frame_dir / "semantic.png",
        frame_dir / "object_id.png",
        frame_dir / "albedo.png",
        frame_dir / "frame_metadata.json",
    ]


def existing_frame_matches_resolution(frame_dir: Path, expected_resolution: Optional[Sequence[int]]) -> bool:
    if expected_resolution is None:
        return True
    from .manifest_utils import read_json
    import bpy  # type: ignore

    metadata_path = frame_dir / "frame_metadata.json"
    expected = [int(expected_resolution[0]), int(expected_resolution[1])]
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        saved_size = metadata.get("image_size")
        if saved_size is not None:
            return [int(saved_size[0]), int(saved_size[1])] == expected

    rgb_path = frame_dir / "rgb.png"
    if not rgb_path.exists():
        return False
    image = bpy.data.images.load(str(rgb_path), check_existing=False)
    try:
        width, height = [int(value) for value in image.size]
        return [width, height] == expected
    finally:
        bpy.data.images.remove(image)


def render_frame_passes(
    frame_dir: Path,
    target_asset: FurnitureAsset,
    camera_payload: Mapping[str, Any],
    samples: int,
    overwrite: bool,
    expected_resolution: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    from .label_utils import blender_depth_range_from_exr, blender_image_mask_stats, blender_target_visibility_fraction
    from .manifest_utils import read_json

    if not overwrite and all(path.exists() for path in required_frame_files(frame_dir)):
        if existing_frame_matches_resolution(frame_dir, expected_resolution):
            return read_json(frame_dir / "frame_metadata.json")
        raise FileExistsError(
            f"{frame_dir} already contains a complete frame at a different resolution. "
            "Use --overwrite or choose a different --output-root."
        )

    ensure_dir(frame_dir)
    configure_camera(camera_payload)
    original_slots = capture_material_slots()

    restore_material_slots(original_slots)
    render_png(frame_dir / "rgb.png", samples=samples)

    restore_material_slots(original_slots)
    depth_path = render_depth(frame_dir, samples=max(1, samples))
    save_depth_npy(depth_path, frame_dir / "depth.npy")

    normal = create_normal_material()
    apply_single_material_to_meshes(normal)
    render_png(frame_dir / "normal.png", samples=1)

    apply_target_mask_materials(target_asset.parts)
    render_png(frame_dir / "target_mask.png", samples=1)

    instance_palette = apply_instance_materials()
    render_png(frame_dir / "instance.png", samples=1)
    shutil.copyfile(frame_dir / "instance.png", frame_dir / "object_id.png")
    write_json(frame_dir / "object_id_palette.json", instance_palette)

    semantic_palette = apply_semantic_materials()
    render_png(frame_dir / "semantic.png", samples=1)
    write_json(frame_dir / "semantic_palette.json", semantic_palette)

    apply_albedo_materials(original_slots)
    render_png(frame_dir / "albedo.png", samples=1)

    restore_material_slots(original_slots)
    mask_stats = blender_image_mask_stats(frame_dir / "target_mask.png", edge_margin_px=3)
    visible_fraction = blender_target_visibility_fraction(target_asset.parts)
    depth_range = blender_depth_range_from_exr(depth_path)
    return {
        "mask_stats": mask_stats,
        "visible_fraction": visible_fraction,
        "depth_range_m": depth_range,
    }
