"""Procedural furniture assets for the object-translation dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


CATEGORY_IDS: Dict[str, int] = {
    "room": 1,
    "wall": 2,
    "floor": 3,
    "chair": 10,
    "armchair": 11,
    "side_table": 12,
    "small_cabinet": 13,
    "sofa": 30,
    "bookcase": 31,
    "table": 32,
    "lamp": 33,
    "stool": 34,
    "static_cabinet": 35,
}

TARGET_CATEGORIES: Tuple[str, ...] = ("chair", "armchair", "side_table", "small_cabinet")


@dataclass
class FurnitureAsset:
    root: Any
    parts: List[Any]
    logical_name: str
    object_id: int
    category: str
    is_target: bool
    signature: Dict[str, Any]


MATERIAL_SPECS: Dict[str, Dict[str, Any]] = {
    "floor_warm_oak": {
        "base_color": (0.58, 0.45, 0.31, 1.0),
        "roughness": 0.72,
        "noise": True,
        "noise_scale": 18.0,
        "noise_strength": 0.09,
    },
    "floor_desaturated_wood": {
        "base_color": (0.47, 0.42, 0.35, 1.0),
        "roughness": 0.78,
        "noise": True,
        "noise_scale": 22.0,
        "noise_strength": 0.06,
    },
    "floor_cool_concrete": {
        "base_color": (0.46, 0.48, 0.47, 1.0),
        "roughness": 0.84,
        "noise": True,
        "noise_scale": 28.0,
        "noise_strength": 0.05,
    },
    "wall_soft_gray": {
        "base_color": (0.67, 0.69, 0.68, 1.0),
        "roughness": 0.80,
        "noise": True,
        "noise_scale": 13.0,
        "noise_strength": 0.035,
    },
    "wall_warm_white": {
        "base_color": (0.76, 0.73, 0.67, 1.0),
        "roughness": 0.82,
        "noise": True,
        "noise_scale": 12.0,
        "noise_strength": 0.035,
    },
    "wall_cool_gray": {
        "base_color": (0.61, 0.65, 0.67, 1.0),
        "roughness": 0.82,
        "noise": True,
        "noise_scale": 14.0,
        "noise_strength": 0.035,
    },
    "baseboard_charcoal": {"base_color": (0.17, 0.18, 0.18, 1.0), "roughness": 0.74},
    "wood_dark": {
        "base_color": (0.30, 0.20, 0.13, 1.0),
        "roughness": 0.66,
        "noise": True,
        "noise_scale": 20.0,
        "noise_strength": 0.05,
    },
    "wood_light": {
        "base_color": (0.70, 0.56, 0.37, 1.0),
        "roughness": 0.64,
        "noise": True,
        "noise_scale": 18.0,
        "noise_strength": 0.06,
    },
    "fabric_green": {
        "base_color": (0.22, 0.43, 0.35, 1.0),
        "roughness": 0.86,
        "noise": True,
        "noise_scale": 32.0,
        "noise_strength": 0.045,
    },
    "fabric_blue": {
        "base_color": (0.20, 0.31, 0.50, 1.0),
        "roughness": 0.84,
        "noise": True,
        "noise_scale": 28.0,
        "noise_strength": 0.04,
    },
    "target_teal": {"base_color": (0.05, 0.55, 0.58, 1.0), "roughness": 0.62},
    "target_clay": {"base_color": (0.62, 0.25, 0.16, 1.0), "roughness": 0.68},
    "target_gold_wood": {"base_color": (0.78, 0.54, 0.20, 1.0), "roughness": 0.60},
    "target_slate": {"base_color": (0.24, 0.28, 0.34, 1.0), "roughness": 0.70},
    "metal_dark": {"base_color": (0.08, 0.085, 0.09, 1.0), "roughness": 0.38, "metallic": 0.5},
    "lamp_shade": {"base_color": (0.86, 0.80, 0.66, 1.0), "roughness": 0.72},
    "accent_red": {"base_color": (0.63, 0.14, 0.10, 1.0), "roughness": 0.66},
    "accent_offwhite": {"base_color": (0.82, 0.80, 0.72, 1.0), "roughness": 0.68},
}


def create_materials() -> Dict[str, Any]:
    import bpy  # type: ignore

    materials: Dict[str, Any] = {}
    for name, spec in MATERIAL_SPECS.items():
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = spec["base_color"]
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is not None:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = spec["base_color"]
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = float(spec.get("roughness", 0.65))
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = float(spec.get("metallic", 0.0))
            if spec.get("noise", False):
                noise = nodes.new(type="ShaderNodeTexNoise")
                noise.inputs["Scale"].default_value = float(spec.get("noise_scale", 18.0))
                noise.inputs["Detail"].default_value = 8.0
                noise.inputs["Roughness"].default_value = 0.55
                ramp = nodes.new(type="ShaderNodeValToRGB")
                strength = float(spec.get("noise_strength", 0.04))
                base = spec["base_color"]
                low = tuple(max(0.0, float(c) - strength) for c in base[:3]) + (1.0,)
                high = tuple(min(1.0, float(c) + strength) for c in base[:3]) + (1.0,)
                ramp.color_ramp.elements[0].position = 0.15
                ramp.color_ramp.elements[0].color = low
                ramp.color_ramp.elements[1].position = 1.0
                ramp.color_ramp.elements[1].color = high
                mat.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
                mat.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        materials[name] = mat
    return materials


def _set_object_properties(obj: Any, object_id: int, category: str, logical_name: str, is_target: bool) -> None:
    obj["object_id"] = int(object_id)
    obj["category_id"] = int(CATEGORY_IDS[category])
    obj["category_name"] = category
    obj["logical_name"] = logical_name
    obj["is_target"] = bool(is_target)
    obj.pass_index = int(object_id)


def _add_bevel(obj: Any, width: float = 0.015, segments: int = 2) -> None:
    bevel = obj.modifiers.new("softened_edges", "BEVEL")
    bevel.width = float(width)
    bevel.segments = int(segments)
    bevel.affect = "EDGES"
    normals = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    normals.keep_sharp = True


def create_empty(name: str, location: Sequence[float] = (0.0, 0.0, 0.0), rotation_z: float = 0.0) -> Any:
    import bpy  # type: ignore

    root = bpy.data.objects.new(name, None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.25
    root.location = tuple(float(v) for v in location)
    root.rotation_euler = (0.0, 0.0, float(rotation_z))
    bpy.context.collection.objects.link(root)
    return root


def add_cube_part(
    parent: Any,
    name: str,
    location: Sequence[float],
    dimensions: Sequence[float],
    material: Any,
    object_id: int,
    category: str,
    logical_name: str,
    is_target: bool,
    rotation_z: float = 0.0,
    bevel_width: float = 0.014,
) -> Any:
    import bpy  # type: ignore

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, float(rotation_z)))
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    obj.parent = parent
    obj.location = tuple(float(v) for v in location)
    obj.dimensions = tuple(float(v) for v in dimensions)
    obj.data.materials.append(material)
    _set_object_properties(obj, object_id, category, logical_name, is_target)
    _add_bevel(obj, bevel_width)
    return obj


def add_cylinder_part(
    parent: Any,
    name: str,
    location: Sequence[float],
    radius: float,
    depth: float,
    material: Any,
    object_id: int,
    category: str,
    logical_name: str,
    is_target: bool,
    vertices: int = 32,
    bevel_width: float = 0.006,
) -> Any:
    import bpy  # type: ignore

    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=float(radius), depth=float(depth), location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    obj.parent = parent
    obj.location = tuple(float(v) for v in location)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    _set_object_properties(obj, object_id, category, logical_name, is_target)
    _add_bevel(obj, bevel_width)
    return obj


def add_cone_part(
    parent: Any,
    name: str,
    location: Sequence[float],
    radius1: float,
    radius2: float,
    depth: float,
    material: Any,
    object_id: int,
    category: str,
    logical_name: str,
    is_target: bool,
    vertices: int = 36,
) -> Any:
    import bpy  # type: ignore

    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=float(radius1),
        radius2=float(radius2),
        depth=float(depth),
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    obj.parent = parent
    obj.location = tuple(float(v) for v in location)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    _set_object_properties(obj, object_id, category, logical_name, is_target)
    _add_bevel(obj, 0.004)
    return obj


def asset_world_bbox(asset: FurnitureAsset) -> List[List[float]]:
    from mathutils import Vector  # type: ignore

    points = []
    for obj in asset.parts:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return []
    arr = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float64)
    min_xyz = arr.min(axis=0)
    max_xyz = arr.max(axis=0)
    from .label_utils import aabb_corners

    return aabb_corners(min_xyz, max_xyz)


def create_room_shell(layout: Mapping[str, Any], materials: Mapping[str, Any]) -> List[FurnitureAsset]:
    width, depth, height = [float(v) for v in layout["dimensions"]]
    floor_mat = materials[str(layout["floor_material"])]
    wall_mat = materials[str(layout["wall_material"])]
    baseboard_mat = materials["baseboard_charcoal"]

    assets: List[FurnitureAsset] = []
    root = create_empty("room_shell_root")
    floor = add_cube_part(root, "floor", (0.0, 0.0, -0.025), (width, depth, 0.05), floor_mat, 1, "floor", "floor", False, bevel_width=0.0)
    north = add_cube_part(
        root,
        "wall_north",
        (0.0, depth * 0.5, height * 0.5),
        (width, 0.08, height),
        wall_mat,
        2,
        "wall",
        "wall_north",
        False,
        bevel_width=0.0,
    )
    west = add_cube_part(
        root,
        "wall_west",
        (-width * 0.5, 0.0, height * 0.5),
        (0.08, depth, height),
        wall_mat,
        3,
        "wall",
        "wall_west",
        False,
        bevel_width=0.0,
    )
    base_n = add_cube_part(
        root,
        "baseboard_north",
        (0.0, depth * 0.5 - 0.065, 0.075),
        (width, 0.05, 0.15),
        baseboard_mat,
        4,
        "wall",
        "baseboard_north",
        False,
        bevel_width=0.004,
    )
    base_w = add_cube_part(
        root,
        "baseboard_west",
        (-width * 0.5 + 0.065, 0.0, 0.075),
        (0.05, depth, 0.15),
        baseboard_mat,
        5,
        "wall",
        "baseboard_west",
        False,
        bevel_width=0.004,
    )
    for logical_name, obj in (("floor", floor), ("wall_north", north), ("wall_west", west), ("baseboard_north", base_n), ("baseboard_west", base_w)):
        assets.append(
            FurnitureAsset(
                root=obj,
                parts=[obj],
                logical_name=logical_name,
                object_id=int(obj["object_id"]),
                category=str(obj["category_name"]),
                is_target=False,
                signature={"asset_type": logical_name, "dimensions": list(obj.dimensions)},
            )
        )
    return assets


def _chair_dimensions(variant: int) -> Dict[str, float]:
    variants = [
        {"w": 0.58, "d": 0.54, "seat_h": 0.44, "back_h": 0.52, "leg": 0.065},
        {"w": 0.52, "d": 0.60, "seat_h": 0.46, "back_h": 0.58, "leg": 0.055},
        {"w": 0.64, "d": 0.50, "seat_h": 0.42, "back_h": 0.48, "leg": 0.075},
    ]
    return variants[variant % len(variants)]


def create_chair(
    logical_name: str,
    object_id: int,
    variant: int,
    materials: Mapping[str, Any],
    is_target: bool,
    material_name: str,
) -> FurnitureAsset:
    dims = _chair_dimensions(variant)
    w, d, seat_h, back_h, leg = dims["w"], dims["d"], dims["seat_h"], dims["back_h"], dims["leg"]
    mat = materials[material_name]
    dark = materials["metal_dark"]
    root = create_empty(f"{logical_name}_root")
    parts: List[Any] = []
    cat = "chair"
    parts.append(add_cube_part(root, f"{logical_name}_seat", (0.0, 0.0, seat_h), (w, d, 0.11), mat, object_id, cat, logical_name, is_target, bevel_width=0.025))
    leg_y = d * 0.35
    leg_x = w * 0.38
    for idx, (x, y) in enumerate([(-leg_x, -leg_y), (leg_x, -leg_y), (-leg_x, leg_y), (leg_x * 0.82, leg_y)]):
        parts.append(
            add_cube_part(
                root,
                f"{logical_name}_leg_{idx}",
                (x, y, seat_h * 0.5),
                (leg, leg, seat_h),
                dark,
                object_id,
                cat,
                logical_name,
                is_target,
                bevel_width=0.011,
            )
        )
    parts.append(add_cube_part(root, f"{logical_name}_back_panel", (0.0, d * 0.48, seat_h + back_h * 0.52), (w * 0.92, 0.075, back_h), mat, object_id, cat, logical_name, is_target, bevel_width=0.022))
    parts.append(add_cube_part(root, f"{logical_name}_back_top_rail", (0.0, d * 0.52, seat_h + back_h + 0.04), (w, 0.09, 0.08), mat, object_id, cat, logical_name, is_target, bevel_width=0.02))
    parts.append(add_cube_part(root, f"{logical_name}_side_brace", (-w * 0.50, 0.08, seat_h * 0.82), (0.05, d * 0.75, 0.055), dark, object_id, cat, logical_name, is_target, rotation_z=-0.08, bevel_width=0.009))
    signature = {"asset_type": "chair", "variant": int(variant), "parameters": dims, "material": material_name}
    return FurnitureAsset(root, parts, logical_name, object_id, cat, is_target, signature)


def create_armchair(
    logical_name: str,
    object_id: int,
    variant: int,
    materials: Mapping[str, Any],
    is_target: bool,
    material_name: str,
) -> FurnitureAsset:
    variants = [
        {"w": 0.78, "d": 0.66, "seat_h": 0.38, "back_h": 0.48, "arm_h": 0.57},
        {"w": 0.88, "d": 0.58, "seat_h": 0.40, "back_h": 0.42, "arm_h": 0.55},
        {"w": 0.74, "d": 0.72, "seat_h": 0.36, "back_h": 0.52, "arm_h": 0.60},
    ]
    dims = variants[variant % len(variants)]
    w, d, seat_h, back_h, arm_h = dims["w"], dims["d"], dims["seat_h"], dims["back_h"], dims["arm_h"]
    mat = materials[material_name]
    accent = materials["accent_offwhite"]
    root = create_empty(f"{logical_name}_root")
    parts: List[Any] = []
    cat = "armchair"
    parts.append(add_cube_part(root, f"{logical_name}_seat_cushion", (0.0, -0.02, seat_h), (w, d, 0.15), mat, object_id, cat, logical_name, is_target, bevel_width=0.035))
    parts.append(add_cube_part(root, f"{logical_name}_back_cushion", (0.0, d * 0.43, seat_h + back_h * 0.52), (w, 0.13, back_h), mat, object_id, cat, logical_name, is_target, bevel_width=0.035))
    parts.append(add_cube_part(root, f"{logical_name}_left_arm", (-w * 0.55, -0.03, arm_h * 0.5), (0.13, d * 1.03, arm_h), mat, object_id, cat, logical_name, is_target, bevel_width=0.03))
    parts.append(add_cube_part(root, f"{logical_name}_right_arm", (w * 0.55, -0.07, arm_h * 0.5), (0.13, d * 0.90, arm_h), mat, object_id, cat, logical_name, is_target, bevel_width=0.03))
    parts.append(add_cube_part(root, f"{logical_name}_front_apron", (0.0, -d * 0.55, 0.22), (w * 0.88, 0.08, 0.18), accent, object_id, cat, logical_name, is_target, bevel_width=0.018))
    for idx, x in enumerate([-w * 0.36, w * 0.36]):
        parts.append(add_cube_part(root, f"{logical_name}_front_foot_{idx}", (x, -d * 0.46, 0.075), (0.11, 0.11, 0.15), materials["wood_dark"], object_id, cat, logical_name, is_target, bevel_width=0.012))
    signature = {"asset_type": "armchair", "variant": int(variant), "parameters": dims, "material": material_name}
    return FurnitureAsset(root, parts, logical_name, object_id, cat, is_target, signature)


def create_side_table(
    logical_name: str,
    object_id: int,
    variant: int,
    materials: Mapping[str, Any],
    is_target: bool,
    material_name: str,
) -> FurnitureAsset:
    variants = [
        {"w": 0.58, "d": 0.48, "h": 0.58, "top": 0.08, "leg": 0.055},
        {"w": 0.50, "d": 0.56, "h": 0.62, "top": 0.075, "leg": 0.05},
        {"w": 0.66, "d": 0.44, "h": 0.54, "top": 0.09, "leg": 0.06},
    ]
    dims = variants[variant % len(variants)]
    w, d, h, top, leg = dims["w"], dims["d"], dims["h"], dims["top"], dims["leg"]
    mat = materials[material_name]
    root = create_empty(f"{logical_name}_root")
    parts: List[Any] = []
    cat = "side_table"
    parts.append(add_cube_part(root, f"{logical_name}_top", (0.0, 0.0, h), (w, d, top), mat, object_id, cat, logical_name, is_target, bevel_width=0.02))
    leg_z = (h - top * 0.5) * 0.5
    for idx, (x, y) in enumerate([(-w * 0.39, -d * 0.36), (w * 0.39, -d * 0.36), (-w * 0.39, d * 0.36), (w * 0.32, d * 0.36)]):
        parts.append(add_cube_part(root, f"{logical_name}_leg_{idx}", (x, y, leg_z), (leg, leg, h - top * 0.5), materials["wood_dark"], object_id, cat, logical_name, is_target, bevel_width=0.009))
    parts.append(add_cube_part(root, f"{logical_name}_lower_shelf", (0.0, 0.03, h * 0.42), (w * 0.78, d * 0.70, 0.055), mat, object_id, cat, logical_name, is_target, bevel_width=0.015))
    parts.append(add_cube_part(root, f"{logical_name}_drawer_front", (0.0, -d * 0.515, h * 0.78), (w * 0.64, 0.035, 0.15), materials["accent_offwhite"], object_id, cat, logical_name, is_target, bevel_width=0.012))
    parts.append(add_cube_part(root, f"{logical_name}_drawer_pull", (w * 0.12, -d * 0.542, h * 0.78), (0.12, 0.025, 0.025), materials["metal_dark"], object_id, cat, logical_name, is_target, bevel_width=0.006))
    signature = {"asset_type": "side_table", "variant": int(variant), "parameters": dims, "material": material_name}
    return FurnitureAsset(root, parts, logical_name, object_id, cat, is_target, signature)


def create_small_cabinet(
    logical_name: str,
    object_id: int,
    variant: int,
    materials: Mapping[str, Any],
    is_target: bool,
    material_name: str,
) -> FurnitureAsset:
    variants = [
        {"w": 0.72, "d": 0.42, "h": 0.82},
        {"w": 0.64, "d": 0.46, "h": 0.76},
        {"w": 0.80, "d": 0.40, "h": 0.70},
    ]
    dims = variants[variant % len(variants)]
    w, d, h = dims["w"], dims["d"], dims["h"]
    mat = materials[material_name]
    root = create_empty(f"{logical_name}_root")
    parts: List[Any] = []
    cat = "small_cabinet"
    parts.append(add_cube_part(root, f"{logical_name}_body", (0.0, 0.0, h * 0.5), (w, d, h), mat, object_id, cat, logical_name, is_target, bevel_width=0.025))
    parts.append(add_cube_part(root, f"{logical_name}_left_door", (-w * 0.255, -d * 0.515, h * 0.52), (w * 0.46, 0.035, h * 0.68), materials["accent_offwhite"], object_id, cat, logical_name, is_target, bevel_width=0.012))
    parts.append(add_cube_part(root, f"{logical_name}_right_drawer_a", (w * 0.25, -d * 0.52, h * 0.68), (w * 0.40, 0.035, h * 0.22), materials["accent_offwhite"], object_id, cat, logical_name, is_target, bevel_width=0.012))
    parts.append(add_cube_part(root, f"{logical_name}_right_drawer_b", (w * 0.25, -d * 0.52, h * 0.38), (w * 0.40, 0.035, h * 0.22), materials["accent_offwhite"], object_id, cat, logical_name, is_target, bevel_width=0.012))
    parts.append(add_cube_part(root, f"{logical_name}_left_handle", (-w * 0.08, -d * 0.555, h * 0.52), (0.035, 0.025, h * 0.22), materials["metal_dark"], object_id, cat, logical_name, is_target, bevel_width=0.004))
    parts.append(add_cube_part(root, f"{logical_name}_drawer_pull_top", (w * 0.25, -d * 0.555, h * 0.68), (w * 0.22, 0.025, 0.025), materials["metal_dark"], object_id, cat, logical_name, is_target, bevel_width=0.004))
    parts.append(add_cube_part(root, f"{logical_name}_drawer_pull_bottom", (w * 0.25, -d * 0.555, h * 0.38), (w * 0.22, 0.025, 0.025), materials["metal_dark"], object_id, cat, logical_name, is_target, bevel_width=0.004))
    parts.append(add_cube_part(root, f"{logical_name}_toe_kick", (0.0, -d * 0.20, 0.055), (w * 0.86, d * 0.50, 0.11), materials["wood_dark"], object_id, cat, logical_name, is_target, bevel_width=0.01))
    signature = {"asset_type": "small_cabinet", "variant": int(variant), "parameters": dims, "material": material_name}
    return FurnitureAsset(root, parts, logical_name, object_id, cat, is_target, signature)


def create_target_asset(
    category: str,
    variant: int,
    materials: Mapping[str, Any],
    object_id: int = 20,
) -> FurnitureAsset:
    material_by_category = {
        "chair": "target_teal",
        "armchair": "target_clay",
        "side_table": "target_gold_wood",
        "small_cabinet": "target_slate",
    }
    material_name = material_by_category[category]
    logical_name = f"{category}_{variant:03d}"
    if category == "chair":
        return create_chair(logical_name, object_id, variant, materials, True, material_name)
    if category == "armchair":
        return create_armchair(logical_name, object_id, variant, materials, True, material_name)
    if category == "side_table":
        return create_side_table(logical_name, object_id, variant, materials, True, material_name)
    if category == "small_cabinet":
        return create_small_cabinet(logical_name, object_id, variant, materials, True, material_name)
    raise ValueError(f"Unsupported target category {category!r}")


def create_sofa(logical_name: str, object_id: int, materials: Mapping[str, Any]) -> FurnitureAsset:
    root = create_empty(f"{logical_name}_root")
    mat = materials["fabric_blue"]
    parts = [
        add_cube_part(root, f"{logical_name}_seat", (0.0, 0.0, 0.35), (1.45, 0.62, 0.20), mat, object_id, "sofa", logical_name, False, bevel_width=0.045),
        add_cube_part(root, f"{logical_name}_back", (0.0, 0.34, 0.72), (1.52, 0.16, 0.74), mat, object_id, "sofa", logical_name, False, bevel_width=0.04),
        add_cube_part(root, f"{logical_name}_left_arm", (-0.83, 0.0, 0.48), (0.18, 0.72, 0.52), mat, object_id, "sofa", logical_name, False, bevel_width=0.035),
        add_cube_part(root, f"{logical_name}_right_arm", (0.83, -0.04, 0.48), (0.18, 0.64, 0.52), mat, object_id, "sofa", logical_name, False, bevel_width=0.035),
        add_cube_part(root, f"{logical_name}_pillow", (-0.35, -0.12, 0.55), (0.38, 0.10, 0.28), materials["accent_red"], object_id, "sofa", logical_name, False, bevel_width=0.025),
    ]
    return FurnitureAsset(root, parts, logical_name, object_id, "sofa", False, {"asset_type": "sofa"})


def create_bookcase(logical_name: str, object_id: int, materials: Mapping[str, Any]) -> FurnitureAsset:
    root = create_empty(f"{logical_name}_root")
    wood = materials["wood_dark"]
    accent = materials["accent_offwhite"]
    parts = [
        add_cube_part(root, f"{logical_name}_back", (0.0, 0.05, 0.78), (0.82, 0.08, 1.56), wood, object_id, "bookcase", logical_name, False, bevel_width=0.018),
        add_cube_part(root, f"{logical_name}_left_side", (-0.45, 0.0, 0.78), (0.08, 0.36, 1.56), wood, object_id, "bookcase", logical_name, False, bevel_width=0.018),
        add_cube_part(root, f"{logical_name}_right_side", (0.45, 0.0, 0.78), (0.08, 0.36, 1.56), wood, object_id, "bookcase", logical_name, False, bevel_width=0.018),
    ]
    for idx, z in enumerate([0.24, 0.58, 0.92, 1.26, 1.54]):
        parts.append(add_cube_part(root, f"{logical_name}_shelf_{idx}", (0.0, 0.0, z), (0.90, 0.38, 0.055), wood, object_id, "bookcase", logical_name, False, bevel_width=0.012))
    for idx, (x, z, h) in enumerate([(-0.20, 0.41, 0.20), (0.16, 0.78, 0.24), (-0.10, 1.12, 0.22), (0.26, 1.40, 0.18)]):
        parts.append(add_cube_part(root, f"{logical_name}_book_{idx}", (x, -0.13, z), (0.12, 0.10, h), accent, object_id, "bookcase", logical_name, False, bevel_width=0.006))
    return FurnitureAsset(root, parts, logical_name, object_id, "bookcase", False, {"asset_type": "bookcase"})


def create_reference_table(logical_name: str, object_id: int, materials: Mapping[str, Any]) -> FurnitureAsset:
    root = create_empty(f"{logical_name}_root")
    wood = materials["wood_light"]
    dark = materials["metal_dark"]
    parts = [add_cube_part(root, f"{logical_name}_top", (0.0, 0.0, 0.58), (1.18, 0.58, 0.09), wood, object_id, "table", logical_name, False, bevel_width=0.02)]
    for idx, (x, y) in enumerate([(-0.48, -0.22), (0.48, -0.22), (-0.48, 0.22), (0.40, 0.22)]):
        parts.append(add_cube_part(root, f"{logical_name}_leg_{idx}", (x, y, 0.29), (0.06, 0.06, 0.58), dark, object_id, "table", logical_name, False, bevel_width=0.008))
    parts.append(add_cube_part(root, f"{logical_name}_cross_rail", (0.0, 0.25, 0.34), (0.95, 0.04, 0.05), dark, object_id, "table", logical_name, False, bevel_width=0.006))
    return FurnitureAsset(root, parts, logical_name, object_id, "table", False, {"asset_type": "reference_table"})


def create_floor_lamp(logical_name: str, object_id: int, materials: Mapping[str, Any]) -> FurnitureAsset:
    root = create_empty(f"{logical_name}_root")
    dark = materials["metal_dark"]
    shade = materials["lamp_shade"]
    parts = [
        add_cylinder_part(root, f"{logical_name}_base", (0.0, 0.0, 0.035), 0.18, 0.07, dark, object_id, "lamp", logical_name, False, vertices=48),
        add_cylinder_part(root, f"{logical_name}_pole", (0.0, 0.0, 0.78), 0.025, 1.48, dark, object_id, "lamp", logical_name, False, vertices=24),
        add_cone_part(root, f"{logical_name}_shade", (0.0, 0.0, 1.57), 0.27, 0.18, 0.32, shade, object_id, "lamp", logical_name, False, vertices=48),
        add_cube_part(root, f"{logical_name}_pull", (0.10, -0.03, 1.36), (0.018, 0.018, 0.22), dark, object_id, "lamp", logical_name, False, bevel_width=0.003),
    ]
    return FurnitureAsset(root, parts, logical_name, object_id, "lamp", False, {"asset_type": "floor_lamp"})


def create_stool(logical_name: str, object_id: int, materials: Mapping[str, Any]) -> FurnitureAsset:
    root = create_empty(f"{logical_name}_root")
    mat = materials["fabric_green"]
    dark = materials["wood_dark"]
    parts = [
        add_cube_part(root, f"{logical_name}_seat", (0.0, 0.0, 0.42), (0.44, 0.38, 0.11), mat, object_id, "stool", logical_name, False, bevel_width=0.03),
    ]
    for idx, (x, y) in enumerate([(-0.16, -0.14), (0.16, -0.14), (-0.16, 0.14), (0.12, 0.14)]):
        parts.append(add_cube_part(root, f"{logical_name}_leg_{idx}", (x, y, 0.21), (0.045, 0.045, 0.42), dark, object_id, "stool", logical_name, False, bevel_width=0.007))
    return FurnitureAsset(root, parts, logical_name, object_id, "stool", False, {"asset_type": "stool"})


def create_static_cabinet(logical_name: str, object_id: int, materials: Mapping[str, Any]) -> FurnitureAsset:
    asset = create_small_cabinet(logical_name, object_id, 1, materials, False, "wood_dark")
    asset.category = "static_cabinet"
    for part in asset.parts:
        part["category_id"] = CATEGORY_IDS["static_cabinet"]
        part["category_name"] = "static_cabinet"
    asset.signature = {"asset_type": "static_cabinet"}
    return asset


def create_static_reference_assets(
    layout_id: int,
    room_dimensions: Sequence[float],
    materials: Mapping[str, Any],
) -> List[FurnitureAsset]:
    width, depth, _height = [float(v) for v in room_dimensions]
    creators = [
        create_sofa,
        create_bookcase,
        create_reference_table,
        create_floor_lamp,
        create_stool if layout_id % 2 == 0 else create_static_cabinet,
    ]
    assets = [creator(f"static_{idx:02d}_{creator.__name__.replace('create_', '')}", 100 + idx, materials) for idx, creator in enumerate(creators)]

    placements = [
        ((0.95, depth * 0.5 - 0.48, 0.0), np.pi),
        ((-width * 0.5 + 0.34, 0.72, 0.0), -np.pi * 0.5),
        ((1.82, 0.95, 0.0), -0.22),
        ((-1.68, -1.52, 0.0), 0.12),
        ((2.05, -1.55, 0.0), 0.38),
    ]
    if layout_id % 3 == 1:
        placements = [
            ((1.35, depth * 0.5 - 0.50, 0.0), np.pi),
            ((-width * 0.5 + 0.33, -0.30, 0.0), -np.pi * 0.5),
            ((2.05, 0.45, 0.0), 0.18),
            ((-1.56, -1.65, 0.0), -0.10),
            ((1.38, -1.78, 0.0), -0.34),
        ]
    elif layout_id % 3 == 2:
        placements = [
            ((0.55, depth * 0.5 - 0.52, 0.0), np.pi),
            ((-width * 0.5 + 0.35, 1.20, 0.0), -np.pi * 0.5),
            ((2.02, 1.18, 0.0), -0.38),
            ((-1.65, -1.35, 0.0), 0.20),
            ((1.92, -1.92, 0.0), 0.18),
        ]

    for asset, (location, yaw) in zip(assets, placements):
        asset.root.location = location
        asset.root.rotation_euler = (0.0, 0.0, yaw)
    return assets


def logical_asset_metadata(asset: FurnitureAsset, is_room_shell: bool = False) -> Dict[str, Any]:
    from .label_utils import bounds_from_corners

    corners = asset_world_bbox(asset)
    min_xyz, max_xyz = bounds_from_corners(corners)
    return {
        "logical_name": asset.logical_name,
        "object_id": int(asset.object_id),
        "category": asset.category,
        "category_id": int(CATEGORY_IDS[asset.category]),
        "is_target": bool(asset.is_target),
        "is_room_shell": bool(is_room_shell),
        "part_names": [obj.name for obj in asset.parts],
        "bbox_corners_world": corners,
        "aabb_min_world": [float(v) for v in min_xyz],
        "aabb_max_world": [float(v) for v in max_xyz],
        "root_matrix_world": np.asarray(asset.root.matrix_world, dtype=np.float64).round(8).tolist(),
        "signature": asset.signature,
    }
