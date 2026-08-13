"""Central configuration for the Lingbot-map memory retention scene.

All positions are in Blender world meters.  The experiment deliberately uses
fixed constants instead of sampling so that the three conditions are strictly
paired and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Dict, List, Sequence, Tuple


EXPERIMENT_NAME = "lingbot_memory_basic_v1"
TARGET_NAME = "target_teal_cylinder"

CONDITIONS = ("always_present", "always_absent", "seen_then_removed")


@dataclass(frozen=True)
class FrameConfig:
    total_frames: int = 96
    anchor_count: int = 8
    first_loop_start: int = 8
    first_loop_end: int = 51
    second_loop_start: int = 52
    second_loop_end: int = 95
    local_window: int = 16
    loop_frames: int = 44

    @property
    def anchor_frames(self) -> range:
        return range(0, self.anchor_count)

    @property
    def first_loop_frames(self) -> range:
        return range(self.first_loop_start, self.first_loop_end + 1)

    @property
    def second_loop_frames(self) -> range:
        return range(self.second_loop_start, self.second_loop_end + 1)


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    lens_mm: float = 20.0
    clip_start: float = 0.05
    clip_end: float = 50.0
    radius_x: float = 3.05
    radius_y: float = 2.90
    height_m: float = 1.15
    look_at: Tuple[float, float, float] = (0.0, 0.0, -0.40)
    loop_start_angle_rad: float = pi
    anchor_angle_offset_rad: float = -0.34
    path_mode: str = "orbit"
    loop_waypoint_angles_rad: Tuple[Tuple[int, float], ...] = ()


@dataclass(frozen=True)
class RenderConfig:
    samples: int = 32
    resolution_percentage: int = 100
    exposure: float = 0.0
    gamma: float = 1.0
    visibility_pixel_threshold: int = 1
    visibility_area_ratio: float = 0.0
    random_seed: int = 1234
    output_root: str = "memory_scene_blender/outputs"


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    base_color: Tuple[float, float, float, float]
    roughness: float = 0.62
    metallic: float = 0.0
    procedural_noise: bool = False
    noise_scale: float = 18.0
    noise_strength: float = 0.08


@dataclass(frozen=True)
class ObjectSpec:
    object_id: int
    name: str
    category: str
    primitive: str
    location: Tuple[float, float, float]
    rotation_euler: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    material: str
    is_target: bool = False


@dataclass(frozen=True)
class SceneProfile:
    key: str
    experiment_name: str
    target_name: str
    frame: FrameConfig
    camera: CameraConfig
    render: RenderConfig
    materials: Dict[str, MaterialSpec]
    objects: Sequence[ObjectSpec]
    lighting: str


FRAME = FrameConfig()
CAMERA = CameraConfig()
SIMPLIFIED_V2_CAMERA = CameraConfig(
    path_mode="piecewise_angles",
    loop_waypoint_angles_rad=(
        (0, pi),
        (34, pi - 2.0 * pi * (34 / 44)),
        (43, pi + 2.0 * pi * (43 / 44) - 4.0 * pi),
    ),
)
RENDER = RenderConfig()
LIGHTING_PROFILE = "basic"


CATEGORY_IDS: Dict[str, int] = {
    "room": 1,
    "occluder": 2,
    "block": 3,
    "cylinder": 4,
    "sphere": 5,
    "table": 6,
    "target": 7,
    "decor": 8,
}


MATERIALS: Dict[str, MaterialSpec] = {
    "floor_warm_gray": MaterialSpec(
        "floor_warm_gray", (0.48, 0.49, 0.44, 1.0), procedural_noise=True, noise_scale=22.0, noise_strength=0.10
    ),
    "wall_soft_gray": MaterialSpec(
        "wall_soft_gray", (0.68, 0.70, 0.68, 1.0), procedural_noise=True, noise_scale=15.0, noise_strength=0.07
    ),
    "partition_concrete": MaterialSpec(
        "partition_concrete", (0.45, 0.47, 0.46, 1.0), procedural_noise=True, noise_scale=28.0, noise_strength=0.06
    ),
    "teal_target": MaterialSpec("teal_target", (0.05, 0.56, 0.58, 1.0), roughness=0.54),
    "muted_teal": MaterialSpec("muted_teal", (0.13, 0.46, 0.48, 1.0), roughness=0.58),
    "brick_red": MaterialSpec("brick_red", (0.63, 0.18, 0.13, 1.0), roughness=0.65),
    "ochre": MaterialSpec("ochre", (0.80, 0.56, 0.20, 1.0), roughness=0.60),
    "navy": MaterialSpec("navy", (0.12, 0.20, 0.37, 1.0), roughness=0.62),
    "sage": MaterialSpec("sage", (0.35, 0.54, 0.36, 1.0), roughness=0.64),
    "plum": MaterialSpec("plum", (0.38, 0.22, 0.42, 1.0), roughness=0.68),
    "off_white": MaterialSpec("off_white", (0.78, 0.76, 0.70, 1.0), roughness=0.55),
    "charcoal": MaterialSpec("charcoal", (0.16, 0.16, 0.15, 1.0), roughness=0.70),
}

BASIC_MATERIALS = MATERIALS

SIMPLIFIED_MATERIALS: Dict[str, MaterialSpec] = {
    "simple_floor_gray": MaterialSpec(
        "simple_floor_gray", (0.50, 0.51, 0.48, 1.0), roughness=0.78, procedural_noise=True, noise_scale=18.0, noise_strength=0.04
    ),
    "simple_wall_gray": MaterialSpec(
        "simple_wall_gray", (0.66, 0.68, 0.66, 1.0), roughness=0.82, procedural_noise=True, noise_scale=12.0, noise_strength=0.03
    ),
    "simple_occluder_gray": MaterialSpec("simple_occluder_gray", (0.36, 0.39, 0.39, 1.0), roughness=0.86),
    "teal_target": MaterialSpec("teal_target", (0.05, 0.56, 0.58, 1.0), roughness=0.72),
}


ROOM_OBJECTS: Sequence[ObjectSpec] = (
    ObjectSpec(1, "floor", "room", "cube", (0.0, 0.0, -0.025), (0.0, 0.0, 0.0), (6.4, 6.4, 0.05), "floor_warm_gray"),
    ObjectSpec(2, "wall_north", "room", "cube", (0.0, 3.2, 1.3), (0.0, 0.0, 0.0), (6.4, 0.08, 2.6), "wall_soft_gray"),
    ObjectSpec(3, "wall_south", "room", "cube", (0.0, -3.2, 1.3), (0.0, 0.0, 0.0), (6.4, 0.08, 2.6), "wall_soft_gray"),
    ObjectSpec(4, "wall_east", "room", "cube", (3.2, 0.0, 1.3), (0.0, 0.0, 0.0), (0.08, 6.4, 2.6), "wall_soft_gray"),
    ObjectSpec(5, "wall_west", "room", "cube", (-3.2, 0.0, 1.3), (0.0, 0.0, 0.0), (0.08, 6.4, 2.6), "wall_soft_gray"),
    ObjectSpec(6, "central_partition", "occluder", "cube", (0.0, 0.0, 0.88), (0.0, 0.0, 0.0), (0.36, 2.35, 1.76), "partition_concrete"),
)


STATIC_OBJECTS: Sequence[ObjectSpec] = (
    ObjectSpec(20, TARGET_NAME, "target", "cylinder", (1.62, -0.74, 0.42), (0.0, 0.0, 0.0), (0.50, 0.50, 0.84), "teal_target", True),
    ObjectSpec(21, "decoy_teal_cylinder", "cylinder", "cylinder", (-1.45, 1.15, 0.33), (0.0, 0.0, 0.0), (0.38, 0.38, 0.66), "muted_teal"),
    ObjectSpec(22, "red_low_block", "block", "cube", (1.25, 1.38, 0.24), (0.0, 0.0, 0.22), (0.72, 0.45, 0.48), "brick_red"),
    ObjectSpec(23, "ochre_tall_block", "block", "cube", (-1.82, -1.22, 0.42), (0.0, 0.0, -0.18), (0.42, 0.48, 0.84), "ochre"),
    ObjectSpec(24, "navy_sphere", "sphere", "sphere", (0.70, 2.05, 0.26), (0.0, 0.0, 0.0), (0.52, 0.52, 0.52), "navy"),
    ObjectSpec(25, "sage_cube", "block", "cube", (-2.10, 0.05, 0.28), (0.0, 0.0, 0.42), (0.56, 0.56, 0.56), "sage"),
    ObjectSpec(26, "plum_sight_blocker", "block", "cube", (1.05, 0.45, 0.60), (0.0, 0.0, 0.04), (0.70, 0.30, 1.20), "plum"),
    ObjectSpec(27, "low_table_top", "table", "cube", (-0.72, -2.05, 0.45), (0.0, 0.0, -0.28), (1.02, 0.58, 0.10), "off_white"),
    ObjectSpec(28, "low_table_leg_a", "table", "cube", (-1.10, -2.23, 0.22), (0.0, 0.0, -0.28), (0.10, 0.10, 0.44), "charcoal"),
    ObjectSpec(29, "low_table_leg_b", "table", "cube", (-0.34, -1.87, 0.22), (0.0, 0.0, -0.28), (0.10, 0.10, 0.44), "charcoal"),
    ObjectSpec(30, "offwhite_wall_marker", "decor", "cube", (2.88, -1.15, 1.25), (0.0, 0.0, 0.0), (0.045, 0.55, 0.32), "off_white"),
)


ALL_OBJECTS: Sequence[ObjectSpec] = (*ROOM_OBJECTS, *STATIC_OBJECTS)


SIMPLIFIED_ROOM_OBJECTS: Sequence[ObjectSpec] = (
    ObjectSpec(1, "floor", "room", "cube", (0.0, 0.0, -0.025), (0.0, 0.0, 0.0), (6.4, 6.4, 0.05), "simple_floor_gray"),
    ObjectSpec(2, "wall_north", "room", "cube", (0.0, 3.2, 1.3), (0.0, 0.0, 0.0), (6.4, 0.08, 2.6), "simple_wall_gray"),
    ObjectSpec(3, "wall_south", "room", "cube", (0.0, -3.2, 1.3), (0.0, 0.0, 0.0), (6.4, 0.08, 2.6), "simple_wall_gray"),
    ObjectSpec(4, "wall_east", "room", "cube", (3.2, 0.0, 1.3), (0.0, 0.0, 0.0), (0.08, 6.4, 2.6), "simple_wall_gray"),
    ObjectSpec(5, "wall_west", "room", "cube", (-3.2, 0.0, 1.3), (0.0, 0.0, 0.0), (0.08, 6.4, 2.6), "simple_wall_gray"),
    ObjectSpec(6, "central_occluder_block", "occluder", "cube", (0.0, 0.0, 0.92), (0.0, 0.0, 0.0), (0.76, 2.92, 1.84), "simple_occluder_gray"),
)

SIMPLIFIED_STATIC_OBJECTS: Sequence[ObjectSpec] = (
    ObjectSpec(20, TARGET_NAME, "target", "cylinder", (1.62, -0.74, 0.42), (0.0, 0.0, 0.0), (0.50, 0.50, 0.84), "teal_target", True),
)

SIMPLIFIED_ALL_OBJECTS: Sequence[ObjectSpec] = (*SIMPLIFIED_ROOM_OBJECTS, *SIMPLIFIED_STATIC_OBJECTS)

SIMPLIFIED_V2_STATIC_OBJECTS: Sequence[ObjectSpec] = (
    ObjectSpec(20, TARGET_NAME, "target", "cylinder", (1.62, 0.74, 0.42), (0.0, 0.0, 0.0), (0.50, 0.50, 0.84), "teal_target", True),
)

SIMPLIFIED_V2_ALL_OBJECTS: Sequence[ObjectSpec] = (*SIMPLIFIED_ROOM_OBJECTS, *SIMPLIFIED_V2_STATIC_OBJECTS)


SCENE_PROFILES: Dict[str, SceneProfile] = {
    "basic": SceneProfile(
        key="basic",
        experiment_name="lingbot_memory_basic_v1",
        target_name=TARGET_NAME,
        frame=FRAME,
        camera=CAMERA,
        render=RENDER,
        materials=BASIC_MATERIALS,
        objects=ALL_OBJECTS,
        lighting="basic",
    ),
    "simplified": SceneProfile(
        key="simplified",
        experiment_name="lingbot_memory_simplified_v1",
        target_name=TARGET_NAME,
        frame=FRAME,
        camera=CAMERA,
        render=RENDER,
        materials=SIMPLIFIED_MATERIALS,
        objects=SIMPLIFIED_ALL_OBJECTS,
        lighting="simplified",
    ),
    "simplified_v2": SceneProfile(
        key="simplified_v2",
        experiment_name="lingbot_memory_simplified_v2",
        target_name=TARGET_NAME,
        frame=FRAME,
        camera=SIMPLIFIED_V2_CAMERA,
        render=RENDER,
        materials=SIMPLIFIED_MATERIALS,
        objects=SIMPLIFIED_V2_ALL_OBJECTS,
        lighting="simplified",
    ),
}

DEFAULT_SCENE_PROFILE = "basic"


def target_present_for_condition(condition: str, frame: int) -> bool:
    """Return whether the real target object exists in a condition at a frame."""
    if condition == "always_present":
        return True
    if condition == "always_absent":
        return False
    if condition == "seen_then_removed":
        return frame <= FRAME.first_loop_end
    raise ValueError(f"Unknown condition: {condition}")


def object_specs_by_name() -> Dict[str, ObjectSpec]:
    return {spec.name: spec for spec in ALL_OBJECTS}


def non_target_layout_signature() -> List[Tuple[int, str, str, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    """Small immutable signature used by sanity checks across conditions."""
    return [
        (spec.object_id, spec.name, spec.category, spec.location, spec.rotation_euler, spec.dimensions)
        for spec in ALL_OBJECTS
        if not spec.is_target
    ]
