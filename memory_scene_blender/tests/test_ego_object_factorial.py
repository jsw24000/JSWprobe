"""Lightweight, Blender-free checks for the ego/object factorial protocol."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from memory_scene_blender.ego_object_factorial.protocol import (
    build_matched_relative_groups,
    camera_from_source_extrinsics,
    linear_alphas,
    motion_conditions,
    project_opencv,
    sequence_id,
    translated_camera_payload,
)
from memory_scene_blender.object_translation.manifest_utils import write_jsonl
from memory_scene_blender.scripts.render_ego_object_x_factorial_preview import render_previews


LEVELS = (-2, -1, 0, 1, 2)
DELTA_M = 0.04


def _base_camera() -> dict:
    angle = np.deg2rad(23.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = [0.35, -0.2, 1.1]
    return {
        "camera_id": "camera_007",
        "blender_camera_to_world": camera_to_world.tolist(),
        "intrinsics": {
            "K": [[520.0, 0.0, 256.0], [0.0, 520.0, 256.0], [0.0, 0.0, 1.0]],
            "width": 512,
            "height": 512,
        },
        "look_at": [0.0, 0.0, 0.8],
    }


class EgoObjectFactorialProtocolTests(unittest.TestCase):
    def test_source_extrinsics_are_kept_but_intrinsics_are_recomputed(self) -> None:
        source = _base_camera()
        source["scene_id"] = "scene_003"
        source["intrinsics"] = {
            "K": [[443.4, 0.0, 256.0], [0.0, 443.4, 0.625], [0.0, 0.0, 1.0]],
            "width": 512,
            "height": 1,
        }
        config = {
            "fov_degrees": 60.0,
            "sensor_width_mm": 36.0,
            "clip_start": 0.05,
            "clip_end": 50.0,
            "look_at": [0.0, 0.0, 0.8],
        }
        rebuilt = camera_from_source_extrinsics(source, config, [512, 512])
        np.testing.assert_allclose(
            rebuilt["blender_camera_to_world"], source["blender_camera_to_world"]
        )
        self.assertEqual(rebuilt["intrinsics"]["width"], 512)
        self.assertEqual(rebuilt["intrinsics"]["height"], 512)
        self.assertEqual(rebuilt["intrinsics"]["principal_point"], [256.0, 256.0])
        self.assertAlmostEqual(rebuilt["intrinsics"]["K"][1][2], 256.0)
        self.assertFalse(rebuilt["source_intrinsics_reused"])

    def test_linear_alpha_has_exact_endpoints_and_uniform_steps(self) -> None:
        alpha = linear_alphas(8)
        np.testing.assert_allclose(alpha, np.arange(8, dtype=np.float64) / 7.0)
        self.assertEqual(float(alpha[0]), 0.0)
        self.assertEqual(float(alpha[-1]), 1.0)
        np.testing.assert_allclose(np.diff(alpha), np.full(7, 1.0 / 7.0))
        with self.assertRaises(ValueError):
            linear_alphas(1)

    def test_factorial_conditions_and_matched_relative_group_counts(self) -> None:
        conditions = motion_conditions(LEVELS, DELTA_M)
        self.assertEqual(len(conditions), 25)
        self.assertEqual(len({(row["ego_level"], row["object_level"]) for row in conditions}), 25)

        sequences = []
        for row in conditions:
            self.assertAlmostEqual(
                row["relative_amplitude_m"],
                row["object_amplitude_m"] - row["ego_amplitude_m"],
            )
            self.assertEqual(row["relative_level"], row["object_level"] - row["ego_level"])
            sequences.append(
                {
                    **row,
                    "scene_id": "scene_003",
                    "object_anchor_id": "anchor_011",
                    "base_camera_id": "camera_007",
                    "sequence_id": sequence_id(
                        "scene_003",
                        "anchor_011",
                        "camera_007",
                        row["ego_level"],
                        row["object_level"],
                    ),
                }
            )

        self.assertEqual(sum(bool(row["is_static"]) for row in conditions), 1)
        self.assertEqual(sum(bool(row["is_compensated_motion"]) for row in conditions), 4)

        groups = build_matched_relative_groups(sequences)
        self.assertEqual(len(groups), 9)
        counts = {row["relative_level"]: row["member_count"] for row in groups}
        self.assertEqual(counts, {-4: 1, -3: 2, -2: 3, -1: 4, 0: 5, 1: 4, 2: 3, 3: 2, 4: 1})
        self.assertEqual(sum(counts.values()), 25)
        relative_minus_one = next(row for row in groups if row["relative_level"] == -1)
        decompositions = {
            (row["ego_level"], row["object_level"])
            for row in relative_minus_one["members"]
        }
        self.assertIn((1, 0), decompositions)
        self.assertIn((0, -1), decompositions)

    def test_camera_intervention_is_pure_world_x_translation(self) -> None:
        base = _base_camera()
        base_matrix = np.asarray(base["blender_camera_to_world"], dtype=np.float64)
        moved = translated_camera_payload(base, 0.08)
        moved_matrix = np.asarray(moved["blender_camera_to_world"], dtype=np.float64)

        np.testing.assert_allclose(moved_matrix[:3, :3], base_matrix[:3, :3], atol=1.0e-8)
        np.testing.assert_allclose(moved_matrix[:3, 3] - base_matrix[:3, 3], [0.08, 0.0, 0.0])
        np.testing.assert_allclose(moved["rotation_matrix"], base_matrix[:3, :3], atol=1.0e-8)
        np.testing.assert_allclose(
            np.asarray(moved["blender_world_to_camera"]) @ moved_matrix,
            np.eye(4),
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            np.asarray(moved["opencv_world_to_camera"])
            @ np.asarray(moved["opencv_camera_to_world"]),
            np.eye(4),
            atol=2.0e-8,
        )
        self.assertFalse(moved["look_at_recomputed"])
        self.assertNotIn("look_at", moved)
        self.assertEqual(moved["initial_look_at"], base["look_at"])

    def test_matched_relative_and_compensated_projections(self) -> None:
        base = _base_camera()
        base_matrix = np.asarray(base["blender_camera_to_world"], dtype=np.float64)
        # Define stable physical points in front of the Blender camera (local -Z),
        # then express them in world coordinates.
        points_camera_blender = np.array(
            [[-0.35, 0.15, -3.2], [0.45, -0.25, -4.1], [0.1, 0.4, -5.0]],
            dtype=np.float64,
        )
        points_world = (
            base_matrix[:3, :3] @ points_camera_blender.T
        ).T + base_matrix[:3, 3]
        intrinsic = base["intrinsics"]["K"]
        static_camera = translated_camera_payload(base, 0.0)
        _xyz_static, uv_static = project_opencv(
            points_world, static_camera["opencv_world_to_camera"], intrinsic
        )

        for alpha in linear_alphas(8):
            displacement = DELTA_M * float(alpha)

            # (+1, 0) and (0, -1) have the same object-minus-camera
            # displacement and therefore the same same-point trajectory.
            pure_ego_camera = translated_camera_payload(base, displacement)
            xyz_pure_ego, uv_pure_ego = project_opencv(
                points_world, pure_ego_camera["opencv_world_to_camera"], intrinsic
            )
            pure_object_points = points_world + np.array([-displacement, 0.0, 0.0])
            xyz_pure_object, uv_pure_object = project_opencv(
                pure_object_points, static_camera["opencv_world_to_camera"], intrinsic
            )
            np.testing.assert_allclose(xyz_pure_ego, xyz_pure_object, atol=2.0e-8)
            np.testing.assert_allclose(uv_pure_ego, uv_pure_object, atol=2.0e-6)

            # Translating camera and object together preserves the target's
            # camera-relative coordinates and projection exactly.
            compensated_points = points_world + np.array([displacement, 0.0, 0.0])
            xyz_compensated, uv_compensated = project_opencv(
                compensated_points, pure_ego_camera["opencv_world_to_camera"], intrinsic
            )
            xyz_static, _unused = project_opencv(
                points_world, static_camera["opencv_world_to_camera"], intrinsic
            )
            np.testing.assert_allclose(xyz_compensated, xyz_static, atol=2.0e-8)
            np.testing.assert_allclose(uv_compensated, uv_static, atol=2.0e-6)

    def test_preview_reads_relative_frame_paths_and_writes_group_summary(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            group = "scene_003__anchor_011__camera_007"
            sequences = []
            frames = []
            displayed = {(0, 0), (1, 0), (0, -1), (1, 1)}
            for condition in motion_conditions(LEVELS, DELTA_M):
                ego_level = int(condition["ego_level"])
                object_level = int(condition["object_level"])
                seq_id = sequence_id(
                    "scene_003", "anchor_011", "camera_007", ego_level, object_level
                )
                sequences.append(
                    {
                        **condition,
                        "group_id": group,
                        "scene_id": "scene_003",
                        "object_anchor_id": "anchor_011",
                        "base_camera_id": "camera_007",
                        "sequence_id": seq_id,
                        "num_frames": 8,
                    }
                )
                if (ego_level, object_level) not in displayed:
                    continue
                for frame_index in (0, 4, 7):
                    relative_dir = Path("frames") / seq_id / f"frame_{frame_index:03d}"
                    absolute_dir = root / relative_dir
                    absolute_dir.mkdir(parents=True, exist_ok=True)
                    rgb_path = absolute_dir / "rgb.png"
                    mask_path = absolute_dir / "target_mask.png"
                    Image.new("RGB", (24, 16), (40 + frame_index, 70, 110)).save(rgb_path)
                    Image.new("L", (24, 16), 255).save(mask_path)
                    path_mapping = {
                        "rgb": (relative_dir / "rgb.png").as_posix(),
                        "target_mask": (relative_dir / "target_mask.png").as_posix(),
                    }
                    frame = {
                        "sequence_id": seq_id,
                        "frame_index": frame_index,
                        "group_id": group,
                    }
                    if frame_index == 0 and (ego_level, object_level) == (0, 0):
                        frame.update(path_mapping)  # Legacy direct-key compatibility.
                    else:
                        frame["frame_paths"] = path_mapping
                    frames.append(frame)

            write_jsonl(root / "manifests" / "sequences.jsonl", sequences)
            write_jsonl(root / "manifests" / "frames.jsonl", frames)
            summary = render_previews(root, thumb_width=48)

            self.assertEqual(summary["group_count"], 1)
            group_summary = summary["groups"][0]
            self.assertTrue(group_summary["has_all_25_conditions"])
            self.assertEqual(group_summary["selected_frame_indices"], [0, 4, 7])
            self.assertEqual(group_summary["missing_rgb_files"], [])
            self.assertEqual(group_summary["missing_target_masks"], [])
            self.assertTrue((root / group_summary["rgb_contact_sheet"]).is_file())
            self.assertTrue((root / group_summary["target_mask_overlay_contact_sheet"]).is_file())
            self.assertTrue((root / "previews" / "preview_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
