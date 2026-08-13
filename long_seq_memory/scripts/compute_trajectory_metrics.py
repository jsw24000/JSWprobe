#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import write_json
from long_seq_memory.trajectory_metrics import rotation_angle_deg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    gt = np.load(run_dir / "gt_poses_c2w.npy")
    pred = np.load(run_dir / "pred_poses_c2w_demo_convention.npy")
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.shape[-2:] == (3, 4):
        pred4 = np.repeat(np.eye(4)[None], pred.shape[0], axis=0)
        pred4[:, :3, :4] = pred
        pred = pred4
    trans_err = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    rot_err = np.array([rotation_angle_deg(pred[i, :3, :3], gt[i, :3, :3]) for i in range(len(gt))])
    np.save(run_dir / "translation_error_m.npy", trans_err)
    np.save(run_dir / "rotation_error_deg.npy", rot_err)
    write_json(
        run_dir / "trajectory_metrics.json",
        {
            "note": "Raw per-frame GT-vs-pred metrics without Sim(3) alignment.",
            "translation_error_m": {
                "mean": float(trans_err.mean()),
                "median": float(np.median(trans_err)),
                "max": float(trans_err.max()),
            },
            "rotation_error_deg": {
                "mean": float(rot_err.mean()),
                "median": float(np.median(rot_err)),
                "max": float(rot_err.max()),
            },
        },
    )
    print(f"Wrote {run_dir / 'trajectory_metrics.json'}")


if __name__ == "__main__":
    main()
