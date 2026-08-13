# Oxford Spires minimal data container

This folder is reserved for a single Oxford Spires sequence, currently:

- Sequence: `2024-03-12-keble-college-02`
- Short scene name expected by Lingbot benchmark output: `keble-college-02`
- Camera: `cam0` only

## Relevant Lingbot files

Local repository files found:

- `lingbot-map/benchmark/datasets/oxford_spires.py`
- `lingbot-map/benchmark/configs/datasets/oxford.yaml`
- `lingbot-map/benchmark/configs/datasets/oxford_long.yaml`
- `lingbot-map/benchmark/configs/oxford.yaml`
- `lingbot-map/benchmark/configs/oxford_long.yaml`
- `lingbot-map/benchmark/evaluate.py`

The current local checkout does not contain `lingbot-map/preprocess/oxford.py`.
The upstream Lingbot script exists at:

- `https://github.com/Robbyant/lingbot-map/blob/main/preprocess/oxford.py`

That script expects a dataset root containing:

```text
calibration/
  cam0.yaml
  cam-lidar-imu.yaml
ground_truth_map/
  keble-college/
    merged-cloud-1cm.pcd
sequences/
  2024-03-12-keble-college-02/
    raw/
      images.zip
    processed/
      trajectory/
        gt-tum.txt
      vilens-slam/
        slam-poses.csv
        undist-clouds.zip
```

It reads only `cam0/*.jpg` from `raw/images.zip`, matches image timestamps to
`processed/trajectory/gt-tum.txt` within `0.1 s`, and writes:

```text
lingbot_ready/
  keble-college-02/
    images/
      000000.png
      ...
    depth/
      000000.npy
      ...
    poses_c2w.txt
    intrinsics.txt
    ground_truth.ply
```

`--images_only` skips TLS depth and `ground_truth.ply`, producing only RGB,
poses, and intrinsics.

## Folder layout here

```text
downloads/
  calibration/
  ground_truth_map/
  sequences/
raw_one_sequence/
  2024-03-12-keble-college-02/
    cam0/
    undist-clouds/
    timestamps_sync/
lingbot_ready/
  keble-college-02/
    images/
```

Use `downloads/` as the HuggingFace target directory.
Use `raw_one_sequence/` only if you manually extract `cam0` images or LiDAR PCDs.
Use `lingbot_ready/` as the output root for Lingbot's Oxford preprocessing.

## Minimal download set

The default pattern file is `dataset_download_one_sequence.yaml`.
It intentionally avoids `sequences/*` and only lists:

- Single-sequence RGB archive: `raw/images.zip`
- `cam0` calibration and camera-LiDAR calibration
- GT trajectory: `processed/trajectory/gt-tum.txt`
- LiDAR SLAM poses and undistorted LiDAR clouds for sync/depth generation
- TLS map for visibility-filtered depth or overlap workflows

The full Oxford Spires dataset is about 1.3 TB, so keep downloads pattern-based.

