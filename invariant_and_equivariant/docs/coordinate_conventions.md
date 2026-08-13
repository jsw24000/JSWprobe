# Coordinate Conventions

The Blender dataset uses meters.

- Blender world coordinates: `+Z` is up.
- Target objects move only in the Blender world ground plane, but labels keep all three coordinates.
- Blender camera coordinates: local `-Z` is forward and local `+Y` is up.
- OpenCV camera coordinates: `+X` right, `+Y` down, `+Z` forward.

The dataset stores:

```text
blender_camera_to_opencv = diag(1, -1, -1, 1)
opencv_world_to_camera = blender_camera_to_opencv @ blender_world_to_camera
```

`camera_000` is the reference camera. Main absolute-position probes use `object_center_ref_camera`.

Important interpretation boundary: the object is physically translated in a 2D ground plane. The label is 3D in the reference camera, but the effective motion rank is expected to be at most two unless visibility/projection effects add apparent higher-dimensional variation.

