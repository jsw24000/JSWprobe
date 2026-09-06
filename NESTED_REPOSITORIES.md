# Nested Repository Provenance

This folder was consolidated into a single top-level JSWprobe repository.
The nested `.git` directories were removed after recording their provenance here.

As of 2026-09-06, `dinov2` and `lingbot-map` have been moved to
`~/Desktop/dinov2` and `~/Desktop/lingbot-map`. Their former top-level paths
are now untracked and ignored in JSWprobe. The Desktop copies are external
dependencies and are not backed up by pushing this repository. The table below
preserves their original provenance; earlier Git commits retain the vendored files.
Model weights and generated experiment outputs remain ignored.

| Path | Original remote | Branch | Commit |
| --- | --- | --- | --- |
| `lingbot-map` | `git@github.com:Robbyant/lingbot-map.git` | `main` | `53f7ef4828f44d274d7c0c0c23d606fb9f53703a` |
| `vggt` | `https://github.com/facebookresearch/vggt.git` | `main` | `a288dd0f14786c93483e45524328726ab7b1b4ce` |
| `cross_view_consistency` | `https://github.com/jsw24000/Cross-view-consistency.git` | `main` | `7a8c70977c4564c086ef4c755d073ac5bd9071f3` |
| `dinov2` | `https://github.com/facebookresearch/dinov2.git` | `main` | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| `InfiniteVGGT` | `https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT.git` | `main` | `7f9a5a26c2e2f9603151d1dcc61fd402ef3f6a27` |
