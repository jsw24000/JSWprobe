# E2 V2 representation-geometry protocol

E2 reads the validated V2 dataset at `memory_scene_blender/outputs/ego_object_factorial_v2/full` and writes only to `dynamic/outputs/e2_full_v2`. It reuses the four E1 adapters and their exact feature definitions. It never writes the source dataset, E1 outputs, model repositories, or checkpoints.

The four panels are:

- `core_x_confirmation`: all 24 physical contexts, `tx_d004`, 600 sequences. It retains E1's homogeneous-family metrics.
- `xy_d004`: the manifest-discovered eight extended contexts, paired `tx_d004` and `ty_d004`, 400 sequences.
- `representation_specialization`: the same sequences as `xy_d004`, with patch primary and VGGT/Omega pool, camera, individual-register, and dense features as auxiliaries.
- `scale_locality`: the eight extended contexts and all six X/Y scale families, 1200 sequences.

The dataset audit constructs `core_tx004`, `core_xy004`, and `core_scale6` at `physical_context_id` level. It first requires bitwise-identical canonical `point_id` and `xyz_object_local` arrays across relevant families, then intersects endpoint validity across all 25 conditions per family. It fails below `min_core_points`; there is no fallback. Each feature shard stores the union of every panel core that can use that sequence. Analysis selects the exact panel core from the stored `point_ids`, so Stage 2 and Stage 3 reuse Stage 1 shards without comparing different physical points.

The configured E2 minimum is 8 physical points. This is an explicit V2 feasibility threshold rather than a silent relaxation: the repaired thin `bg_003__side_table_v00` context has 11 points for `core_tx004`, 11 for `core_xy004`, and 9 for `core_scale6` under the inherited 8-pixel boundary rule and 192 canonical tracks. Every other context has at least 24, 27, and 25 respectively. The audit records every count and fails if any panel core falls below 8.

For each family, E2 calls E1 `compute_metrics()` and retains `TC_e`, `TC_o`, `scale_e`, `scale_o`, `S_e`, `S_o`, signed/absolute cosine, `D_cause`, `cause_cos`, `C_resp`, `C_norm`, and per-level compensated responses. `D_cause_legacy` is the identical mean over the 20 nonzero same-r pairs. E2 additionally reports `v_r=(v_o-v_e)/2`, `v_c=v_e+v_o`, their norms, `M_r`, `M_c`, and an absolute log ego/object magnitude ratio.

For paired X/Y families, `J_e=[v_e^x,v_e^y]` and `J_o=[v_o^x,v_o^y]`. Exact singular values, continuous `rho=sigma2/(sigma1+eps)`, numerical ranks, available principal angles, overlap, and conditioning flags are saved. Two-angle interpretation is allowed only when both matrices are rank 2 and meet `well_conditioned_rho`. Continuous ratios carry more scientific weight than the configured binary rank threshold.

`D_c_given_r` preserves every one of the 20 nonzero-r condition pairs and its `r`, common coordinates, `delta_c`, causes, cosine and response norms. It measures representation variation along the common/world coordinate while relative target-camera translation is fixed. `C_norm` measures compensated response at r=0. These are related controls, not the same metric. Matched r does not imply identical RGB because camera motion changes the background/reference frame.

Scale analysis compares centered derivative estimates across 0.02/0.04/0.06 m separately for X/Y and ego/object. The equivalent-trajectory audit checks transforms, canonical projected tracks, metadata-derived physical amplitudes, and RGB hashes when different family/level combinations realize the same trajectory. Finite differences are called tangent-like only when these locality controls support that wording.

The primary descriptive unit is `physical_context_id`; point/register distributions are diagnostics. X/Y results are paired on the same eight contexts. No point-level significance tests or p-values are produced. The target/background design is partially crossed, so the 24 contexts are not described as fully independent scenes. Raw norms and block indices are not treated as calibrated or homologous across architectures.

Metric outputs are written under `metrics/<panel>/`: `family_metrics.csv`, `local_geometry.csv`, `matched_common_mode_pairs.csv`, `matched_common_mode_aggregates.csv`, `xy_subspace_metrics.csv`, `xy_subspace_context_metrics.csv`, `scale_consistency_metrics.csv`, `scale_consistency_context_metrics.csv`, `projection_diagnostics.csv`, `projection_context_summary.csv`, `input_rgb_controls.csv`, and `equivalent_trajectory_audit.csv`. Unit-level tables are diagnostics; the corresponding context tables drive primary plots.

Planning performs no model forward:

```bash
python dynamic/scripts/plan_e2.py --config dynamic/configs/e2_full_v2.yaml --panel all
```

Later manual execution:

```bash
# Panel A bounded smoke
bash dynamic/scripts/run_e2_pipeline.sh --config dynamic/configs/e2_full_v2.yaml --panel core_x_confirmation --stage smoke

# Panel A full extraction and analysis
bash dynamic/scripts/run_e2_pipeline.sh --config dynamic/configs/e2_full_v2.yaml --panel core_x_confirmation --stage full

# Panels B/C. The shared cache reuses the eight tx_d004 context shards from A.
bash dynamic/scripts/run_e2_pipeline.sh --config dynamic/configs/e2_full_v2.yaml --panel xy_d004 --stage full
bash dynamic/scripts/run_e2_pipeline.sh --config dynamic/configs/e2_full_v2.yaml --panel representation_specialization --stage full

# Panel D; existing d004 shards are reused.
bash dynamic/scripts/run_e2_pipeline.sh --config dynamic/configs/e2_full_v2.yaml --panel scale_locality --stage full

# Final combined report after all analyses exist
/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python dynamic/scripts/build_e2_report.py \
  --config dynamic/configs/e2_full_v2.yaml \
  --panels core_x_confirmation xy_d004 representation_specialization scale_locality
```

Do not run multiple writers against the same E2 output root. Extraction takes a nonblocking OS advisory lock at `features/.writer.lock`, so a second writer fails before touching shards; the lock is released automatically on process exit. A shard is written atomically and reused only when its config, checkpoint, model-source, RGB, preprocessing, sequence identity, and exact point IDs match. Every full pipeline invocation first extracts and validates its bounded smoke set.
