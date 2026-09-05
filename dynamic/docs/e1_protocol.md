# E1 analysis protocol

For group g and representation/layer/regime, use
`dz(e,o)=z(I7;e,o)-z(I7;0,0)` at matched physical point IDs.
All reported motion levels denote multiples of δ=0.04 m.
Never use within-sequence `I7-I0` as the primary feature delta.

1. `ve1=(dz(1,0)-dz(-1,0))/(2δ)` and
   `ve2=(dz(2,0)-dz(-2,0))/(4δ)`, likewise vo.
   Report cosine inner/outer (`TC_e`, `TC_o`) and norm ratios
   `scale_e`, `scale_o`. Low consistency means a local tangent interpretation
   is weak; changes can reflect nonlinearity or finite-difference curvature.
2. Inner tangents: signed cos(ve,vo), its absolute value, and
   `S_e=||ve||`, `S_o=||vo||` per metre. Antiparallel is still the same axis.
   Falling ego sensitivity with retained object response is distinct from
   orthogonalization. Sensitivity scales are architecture/layer-dependent;
   cross-model raw norms alone do not establish stronger information content.
3. All same-r pairs with nonzero r and multiple decompositions:
   `D=||dzA-dzB||/(0.5*(||dzA||+||dzB||)+1e-8)` and cosine(dzA,dzB).
   There are 20 unordered pairs/group: r=±1 contributes 6 each, ±2 contributes
   3 each, ±3 contributes 1 each. r=±4 has no twin and is excluded.
   Pairwise metrics are averaged **within each physical point**, then median
   across points within group. Pair-level group distributions and both delta
   norms are saved separately, so large ratios with weak response can be audited.
   D is in [0,2] up to epsilon/roundoff; it is not a causal classifier score.
4. r=0 is separate: e∈{-2,-1,1,2}. Save `C_resp_e=||dz(e,e)||` and
   `C_norm_e=C_resp_e/(0.5*(||dz(e,0)||+||dz(0,e)||)+eps)`.
   The compact overview also takes their mean within point. A response despite
   matched target trajectory is evidence of context sensitivity, not proof of
   world-centric physical understanding.

For each metric first calculate per point, then group median/Q25/Q75 and mean.
Cross-group curves summarize group medians; each group appears individually.
Registers are separate units 0…15: metrics per register, then their median per
group. Heatmaps show each register's median across groups. Camera and pool have
one entity vector per condition. Dense auxiliary reports only D/cosine,
compensated response and sensitivities. Zero tangent norms give undefined
cosines rather than invented zero values, with counts in analysis provenance.

No p-values, no treating points as independent observations. Four groups are
nested in two scenes and inherit train splits; neither generalization nor
population-level claims follow. Same-r removes target projection differences,
but global appearance, background, shadows and lighting can still distinguish
conditions. DINO Single measures how much is present without explicit
multi-frame aggregation. Prefer Pair vs Full for context comparisons because
I7 uses the same non-first token role.

Pre/post routing is observational. Even register-first separation followed by
patch change is only a candidate for future activation-swap experiments.
Selected layers are sparse; this pilot cannot establish the exact layer where
information first appeared. Dense spatial-resolution agreement is auxiliary,
not a guarantee that differences are free of resolution or receptive-field effects.

## Artifact gates

Audit: local inputs, full factorial, fixed point identity, current file hashes,
all frames present, reprojection consistency, same-r final UV error, source
validation pass, both checkpoint hashes and source versions, coordinate tests.
Smoke: one group and 10 requested conditions, 40 shards, all required keys and
shapes, finite values, exact point IDs, correct frame indices, register-only
patch equality, DenseHead and DINO runtime shape, then a saved PASS gate.
Full: all 100×4 shards, same validations and hashes, then metrics and figures.
Use sequence-level atomic NPZ writes and exact compatibility gates for reuse.
Do not modify sources or lower frames/resolution to avoid a failed gate.
