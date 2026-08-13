# Lingbot-map Memory Token Trace

This document records the source-level trace used by `memory_objectness_probe`.
No Lingbot-map source files were modified.

## Local Source And Checkpoint

- Source tree used by the runner: `../lingbot-map` relative to `memory_objectness_probe/`.
- Checkpoint used by default: `../lingbot-map/checkpoints/lingbot-map.pt`.
- The conda environment on this machine is named `lingbot-map`, not `lingbot`.
- The environment's default installed `lingbot_map` points elsewhere, so
  `LingbotRunner` inserts the CLI `--lingbot_root` at the front of `sys.path`
  and checks `lingbot_map.__file__` before loading the model.

## Relevant Source Files

- `lingbot_map/models/gct_stream.py`
  - class: `GCTStream`
  - functions: `_aggregate_features`, `inference_streaming`, `clean_kv_cache`
- `lingbot_map/models/gct_base.py`
  - class: `GCTBase`
  - function: `forward`
- `lingbot_map/aggregator/base.py`
  - class: `AggregatorBase`
  - functions: `_embed_images`, `_get_positions`, `forward`
- `lingbot_map/aggregator/stream.py`
  - class: `AggregatorStream`
  - functions: `_setup_special_tokens`, `_prepare_special_tokens`,
    `_process_causal_stream`
- `lingbot_map/layers/attention.py`
  - classes: `FlashInferAttention`, `SDPAAttention`
  - functions: `append_frame` call path, KV eviction path
- `lingbot_map/layers/flashinfer_cache.py`
  - class: `FlashInferKVCacheManager`
  - functions: `append_frame`, `evict_frames`

## Six Compact Tokens

The streaming aggregator defines six special tokens per frame:

1. `camera`
2. `register0`
3. `register1`
4. `register2`
5. `register3`
6. `scale`

This order is confirmed by `AggregatorStream._setup_special_tokens` and
`AggregatorStream._prepare_special_tokens`. The code constructs:

```text
special_tokens = torch.cat([camera_token, register_token, scale_token], dim=1)
patch_start_idx = 1 + num_register_tokens + 1
num_special_tokens = 1 + num_register_tokens + 1
```

With the default `num_register_tokens=4`, `patch_start_idx == 6`. Patch tokens
begin after these six tokens.

## Tensor Shapes

`GCTBase.forward` calls `_aggregate_features(images, ...)`, then prediction
heads consume the returned token list.

`GCTStream._aggregate_features` calls:

```text
self.aggregator(..., selected_idx=[4, 11, 17, 23], ...)
```

`AggregatorBase.forward` returns:

```text
output_list, patch_start_idx
```

Each selected item in `output_list` has shape:

```text
[B, S, P, 2C]
```

For the default checkpoint/model:

- `B = 1` during extraction.
- `S = 8` for the initial scale-frame batch, then `S = 1` for streaming frames.
- `P = 6 + patch_count`.
- `C = 1024`, so the selected representation dimension is `D = 2C = 2048`.
- The compact memory probe tensor is therefore:

```text
output[0, local_frame_index, :6, :] -> [6, 2048]
```

Saved feature keys are normalized as:

```text
block04, block11, block17, block23
```

## Frame ID Mapping

`AggregatorStream` maintains `total_frames_processed`.

- Before the first scale-frame call, `total_frames_processed == 0`.
- `inference_streaming` first processes `num_scale_frames` frames together.
  With the default `num_scale_frames=8`, those records correspond to frames
  `0..7`.
- Remaining frames are processed one-by-one. For frame 15, the wrapper sees
  `before_total_frames_processed == 15` and records the single output as frame
  ID `15`.

The KV cache itself stores K/V tensors by append order rather than an explicit
public frame-id field. `FlashInferKVCacheManager.append_frame` and the SDPA
cache append per-block K/V, and eviction can move old special-token K/V into a
special stream. For this probe we do not read the KV cache as the feature
source; we hook the model output from `_aggregate_features`, where the frame ID
can be recovered from `total_frames_processed` and the current local sequence
length.

## Hook Position

The final hook/wrapper position is:

```text
GCTStream._aggregate_features return value
```

The wrapper records the selected block outputs immediately after each streaming
forward call and stores the first six token rows. This is experiment-side
monkey patching only; no permanent model logic change is required.

## Flush Frames

No flush frames are required for the default experiment. The frame-15 six-token
representation is available while processing original frame 15. The extractor
still records `flush_frames_used`, and a CLI `--flush_frames` option exists for
future variants, but the default is `0`.

If flush frames are ever used, the extractor appends repeated final RGB inputs
only to drive streaming state forward. The saved feature still uses the explicit
record for original `probe_frame`, not the flush frame record.
