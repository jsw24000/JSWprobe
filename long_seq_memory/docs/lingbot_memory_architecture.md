# LingBot-Map Memory Architecture Audit

Audit date: 2026-07-31  
Source used: `/home/3dsm/Desktop/JSWprobe/lingbot-map`  
Requested path note: `Lingbot-map` does not exist in this workspace; the actual source directory is `lingbot-map`.

This document is based on the current local source, not on the paper alone.

## Entrypoints

- Benchmark wrapper: `benchmark/methods/lingbot_map.py`
  - Builds `lingbot_map.models.gct_stream.GCTStream` for streaming mode.
  - Builds `lingbot_map.models.gct_stream_window.GCTStream` for windowed mode.
  - Loads checkpoint with `torch.load(...); ckpt.get("model", ckpt)`.
  - Default method config is `benchmark/configs/methods/lingbot_map.yaml`.
- Demo runner: `demo.py`
  - Uses the official image preprocessing function.
  - Supports `streaming` and `windowed`.
  - Exposes `--use_sdpa`, `--keyframe_interval`, `--kv_cache_sliding_window`, `--num_scale_frames`, and `--max_frame_num`.
- Core streaming model: `lingbot_map/models/gct_stream.py`
  - `GCTStream.inference_streaming()` runs scale frames first, then one frame at a time.
  - `GCTStream.clean_kv_cache()` resets aggregator and camera-head caches.

## Image Preprocessing

Official preprocessing is `lingbot_map/utils/load_fn.py::load_and_preprocess_images`.

- Image list is sorted lexicographically by path in `demo.py`.
- Images are converted to RGB and `torchvision.transforms.ToTensor()` gives `[0, 1]`.
- Canonical demo mode is `crop`.
- In crop mode:
  - width is resized to `image_size` (default 518);
  - height is resized with aspect ratio and rounded to a multiple of `patch_size` (default 14);
  - if resized height exceeds `image_size`, it is center-cropped to `image_size`.
- Aggregator then normalizes with ImageNet/ResNet mean and std in `AggregatorBase._embed_images()`.
- For Oxford frames with native 1440x1080 and default 518/14, the expected processed size is 518x392, giving 37x28 = 1036 image patch tokens per frame.

## Model Call Chain

For streaming mode:

1. `scripts/run_reconstruction.py` or benchmark wrapper loads `GCTStream`.
2. `GCTStream.inference_streaming(images, num_scale_frames, keyframe_interval, output_device)`.
3. `GCTStream.forward(...)` from `GCTBase.forward`.
4. `GCTStream._aggregate_features(...)`.
5. `AggregatorStream.forward(...)`.
6. Alternating per-frame and global blocks:
   - frame blocks: `Block` with ordinary per-frame attention;
   - global blocks: `FlashInferBlock` by default, or `SDPABlock` if `use_sdpa=True`.
7. Prediction heads:
   - `CameraCausalHead` predicts camera pose encoding from camera tokens;
   - DPT heads predict depth and world points if dense outputs are enabled.

The aggregator returns selected block outputs at block group indices `[4, 11, 17, 23]`, each as concatenated frame/global features `[B, S, P, 2C]`.

## Token Layout

Current `AggregatorStream._setup_special_tokens()` defines per-frame special tokens:

| Local index | Type |
| --- | --- |
| 0 | camera |
| 1..4 | register tokens |
| 5 | scale |
| 6.. | image patch tokens |

Therefore:

- `num_register_tokens = 4`
- `num_special_tokens = 6`
- `patch_start_idx = 6`
- `tokens_per_frame = 6 + num_patches`
- For the Oxford processed shape above, `tokens_per_frame = 1042`.

The FlashInfer cache comments and `append_frame()` confirm the same order:
`[camera, reg0, ..., regN, scale, patch0, ...]`.

## Trajectory Memory Layout

There are two relevant memory systems.

### Aggregator Global Attention

Default backend is FlashInfer (`use_sdpa=False`).

Local runtime note: on this machine, FlashInfer JIT fails on the Blackwell GPU
with `SM 12.x requires CUDA >= 12.9`. The completed smoke and focused-loop runs
therefore use LingBot-Map's SDPA fallback (`use_sdpa=True`). The architecture
below still records the default FlashInfer design because it is the upstream
default and explains the intended cache layout.

`FlashInferKVCacheManager` uses two streams per layer:

- Patch stream:
  - one recyclable patch page per frame;
  - first `scale_frames` patch pages are retained;
  - latest `kv_cache_sliding_window` patch pages are retained;
  - older patch pages are evicted/recycled.
- Special stream:
  - append-only;
  - stores the 6 special tokens for every appended frame;
  - never recycled during a sequence;
  - visible page order is `scale_patch_pages + live_window_patch_pages + all_special_pages`.

For long-loop analysis, this means frame 3403 will not retain image patch tokens when frame 4414 is processed unless it is inside the live patch window. It can still be present through its special tokens: camera, register tokens, and scale token.

The SDPA path stores dense dict caches (`k_i`, `v_i`, `k_i_special`, `v_i_special`) and evicts old full-frame tokens into special-token caches. It is easier to inspect, but still uses fused `torch.nn.functional.scaled_dot_product_attention` and does not return attention weights.

### Camera Causal Head

`CameraCausalHead` extracts `tokens[:, :, 0]`, so it uses only aggregator camera tokens. It has a separate causal attention stack and separate KV cache list, one dict per pose-refinement iteration. This is camera-token-only memory, not image patch memory.

## Local Window and Old Memory Boundary

Defaults in this experiment:

- `num_scale_frames = 8`
- `kv_cache_sliding_window = 64`
- `kv_cache_cross_frame_special = True`
- `kv_cache_include_scale_frames = True`

For aggregator FlashInfer memory:

- patch-visible frames are first 8 scale frames plus the latest 64 frames;
- old trajectory memory is the append-only special stream after excluding current/local/scale frames for analysis purposes;
- old memory token count per old source frame is 6 unless `kv_cache_camera_only` is enabled, which it is not in the benchmark/demo construction.

The analysis must not mix live local patch pages with old special trajectory memory.

## Keyframe, VO, Chunk, and Reset Behavior

`GCTStream.inference_streaming()` documents keyframe mode. It calls `_set_skip_append(True)` for non-keyframes, then resets it after the frame.

Important static finding: in the current aggregator code path, this skip flag does not appear to prevent aggregator global-cache append:

- `_set_skip_append()` writes `self.aggregator.kv_cache["_skip_append"]`.
- FlashInfer global blocks receive `kv_cache=manager`, not the dict.
- `FlashInferBlock.forward()` calls `manager.append_frame(...)` unconditionally.
- `SDPAAttention.forward()` also appends into the dict without checking `_skip_append`.

The skip flag is honored by `CausalAttention`, which is used by the camera head. Therefore, until runtime hooks prove otherwise, treat `keyframe_interval` as not reducing aggregator trajectory-memory writes. Use `keyframe_interval=1` for clean focused analysis.

Windowed mode in `gct_stream_window.py` has chunk/window alignment and optional flow-based keyframe logic. The focused loop experiment should start with plain streaming mode, not windowed mode, so frame 3403 and frame 4414 are not separated by window reset/alignment.

## Temporal RoPE

Aggregator 3D RoPE is created in `AggregatorStream._init_3d_rope()` with `max_frame_num`.

During streaming, `_process_causal_stream()` uses:

- `f_start = self.total_frames_processed`
- `f_end = self.total_frames_processed + S_global`

The RoPE table is sliced with `[f_start, f_end)`. For focused loop and full 5968-frame runs, `max_frame_num=1024` is unsafe. The experiment configs set `max_frame_num=7000`.

Camera-head 3D RoPE is disabled by default in the current constructor path (`enable_camera_3d_rope=False` unless explicitly changed).

## Attention Modules and Weights

Global aggregator attention:

- FlashInfer path:
  - `FlashInferBlock.attn_pre()` computes normed, RoPE-applied Q/K/V for the current frame.
  - `FlashInferKVCacheManager.append_frame()` writes K/V into patch and special streams.
  - `FlashInferKVCacheManager.compute_attention()` runs FlashInfer paged attention.
  - It does not return attention weights.
- SDPA path:
  - `SDPAAttention.forward()` builds Q/K/V, applies RoPE, updates cache, and calls `F.scaled_dot_product_attention`.
  - It also does not return attention weights.

Camera-head attention:

- `CameraBlock` calls `CausalAttention`.
- This path has explicit mask construction and a `_skip_append` branch.
- It still uses `F.scaled_dot_product_attention` when fused attention is enabled.

Direct attention weights are therefore not safely available from normal forward. Recommended extraction is:

1. Hook post-norm, post-RoPE Q from the current frame.
2. Hook the exact visible K/V sequence from the cache manager or SDPA cache.
3. Build a source-frame memory index from the cache layout.
4. Recompute only selected `current query -> old trajectory memory key` logits and softmax offline.

`long_seq_memory/src/long_seq_memory/interaction_hooks.py` implements selected-frame hooks:

- current Q/K/V via `FlashInferBlock.attn_pre`;
- visible memory K/V via `FlashInferKVCacheManager.compute_attention`.
- SDPA visible memory Q/K/V via a wrapper around `SDPAAttention.forward`.

Because selected event-frame K/V can be hundreds of MB per layer, configs must keep `selected_frames` and `selected_layers` small.

## Source-Frame Tracing

The current LingBot-Map cache does not store source frame IDs next to each token. Source tracing must be maintained externally.

For FlashInfer aggregator:

- patch page source frames are determined by append order and the manager's `scale_patch_pages` / `live_window_patch_pages`;
- special stream source frame is determined by append order and `num_special_tokens=6`;
- special token local type is local index modulo 6.

`memory_index.py` encodes this mapping. For focused run `start_frame=3200`, temporal position 0 corresponds to absolute frame 3200, so frame 3403 has temporal position 203.

## Recommended Hook Points

Safe first hook points:

- `FlashInferBlock.attn_pre` for current-frame Q/K/V after normalization and RoPE.
- `FlashInferKVCacheManager.compute_attention` for visible memory K/V before attention.
- `FlashInferKVCacheManager.build_visible_page_table` for page order diagnostics.
- `CameraCausalHead.trunk_fn` if camera-query-only memory analysis is needed.

Avoid modifying normal model outputs. Use wrappers and monkey patches in `long_seq_memory`.

## Safe Focused Loop Mode

Use:

- streaming mode;
- `start_frame=3200`, `end_frame=4600`, `stride=1`;
- `keyframe_interval=1`;
- `num_scale_frames=8`;
- `kv_cache_sliding_window=64`;
- `max_frame_num=7000`;
- selected layers only, initially `[23]` or `[17, 23]`;
- selected current frames around 4414 only for Q/K/V capture.

Before interpreting retrieval scores, verify:

- frame 3403 is present in old special memory when frame 4414 runs;
- frame 3403 is not being counted as a live patch-window frame;
- no `clean_kv_cache()` or window reset occurred between 3403 and 4414;
- post-RoPE Q/K are used for offline attention;
- visible memory source mapping agrees with `memory_state_summary.jsonl`.
