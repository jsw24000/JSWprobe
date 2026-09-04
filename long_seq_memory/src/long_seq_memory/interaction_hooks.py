from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import numpy as np


@dataclass
class QKVRecord:
    frame_id: int
    layer_id: int
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    capture_kind: str = "current_qkv"


@dataclass
class QKVHookStore:
    selected_frames: set[int] = field(default_factory=set)
    selected_layers: set[int] = field(default_factory=set)
    raw_selected_frames: set[int] | None = None
    raw_selected_layers: set[int] | None = None
    storage_dtype: str = "float16"
    aggregate_callback: Callable[[QKVRecord], None] | None = None
    current_frame_id: int | None = None
    records: list[QKVRecord] = field(default_factory=list)

    def should_capture(self, layer_id: int) -> bool:
        return (
            self.current_frame_id is not None
            and self.current_frame_id in self.selected_frames
            and (not self.selected_layers or layer_id in self.selected_layers)
        )

    def add(self, layer_id: int, q: Any, k: Any, v: Any) -> None:
        if not self.should_capture(layer_id):
            return
        record = QKVRecord(
            frame_id=int(self.current_frame_id),
            layer_id=int(layer_id),
            q=self._to_numpy(q),
            k=self._to_numpy(k),
            v=self._to_numpy(v),
        )
        if self._should_save_raw(layer_id):
            self.records.append(record)

    def add_memory(self, layer_id: int, q: Any, k: Any, v: Any) -> None:
        if not self.should_capture(layer_id):
            return
        if self.aggregate_callback is not None:
            self.aggregate_callback(
                QKVRecord(
                    frame_id=int(self.current_frame_id),
                    layer_id=int(layer_id),
                    q=q,
                    k=k,
                    v=v,
                    capture_kind="visible_memory_qkv",
                )
            )
        if self._should_save_raw(layer_id):
            self.records.append(
                QKVRecord(
                    frame_id=int(self.current_frame_id),
                    layer_id=int(layer_id),
                    q=self._to_numpy(q),
                    k=self._to_numpy(k),
                    v=self._to_numpy(v),
                    capture_kind="visible_memory_qkv",
                )
            )

    def _should_save_raw(self, layer_id: int) -> bool:
        if self.raw_selected_frames is not None and self.current_frame_id not in self.raw_selected_frames:
            return False
        if self.raw_selected_layers is not None and self.raw_selected_layers and layer_id not in self.raw_selected_layers:
            return False
        return True

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        import torch

        arr = tensor.detach().cpu()
        if str(self.storage_dtype).lower() == "float32":
            return arr.float().numpy()
        if arr.dtype == torch.bfloat16:
            arr = arr.to(dtype=torch.float16)
        elif str(self.storage_dtype).lower() in {"float16", "fp16", "half", "bfloat16"}:
            arr = arr.to(dtype=torch.float16)
        return arr.numpy()

    def save_npz(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, record in enumerate(self.records):
            np.savez_compressed(
                out_dir / f"qkv_frame{record.frame_id:06d}_layer{record.layer_id:02d}_{idx:04d}.npz",
                frame_id=record.frame_id,
                layer_id=record.layer_id,
                capture_kind=record.capture_kind,
                q=record.q,
                k=record.k,
                v=record.v,
            )


@dataclass
class RepresentationHookStore:
    output_dir: Path
    selected_frames: set[int] = field(default_factory=set)
    selected_layers: set[int] | None = None
    storage_dtype: str = "float16"
    num_special_tokens: int = 6
    current_frame_ids: list[int] = field(default_factory=list)
    saved: list[dict[str, Any]] = field(default_factory=list)

    def set_current_frame_ids(self, frame_ids: list[int] | tuple[int, ...]) -> None:
        self.current_frame_ids = [int(frame_id) for frame_id in frame_ids]

    def should_capture_layer(self, layer_id: int) -> bool:
        return self.selected_layers is None or int(layer_id) in self.selected_layers

    def should_capture_current(self) -> bool:
        return any(frame_id in self.selected_frames for frame_id in self.current_frame_ids)

    def add(self, layer_id: int, tokens: Any) -> None:
        if not self.should_capture_layer(layer_id) or not self.should_capture_current():
            return
        if not self.current_frame_ids:
            return
        import torch

        if not hasattr(tokens, "detach"):
            return
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            return
        frame_count = len(self.current_frame_ids)
        if int(tokens.shape[1]) % frame_count != 0:
            return
        tokens_per_frame = int(tokens.shape[1]) // frame_count
        arr = tokens.detach().cpu()
        if str(self.storage_dtype).lower() == "float32":
            arr = arr.float()
        elif arr.dtype == torch.bfloat16:
            arr = arr.to(dtype=torch.float16)
        elif str(self.storage_dtype).lower() in {"float16", "fp16", "half", "bfloat16"}:
            arr = arr.to(dtype=torch.float16)
        arr_np = arr.numpy()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for offset, frame_id in enumerate(self.current_frame_ids):
            if frame_id not in self.selected_frames:
                continue
            start = offset * tokens_per_frame
            end = start + tokens_per_frame
            out_path = self.output_dir / f"x_frame{frame_id:06d}_layer{int(layer_id):02d}.npz"
            np.savez_compressed(
                out_path,
                frame_id=np.asarray(frame_id, dtype=np.int64),
                layer_id=np.asarray(int(layer_id), dtype=np.int64),
                capture_kind=np.asarray("block_output"),
                num_special_tokens=np.asarray(self.num_special_tokens, dtype=np.int64),
                tokens=arr_np[0, start:end],
            )
            self.saved.append(
                {
                    "frame_id": frame_id,
                    "layer_id": int(layer_id),
                    "path": str(out_path),
                    "tokens_shape": [int(tokens_per_frame), int(arr_np.shape[-1])],
                    "num_special_tokens": int(self.num_special_tokens),
                }
            )


@dataclass
class SDPACausalInterventionConfig:
    name: str
    mode: str
    layers: set[int]
    frame_ids: set[int] | None = None
    num_special_tokens: int = 6
    include_old_memory_special_keys: bool = False
    apply_when_old_memory_present: bool = True
    current_frame_id: int | None = None

    def applies_to_current_frame(self) -> bool:
        return self.frame_ids is None or self.current_frame_id in self.frame_ids


def _parse_layer_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return set()
        return {int(item) for item in value.split(",") if item.strip()}
    return {int(item) for item in value}


def representation_layers_from_config(value: Any, num_layers: int) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    layers = _parse_layer_set(value)
    return {layer for layer in layers if 0 <= layer < num_layers}


def causal_intervention_from_config(cfg: dict[str, Any]) -> SDPACausalInterventionConfig | None:
    raw = cfg.get("causal_intervention", {}) or {}
    if not raw.get("enabled", False):
        return None
    mode = str(raw.get("mode", "")).strip()
    if mode not in {"special_to_old_memory", "image_to_old_memory", "image_to_live_special"}:
        raise ValueError(
            "causal_intervention.mode must be one of "
            "special_to_old_memory, image_to_old_memory, image_to_live_special"
        )
    layers = _parse_layer_set(raw.get("layers", []))
    if not layers:
        raise ValueError("causal_intervention.layers must be a non-empty list for causal ablations")
    frames_raw = raw.get("frames", None)
    if frames_raw is None:
        raise ValueError("causal_intervention.frames must be set to a frame list or 'all'")
    frame_ids = None if isinstance(frames_raw, str) and frames_raw.strip().lower() == "all" else _parse_layer_set(frames_raw)
    if frame_ids is not None and not frame_ids:
        raise ValueError("causal_intervention.frames must not be empty")
    return SDPACausalInterventionConfig(
        name=str(raw.get("name", mode)),
        mode=mode,
        layers=layers,
        frame_ids=frame_ids,
        num_special_tokens=int(raw.get("num_special_tokens", cfg.get("memory_policy", {}).get("patch_start_idx", 6))),
        include_old_memory_special_keys=bool(raw.get("include_old_memory_special_keys", False)),
        apply_when_old_memory_present=bool(raw.get("apply_when_old_memory_present", True)),
    )


def install_block_output_hooks(model: Any, store: RepresentationHookStore) -> list[Callable[[], None]]:
    removers: list[Callable[[], None]] = []
    blocks = getattr(getattr(model, "aggregator", None), "global_blocks", [])
    for layer_id, block in enumerate(blocks):
        if not store.should_capture_layer(layer_id):
            continue

        def hook(_module, _inputs, output, _layer=layer_id):
            store.add(_layer, output)

        handle = block.register_forward_hook(hook)
        removers.append(handle.remove)
    return removers


def install_flashinfer_attn_pre_hooks(model: Any, store: QKVHookStore) -> list[Callable[[], None]]:
    """Monkey-patch FlashInferBlock.attn_pre to capture post-norm, post-RoPE Q/K/V.

    The hook is intentionally narrow. It records the tensors returned by
    `FlashInferBlock.attn_pre`, which are already formatted as [tokens, heads, dim]
    and have RoPE applied by LingBot-Map's own code path.
    """
    removers: list[Callable[[], None]] = []
    blocks = getattr(getattr(model, "aggregator", None), "global_blocks", [])

    for layer_id, block in enumerate(blocks):
        if not hasattr(block, "attn_pre"):
            continue
        original = block.attn_pre

        def wrapped(self, x, pos=None, enable_3d_rope=False, *, _orig=original, _layer=layer_id):
            q, k, v = _orig(x, pos=pos, enable_3d_rope=enable_3d_rope)
            store.add(_layer, q, k, v)
            return q, k, v

        block.attn_pre = MethodType(wrapped, block)

        def make_remover(target, orig):
            return lambda: setattr(target, "attn_pre", orig)

        removers.append(make_remover(block, original))

    return removers


def install_flashinfer_memory_hooks(flashinfer_cache_module: Any, store: QKVHookStore) -> Callable[[], None]:
    """Patch FlashInferKVCacheManager.compute_attention to save visible K/V.

    This captures the exact visible K/V sequence used by FlashInfer for selected
    frames/layers by calling the manager's dense gather helper immediately before
    the original attention call. It is memory-heavy and should only be enabled for
    a small selected frame/layer set.
    """
    cls = flashinfer_cache_module.FlashInferKVCacheManager
    original = cls.compute_attention

    def wrapped(self, block_idx: int, q):
        if store.should_capture(block_idx):
            k_flat, v_flat = self._gather_kv(block_idx)
            store.add_memory(block_idx, q, k_flat, v_flat)
        return original(self, block_idx, q)

    cls.compute_attention = wrapped
    return lambda: setattr(cls, "compute_attention", original)


def install_sdpa_memory_hooks(attention_module: Any, store: QKVHookStore) -> Callable[[], None]:
    """Patch SDPAAttention.forward to capture current Q and visible cached K/V.

    The original forward still runs first. The hook then reconstructs the same
    post-norm, post-RoPE Q from the original input and reads the updated cache
    after append/eviction, matching the visible K/V sequence used by SDPA.
    """
    cls = attention_module.SDPAAttention
    apply_rotary_emb = attention_module.apply_rotary_emb
    original = cls.forward

    def wrapped(
        self,
        x,
        pos=None,
        enable_ulysses_cp=False,
        num_patches=None,
        num_special=None,
        num_frames=None,
        enable_3d_rope=False,
        kv_cache=None,
        global_idx=0,
        num_frame_per_block=1,
        num_frame_for_scale=-1,
        num_register_tokens=4,
    ):
        out = original(
            self,
            x,
            pos=pos,
            enable_ulysses_cp=enable_ulysses_cp,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
            kv_cache=kv_cache,
            global_idx=global_idx,
            num_frame_per_block=num_frame_per_block,
            num_frame_for_scale=num_frame_for_scale,
            num_register_tokens=num_register_tokens,
        )
        if kv_cache is not None and store.should_capture(global_idx):
            visible = getattr(self, "_long_seq_last_visible_qkv", None)
            if visible is not None:
                q, k_full, v_full = visible
                delattr(self, "_long_seq_last_visible_qkv")
            else:
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, _, _ = qkv.unbind(0)
                q = self.q_norm(q)
                if self.rope is not None and not enable_3d_rope:
                    q = self.rope(q, pos)
                elif self.rope is not None and enable_3d_rope:
                    q = apply_rotary_emb(q, pos)

                k_cached = kv_cache[f"k_{global_idx}"].clone()
                v_cached = kv_cache[f"v_{global_idx}"].clone()
                a, b, c, d, e = k_cached.shape
                k_full = k_cached.reshape(a, b, c * d, e)
                v_full = v_cached.reshape(a, b, c * d, e)
                if f"k_{global_idx}_special" in kv_cache and kv_cache[f"k_{global_idx}_special"] is not None:
                    special_k = kv_cache[f"k_{global_idx}_special"]
                    special_v = kv_cache[f"v_{global_idx}_special"]
                    sa, sb, sc, sd, se = special_k.shape
                    k_full = np_or_torch_cat(special_k.reshape(sa, sb, sc * sd, se), k_full)
                    v_full = np_or_torch_cat(special_v.reshape(sa, sb, sc * sd, se), v_full)
            store.add_memory(global_idx, q, k_full, v_full)
        return out

    cls.forward = wrapped
    return lambda: setattr(cls, "forward", original)


def install_sdpa_skip_append_patch(attention_module: Any) -> Callable[[], None]:
    """Make SDPAAttention honor `_skip_append` for non-keyframes.

    LingBot's CausalAttention path already implements this branch, but the SDPA
    aggregator path in the checked local source always appends. The patch lives
    in this experiment package so the upstream source remains read-only.
    """
    cls = attention_module.SDPAAttention
    apply_rotary_emb = attention_module.apply_rotary_emb
    torch = attention_module.torch
    F = attention_module.F
    original = cls.forward

    def wrapped(
        self,
        x,
        pos=None,
        enable_ulysses_cp=False,
        num_patches=None,
        num_special=None,
        num_frames=None,
        enable_3d_rope=False,
        kv_cache=None,
        global_idx=0,
        num_frame_per_block=1,
        num_frame_for_scale=-1,
        num_register_tokens=4,
    ):
        if kv_cache is None or not bool(kv_cache.get("_skip_append", False)):
            return original(
                self,
                x,
                pos=pos,
                enable_ulysses_cp=enable_ulysses_cp,
                num_patches=num_patches,
                num_special=num_special,
                num_frames=num_frames,
                enable_3d_rope=enable_3d_rope,
                kv_cache=kv_cache,
                global_idx=global_idx,
                num_frame_per_block=num_frame_per_block,
                num_frame_for_scale=num_frame_for_scale,
                num_register_tokens=num_register_tokens,
            )

        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and not enable_3d_rope:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        elif self.rope is not None and enable_3d_rope:
            q = apply_rotary_emb(q, pos)
            k = apply_rotary_emb(k, pos)

        key = f"k_{global_idx}"
        value = f"v_{global_idx}"
        cached_k = kv_cache.get(key)
        cached_v = kv_cache.get(value)
        current_k = k.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
        current_v = v.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
        if cached_k is None:
            k_visible = current_k
            v_visible = current_v
        else:
            k_visible = torch.cat((cached_k, current_k), dim=2)
            v_visible = torch.cat((cached_v, current_v), dim=2)

        a, b, c, d, e = k_visible.shape
        k_full = k_visible.reshape(a, b, c * d, e)
        v_full = v_visible.reshape(a, b, c * d, e)
        special_key = f"k_{global_idx}_special"
        special_value = f"v_{global_idx}_special"
        if special_key in kv_cache and kv_cache[special_key] is not None:
            special_k = kv_cache[special_key]
            special_v = kv_cache[special_value]
            sa, sb, sc, sd, se = special_k.shape
            k_full = torch.cat([special_k.reshape(sa, sb, sc * sd, se), k_full], dim=2)
            v_full = torch.cat([special_v.reshape(sa, sb, sc * sd, se), v_full], dim=2)

        self._long_seq_last_visible_qkv = (q, k_full, v_full)
        q_seq_len = q.shape[2]
        out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, q_seq_len, self.num_heads * self.head_dim)
        out = self.proj(out)
        return self.proj_drop(out)

    cls.forward = wrapped
    return lambda: setattr(cls, "forward", original)


def install_sdpa_causal_intervention_patch(
    attention_module: Any,
    intervention: SDPACausalInterventionConfig,
) -> Callable[[], None]:
    """Patch SDPAAttention.forward with a selective additive attention mask.

    The wrapper intentionally mirrors LingBot's SDPA cache update path, including
    the local `_skip_append` branch used by this experiment for keyframe-only KV
    writes. It only changes the attention call for configured layers.
    """
    cls = attention_module.SDPAAttention
    apply_rotary_emb = attention_module.apply_rotary_emb
    torch = attention_module.torch
    F = attention_module.F
    original = cls.forward

    def q_selector(q_seq_len: int, tokens_per_frame: int, device: Any) -> Any:
        local = torch.arange(q_seq_len, device=device) % tokens_per_frame
        if intervention.mode == "special_to_old_memory":
            return local < intervention.num_special_tokens
        return local >= intervention.num_special_tokens

    def kv_selector(old_token_count: int, live_token_count: int, tokens_per_frame: int, device: Any) -> Any:
        if intervention.mode in {"special_to_old_memory", "image_to_old_memory"}:
            return torch.arange(old_token_count, device=device, dtype=torch.long)
        live = torch.arange(live_token_count, device=device, dtype=torch.long)
        live_special = live[(live % tokens_per_frame) < intervention.num_special_tokens] + old_token_count
        if not intervention.include_old_memory_special_keys:
            return live_special
        old = torch.arange(old_token_count, device=device, dtype=torch.long)
        return torch.cat([old, live_special], dim=0)

    def build_attn_mask(q: Any, k_full: Any, old_token_count: int, tokens_per_frame: int, layer_id: int) -> tuple[Any | None, dict[str, Any]]:
        q_seq_len = int(q.shape[2])
        kv_seq_len = int(k_full.shape[2])
        live_token_count = max(kv_seq_len - old_token_count, 0)
        active_layer = int(layer_id) in intervention.layers
        active_frame = intervention.applies_to_current_frame()
        has_old_memory = old_token_count > 0
        stats: dict[str, Any] = {
            "intervention": intervention.name,
            "mode": intervention.mode,
            "current_frame_id": intervention.current_frame_id,
            "layer_id": int(layer_id),
            "active_layer": bool(active_layer),
            "active_frame": bool(active_frame),
            "applied": False,
            "q_seq_len": q_seq_len,
            "kv_seq_len": kv_seq_len,
            "old_memory_token_count": int(old_token_count),
            "live_token_count": int(live_token_count),
            "tokens_per_frame": int(tokens_per_frame),
            "num_special_tokens": int(intervention.num_special_tokens),
            "include_old_memory_special_keys": bool(intervention.include_old_memory_special_keys),
        }
        if not active_layer:
            stats["skip_reason"] = "layer_not_selected"
            return None, stats
        if not active_frame:
            stats["skip_reason"] = "frame_not_selected"
            return None, stats
        if intervention.apply_when_old_memory_present and not has_old_memory:
            stats["skip_reason"] = "no_old_memory"
            return None, stats
        q_mask = q_selector(q_seq_len, tokens_per_frame, q.device)
        q_indices = torch.nonzero(q_mask, as_tuple=False).flatten()
        k_indices = kv_selector(old_token_count, live_token_count, tokens_per_frame, q.device)
        stats["selected_query_tokens"] = int(q_indices.numel())
        stats["selected_key_tokens"] = int(k_indices.numel())
        stats["masked_pairs"] = int(q_indices.numel() * k_indices.numel())
        if q_indices.numel() == 0 or k_indices.numel() == 0:
            stats["skip_reason"] = "empty_query_or_key_selection"
            return None, stats
        mask = torch.zeros((q_seq_len, kv_seq_len), device=q.device, dtype=q.dtype)
        mask[q_indices[:, None], k_indices[None, :]] = float("-inf")
        stats["applied"] = True
        return mask, stats

    def wrapped(
        self,
        x,
        pos=None,
        enable_ulysses_cp=False,
        num_patches=None,
        num_special=None,
        num_frames=None,
        enable_3d_rope=False,
        kv_cache=None,
        global_idx=0,
        num_frame_per_block=1,
        num_frame_for_scale=-1,
        num_register_tokens=4,
    ):
        if kv_cache is None:
            return original(
                self,
                x,
                pos=pos,
                enable_ulysses_cp=enable_ulysses_cp,
                num_patches=num_patches,
                num_special=num_special,
                num_frames=num_frames,
                enable_3d_rope=enable_3d_rope,
                kv_cache=kv_cache,
                global_idx=global_idx,
                num_frame_per_block=num_frame_per_block,
                num_frame_for_scale=num_frame_for_scale,
                num_register_tokens=num_register_tokens,
            )

        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and not enable_3d_rope:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        elif self.rope is not None and enable_3d_rope:
            q = apply_rotary_emb(q, pos)
            k = apply_rotary_emb(k, pos)

        camera_token_idx = 0
        scale_token_idx = camera_token_idx + num_register_tokens + 1
        key = f"k_{global_idx}"
        value = f"v_{global_idx}"
        skip_append = bool(kv_cache.get("_skip_append", False))
        cached_k = kv_cache.get(key)
        cached_v = kv_cache.get(value)

        if skip_append:
            current_k = k.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
            current_v = v.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
            if cached_k is None:
                k_visible = current_k
                v_visible = current_v
            else:
                k_visible = torch.cat((cached_k, current_k), dim=2)
                v_visible = torch.cat((cached_v, current_v), dim=2)
        else:
            if cached_k is None:
                kv_cache[key] = k.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
                kv_cache[value] = v.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
            else:
                cached_tokens_per_frame = int(cached_k.shape[3])
                num_frame_per_block = int(k.shape[2]) // cached_tokens_per_frame
                kv_cache[key] = torch.cat(
                    (
                        cached_k,
                        k.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim),
                    ),
                    dim=2,
                )
                kv_cache[value] = torch.cat(
                    (
                        cached_v,
                        v.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim),
                    ),
                    dim=2,
                )
            self._apply_kv_cache_eviction(
                kv_cache, global_idx, camera_token_idx, scale_token_idx, num_register_tokens
            )
            k_visible = kv_cache[key].clone()
            v_visible = kv_cache[value].clone()

        a, b, c, d, e = k_visible.shape
        tokens_per_frame = int(d)
        k_full = k_visible.reshape(a, b, c * d, e)
        v_full = v_visible.reshape(a, b, c * d, e)
        old_token_count = 0
        special_key = f"k_{global_idx}_special"
        special_value = f"v_{global_idx}_special"
        if special_key in kv_cache and kv_cache[special_key] is not None:
            special_k = kv_cache[special_key]
            special_v = kv_cache[special_value]
            sa, sb, sc, sd, se = special_k.shape
            old_token_count = int(sc * sd)
            k_full = torch.cat([special_k.reshape(sa, sb, sc * sd, se), k_full], dim=2)
            v_full = torch.cat([special_v.reshape(sa, sb, sc * sd, se), v_full], dim=2)

        attn_mask, stats = build_attn_mask(q, k_full, old_token_count, tokens_per_frame, int(global_idx))
        stats["skip_append"] = bool(skip_append)
        self._long_seq_last_intervention_stats = stats
        self._long_seq_last_visible_qkv = (q, k_full, v_full)
        q_seq_len = q.shape[2]
        out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, q_seq_len, self.num_heads * self.head_dim)
        out = self.proj(out)
        return self.proj_drop(out)

    cls.forward = wrapped
    return lambda: setattr(cls, "forward", original)


def np_or_torch_cat(left, right):
    import torch

    return torch.cat([left, right], dim=2)
