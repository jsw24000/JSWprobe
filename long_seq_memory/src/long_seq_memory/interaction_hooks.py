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


def np_or_torch_cat(left, right):
    import torch

    return torch.cat([left, right], dim=2)
