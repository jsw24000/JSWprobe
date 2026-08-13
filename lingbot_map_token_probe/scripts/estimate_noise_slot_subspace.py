#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from noise_slot_utils import append_run_log, orthonormalize_basis, resolve_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate noise-slot SVD positional subspaces.")
    parser.add_argument("--config", default="configs/noise_slot_svd_debias.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    output_root = Path(cfg["output_root"])
    append_run_log(cfg, "Starting estimate_noise_slot_subspace.py")

    for stage in cfg["stages_resolved"]:
        token_path = output_root / "02_noise_slot_tokens" / stage / f"noise_slot_tokens_{stage}.pt"
        if not token_path.is_file():
            append_run_log(cfg, f"Warning: missing noise-slot tokens for {stage}: {token_path}")
            continue
        payload = torch.load(token_path, map_location="cpu", weights_only=False)
        z = payload["tokens"].float()
        if z.ndim != 4:
            append_run_log(cfg, f"Warning: skipping {stage}; unsupported shape {tuple(z.shape)}")
            continue

        append_run_log(cfg, f"Estimating SVD for {stage}: noise-slot tokens={tuple(z.shape)}")
        if cfg.get("normalize_before_svd", "l2") == "l2":
            z_norm = F.normalize(z, dim=-1, eps=1e-8)
        elif cfg.get("normalize_before_svd") in (None, "none"):
            z_norm = z
        else:
            raise ValueError(f"Unsupported normalize_before_svd: {cfg.get('normalize_before_svd')}")

        p = z_norm.mean(dim=0)  # [H,W,C]
        h, w, c = p.shape
        p_flat = p.reshape(h * w, c)
        if cfg.get("center_before_svd", True):
            mean = p_flat.mean(dim=0)
            x = p_flat - mean
        else:
            mean = torch.zeros(c, dtype=p_flat.dtype)
            x = p_flat

        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        energy = s.square()
        evr = energy / energy.sum().clamp_min(1e-12)

        stage_dir = output_root / "03_position_subspaces" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        max_rank = vh.shape[0]
        for k in cfg.get("k_list", [1, 2, 4, 8, 16]):
            k = int(k)
            if c < k or max_rank < k:
                append_run_log(cfg, f"Skipping {stage} k={k}: C={c}, rank={max_rank}")
                continue
            basis = vh[:k].T.contiguous()
            basis = orthonormalize_basis(basis)
            out = {
                "stage_name": stage,
                "method": "noise_slot_svd",
                "k": k,
                "basis": basis,
                "mean": mean.contiguous(),
                "mean_noise_slot_field": p.contiguous(),
                "singular_values": s.contiguous(),
                "explained_variance": energy.contiguous(),
                "explained_variance_ratio": evr.contiguous(),
                "explained_variance_ratio_topk": float(evr[:k].sum().item()),
                "token_hw": [int(h), int(w)],
                "num_noise_slot_samples": int(z.shape[0]),
                "noise_types": cfg.get("noise_types", []),
                "noise_seeds": cfg.get("noise_seeds", []),
                "slot_indices": cfg.get("slot_indices_resolved", []),
                "normalize_before_svd": cfg.get("normalize_before_svd", "l2"),
                "center_before_svd": bool(cfg.get("center_before_svd", True)),
            }
            out_path = stage_dir / f"noise_slot_svd_k{k:02d}.pt"
            torch.save(out, out_path)
            append_run_log(
                cfg,
                f"Saved {stage} k={k}: topk_evr={out['explained_variance_ratio_topk']:.6f} -> {out_path}",
            )

    append_run_log(cfg, "Finished estimate_noise_slot_subspace.py")


if __name__ == "__main__":
    main()
