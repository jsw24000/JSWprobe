#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from noise_slot_utils import (
    append_run_log,
    generate_noise_image,
    load_lingbot_aggregator_model,
    resolve_config,
    run_aggregator_tokens,
    save_noise_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate noise-slot replacement token samples.")
    parser.add_argument("--config", default="configs/noise_slot_svd_debias.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    output_root = Path(cfg["output_root"])
    append_run_log(cfg, "Starting generate_noise_slot_samples.py")
    append_run_log(
        cfg,
        f"Stages={cfg['stages_resolved']}, slots={cfg['slot_indices_resolved']}, "
        f"noise_types={cfg['noise_types']}, seeds={cfg['noise_seeds']}, "
        f"samples={len(cfg['sample_specs'])}",
    )

    save_noise_examples(cfg)

    frame_paths = [Path(p) for p in cfg["frame_paths"]]
    frame_indices = cfg.get("frame_indices", list(range(len(frame_paths))))
    replacement_dir = output_root / "01_noise_inputs" / "replacements"
    replacement_dir.mkdir(parents=True, exist_ok=True)

    model, device, dtype = load_lingbot_aggregator_model(cfg)

    storage_dtype_name = str(cfg.get("storage_dtype", "float16"))
    storage_dtype = torch.float16 if storage_dtype_name == "float16" else torch.float32
    append_run_log(cfg, f"Noise-slot token storage dtype: {storage_dtype_name}")

    stage_samples = {stage: [] for stage in cfg["stages_resolved"]}
    stage_payload_meta = {stage: None for stage in cfg["stages_resolved"]}
    sample_index = []

    for spec in cfg["sample_specs"]:
        sample_id = int(spec["sample_id"])
        slot = int(spec["slot_index_in_sampled_sequence"])
        noise_type = str(spec["noise_type"])
        seed = int(spec["seed"])

        with Image.open(frame_paths[slot]) as img:
            size = img.size
        noise_img = generate_noise_image(noise_type, seed, size)
        replacement_path = replacement_dir / (
            f"sample_{sample_id:04d}_slot{slot:03d}_{noise_type}_seed{seed:03d}.png"
        )
        noise_img.save(replacement_path)

        seq_paths = list(frame_paths)
        seq_paths[slot] = replacement_path
        append_run_log(
            cfg,
            f"Forward sample {sample_id + 1}/{len(cfg['sample_specs'])}: "
            f"slot={slot}, original_frame={frame_indices[slot]}, noise={noise_type}, seed={seed}",
        )
        tokens_by_stage = run_aggregator_tokens(cfg, model, device, dtype, seq_paths)

        row = {
            "sample_id": sample_id,
            "slot_index_in_sampled_sequence": slot,
            "original_frame_index": int(frame_indices[slot]),
            "noise_type": noise_type,
            "seed": seed,
            "replaced_image_path": str(replacement_path),
            "sequence_length": len(frame_paths),
        }
        sample_index.append(row)

        for stage in cfg["stages_resolved"]:
            if stage not in tokens_by_stage:
                append_run_log(cfg, f"Warning: {stage} missing from forward output; skipping sample.")
                continue
            token_map = tokens_by_stage[stage][slot].detach().cpu().to(dtype=storage_dtype)
            stage_samples[stage].append(token_map)
            stage_payload_meta[stage] = {
                "token_hw": [int(token_map.shape[0]), int(token_map.shape[1])],
                "channels": int(token_map.shape[-1]),
            }

        del tokens_by_stage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    token_root = output_root / "02_noise_slot_tokens"
    token_root.mkdir(parents=True, exist_ok=True)
    for stage in cfg["stages_resolved"]:
        stage_dir = token_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        samples = stage_samples[stage]
        if not samples:
            append_run_log(cfg, f"Warning: no samples collected for {stage}")
            continue
        tokens = torch.stack(samples, dim=0).contiguous()
        payload = {
            "stage_name": stage,
            "tokens": tokens,
            "token_hw": [int(tokens.shape[1]), int(tokens.shape[2])],
            "num_samples": int(tokens.shape[0]),
            "sample_index": sample_index,
            "normalize": "none",
            "source": "noise_slot_svd",
            "storage_dtype": storage_dtype_name,
            "config": {
                "noise_types": cfg["noise_types"],
                "noise_seeds": cfg["noise_seeds"],
                "slot_indices": cfg["slot_indices_resolved"],
                "max_slots": cfg.get("max_slots"),
                "max_noise_samples": cfg.get("max_noise_samples"),
            },
        }
        out_pt = stage_dir / f"noise_slot_tokens_{stage}.pt"
        torch.save(payload, out_pt)
        (stage_dir / "sample_index.json").write_text(json.dumps(sample_index, indent=2))
        append_run_log(cfg, f"Saved {stage}: tokens={tuple(tokens.shape)} -> {out_pt}")

    append_run_log(cfg, "Finished generate_noise_slot_samples.py")


if __name__ == "__main__":
    main()
