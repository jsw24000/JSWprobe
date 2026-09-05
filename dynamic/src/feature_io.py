import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import yaml


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def json_safe(x):
    if isinstance(x, dict): return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return json_safe(x.tolist())
    if isinstance(x, np.generic): return json_safe(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def arguments(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/e1_pilot.yaml'))
    p.add_argument('--dataset-root'); p.add_argument('--output-root'); p.add_argument('--device')
    p.add_argument('--dinov3-checkpoint'); p.add_argument('--vggt-omega-checkpoint')
    p.add_argument('--dinov3-repo'); p.add_argument('--vggt-omega-repo')
    return p


def configuration(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ['dataset_root', 'output_root', 'device']:
        if getattr(args, key, None): cfg[key] = getattr(args, key)
    for model in ['dinov3', 'vggt_omega']:
        for key in ['repo', 'checkpoint']:
            if getattr(args, model + '_' + key, None): cfg[model][key] = getattr(args, model + '_' + key)
    return cfg


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def save_features(path, arrays, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    for k, v in arrays.items():
        if np.issubdtype(v.dtype, np.floating) and not np.isfinite(v).all():
            raise ValueError(f'Nonfinite feature: {k}')
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays, metadata_json=np.array(json.dumps(json_safe(metadata))))
    os.replace(tmp, path)
