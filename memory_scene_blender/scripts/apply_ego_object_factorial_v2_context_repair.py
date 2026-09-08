#!/usr/bin/env python3
"""Validate and transactionally install a scoped V2 context replacement."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from memory_scene_blender.ego_object_v2.protocol import matched_relative_groups
from memory_scene_blender.ego_object_v2.reports import write_reports
from memory_scene_blender.ego_object_v2.validation import validate_dataset
from memory_scene_blender.object_translation.manifest_utils import write_json, write_jsonl

ALLOWED_CONTEXTS = {'bg_003__chair_v01', 'bg_003__side_table_v00'}
CONTEXT_MANIFESTS = ('scenes', 'contexts', 'anchors', 'base_cameras', 'groups',
                     'sequences', 'frames', 'track_sets')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def inspect_replacement(full_root, replacement_root, min_mask_ratio):
    full_marker = read_json(full_root/'.ego_object_factorial_v2_dataset')
    if full_marker.get('dataset_name') != 'ego_object_factorial_v2':
        raise ValueError('Dataset root is not a V2 dataset')
    if full_root == replacement_root or full_root in replacement_root.parents:
        raise ValueError('Replacement root must be a separate sibling tree')
    marker = read_json(replacement_root/'.ego_object_factorial_v2_dataset')
    if marker.get('dataset_name') != 'ego_object_factorial_v2':
        raise ValueError('Replacement root is not a V2 dataset')
    summary = read_json(replacement_root/'dataset_summary.json')
    context_ids = set(summary.get('subset_context_ids', []))
    if context_ids != ALLOWED_CONTEXTS:
        raise ValueError(f'Replacement must contain exactly {sorted(ALLOWED_CONTEXTS)}, got {sorted(context_ids)}')
    expected = {'contexts': 2, 'groups': 7, 'sequences': 175, 'frames': 1400}
    if summary.get('counts') != expected or not summary.get('complete') or summary.get('dry_run'):
        raise ValueError(f'Incomplete rendered replacement: {summary}')
    actual_counts = {
        'contexts': len(read_jsonl(replacement_root/'manifests/contexts.jsonl')),
        'groups': len(read_jsonl(replacement_root/'manifests/groups.jsonl')),
        'sequences': len(read_jsonl(replacement_root/'manifests/sequences.jsonl')),
        'frames': len(read_jsonl(replacement_root/'manifests/frames.jsonl')),
    }
    if actual_counts != expected:
        raise ValueError(f'Replacement manifest counts do not match: {actual_counts}')
    default_scale = summary.get('camera_distance_scale')
    overrides = summary.get('context_camera_distance_scales', {})
    scales = {cid: overrides.get(cid, default_scale) for cid in context_ids}
    if any(scale is None or not 0.0 < float(scale) < 1.0 for scale in scales.values()):
        raise ValueError(f'Replacement did not use a nearer-camera scale for every context: {scales}')

    old_canonical = full_root/'canonical_targets'
    new_canonical = replacement_root/'canonical_targets'
    for target_id in ('chair_v01', 'side_table_v00'):
        rel = Path(target_id)/'canonical_surface_points.npz'
        with np.load(old_canonical/rel) as old, np.load(new_canonical/rel) as new:
            if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
                raise ValueError(f'Canonical point identity drift: {target_id}')

    groups = read_jsonl(replacement_root/'manifests/groups.jsonl')
    family_counts = {cid: sum(g['context_id'] == cid for g in groups) for cid in context_ids}
    if family_counts != {'bg_003__chair_v01': 1, 'bg_003__side_table_v00': 6}:
        raise ValueError(f'Unexpected family coverage: {family_counts}')
    frames = read_jsonl(replacement_root/'manifests/frames.jsonl')
    if {frame['context_id'] for frame in frames} != context_ids:
        raise ValueError('Replacement frame identities do not match its context set')
    ratios = []
    for frame in frames:
        ratio = float(frame['target_mask_area_ratio'])
        ratios.append(ratio)
        if ratio < min_mask_ratio:
            raise ValueError(f"Small mask {ratio:.6f}: {frame['frame_id']}")
        if frame['target_truncated'] or float(frame['target_edge_touch_ratio']) > 0.01:
            raise ValueError(f"Truncated/edge-touching target: {frame['frame_id']}")
        for key in ('rgb','depth_exr','depth_npy','native_depth_npy','target_mask','frame_metadata'):
            path = replacement_root/frame[key]
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing rendered artifact: {path}')
    return dict(context_ids=sorted(context_ids), counts=expected,
                camera_distance_scales=scales,
                minimum_mask_area_ratio=min(ratios), maximum_mask_area_ratio=max(ratios))


def merged_rows(full_root, replacement_root, name, context_ids):
    old = read_jsonl(full_root/'manifests'/f'{name}.jsonl')
    new = read_jsonl(replacement_root/'manifests'/f'{name}.jsonl')
    def context_id(row):
        # track_sets follows the older schema and carries scene_id=context_id.
        value = row.get('context_id', row.get('scene_id'))
        if value is None:
            raise ValueError(f'{name} row has neither context_id nor scene_id')
        return value
    by_context = {}
    for row in new:
        by_context.setdefault(context_id(row), []).append(row)
    result = []
    inserted = set()
    for row in old:
        cid = context_id(row)
        if cid in context_ids:
            if cid not in inserted:
                result.extend(by_context[cid]); inserted.add(cid)
        else:
            result.append(row)
    if inserted != context_ids:
        raise ValueError(f'Full manifests lack contexts: {sorted(context_ids-inserted)}')
    return result


def apply_replacement(full_root, replacement_root, audit):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = full_root.parent/f'.context_repair_backup_{stamp}'
    context_ids = set(audit['context_ids'])
    staged = {}
    for name in CONTEXT_MANIFESTS:
        staged[name] = merged_rows(full_root, replacement_root, name, context_ids)
    staged['matched_relative_groups'] = matched_relative_groups(staged['sequences'])

    top_files = ('selection_report.json','provenance.json','dataset_summary.json','validation_summary.json')
    backup.mkdir()
    try:
        shutil.move(str(full_root/'manifests'), str(backup/'manifests'))
        shutil.move(str(full_root/'reports'), str(backup/'reports'))
        for name in top_files:
            path = full_root/name
            if path.exists(): shutil.move(str(path), str(backup/name))
        for cid in sorted(context_ids):
            shutil.move(str(full_root/'contexts'/cid), str(backup/cid))
            shutil.move(str(replacement_root/'contexts'/cid), str(full_root/'contexts'/cid))

        (full_root/'manifests').mkdir()
        for name, rows in staged.items(): write_jsonl(full_root/'manifests'/f'{name}.jsonl', rows)
        for name in ('targets','backgrounds','motion_families'):
            shutil.copy2(backup/'manifests'/f'{name}.jsonl', full_root/'manifests'/f'{name}.jsonl')
        shutil.copy2(backup/'manifests'/'splits.json', full_root/'manifests'/'splits.json')

        selection = read_json(backup/'selection_report.json')
        replacements = {r['context_id']:r for r in read_json(replacement_root/'selection_report.json')['contexts']}
        selection['contexts'] = [replacements.get(r['context_id'], r) for r in selection['contexts']]
        write_json(full_root/'selection_report.json', selection)
        provenance = read_json(backup/'provenance.json')
        provenance.setdefault('context_repairs', []).append(dict(timestamp_utc=stamp, **audit,
            replacement_provenance=read_json(replacement_root/'provenance.json')))
        write_json(full_root/'provenance.json', provenance)
        shutil.copy2(backup/'dataset_summary.json', full_root/'dataset_summary.json')
        plan = read_json(full_root/'context_plan.json')
        manifests = {name: read_jsonl(full_root/'manifests'/f'{name}.jsonl')
                     for name in (*CONTEXT_MANIFESTS, 'matched_relative_groups','targets','backgrounds','motion_families')}
        write_reports(full_root, plan, manifests)
        result = validate_dataset(full_root, geometry_only=False)
        if not result['ok']:
            raise RuntimeError('Merged full dataset validation failed; replacement rolled back')
    except Exception:
        for cid in sorted(context_ids):
            current = full_root/'contexts'/cid
            if current.exists(): shutil.move(str(current), str(replacement_root/'contexts'/cid))
            if (backup/cid).exists(): shutil.move(str(backup/cid), str(current))
        for name in ('manifests','reports'):
            current = full_root/name
            if current.exists(): shutil.rmtree(current)
            if (backup/name).exists(): shutil.move(str(backup/name), str(current))
        for name in top_files:
            current = full_root/name
            if current.exists(): current.unlink()
            if (backup/name).exists(): shutil.move(str(backup/name), str(current))
        backup.rmdir()
        raise
    shutil.rmtree(backup)
    write_json(full_root/'context_repair_summary.json', dict(ok=True,timestamp_utc=stamp,**audit))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--replacement-root',type=Path,required=True)
    p.add_argument('--min-mask-area-ratio',type=float,default=0.02)
    p.add_argument('--apply',action='store_true',help='Install after all replacement checks pass')
    args=p.parse_args()
    full=args.dataset_root.resolve(); replacement=args.replacement_root.resolve()
    audit=inspect_replacement(full,replacement,args.min_mask_area_ratio)
    if not args.apply:
        print(json.dumps(dict(ok=True,applied=False,**audit),indent=2)); return
    result=apply_replacement(full,replacement,audit)
    print(json.dumps(dict(ok=True,applied=True,audit=audit,validation=result),indent=2))


if __name__=='__main__': main()
