"""Compact coverage and projected-motion audit reports; no scientific claims."""
import csv
from collections import Counter
import numpy as np
from memory_scene_blender.object_translation.manifest_utils import write_json


def csv_rows(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty'])
        w.writeheader();w.writerows(rows)


def write_reports(root, plan, manifests):
    out=root/'reports';out.mkdir(exist_ok=True)
    contexts=manifests['contexts']
    write_json(out/'coverage_summary.json',dict(
        planned_full_counts=plan['expected_full_counts'],context_count=len(contexts),
        target_counts=dict(Counter(c['target_id'] for c in contexts)),
        background_counts=dict(Counter(c['background_id'] for c in contexts)),
        category_counts=dict(Counter(c['target_category'] for c in contexts)),
        extension_subset_ids=[c['context_id'] for c in plan['contexts'] if c['extension']],
        motion_family_coverage=dict(Counter(g['motion_family_id'] for g in manifests['groups']))))
    csv_rows(out/'coverage_matrix.csv',[dict(target_id=t['target_id'],**{
        b['background_id']:int(any(c['target_id']==t['target_id'] and c['background_id']==b['background_id'] for c in plan['contexts']))
        for b in plan['backgrounds']}) for t in plan['targets']])
    csv_rows(out/'selection_summary.csv',[dict(context_id=g['context_id'],physical_context_id=g['physical_context_id'],
        motion_family_id=g['motion_family_id'],anchor_id=g['object_anchor_id'],base_camera_id=g['base_camera_id'],
        min_visible_fraction=g['camera_selection']['visible_surface_fraction'],
        min_bbox_margin_px=g['camera_selection']['bbox_margin_px']) for g in manifests['groups']])
    rows=[]
    for s in manifests['sequences']:
        if s['ego_level']!=0 or abs(s['object_level']) not in (1,2): continue
        with np.load(root/s['tracks_path']) as tr:
            common=tr['visible'][0]&tr['visible'][-1]
            d=np.linalg.norm(tr['projected_uv'][-1,common]-tr['projected_uv'][0,common],axis=1)
        median=float(np.median(d)) if len(d) else None
        rows.append(dict(physical_context_id=s['physical_context_id'],group_id=s['group_id'],motion_family_id=s['motion_family_id'],
            axis_label='X' if s['motion_axis_world'][0] else 'Y',delta_m=s['delta_m'],object_level=s['object_level'],
            common_visible_points=len(d),median_pixel_displacement=median,min_px=float(d.min()) if len(d) else None,
            max_px=float(d.max()) if len(d) else None,median_displacement_div14=median/14 if median is not None else None,
            median_displacement_div16=median/16 if median is not None else None))
    csv_rows(out/'projected_motion_scale.csv',rows)
