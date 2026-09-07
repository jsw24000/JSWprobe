"""Blender-free fail-closed integration checks on an existing geometry smoke.

Input stays read-only: each mutation replaces a copied manifest in a temporary
hard-linked tree. Hard-linked NPZ/render files are never changed.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from memory_scene_blender.ego_object_v2.validation import validate_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-report',type=Path,required=True)
    a=p.parse_args()
    if a.output_report.exists():raise FileExistsError(a.output_report)
    cases=[('duplicate_sequence','sequences',lambda rows:rows.append(dict(rows[0]))),
           ('wrong_relative_level','sequences',lambda rows:rows[0].update(relative_level=99)),
           ('axis_family_mismatch','groups',lambda rows:rows[0].update(motion_axis_world=[0,0,1])),
           ('extension_camera_mismatch','groups',lambda rows:rows[0].update(base_camera_id='camera_invalid')),
           ('absolute_path','sequences',lambda rows:rows[0].update(canonical_points_path='/tmp/invalid_v2_canonical.npz')),
           ('partial_plan','groups',lambda rows:rows.pop())]
    with tempfile.TemporaryDirectory(prefix='v2_validation_baseline_') as tmp:
        baseline=Path(tmp)/'dataset'
        shutil.copytree(a.dataset_root,baseline,copy_function=os.link)
        (baseline/'validation_summary.json').unlink(missing_ok=True)
        result=validate_dataset(baseline,geometry_only=True)
        assert result['ok'],result['errors']
    results=[]
    for name,manifest,mutate in cases:
        with tempfile.TemporaryDirectory(prefix='v2_validation_') as tmp:
            root=Path(tmp)/'dataset'
            shutil.copytree(a.dataset_root,root,copy_function=os.link)
            # Unlink the output before the validator writes it; protect the source inode.
            (root/'validation_summary.json').unlink(missing_ok=True)
            path=root/'manifests'/f'{manifest}.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            mutate(rows)
            path.unlink()
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result=validate_dataset(root,geometry_only=True)
            expected={'duplicate_sequence':'schema_unique_ids','wrong_relative_level':'interventions_endpoints_frame0_same_r_compensation',
                      'axis_family_mismatch':'motion_coverage_counts_no_partial_full','extension_camera_mismatch':'canonical_identity_extension_pairing',
                      'absolute_path':'json_finiteness_relative_paths','partial_plan':'motion_coverage_counts_no_partial_full'}[name]
            assert not result['ok'] and result['checks'][expected]['status']=='failed',(name,result['errors'])
            results.append(dict(case=name,rejected=True,error_counts=result['error_counts']))
            print('rejected:',name,flush=True)
    a.output_report.parent.mkdir(parents=True,exist_ok=True)
    a.output_report.write_text(json.dumps(dict(ok=True,cases=results),indent=2)+'\n')


if __name__=='__main__':main()
