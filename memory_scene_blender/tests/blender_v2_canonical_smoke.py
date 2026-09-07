"""Blender-only canonical identity smoke: construct 24 scenes, render zero frames.

Run with Blender --background --python-exit-code 1 --python this_file --
--output-report /a/fresh/canonical_identity.json.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from memory_scene_blender.ego_object_v2.protocol import make_plan
from memory_scene_blender.ego_object_v2.geometry import build_scene_bundle_explicit
from memory_scene_blender.ego_object_factorial.geometry import sample_target_surface_points
from memory_scene_blender.object_translation.scene_builder import set_target_position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-report', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output_report.exists():
        raise FileExistsError(args.output_report)
    cfg = json.loads((ROOT / 'memory_scene_blender/configs/ego_object_factorial_v2.yaml').read_text())
    base = json.loads((ROOT / cfg['base_config']).read_text())
    plan = make_plan(cfg)
    canonical, checked = {}, []
    for context in plan['contexts']:
        bundle = build_scene_bundle_explicit(base, cfg['modes']['full'], **{
            key: context[key] for key in ('context_id', 'room_layout_index', 'static_layout_id',
                                         'target_category', 'target_variant', 'context_seed')})
        set_target_position(bundle.target_asset, [0, 0])
        points = sample_target_surface_points(bundle.target_asset, cfg['tracks']['num_surface_points'],
                                              context['canonical_sampling_seed'])
        if context['target_id'] in canonical:
            assert all(np.array_equal(value, canonical[context['target_id']][key])
                       for key, value in points.items()), context['context_id']
        canonical[context['target_id']] = points
        checked.append(context['context_id'])
        print('canonical identity:', context['context_id'], flush=True)
    report = dict(ok=True, rendered_frames=0, contexts=len(checked), target_identities=len(canonical),
                  checked_contexts=checked)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
