"""Blender-free deterministic V2 design and identifiers."""
import hashlib
import re
from collections import defaultdict
import numpy as np
from memory_scene_blender.ego_object_factorial.protocol import condition_key, level_token

CATEGORIES = ('chair', 'armchair', 'side_table', 'small_cabinet')
LEVELS = [-2, -1, 0, 1, 2]
FULL_COUNTS = dict(contexts=24, groups=64, sequences=1600, frames=12800)


def stable_seed(seed, identity):
    return int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:4], 'big')


def canonical_path(target_id):
    if target_id not in {f'{c}_v{v:02d}' for c in CATEGORIES for v in range(3)}:
        raise ValueError(f'Unknown target {target_id}')
    return f'canonical_targets/{target_id}/canonical_surface_points.npz'


def motion_family(family_id, row):
    match = re.fullmatch(r't([xy])_d(002|004|006)', family_id)
    if not match:
        raise ValueError(f'Invalid motion family {family_id}')
    axis = np.asarray(row['axis_world'], dtype=float)
    expected = [1, 0, 0] if match[1] == 'x' else [0, 1, 0]
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.array_equal(axis, expected):
        raise ValueError('V2 families require the declared horizontal X/Y unit axis; Z is unsupported')
    if not np.isclose(row['delta_m'], int(match[2]) / 100, rtol=0, atol=1e-12):
        raise ValueError('Family ID and delta disagree')
    return dict(motion_family_id=family_id, axis_world=axis.tolist(), delta_m=float(row['delta_m']))


def make_plan(config):
    families = [motion_family(k, v) for k, v in sorted(config['motion_families'].items())]
    if {r['motion_family_id'] for r in families} != {f't{a}_d00{d}' for a in 'xy' for d in (2,4,6)}:
        raise ValueError('Exactly six motion families required')
    backgrounds = config['backgrounds']
    if [b['background_id'] for b in backgrounds] != [f'bg_{b:03d}' for b in range(6)]:
        raise ValueError('Six ordered background templates required')
    if len({(b['room_layout_index'], b['static_layout_id'] % 3) for b in backgrounds}) != 6:
        raise ValueError('Background templates repeat effective room/static layouts')
    targets = [dict(target_id=f'{c}_v{v:02d}', target_category=c, target_variant=v,
                    canonical_points_path=canonical_path(f'{c}_v{v:02d}'),
                    canonical_sampling_seed=stable_seed(config['seed'], f'{c}_v{v:02d}'))
               for c in CATEGORIES for v in range(3)]
    contexts = []
    for c, category in enumerate(CATEGORIES):
        for v in range(3):
            target = targets[c*3+v]
            for offset in (0,1):
                b = (2*v+c+offset) % 6
                cid = f"bg_{b:03d}__{target['target_id']}"
                # Two contexts/category, with all six backgrounds represented.
                extension = v == c % 2
                contexts.append(dict(**target, **backgrounds[b], context_id=cid, scene_id=cid,
                    context_seed=stable_seed(config['seed'], cid), extension=extension,
                    motion_family_ids=[f['motion_family_id'] for f in families] if extension else ['tx_d004']))
    return dict(algorithm='b=(2*v+c+offset)%6; extension v=c%2', seed=config['seed'],
                targets=targets, backgrounds=backgrounds, contexts=contexts,
                motion_families=families, expected_full_counts=FULL_COUNTS)


def group_id(physical_context_id, family_id):
    return f'{physical_context_id}__{family_id}'


def sequence_id(group, e, o):
    return f'{group}__{condition_key(e,o)}'


def matched_relative_groups(sequences):
    groups = defaultdict(list)
    for s in sequences:
        groups[s['group_id'], s['relative_level']].append(s)
    result = []
    for (gid, r), members in sorted(groups.items()):
        first = members[0]
        result.append({**{k:first[k] for k in ('group_id','physical_context_id','motion_family_id',
            'context_id','scene_id','target_id','background_id','object_anchor_id','base_camera_id')},
            'matched_relative_group_id':f'{gid}__r_{level_token(r)}', 'relative_level':r,
            'member_count':len(members), 'members':[{k:s[k] for k in ('sequence_id','ego_level',
            'object_level','ego_amplitude_m','object_amplitude_m')} for s in members]})
    return result
