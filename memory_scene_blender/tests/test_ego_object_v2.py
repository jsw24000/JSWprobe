"""Blender-free V2 crossing, identities, motion math and safety regression tests."""
import copy
import itertools
import json
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
from memory_scene_blender.ego_object_v2.protocol import (
    make_plan, motion_family, stable_seed, canonical_path, group_id, sequence_id,
    matched_relative_groups, FULL_COUNTS, LEVELS)
from memory_scene_blender.ego_object_factorial.protocol import translated_camera_payload, project_opencv, motion_conditions
from memory_scene_blender.tests.test_ego_object_factorial import _base_camera
from memory_scene_blender.scripts.generate_ego_object_factorial_v2 import parse_args, generate_dataset

ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/'memory_scene_blender/configs/ego_object_factorial_v2.yaml').read_text())


class V2Protocol(unittest.TestCase):
    def test_crossing_and_extension_balance(self):
        p=make_plan(CONFIG);c=p['contexts'];ext=[x for x in c if x['extension']]
        self.assertEqual(len(c),24)
        self.assertEqual(len({x['context_id'] for x in c}),24)
        self.assertEqual(set(Counter(x['target_id'] for x in c).values()),{2})
        self.assertEqual(set(Counter(x['background_id'] for x in c).values()),{4})
        for b in p['backgrounds']:
            self.assertEqual(len({x['target_category'] for x in c if x['background_id']==b['background_id']}),4)
        self.assertEqual(len(ext),8)
        self.assertEqual(len({x['background_id'] for x in ext}),6)
        self.assertEqual(set(Counter(x['target_category'] for x in ext).values()),{2})
        self.assertEqual(p,make_plan(copy.deepcopy(CONFIG)))

    def test_motion_schema(self):
        for a in 'xy':
            for d in [2,4,6]:
                fid=f't{a}_d00{d}'
                row=motion_family(fid,CONFIG['motion_families'][fid])
                self.assertEqual(row['delta_m'],d/100)
        for fid,row in [('tz_d004',dict(axis_world=[0,0,1],delta_m=.04)),
                        ('tx_d004',dict(axis_world=[1,0,1],delta_m=.04)),
                        ('tx_d004',dict(axis_world=[1,0,0],delta_m=.02))]:
            with self.assertRaises(ValueError):motion_family(fid,row)

    def test_actual_background_uniqueness(self):
        c=copy.deepcopy(CONFIG)
        for i,b in enumerate(c['backgrounds']):b['room_layout_index']=i%3
        with self.assertRaisesRegex(ValueError,'repeat'):make_plan(c)

    def test_xy_camera_and_arbitrary_horizontal_same_r_compensation(self):
        base=_base_camera();mat=np.array(base['blender_camera_to_world'])
        pts=np.array([[0,0,-3],[.2,.1,-4],[-.3,.4,-5]])@mat[:3,:3].T+mat[:3,3]
        for axis in ([1,0,0],[0,1,0],[.6,.8,0],[-.8,.6,0]):
            axis=np.array(axis)
            for alpha in np.linspace(0,1,8):
                shift=.06*alpha
                cam=translated_camera_payload(base,shift,axis)
                np.testing.assert_allclose(np.array(cam['blender_camera_to_world'])[:3,:3],mat[:3,:3])
                np.testing.assert_allclose(np.array(cam['blender_camera_to_world'])[:3,3]-mat[:3,3],shift*axis,atol=2e-8)
                static=translated_camera_payload(base,0,axis)
                a=project_opencv(pts,cam['opencv_world_to_camera'],base['intrinsics']['K'])
                b=project_opencv(pts-axis*shift,static['opencv_world_to_camera'],base['intrinsics']['K'])
                c=project_opencv(pts+axis*shift,cam['opencv_world_to_camera'],base['intrinsics']['K'])
                d=project_opencv(pts,static['opencv_world_to_camera'],base['intrinsics']['K'])
                for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=2e-6)
                for x,y in zip(c,d):np.testing.assert_allclose(x,y,atol=2e-6)

    def test_group_sequence_and_matched_ids(self):
        seqs=[]
        for f in make_plan(CONFIG)['motion_families']:
            gid=group_id('physical',f['motion_family_id'])
            for condition in motion_conditions(LEVELS,f['delta_m']):
                seqs.append(dict(condition,group_id=gid,sequence_id=sequence_id(gid,condition['ego_level'],condition['object_level']),
                    physical_context_id='physical',motion_family_id=f['motion_family_id'],context_id='context',scene_id='context',
                    target_id='chair_v00',background_id='bg_000',object_anchor_id='anchor',base_camera_id='camera'))
        self.assertEqual(len({s['sequence_id'] for s in seqs}),150)
        matched=matched_relative_groups(seqs)
        self.assertEqual(len(matched),54)
        for r in matched:
            self.assertEqual(r['member_count'],5-abs(r['relative_level']))
            self.assertTrue(all(r['motion_family_id'] in s['sequence_id'] for s in r['members']))

    def test_canonical_seed_and_path_shared_across_backgrounds(self):
        contexts=make_plan(CONFIG)['contexts']
        for tid,rows in itertools.groupby(sorted(contexts,key=lambda x:x['target_id']),key=lambda x:x['target_id']):
            rows=list(rows);self.assertEqual(len(rows),2)
            self.assertEqual({r['canonical_sampling_seed'] for r in rows},{stable_seed(CONFIG['seed'],tid)})
            self.assertEqual({r['canonical_points_path'] for r in rows},{canonical_path(tid)})
        self.assertEqual(len({c['canonical_sampling_seed'] for c in contexts}),12)

    def test_native_axial_depth_to_pixel_ray_range(self):
        from memory_scene_blender.ego_object_v2.rendering import axial_to_range
        converted=axial_to_range(np.full((2,2),2.),np.eye(3))
        self.assertEqual(converted.dtype,np.float32)
        self.assertAlmostEqual(float(converted[0,0]),np.sqrt(6),places=6)
        self.assertAlmostEqual(float(converted[1,1]),np.sqrt(22),places=6)

    def test_full_count_oracle(self):
        p=make_plan(CONFIG);n=sum(len(c['motion_family_ids']) for c in p['contexts'])
        self.assertEqual(dict(contexts=len(p['contexts']),groups=n,sequences=n*25,frames=n*25*8),FULL_COUNTS)
        self.assertEqual(CONFIG['expected_full_counts'],FULL_COUNTS)
        self.assertEqual(CONFIG['modes']['full']['samples'],8)

    def test_full_gate_before_blender_or_writes(self):
        with self.assertRaisesRegex(ValueError,'allow-full'):
            generate_dataset(parse_args(['--mode','full']))
        with self.assertRaisesRegex(ValueError,'smoke --dry-run'):
            generate_dataset(parse_args(['--smoke-all-families']))


if __name__=='__main__':unittest.main()
