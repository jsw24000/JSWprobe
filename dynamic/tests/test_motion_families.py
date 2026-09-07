import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset import PilotDataset, select_motion_families


class MotionFamilyAdapter(unittest.TestCase):
    def test_v1_absent_filter_unchanged(self):
        rows=[dict(sequence_id='v1',delta_m=.04,motion_axis_world=[1,0,0])]
        self.assertIs(select_motion_families(rows),rows)

    def test_heterogeneous_filter_and_guard(self):
        rows=[dict(sequence_id=f,motion_family_id=f,delta_m=d,motion_axis_world=a)
              for f,d,a in [('tx_d004',.04,[1,0,0]),('ty_d004',.04,[0,1,0]),('tx_d002',.02,[1,0,0])]]
        with self.assertRaisesRegex(ValueError,'homogeneous'):select_motion_families(rows)
        self.assertEqual(select_motion_families(rows,['tx_d004']),rows[:1])
        with self.assertRaises(ValueError):select_motion_families(rows,['tx_d004','ty_d004'])
        with self.assertRaises(ValueError):select_motion_families(rows,['invalid'])
        with self.assertRaises(ValueError):select_motion_families(rows,[])

    def test_filter_before_groups_and_frames(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifests').mkdir()
            seqs=[dict(sequence_id=f,group_id=f,motion_family_id=f,delta_m=.04,motion_axis_world=a,frame_ids=[f+'0'])
                  for f,a in [('tx_d004',[1,0,0]),('ty_d004',[0,1,0])]]
            (root/'manifests/sequences.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in seqs))
            (root/'manifests/frames.jsonl').write_text(''.join(json.dumps(dict(frame_id=s['frame_ids'][0]))+'\n' for s in seqs))
            d=PilotDataset(dict(dataset_root=tmp,input_resolution=[512,512],motion_family_ids=['tx_d004']))
            self.assertEqual(list(d.groups),['tx_d004']);self.assertEqual(list(d.frames),['tx_d0040'])
