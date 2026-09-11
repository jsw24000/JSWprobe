import argparse,copy,sys,unittest
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.e2_dataset import E2Dataset,EXPECTED_FAMILIES,intersect_canonical,validate_cache_metadata
from src.feature_io import configuration,exclusive_file_lock,records_hash

PANELS={
 'core_x_confirmation':{'families':['tx_d004'],'context_rule':'all','core':'core_tx004'},
 'xy_d004':{'families':['tx_d004','ty_d004'],'context_rule':'extended','core':'core_xy004'},
 'scale_locality':{'families':list(EXPECTED_FAMILIES),'context_rule':'extended','core':'core_scale6'}}

def fake_dataset():
    d=E2Dataset.__new__(E2Dataset);d.cfg={'panels':copy.deepcopy(PANELS),'stages':{'stage2':['core_x_confirmation','xy_d004']}}
    d.sequences=[];d.by_physical=defaultdict(lambda:defaultdict(list))
    for pc,families in [('extended',EXPECTED_FAMILIES),('x_only',('tx_d004',))]:
        for family in families:
            for e in range(-2,3):
                for o in range(-2,3):
                    s={'sequence_id':f'{pc}-{family}-{e}-{o}','physical_context_id':pc,'motion_family_id':family,'ego_level':e,'object_level':o}
                    d.sequences.append(s);d.by_physical[pc][family].append(s)
    return d

class E2DatasetTests(unittest.TestCase):
    def test_family_discovery_and_panel_pairing(self):
        d=fake_dataset();self.assertEqual(d.extended_contexts(),['extended'])
        self.assertEqual(len(d.panel_sequences('core_x_confirmation')),50)
        xy=d.panel_sequences('xy_d004');self.assertEqual(len(xy),50);self.assertEqual({s['physical_context_id'] for s in xy},{'extended'})
        self.assertEqual(len(d.panel_sequences('scale_locality')),150)
        self.assertEqual(len(d.sequences_for_request('stage2')),75)

    def test_smoke_is_bounded_and_analytic(self):
        d=fake_dataset();rows=d.panel_sequences('xy_d004',smoke=True)
        self.assertEqual(len(rows),20);self.assertEqual({s['physical_context_id'] for s in rows},{'extended'})

    def test_cross_family_intersection_and_failures(self):
        ids=np.arange(6);xyz=np.eye(6,3);a=np.array([1,1,1,1,0,0]);b=np.array([0,1,1,1,1,0])
        selected,common=intersect_canonical([(ids,xyz,a),(ids,xyz,b)],3,4)
        self.assertEqual(set(selected),{1,2,3});self.assertEqual(int(common.sum()),3)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):intersect_canonical([(ids,xyz,a),(ids+1,xyz,b)],1,4)
        with self.assertRaisesRegex(ValueError,'only 3'):intersect_canonical([(ids,xyz,a),(ids,xyz,b)],4,4)

    def test_cache_mismatch_rejected(self):
        self.assertTrue(validate_cache_metadata({'config_hash':'a','point_ids':[1]}, {'config_hash':'a','point_ids':[1]}))
        with self.assertRaisesRegex(ValueError,'config_hash'):validate_cache_metadata({'config_hash':'old'},{'config_hash':'new'})

    def test_e2_inherits_exact_e1_model_config(self):
        root=Path(__file__).resolve().parents[2]
        def args(path):return argparse.Namespace(config=str(path),dataset_root=None,output_root=None,device=None,models=None,
            dinov3_checkpoint=None,dinov3_repo=None,vggt_omega_checkpoint=None,vggt_omega_repo=None,
            dinov2_checkpoint=None,dinov2_repo=None,vggt_checkpoint=None,vggt_repo=None)
        e1=configuration(args(root/'dynamic/configs/e1_four_models.yaml'));e2=configuration(args(root/'dynamic/configs/e2_full_v2.yaml'))
        for model in e1['models']:self.assertEqual(e2[model],e1[model])

    def test_extraction_writer_lock(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with exclusive_file_lock(Path(tmp)/'writer.lock'):
                second=exclusive_file_lock(Path(tmp)/'writer.lock')
                with self.assertRaisesRegex(RuntimeError,'Another extraction writer'):second.__enter__()

    def test_selected_manifest_hash_is_order_stable(self):
        rows=[{'path':'b','sha256':'2'},{'path':'a','sha256':'1'}]
        self.assertEqual(records_hash(rows),records_hash(list(reversed(rows))))
        self.assertNotEqual(records_hash(rows),records_hash(rows+[{'path':'c','sha256':'3'}]))

if __name__=='__main__':unittest.main()
