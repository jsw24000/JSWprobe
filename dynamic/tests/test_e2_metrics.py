import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.e2_metrics import family_metrics,response_vectors,xy_subspace,scale_consistency
from src.metrics import compute_metrics

def grid(fn):return {(e,o):np.asarray([fn(e,o)],np.float32) for e in range(-2,3) for o in range(-2,3)}

class E2MetricOracles(unittest.TestCase):
    def test_pure_relative(self):
        z=grid(lambda e,o:[o-e,2*(o-e)])
        m,p=family_metrics(z,.04)
        self.assertAlmostEqual(float(m['cos_signed'][0]),-1,6);self.assertAlmostEqual(float(m['M_c'][0]),0,6)
        self.assertAlmostEqual(float(m['M_r'][0]),1,6);self.assertAlmostEqual(float(m['C_norm'][0]),0,6)
        self.assertTrue(all(np.allclose(x['D_c_given_r'],0) for x in p))

    def test_pure_common_and_mixed(self):
        common=grid(lambda e,o:[e+o,2*(e+o)])
        m,p=family_metrics(common,.04)
        self.assertAlmostEqual(float(m['cos_signed'][0]),1,6);self.assertAlmostEqual(float(m['M_c'][0]),1,6)
        self.assertAlmostEqual(float(m['M_r'][0]),0,6);self.assertGreater(float(np.mean([x['D_c_given_r'] for x in p])),0)
        mixed=grid(lambda e,o:[o-e,o+e]);m,_=family_metrics(mixed,.04)
        self.assertAlmostEqual(float(m['M_c'][0]),2**-.5,5);self.assertAlmostEqual(float(m['M_r'][0]),2**-.5,5)

    def test_full_rank_and_rank_collapse(self):
        full=xy_subspace(np.array([-1.,0]),np.array([0.,-1]),np.array([1.,0]),np.array([0.,1]))
        self.assertEqual((full['rank_e'],full['rank_o']),(2,2));self.assertTrue(full['well_conditioned_2d'])
        self.assertTrue(np.allclose(full['principal_angles_deg'],0));self.assertAlmostEqual(full['subspace_overlap'],1)
        collapsed=xy_subspace(np.array([1.,0]),np.array([2.,0]),np.array([0.,1]),np.array([0.,2]))
        self.assertEqual((collapsed['rank_e'],collapsed['rank_o']),(1,1));self.assertFalse(collapsed['well_conditioned_2d'])
        self.assertLess(collapsed['rho_e'],1e-6);self.assertEqual(len(collapsed['principal_angles_deg']),1)

    def test_legacy_exact_regression_and_scale(self):
        z=grid(lambda e,o:[e,o,e*o])
        e2,_=family_metrics(z,.04);e1,_=compute_metrics(z,.04)
        for key,value in e1.items():np.testing.assert_array_equal(e2[key],value)
        self.assertTrue(np.array_equal(e2['D_cause_legacy'],e1['D_cause']))
        rows=scale_consistency({.02:np.array([[1.,0]]),.04:np.array([[2.,0]]),.06:np.array([[0.,1]])})
        self.assertEqual(len(rows),3);self.assertAlmostEqual(float(rows[0]['cosine'][0]),1)

if __name__=='__main__':unittest.main()
