import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.spatial_sampling import sanity_checks,occupancy,SpatialTransform,boundary_distances
from src.metrics import compute_metrics

class Correctness(unittest.TestCase):
    def test_coordinates(self):self.assertTrue(sanity_checks()['passed'])
    def test_mask_occupancy(self):
        mask=np.zeros((512,512),bool);mask[16:32,32:48]=True
        q=occupancy(mask,16);self.assertEqual(q[1,2],1);self.assertEqual(q.sum(),1)
    def test_relative_only_oracle(self):
        z={(e,o):np.array([[o-e,0,0]],dtype=np.float32) for e in range(-2,3) for o in range(-2,3)}
        m,p=compute_metrics(z,.04)
        self.assertEqual(len(p),20)
        for k in ['TC_e','TC_o','scale_e','scale_o','cos_abs']:self.assertAlmostEqual(float(m[k][0]),1,5)
        self.assertAlmostEqual(float(m['cos_signed'][0]),-1,5)
        self.assertAlmostEqual(float(m['D_cause'][0]),0,5)
        self.assertAlmostEqual(float(m['C_norm'][0]),0,5)
    def test_factorized_oracle_and_static_invariance(self):
        z={(e,o):np.array([[e,o,0]],dtype=np.float32) for e in range(-2,3) for o in range(-2,3)}
        m,_=compute_metrics(z,.04);shifted,_=compute_metrics({k:v+20 for k,v in z.items()},.04)
        self.assertAlmostEqual(float(m['cos_abs'][0]),0,5);self.assertGreater(float(m['D_cause'][0]),.5)
        for k in m:np.testing.assert_allclose(m[k],shifted[k],atol=1e-6)
    def test_zero_signal_cosines_undefined(self):
        z={(e,o):np.zeros((1,3),np.float32) for e in range(-2,3) for o in range(-2,3)}
        m,_=compute_metrics(z,.04);self.assertTrue(np.isnan(m['TC_e'][0]))
if __name__=='__main__':unittest.main()
