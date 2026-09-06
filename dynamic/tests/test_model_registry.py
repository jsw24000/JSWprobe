import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.model_registry import MODELS,regimes_for,expected_shapes,shards_per_sequence,transform_for
from src.spatial_sampling import boundary_distances,sample_lattice
import torch

class FourModelTests(unittest.TestCase):
    def test_schema(self):
        c={'models':list(MODELS),**{n:{'layers':[5,11,17,23]} for n in MODELS}}
        self.assertEqual(shards_per_sequence(c),8)
        self.assertEqual(expected_shapes('vggt',c,26)['L23__post__register'],(4,1024))
        self.assertEqual(expected_shapes('vggt_omega',c,26)['L23__post__register'],(16,1024))
        self.assertEqual(expected_shapes('vggt',c,26)['L23__fused__dense'],(26,128))
        self.assertEqual(regimes_for('dinov2'),['Single'])
        c['models']=['dinov2','vggt'];self.assertEqual(shards_per_sequence(c),4)
        c['dinov2']['regimes']=['Single','Pair']
        with self.assertRaises(ValueError):shards_per_sequence(c)
    def test_padding_preserves_pixels_and_physical_distance(self):
        t=transform_for('vggt');old=transform_for('dinov3')
        a=np.arange(512*512*3,dtype=np.uint8).reshape(512,512,3)
        mask=np.zeros((512,512),np.uint8);mask[100:300,100:300]=255
        with tempfile.TemporaryDirectory() as tmp:
            rgb=Path(tmp)/'rgb.png';mp=Path(tmp)/'mask.png';Image.fromarray(a).save(rgb);Image.fromarray(mask).save(mp)
            image=t.images([rgb])[0].permute(1,2,0).numpy()
            np.testing.assert_array_equal(image[3:-3,3:-3],a.astype(np.float32)/255)
            self.assertTrue((image[:3]==1).all())
            uv=np.array([[150.,150.],[200.,200.]])
            np.testing.assert_allclose(t.to_input(uv),uv+3)
            np.testing.assert_allclose(boundary_distances(t.mask(mp),t.to_input(uv)),boundary_distances(old.mask(mp),uv))
        # original UV 4,4 -> padded UV 7,7 -> first patch center.
        np.testing.assert_allclose(t.grid(np.array([[4.,4.],[18.,4.]]),(37,37)),[[0,0],[1,0]])
        lattice=torch.arange(37*37).reshape(1,37,37).float()
        np.testing.assert_allclose(sample_lattice(lattice,np.array([[4.,4.],[18.,4.]]),t).numpy()[:,0],[0,1],atol=1e-5)
        np.testing.assert_allclose(t.grid(np.array([[-3.,-3.],[515.,515.]]),(37,37)),[[-.5,-.5],[36.5,36.5]])
if __name__=='__main__':unittest.main()
