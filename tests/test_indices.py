
import numpy as np
from defor.indices import swir_nir_index, nir_minus_red_index

def test_swir_nir_basic():
    swir = np.array([[2.0, 0.0],[1.0, 3.0]], dtype=np.float32)
    nir  = np.array([[1.0, 2.0],[0.0, 3.0]], dtype=np.float32)
    idx = swir_nir_index(swir, nir)
    assert np.isfinite(idx[0,0]) and np.isfinite(idx[0,1])
    assert idx.shape == swir.shape

def test_nir_minus_red():
    nir = np.array([[0.5, 0.3]], dtype=np.float32)
    red = np.array([[0.2, 0.1]], dtype=np.float32)
    idx = nir_minus_red_index(nir, red)
    np.testing.assert_allclose(idx, np.array([[0.3, 0.2]], dtype=np.float32))
