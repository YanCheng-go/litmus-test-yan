import numpy as np
from deforestation.indices import swir_nir_index, nir_minus_red_index, compute_index


def test_swir_nir_index_basic():
    swir = np.array([[2., 0.], [1., 3.]], dtype=float)
    nir  = np.array([[1., 0.], [2., 0.]], dtype=float)
    out = swir_nir_index(swir, nir)
    assert out.shape == swir.shape
    assert np.isfinite(out[0,0]) and np.all(np.isfinite(out[np.isfinite(out)]))


def test_nir_minus_red_index_basic():
    nir = np.array([[0.6, 0.4]], dtype=float)
    red = np.array([[0.1, 0.5]], dtype=float)
    out = nir_minus_red_index(nir, red)
    np.testing.assert_allclose(out, np.array([[0.5, -0.1]], dtype=np.float32))


def test_compute_index_switch():
    nir = np.ones((2,2), float)
    red = np.zeros((2,2), float)
    swir = np.ones((2,2), float)
    assert compute_index(nir, red, swir, "nir_minus_red").dtype == np.float32
