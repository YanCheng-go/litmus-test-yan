import numpy as np
from deforestation.thresholding import otsu_threshold, classify_forest


def test_otsu_returns_none_on_small_sample():
    x = np.array([[1., 2.]], dtype=float)
    assert otsu_threshold(x) is None


def test_classify_uses_percentile_fallback():
    x = np.array([[0., 1., 2., 3.]], dtype=float)
    valid = np.array([[True, True, True, True]])
    forest = classify_forest(x, valid_mask=valid, fallback_percentile=50)
    assert forest.sum() == 2  # values < median
