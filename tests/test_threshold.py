
import numpy as np
from defor.threshold import otsu_threshold, classify_forest

def test_otsu_bimodal():
    a = np.hstack([np.full(500, 0.2), np.full(500, 0.8)]).astype(np.float32)
    thr = otsu_threshold(a.reshape(20,50))
    assert 0.3 < thr < 0.7

def test_classify_forest_percentile_fallback():
    img = np.array([[0.1, 0.2, 0.3, np.nan]], dtype=np.float32)
    mask = np.array([[True, True, True, False]])
    forest = classify_forest(img, valid_mask=mask, fallback_percentile=50.0)
    assert forest.dtype == bool
