from __future__ import annotations
from typing import Optional
import numpy as np


def otsu_threshold(x: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Optional[float]:
    finite = np.isfinite(x)
    if valid_mask is not None:
        finite = finite & valid_mask
    v = x[finite]
    if v.size < 256:
        return None
    hist, bin_edges = np.histogram(v, bins=256)
    if hist.sum() == 0:
        return None

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    m = np.cumsum(hist * mids)
    mu1 = np.divide(m, w1, out=np.zeros_like(m, dtype=float), where=w1 > 0)
    mu2 = np.divide(m[-1] - m, w2, out=np.zeros_like(m, dtype=float), where=w2 > 0)
    between = (w1[:-1] * w2[1:] * (mu1[:-1] - mu2[1:]) ** 2)
    if not np.isfinite(between).any():
        return None
    k = int(np.argmax(between))
    return float((bin_edges[k] + bin_edges[k + 1]) / 2.0)


def classify_forest(index_img: np.ndarray, valid_mask: Optional[np.ndarray], fallback_percentile: float = 50.0) -> np.ndarray:
    thr = otsu_threshold(index_img, valid_mask=valid_mask)
    if valid_mask is None:
        valid_mask = np.isfinite(index_img)
    else:
        valid_mask = valid_mask & np.isfinite(index_img)
    if thr is None:
        vals = index_img[valid_mask]
        if vals.size == 0:
            return np.zeros_like(index_img, dtype=bool)
        thr = np.percentile(vals, fallback_percentile)
    forest = (index_img < thr) & valid_mask
    return forest
