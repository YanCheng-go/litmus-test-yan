from __future__ import annotations
import numpy as np

EPS = 1e-6


def swir_nir_index(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute 1.5 * SWIR / NIR, handling div-by-zero and invalids."""
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = 1.5 * (swir / (nir + EPS))
    idx[~np.isfinite(idx)] = np.nan
    return idx.astype("float32")


def nir_minus_red_index(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    idx = nir - red
    idx[~np.isfinite(idx)] = np.nan
    return idx.astype("float32")


def compute_index(nir: np.ndarray, red: np.ndarray, swir: np.ndarray, kind: str = "swir_nir") -> np.ndarray:
    if kind == "swir_nir":
        return swir_nir_index(swir, nir)
    if kind == "nir_minus_red":
        return nir_minus_red_index(nir, red)
    raise ValueError(f"Unknown index: {kind}")
