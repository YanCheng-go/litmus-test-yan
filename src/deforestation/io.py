from __future__ import annotations
import glob
import os
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import rasterio

EPS = 1e-6
FILENAME_TIME_RE = re.compile(r'_(\d{4})[_-]([12])(?=[^.]*\.(?:tif|tiff)$)', re.IGNORECASE)


def discover_inputs(patterns: List[str]) -> List[str]:
    """Expand glob patterns and sort by (year, half) inferred from filenames."""
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit("No input files found.")

    def key(f: str):
        m = FILENAME_TIME_RE.search(os.path.basename(f))
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (9_999_999, f)

    return sorted(set(files), key=key)


def read_bands_align(
    paths: List[str],
    nir_idx: int,
    swir_idx: int,
    blue_idx: int,
    green_idx: int,
    red_idx: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict]:
    """Read required bands from each path. Assumes same grid/CRS across inputs."""
    nir_list: List[np.ndarray] = []
    swir_list: List[np.ndarray] = []
    blue_list: List[np.ndarray] = []
    green_list: List[np.ndarray] = []
    red_list: List[np.ndarray] = []
    profile: Optional[Dict] = None

    for p in paths:
        with rasterio.open(p) as ds:
            if profile is None:
                profile = ds.profile.copy()
                profile.update(count=1)
            def _read(idx: int) -> np.ndarray:
                arr = ds.read(idx).astype("float32")
                mask = ds.read_masks(idx) == 0
                arr[mask] = np.nan
                return arr
            nir_list.append(_read(nir_idx))
            swir_list.append(_read(swir_idx))
            blue_list.append(_read(blue_idx))
            green_list.append(_read(green_idx))
            red_list.append(_read(red_idx))
    assert profile is not None
    return nir_list, swir_list, blue_list, green_list, red_list, profile


def write_geotiff(path: str, profile: Dict, array: np.ndarray):
    """Write a single-band GeoTIFF with dtype inferred from the array."""
    prof = profile.copy()
    dtype = str(array.dtype)
    nodata = None
    if array.dtype == np.uint8:
        dtype = "uint8"; nodata = 0
    elif array.dtype == np.int32:
        dtype = "int32"; nodata = 0
    elif array.dtype == np.float32:
        dtype = "float32"; nodata = np.nan
    prof.update(dtype=dtype, nodata=nodata, count=1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array, 1)
