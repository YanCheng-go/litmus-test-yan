
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple, Optional
import math

import numpy as np
import rasterio
from rasterio.io import DatasetReader

from .types import Bands, TimeStamp

def discover_inputs(patterns: Sequence[str]) -> List[Path]:
    """Expand shell-style patterns and sort by (year, half) if encodable."""
    files: List[Path] = []
    for patt in patterns:
        files.extend(Path().glob(patt))
    files = list({f.resolve() for f in files})  # unique
    if not files:
        raise SystemExit("No input files found.")

    def sort_key(p: Path):
        ts = TimeStamp.from_path(p)
        return (ts.year, ts.half, p.name) if ts else (math.inf, 0, p.name)

    files.sort(key=sort_key)
    return files

def _single_band_profile(ds: DatasetReader) -> dict:
    prof = ds.profile.copy()
    prof.update(count=1, nodata=0)
    return prof

def _read_band(ds: DatasetReader, idx: int) -> np.ndarray:
    arr = ds.read(idx).astype("float32")
    mask = ds.read_masks(idx) == 0
    arr[mask] = np.nan
    return arr

def _ensure_contains_bands(ds: DatasetReader, bands: Bands) -> None:
    for idx in (bands.nir, bands.swir1, bands.blue, bands.green, bands.red):
        if not (1 <= idx <= ds.count):
            raise SystemExit(f"Band index {idx} not in dataset with {ds.count} bands: {ds.name}")

def read_bands_align(paths: Sequence[Path], bands: Bands) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], dict
]:
    """Read specified bands from each path. All rasters must be on same grid."""
    nir_list: List[np.ndarray] = []
    swir_list: List[np.ndarray] = []
    blue_list: List[np.ndarray] = []
    green_list: List[np.ndarray] = []
    red_list: List[np.ndarray] = []
    profile: Optional[dict] = None

    for p in paths:
        with rasterio.open(p) as ds:
            _ensure_contains_bands(ds, bands)
            if profile is None:
                profile = _single_band_profile(ds)
            nir = _read_band(ds, bands.nir)
            swir = _read_band(ds, bands.swir1)
            blue = _read_band(ds, bands.blue)
            green = _read_band(ds, bands.green)
            red = _read_band(ds, bands.red)
            nir_list.append(nir)
            swir_list.append(swir)
            blue_list.append(blue)
            green_list.append(green)
            red_list.append(red)

    assert profile is not None
    return nir_list, swir_list, blue_list, green_list, red_list, profile

def write_geotiff(path: Path, profile: dict, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    if array.dtype == np.uint8:
        prof.update(dtype="uint8", nodata=0)
    elif array.dtype == np.int32:
        prof.update(dtype="int32", nodata=0)
    else:
        prof.update(dtype=str(array.dtype))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array, 1)
