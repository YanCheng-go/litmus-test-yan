"""Simple deforestation detection using NDVI thresholding.

- Load one or more satellite images (GeoTIFF format).
- Compute index = 1.5 * swir / nir (high = bare, low = forest).
- Thresholds per image to produce forest masks.
- Detects first time a pixel transitions from forest to non-forest (with persistence).
- Outputs:
    - deforestation_mask.tif (unit8: 1 deforested, 0 otherwise)
    - deforestation_time.tif (int32: yyyyi half (1/2) encoded as 20161; 0=none)
    - forest_mask_<stamp>.tif for debugging when --dump-forest-masks is used

Usage example (from project root, after downloading data):
    python deforestation.py \
        --inputs data/Peninsular_Malaysia_*.tif \
        --nir-band 5 --swir-band 6 \
        --blue-band 2 --green-band 3 --red-band 4 \
        --outdir outputs \
        --persistence 0 \
        --dump-forest-masks \
        --percentile-thr 50
"""

from __future__ import annotations
import argparse
import glob
import os
import re
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from utils import setup_logger

logger = setup_logger(logs_dir="./logs", name="deforestation")

# TODO: non data in the last time step was classified as non-forest...

EPS = 1e-6

FILENAME_TIME_RE = re.compile(
    r"(?:^|[/\\])[^/\\]*_(\d{4})_(\d)[^/\\]*\.(?:tif|tiff)$",
    re.IGNORECASE
)

BANDS = {
    1: 'coastal',
    2: 'blue',
    3: 'green',
    4: 'red',
    5: 'nir',
    6: 'swir1',
    7: 'swir2',
    8: 'thermal1',
    9: 'thermal2'
}

# TODO: use a dictionary to specify band indices per sensor (Landsat, Sentinel-2, etc.)
def parse_args():
    ap = argparse.ArgumentParser(description="Basic deforestation detector (SWIR/NIR).")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input satellite images (GeoTIFF).")
    ap.add_argument("--nir-band", type=int, default=5, help="1-based band index for NIR (L8 B5). Default: 5")
    ap.add_argument("--swir-band", type=int, default=6, help="1-based band index for SWIR (L8 B6). Default: 6")
    ap.add_argument("--blue-band", type=int, default=2, help="1-based band index for Blue (L8 B2). Default: 2")
    ap.add_argument("--green-band", type=int, default=3, help="1-based band index for Green (L8 B3). Default: 3")
    ap.add_argument("--red-band", type=int, default=4, help="1-based band index for Red (L8 B4). Default: 4")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory. Default: outputs")
    ap.add_argument("--persistence", type=int, default=0, help="Number of consecutive non-forest detections to confirm deforestation. Default: 1")
    ap.add_argument("--dump-forest-masks", action="store_true", help="If set, dumps forest masks")
    ap.add_argument("--percentile-thr", type=float, default=50.0, help="Fallback percentile threshold(0-100) if Otsu fails. Default: 50")
    ap.add_argument("--require-nonforest-until-end", action="store_true", help="Only mark deforestation if pixel stays non-forest until the last image.")
    ap.add_argument("--use-water-mask", action="store_true", help="Mask water per time step.")
    ap.add_argument("--use-cloud-mask", action="store_true", help="Mask clouds/shadows per time step.")
    ap.add_argument("--index", choices=["swir_nir", "nir_minus_red"], default="swir_nir", help="Index choice. Default: swir_nir (1.5*SWIR/NIR).")
    return ap.parse_args()

def discover_inputs(patterns: List[str]) -> List[str]:
    """Expand input patterns, sort by (year, half) if possible.
    Arguments:
        patterns: list of glob patterns
    Returns:
        sorted list of file paths
    Raises:
        SystemExit if no files found
    """
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit("No input files found.")
    # sort by (year, half); fallback to filename if pattern doesn't match
    def key(f):
        m = FILENAME_TIME_RE.match(os.path.basename(f))
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (9_999_999, f)
    files = sorted(set(files), key=key)
    return files

# TODO: add resampling/reprojection to common grid if needed
def read_bands_align(paths: List[str], nir_idx: int, swir_idx: int, blue_idx: int, green_idx: int, red_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], dict]:
    """Read NIR and SWIR bands for each path. Returns lists of 2D arrays (NIRs, SWIRs) and a common profile.
    Arguments:
        paths: list of file paths
        nir_idx: 1-based band index for NIR
        swir_idx: 1-based band index for SWIR
        blue_idx: 1-based band index for Blue
        green_idx: 1-based band index for Green
        red_idx: 1-based band index for Red
    Raises:
        SystemExit if any file cannot be read
    Returns:
        (nir_list, swir_list, blue_list, green_list, red_list, profile)
    """
    nir_list, swir_list, blue_list, green_list, red_list = [], [], [], [], []
    profile = None

    # TODO: Use multithreading for faster I/O
    for p in paths:
        with rasterio.open(p) as ds:
            if profile is None:
                profile = ds.profile.copy()
                # keep single-band profile for outputs
                profile.update(count=1, dtype="uint8")
            # read bands (1-based in rasterio)
            nir = ds.read(nir_idx).astype("float32")
            swir = ds.read(swir_idx).astype("float32")
            blue = ds.read(blue_idx).astype("float32")
            green = ds.read(green_idx).astype("float32")
            red = ds.read(red_idx).astype("float32")
            # apply masks/nodata as NaN
            mask = ds.read_masks(nir_idx) == 0
            nir[mask] = np.nan
            mask2 = ds.read_masks(swir_idx) == 0
            swir[mask2] = np.nan
            mask3 = ds.read_masks(blue_idx) == 0
            blue[mask3] = np.nan
            mask4 = ds.read_masks(green_idx) == 0
            green[mask4] = np.nan
            mask5 = ds.read_masks(red_idx) == 0
            red[mask5] = np.nan
            blue_list.append(blue)
            green_list.append(green)
            red_list.append(red)
            nir_list.append(nir)
            swir_list.append(swir)
    return nir_list, swir_list, blue_list, green_list, red_list, profile

def swir_nir_index(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute 1.5 * SWIR / NIR safely. 0/0 = NaN; x/0 = inf; -x/0 = -inf
    Arguments:
        swir: 2D array
        nir: 2D array
    Returns:
        2D array of float32
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = 1.5 * (swir / (nir + EPS))
    idx[~np.isfinite(idx)] = np.nan
    return idx.astype("float32")

def nir_minus_red_index(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NIR - Red.
    Arguments:
        nir: 2D array
        red: 2D array
    Returns:
        2D array of float32
    """
    idx = nir - red
    idx[~np.isfinite(idx)] = np.nan
    return idx.astype("float32")

def compute_index(nir: np.ndarray, red: np.ndarray, swir: np.ndarray, kind: str = "swir_nir") -> np.ndarray:
    """Compute the specified index.
    Arguments:
        index_name: "swir_nir" or "nir_minus_red
        swir: 2D array
        nir: 2D array
        red: 2D array (only for "nir_minus_red")
    Returns:
        2D array of float32
    Raises:
        ValueError if unknown index
    """
    if kind == "swir_nir":
        return swir_nir_index(swir, nir)
    elif kind == "nir_minus_red":
        return nir_minus_red_index(nir, red)
    else:
        raise ValueError(f"Unknown index: {kind}")

def otsu_threshold(x: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Simple Otsu on finite values; returns None if not enough valid data.
    Arguments:
        x: 2D array
        valid_mask: optional boolean mask to further filter valid pixels
    Returns:
        threshold value or None
    """
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
    k = np.argmax(between)
    return float((bin_edges[k] + bin_edges[k + 1]) / 2.0)

def classify_forest(index_img: np.ndarray, valid_mask: Optional[np.ndarray], fallback_percentile: float = 50.0) -> np.ndarray:
    """Return boolean mask: True = forest (low index).
    Arguments:
        index_img: 2D array
        valid_mask: optional boolean mask to filter valid pixels
        fallback_percentile: percentile (0-100) to use if Otsu fails
    Returns:
        boolean mask
    """
    thr = otsu_threshold(index_img, valid_mask=valid_mask)
    finite_valid = np.isfinite(index_img) & (valid_mask if valid_mask is not None else True)
    if thr is None:
        vals = index_img[finite_valid]
        if vals.size == 0:
            return np.zeros_like(index_img, dtype=bool)
        thr = np.percentile(vals, fallback_percentile)
    forest = (index_img < thr) & finite_valid
    return forest

# TODO: Better water/cloud masking
def build_water_mask(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Simple water mask: NIR < 0.1 and SWIR < 0.1 (scaled 0-1).
    Arguments:
        nir: 2D array
        swir: 2D array
    Returns:
        boolean mask: True = water
    """
    water_any = (nir < 0.1) & (swir < 0.1) & np.isfinite(nir) & np.isfinite(swir)
    return water_any

def build_cloud_masks(blue_list, green_list, red_list, swir1_list,
                      vis_thr=0.30, swir_thr=0.20):
    """Boolean cloud mask per image using a simple spectral rule."""
    masks = []
    for b, g, r, s1 in zip(blue_list, green_list, red_list, swir1_list):
        vis = np.nanmean(np.stack([b, g, r], 0), axis=0)
        cloud = (vis > vis_thr) & (b > g) & (s1 > swir_thr)
        masks.append(cloud)
    return masks

def build_shadow_masks(blue_list, green_list, red_list, nir_list, swir1_list,
                       dark_vis_thr=0.12, dark_nir_thr=0.12, dark_swir_thr=0.10):
    """Very crude cloud-shadow mask (dark in VIS+NIR+SWIR)."""
    masks = []
    for b, g, r, n, s1 in zip(blue_list, green_list, red_list, nir_list, swir1_list):
        vis = np.nanmean(np.stack([b, g, r], 0), axis=0)
        shadow = (vis < dark_vis_thr) & (n < dark_nir_thr) & (s1 < dark_swir_thr)
        masks.append(shadow)
    return masks

def build_valid_masks(nir_list, swir_list, blue_list, green_list, red_list,
                      use_water=False, use_clouds=False) -> List[np.ndarray]:
    """Build valid data masks per image, optionally masking water and clouds.
    Arguments:
        nir_list: list of 2D arrays
        swir_list: list of 2D arrays
        blue_list: list of 2D arrays
        green_list: list of 2D arrays
        red_list: list of 2D arrays
        use_water: if True, mask water
        use_clouds: if True, mask clouds/shadows
    Returns:
        list of boolean masks: True = valid data
    """
    valids = []
    water_masks = [np.zeros_like(nir_list[0], dtype=bool)] * len(nir_list)
    cloud_masks = [np.zeros_like(nir_list[0], dtype=bool)] * len(nir_list)
    shadow_masks = [np.zeros_like(nir_list[0], dtype=bool)] * len(nir_list)

    if use_water:
        water_masks = [build_water_mask(n, s) for n, s in zip(nir_list, swir_list)]

    if use_clouds:
        cloud_masks = build_cloud_masks(blue_list, green_list, red_list, swir_list)
        shadow_masks = build_shadow_masks(blue_list, green_list, red_list, nir_list, swir_list)

    for i, (n, s, b, g, r, w, c, sh) in enumerate(zip(nir_list, swir_list, blue_list, green_list, red_list, water_masks, cloud_masks, shadow_masks)):
        finite = np.isfinite(n) & np.isfinite(s)
        # you can choose to also require visible bands finite if you like:
        # finite &= np.isfinite(b) & np.isfinite(g) & np.isfinite(r)
        bad = w | c | sh
        valids.append(finite & (~bad))
    return valids

def detect_deforestation(forest_stack: np.ndarray, years: List[int], halves: List[int], persistence: int = 1, valid_stack: Optional[np.ndarray] = None, require_nonforest_until_end: bool = False,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect first time of deforestation (forest -> non-forest) with optional persistence.
    Arguments:
        forest_stack: [T,H,W] boolean array (True=forest)
        years: list of years per time step
        halves: list of halves (1/2) per time step
        persistence: number of consecutive non-forest steps to confirm deforestation
        valid_stack: [T,H,W] boolean array (True=valid data)
        require_nonforest_until_end: if True, deforestation is confirmed only if the pixel remains non-forest until the last time step
    Returns:
        defo_mask: [H,W] uint8 (1=deforested, 0=otherwise)
        defo_time: [H,W] int32 (yyyyi half encoded as 20161; 0=none)
    Notes:
        - If exclude is given, those pixels are never marked as deforested.
        - If a pixel is forest in the last time step, it is never marked as deforested.
    """
    T, H, W = forest_stack.shape
    defo_mask = np.zeros((H, W), dtype=np.uint8)
    defo_time = np.zeros((H, W), dtype=np.int32)

    if valid_stack is None:
        valid_stack = np.ones_like(forest_stack, dtype=bool)

    for t in range(1, T):
        # must be valid at t-1 and t
        base = (forest_stack[t-1] == True) & (forest_stack[t] == False) & valid_stack[t-1] & valid_stack[t]
        if not base.any():
            continue

        # persistence over the following steps
        persists = np.ones_like(base, dtype=bool)
        for k in range(1, persistence + 1):
            if t + k >= T:
                persists &= False  # cannot confirm persistence beyond series
            else:
                persists &= (forest_stack[t + k] == False) & valid_stack[t + k]

        cand = base & persists

        if require_nonforest_until_end:
            stay_clear = np.ones_like(base, dtype=bool)
            for k in range(t, T):
                stay_clear &= (forest_stack[k] == False) & valid_stack[k]
            cand &= stay_clear

        newly_set = cand & (defo_mask == 0)
        if newly_set.any():
            defo_mask[newly_set] = 1
            stamp = years[t] * 10 + halves[t]
            defo_time[newly_set] = stamp

    return defo_mask, defo_time

def write_geotiff(path: str, profile: dict, array: np.ndarray):
    """Write single-band GeoTIFF with given profile and array.
    Arguments:
        path: output file path
        profile: rasterio profile dict
        array: 2D array to write
    Notes:
        - profile is updated to match array dtype and count=1.
        - Creates parent directories if needed.
    """
    prof = profile.copy()
    # dtype & nodata by array type
    if array.dtype == np.uint8:
        prof.update(dtype="uint8", nodata=0, count=1)
    elif array.dtype == np.int32:
        prof.update(dtype="int32", nodata=0, count=1)
    else:
        prof.update(dtype=str(array.dtype), count=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array, 1)

def main():
    args = parse_args()
    paths = discover_inputs(args.inputs)
    if len(paths) < 2:
        raise SystemExit("Need at least 2 time steps to detect change.")

    # parse time ordering
    years, halves = [], []
    for p in paths:
        m = FILENAME_TIME_RE.match(os.path.basename(p))
        if m:
            years.append(int(m.group(1)))
            halves.append(int(m.group(2)))
        else:
            # fallback: monotonic increasing
            years.append(2000)
            halves.append(len(halves) + 1)

    nir_list, swir_list, blue_list, green_list, red_list, profile = read_bands_align(
        paths,
        nir_idx=args.nir_band,
        swir_idx=args.swir_band,
        blue_idx=args.blue_band,
        green_idx=args.green_band,
        red_idx=args.red_band
    )

    valid_list = build_valid_masks(
        nir_list, swir_list, blue_list, green_list, red_list,
        use_water=args.use_water_mask, use_clouds=args.use_cloud_mask
    )

    forest_masks = []
    for i, (nir, swir, red, valid) in enumerate(zip(nir_list, swir_list, red_list, valid_list)):
        idx = compute_index(nir, red, swir, kind=args.index)

        # save into a new raster
        os.makedirs('./tmp', exist_ok=True)
        with rasterio.open(f"./tmp/{i}.tif", "w", driver="GTiff", height=idx.shape[0], width=idx.shape[1], count=1,
                           dtype="float32", crs="+proj=latlong",
                           transform=rasterio.transform.from_origin(0, 0, 1, 1), ) as dst:
            dst.write(idx.astype("float32"), 1)

        forest = classify_forest(idx, valid_mask=valid, fallback_percentile=args.percentile_thr)
        forest_masks.append(forest)
        if args.dump_forest_masks:
            out = os.path.join(args.outdir, f"forest_mask_{years[i]}_{halves[i]}.tif")
            write_geotiff(out, profile, forest.astype("uint8"))

    forest_stack = np.stack(forest_masks, axis=0).astype(bool)
    valid_stack = np.stack(valid_list, axis=0).astype(bool)

    defo_mask, defo_time = detect_deforestation(
        forest_stack, years, halves,
        persistence=args.persistence,
        valid_stack=valid_stack,
        require_nonforest_until_end=args.require_nonforest_until_end
    )

    write_geotiff(os.path.join(args.outdir, "deforestation_mask.tif"), profile, defo_mask)
    write_geotiff(os.path.join(args.outdir, "deforestation_time.tif"), profile, defo_time)

    # quick stats
    changed = int(defo_mask.sum())
    total = int(np.isfinite(nir_list[0]).sum())
    print(f"[OK] Deforested pixels: {changed:,} / {total:,} ({100*changed/max(total,1):.3f}%)")
    print(f"[OK] Outputs in: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()