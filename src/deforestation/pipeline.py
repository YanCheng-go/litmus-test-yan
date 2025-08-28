from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np

from .io import discover_inputs, read_bands_align, write_geotiff, FILENAME_TIME_RE
from .indices import compute_index
from .thresholding import classify_forest
from .masking import build_valid_masks
from .change import detect_deforestation


def run_pipeline(
    inputs: List[str],
    nir_band: int,
    swir_band: int,
    blue_band: int,
    green_band: int,
    red_band: int,
    outdir: str,
    persistence: int,
    dump_forest_masks: bool,
    percentile_thr: float,
    require_nonforest_until_end: bool,
    use_water_mask: bool,
    use_cloud_mask: bool,
    index_kind: str,
) -> Tuple[str, str, str]:
    paths = discover_inputs(inputs)
    if len(paths) < 2:
        raise SystemExit("Need at least 2 time steps to detect change.")

    years, halves = [], []
    for p in paths:
        base = os.path.basename(p)
        m = FILENAME_TIME_RE.search(base)
        if m:
            years.append(int(m.group(1)))
            halves.append(int(m.group(2)))
        else:
            years.append(2000)
            halves.append(len(halves) + 1)

    nir_list, swir_list, blue_list, green_list, red_list, profile = read_bands_align(
        paths, nir_band, swir_band, blue_band, green_band, red_band
    )

    valid_list = build_valid_masks(
        nir_list, swir_list, blue_list, green_list, red_list,
        use_water=use_water_mask, use_clouds=use_cloud_mask
    )

    forest_masks = []
    for i, (nir, swir, red, valid) in enumerate(zip(nir_list, swir_list, red_list, valid_list)):
        idx = compute_index(nir, red, swir, kind=index_kind)
        forest = classify_forest(idx, valid_mask=valid, fallback_percentile=percentile_thr)
        forest_masks.append(forest)
        if dump_forest_masks:
            os.makedirs(outdir, exist_ok=True)
            write_geotiff(os.path.join(outdir, f"forest_mask_{years[i]}_{halves[i]}.tif"), profile, forest.astype("uint8"))

    forest_stack = np.stack(forest_masks, axis=0).astype(bool)
    valid_stack = np.stack(valid_list, axis=0).astype(bool)

    defo_mask, defo_time, uncertainty_gap_years = detect_deforestation(
        forest_stack, years, halves,
        persistence=persistence,
        valid_stack=valid_stack,
        require_nonforest_until_end=require_nonforest_until_end,
    )

    os.makedirs(outdir, exist_ok=True)
    mask_path = os.path.join(outdir, "deforestation_mask.tif")
    time_path = os.path.join(outdir, "deforestation_time.tif")
    unc_path  = os.path.join(outdir, "deforestation_uncertainty_years.tif")

    write_geotiff(mask_path, profile, defo_mask)
    write_geotiff(time_path, profile, defo_time)
    write_geotiff(unc_path, profile, uncertainty_gap_years.astype("float32"))

    # quick stats
    changed = int(defo_mask.sum())
    total = int(np.isfinite(nir_list[0]).sum())
    print(f"[OK] Deforested pixels: {changed:,} / {total:,} ({100*changed/max(total,1):.3f}%)")

    return mask_path, time_path, unc_path
