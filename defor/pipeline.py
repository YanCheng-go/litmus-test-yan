
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np

from .types import Bands, Options, TimeStamp
from .io import discover_inputs, read_bands_align, write_geotiff
from .indices import compute_index
from .masks import build_valid_masks
from .threshold import classify_forest
from .change import detect_deforestation

def run_deforestation(inputs: list[str], outdir: Path, bands: Bands, opts: Options, dump_forest_masks: bool = False) -> dict:
    paths = discover_inputs(inputs)
    if len(paths) < 2:
        raise SystemExit("Need at least 2 time steps to detect change.")

    timestamps: List[TimeStamp] = []
    fallback_year = 2000
    half_counter = 0
    for p in paths:
        ts = TimeStamp.from_path(p)
        if ts is None:
            half_counter += 1
            ts = TimeStamp(fallback_year, 1 if (half_counter % 2) else 2)
        timestamps.append(ts)

    nir_list, swir_list, blue_list, green_list, red_list, profile = read_bands_align(paths, bands)
    valid_list = build_valid_masks(nir_list, swir_list, blue_list, green_list, red_list,
                                   use_water=opts.use_water_mask, use_clouds=opts.use_cloud_mask)

    forest_masks = []
    for nir, swir, red, valid in zip(nir_list, swir_list, red_list, valid_list):
        idx = compute_index(nir, red, swir, kind=opts.index)
        forest_masks.append(classify_forest(idx, valid_mask=valid, fallback_percentile=opts.percentile_thr))

    forest_stack = np.stack(forest_masks, axis=0).astype(bool)
    valid_stack = np.stack(valid_list, axis=0).astype(bool)

    defo_mask, defo_time = detect_deforestation(
        forest_stack, timestamps, persistence=opts.persistence, valid_stack=valid_stack,
        require_nonforest_until_end=opts.require_nonforest_until_end
    )

    outdir.mkdir(parents=True, exist_ok=True)
    write_geotiff(outdir / "deforestation_mask.tif", profile, defo_mask)
    write_geotiff(outdir / "deforestation_time.tif", profile, defo_time)

    stats = {
        "deforested_pixels": int(defo_mask.sum()),
        "total_pixels": int(np.isfinite(nir_list[0]).sum()),
        "outdir": str(outdir.resolve()),
    }
    return stats
