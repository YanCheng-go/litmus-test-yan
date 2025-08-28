from __future__ import annotations
from typing import List
import numpy as np


def build_water_mask(nir: np.ndarray, swir: np.ndarray, thr: float = 0.1) -> np.ndarray:
    return (nir < thr) & (swir < thr) & np.isfinite(nir) & np.isfinite(swir)


def build_cloud_masks(blue_list, green_list, red_list, swir1_list, vis_thr=0.30, swir_thr=0.20):
    masks = []
    for b, g, r, s1 in zip(blue_list, green_list, red_list, swir1_list):
        vis = np.nanmean(np.stack([b, g, r], 0), axis=0)
        cloud = (vis > vis_thr) & (b > g) & (s1 > swir_thr)
        masks.append(cloud)
    return masks


def build_shadow_masks(blue_list, green_list, red_list, nir_list, swir1_list,
                       dark_vis_thr=0.12, dark_nir_thr=0.12, dark_swir_thr=0.10):
    masks = []
    for b, g, r, n, s1 in zip(blue_list, green_list, red_list, nir_list, swir1_list):
        vis = np.nanmean(np.stack([b, g, r], 0), axis=0)
        shadow = (vis < dark_vis_thr) & (n < dark_nir_thr) & (s1 < dark_swir_thr)
        masks.append(shadow)
    return masks


def build_valid_masks(nir_list, swir_list, blue_list, green_list, red_list,
                      use_water=False, use_clouds=False) -> List[np.ndarray]:
    valids = []
    water_masks = [np.zeros_like(nir_list[0], dtype=bool)] * len(nir_list)
    cloud_masks = [np.zeros_like(nir_list[0], dtype=bool)] * len(nir_list)
    shadow_masks = [np.zeros_like(nir_list[0], dtype=bool)] * len(nir_list)

    if use_water:
        water_masks = [build_water_mask(n, s) for n, s in zip(nir_list, swir_list)]

    if use_clouds:
        cloud_masks = build_cloud_masks(blue_list, green_list, red_list, swir_list)
        shadow_masks = build_shadow_masks(blue_list, green_list, red_list, nir_list, swir_list)

    for n, s, b, g, r, w, c, sh in zip(nir_list, swir_list, blue_list, green_list, red_list, water_masks, cloud_masks, shadow_masks):
        finite = np.isfinite(n) & np.isfinite(s)
        bad = w | c | sh
        valids.append(finite & (~bad))
    return valids
