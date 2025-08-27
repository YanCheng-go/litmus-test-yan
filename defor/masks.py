
from __future__ import annotations
from typing import List, Sequence
import numpy as np

def build_water_mask(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    return (nir < 0.1) & (swir < 0.1) & np.isfinite(nir) & np.isfinite(swir)

def build_cloud_masks(
    blue_list: Sequence[np.ndarray],
    green_list: Sequence[np.ndarray],
    red_list: Sequence[np.ndarray],
    swir1_list: Sequence[np.ndarray],
    vis_thr: float = 0.30,
    swir_thr: float = 0.20,
) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for b, g, r, s1 in zip(blue_list, green_list, red_list, swir1_list):
        vis = np.nanmean(np.stack([b, g, r], 0), axis=0)
        cloud = (vis > vis_thr) & (b > g) & (s1 > swir_thr)
        masks.append(cloud)
    return masks

def build_shadow_masks(
    blue_list: Sequence[np.ndarray],
    green_list: Sequence[np.ndarray],
    red_list: Sequence[np.ndarray],
    nir_list: Sequence[np.ndarray],
    swir1_list: Sequence[np.ndarray],
    dark_vis_thr: float = 0.12,
    dark_nir_thr: float = 0.12,
    dark_swir_thr: float = 0.10,
) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for b, g, r, n, s1 in zip(blue_list, green_list, red_list, nir_list, swir1_list):
        vis = np.nanmean(np.stack([b, g, r], 0), axis=0)
        shadow = (vis < dark_vis_thr) & (n < dark_nir_thr) & (s1 < dark_swir_thr)
        masks.append(shadow)
    return masks

def build_valid_masks(
    nir_list: Sequence[np.ndarray],
    swir_list: Sequence[np.ndarray],
    blue_list: Sequence[np.ndarray],
    green_list: Sequence[np.ndarray],
    red_list: Sequence[np.ndarray],
    use_water: bool = False,
    use_clouds: bool = False,
) -> List[np.ndarray]:
    T = len(nir_list)
    water_masks: List[np.ndarray] = [np.zeros_like(nir_list[0], dtype=bool) for _ in range(T)]
    cloud_masks: List[np.ndarray] = [np.zeros_like(nir_list[0], dtype=bool) for _ in range(T)]
    shadow_masks: List[np.ndarray] = [np.zeros_like(nir_list[0], dtype=bool) for _ in range(T)]

    if use_water:
        water_masks = [build_water_mask(n, s) for n, s in zip(nir_list, swir_list)]
    if use_clouds:
        cloud_masks = build_cloud_masks(blue_list, green_list, red_list, swir_list)
        shadow_masks = build_shadow_masks(blue_list, green_list, red_list, nir_list, swir_list)

    valids: List[np.ndarray] = []
    for n, s, w, c, sh in zip(nir_list, swir_list, water_masks, cloud_masks, shadow_masks):
        finite = np.isfinite(n) & np.isfinite(s)
        bad = w | c | sh
        valids.append(finite & (~bad))
    return valids
