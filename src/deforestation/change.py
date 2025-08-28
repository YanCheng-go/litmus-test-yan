from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def forward_fill_bounded_bool(state: np.ndarray, mask: np.ndarray, max_gap: int = 99) -> Tuple[np.ndarray, np.ndarray]:
    T, H, W = state.shape
    filled_state = state.copy()
    filled_mask = np.zeros_like(state, dtype=bool)

    gap = np.zeros((H, W), dtype=np.int16)
    have_prev = mask[0].copy()

    for t in range(1, T):
        gap = np.where(mask[t], 0, np.where(have_prev, gap + 1, gap))
        carry = (~mask[t]) & have_prev & (gap <= max_gap)
        if carry.any():
            filled_state[t][carry] = filled_state[t-1][carry]
            filled_mask[t][carry] = True
        have_prev = have_prev | mask[t]

    return filled_state, filled_mask


def detect_deforestation(
    forest_stack: np.ndarray,
    years: List[int],
    halves: List[int],
    persistence: int = 0,
    valid_stack: Optional[np.ndarray] = None,
    require_nonforest_until_end: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, H, W = forest_stack.shape
    years = np.asarray(years, dtype=np.int32)
    halves = np.asarray(halves, dtype=np.int32)

    if valid_stack is None:
        valid_stack = np.ones_like(forest_stack, dtype=bool)

    orig_forest = forest_stack.copy()

    dt_years = np.zeros(T, dtype=np.float32)
    if T > 1:
        dy = years[1:] - years[:-1]
        dh = halves[1:] - halves[:-1]
        dt_years[1:] = dy + 0.5 * dh

    forest_ff, _ = forward_fill_bounded_bool(forest_stack, valid_stack)

    defo_mask = np.zeros((H, W), dtype=np.uint8)
    defo_time = np.zeros((H, W), dtype=np.int32)
    confirm_idx = -np.ones((H, W), dtype=np.int32)

    for t in range(1, T):
        base = (forest_ff[t-1] == True) & (forest_ff[t] == False)
        if not base.any():
            continue
        persists = np.ones_like(base, dtype=bool)
        for k in range(1, persistence + 1):
            if t + k >= T:
                persists &= False
            else:
                persists &= (forest_ff[t + k] == False) & valid_stack[t + k]
        cand = base & persists
        if require_nonforest_until_end:
            tail_ok = np.all((forest_ff[t:] == False) & valid_stack[t:], axis=0)
            cand &= tail_ok
        newly = cand & (defo_mask == 0)
        if newly.any():
            defo_mask[newly] = 1
            defo_time[newly] = (years[t] * 10 + halves[t]).astype(np.int32)
            confirm_idx[newly] = t

    closest_idx = -np.ones((H, W), dtype=np.int32)
    prev_valid_idx_at_closest = -np.ones((H, W), dtype=np.int32)

    last_valid_idx = -np.ones((H, W), dtype=np.int32)
    last_valid_state = np.zeros((H, W), dtype=bool)

    for t in range(T):
        curr_valid = valid_stack[t]
        curr_nf = curr_valid & (orig_forest[t] == False)
        ok = (
            (closest_idx < 0) &
            curr_nf &
            (last_valid_idx >= 0) &
            (last_valid_state == True) &
            (last_valid_idx < (t - 1))
        )
        if ok.any():
            closest_idx[ok] = t
            prev_valid_idx_at_closest[ok] = last_valid_idx[ok]
        if curr_valid.any():
            last_valid_idx = np.where(curr_valid, t, last_valid_idx)
            last_valid_state = np.where(curr_valid, orig_forest[t], last_valid_state)

    uncertainty_gap_years = np.zeros((H, W), dtype=np.float32)
    inv_dt = (~valid_stack).astype(np.float32) * dt_years.reshape(T, 1, 1)
    inv_cs = inv_dt.cumsum(axis=0)

    good = (
        (confirm_idx >= 0) &
        (closest_idx >= 0) &
        (prev_valid_idx_at_closest >= 0) &
        (closest_idx <= confirm_idx)
    )
    if good.any():
        idx_c = closest_idx.reshape(1, H, W)
        idx_p = prev_valid_idx_at_closest.reshape(1, H, W)
        cs_c = np.take_along_axis(inv_cs, idx_c, axis=0)[0]
        cs_p = np.take_along_axis(inv_cs, idx_p, axis=0)[0]
        gap = cs_c - cs_p
        gap = np.maximum(gap, 0.0)
        uncertainty_gap_years[good] = gap[good].astype(np.float32)

    return defo_mask, defo_time, uncertainty_gap_years
