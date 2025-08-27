
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
from .types import TimeStamp

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
            filled_state[t][carry] = filled_state[t - 1][carry]
            filled_mask[t][carry] = True
        have_prev = have_prev | mask[t]
    return filled_state, filled_mask

def detect_deforestation(
    forest_stack: np.ndarray,
    timestamps: Sequence[TimeStamp],
    persistence: int = 1,
    valid_stack: Optional[np.ndarray] = None,
    require_nonforest_until_end: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    T, H, W = forest_stack.shape
    defo_mask = np.zeros((H, W), dtype=np.uint8)
    defo_time = np.zeros((H, W), dtype=np.int32)
    if valid_stack is None:
        valid_stack = np.ones_like(forest_stack, dtype=bool)

    forest_ff, _ = forward_fill_bounded_bool(forest_stack, valid_stack)

    for t in range(1, T):
        base = (forest_ff[t - 1] == True) & (forest_ff[t] == False) & valid_stack[t - 1] & valid_stack[t]
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
            stay_clear = np.ones_like(base, dtype=bool)
            for k in range(t, T):
                stay_clear &= (forest_ff[k] == False) & valid_stack[k]
            cand &= stay_clear
        newly_set = cand & (defo_mask == 0)
        if newly_set.any():
            defo_mask[newly_set] = 1
            defo_time[newly_set] = timestamps[t].encoded
    return defo_mask, defo_time
