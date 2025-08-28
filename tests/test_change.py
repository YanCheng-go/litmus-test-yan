import numpy as np
from deforestation.change import forward_fill_bounded_bool, detect_deforestation


def test_forward_fill():
    state = np.array([
        [[True, True]],
        [[True, True]],
        [[False, True]],
        [[False, True]],
    ])
    valid = np.array([
        [[True, True]],
        [[False, True]],
        [[False, True]],
        [[True, True]],
    ])
    ff, filled = forward_fill_bounded_bool(state, valid, max_gap=5)
    assert ff.shape == state.shape
    assert filled.any()


def test_detect_deforestation_simple():
    # 4 timesteps, one pixel becomes non-forest at t=2 and remains
    forest = np.array([
        [[True]],
        [[True]],
        [[False]],
        [[False]],
    ])
    years = [2016, 2016, 2017, 2017]
    halves = [1, 2, 1, 2]
    valid = np.ones_like(forest, dtype=bool)
    m, t, u = detect_deforestation(forest, years, halves, persistence=0, valid_stack=valid)
    assert m.dtype == np.uint8 and m[0,0] == 1
    assert t[0,0] == 20171  # first non-forest time encoded as year*10 + half
    assert u[0,0] >= 0.0
