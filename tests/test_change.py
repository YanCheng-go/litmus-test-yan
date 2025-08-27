
import numpy as np
from defor.change import detect_deforestation
from defor.types import TimeStamp

def test_detect_deforestation_simple():
    # T=3, H=W=2
    forest_stack = np.array([
        [[1,1],[1,1]],
        [[0,1],[1,0]],
        [[0,0],[1,0]],
    ], dtype=bool)
    ts = [TimeStamp(2020,1), TimeStamp(2020,2), TimeStamp(2021,1)]
    valid = np.ones_like(forest_stack, dtype=bool)
    m,t = detect_deforestation(forest_stack, ts, persistence=1, valid_stack=valid, require_nonforest_until_end=False)
    assert m.sum() >= 1
    assert t.dtype == np.int32
