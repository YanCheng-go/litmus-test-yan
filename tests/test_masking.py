import numpy as np
from deforestation.masking import build_water_mask, build_valid_masks


def test_build_water_mask():
    nir = np.array([[0.05, 0.5]])
    swir = np.array([[0.05, 0.5]])
    mask = build_water_mask(nir, swir)
    assert mask.shape == nir.shape
    assert mask[0,0] and not mask[0,1]


def test_build_valid_masks_shapes():
    z = np.ones((3,3), float)
    lists = ([z],[z],[z],[z],[z])
    valids = build_valid_masks(*lists, use_water=False, use_clouds=False)
    assert len(valids) == 1 and valids[0].shape == z.shape
