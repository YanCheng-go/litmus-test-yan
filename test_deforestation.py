# tests/test_deforestation.py
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

SCRIPT = Path(__file__).parent / "deforestation.py"

def _write_scene(path: Path, swir: np.ndarray, nir: np.ndarray, red: np.ndarray,
                 blue: np.ndarray, green: np.ndarray):
    """Write a 6-band GeoTIFF matching the band indices expected by the script:
       1: coastal (dummy), 2: blue, 3: green, 4: red, 5: nir, 6: swir.
    """
    H, W = swir.shape
    transform = from_origin(0, 0, 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        width=W, height=H,
        count=6,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform
    ) as dst:
        # band 1 (coastal) dummy
        dst.write(np.zeros((H, W), dtype=np.float32), 1)
        dst.write(blue.astype(np.float32), 2)
        dst.write(green.astype(np.float32), 3)
        dst.write(red.astype(np.float32), 4)
        dst.write(nir.astype(np.float32), 5)
        dst.write(swir.astype(np.float32), 6)


def _run_cli(tmpdir: Path, inputs_glob: str, outdir: Path, extra_args=None):
    args = [
        sys.executable, str(SCRIPT),
        "--inputs", inputs_glob,
        "--outdir", str(outdir),
        "--percentile-thr", "50",  # force percentile fallback (images are tiny)
        "--index", "swir_nir"      # default index used in tests
    ]
    if extra_args:
        args.extend(extra_args)
    res = subprocess.run(args, cwd=tmpdir, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"CLI failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res


def _make_grid_values():
    """Return swir values that produce two separable classes for 1.5 * (SWIR/NIR):
       forest index ~ 0.10  (SWIR ≈ 0.0667), non-forest index ~ 0.20 (SWIR ≈ 0.1333),
       and a 'very non-forest' index ~ 0.90 (SWIR = 0.6).
       NIR is 1.0 everywhere.
    """
    H, W = 3, 3
    nir = np.ones((H, W), dtype=np.float32)

    idx_forest = 0.10
    idx_nonforest = 0.20
    idx_very_nonforest = 0.90

    swir_forest = np.full((H, W), idx_nonforest / 1.5, dtype=np.float32)      # start as non-forest
    # carve a few forest pixels (lower index)
    swir_forest[0, 0] = idx_forest / 1.5
    swir_forest[1, 1] = idx_forest / 1.5
    swir_forest[2, 2] = idx_forest / 1.5
    swir_forest[0, 2] = idx_forest / 1.5

    # helper bands (not used for clouds/water in these tests)
    red = np.full_like(nir, 0.2, dtype=np.float32)
    blue = np.full_like(nir, 0.2, dtype=np.float32)
    green = np.full_like(nir, 0.2, dtype=np.float32)

    # an explicit 'very non-forest' swir value
    swir_very_nonforest = np.full_like(nir, idx_very_nonforest / 1.5, dtype=np.float32)

    return nir, red, blue, green, swir_forest, swir_very_nonforest


def test_cli_deforestation_basic_persistence0(tmp_path: Path):
    """
    With two time steps, persistence=0 should detect a forest->non-forest change
    at (1,1) and timestamp it with the second scene's stamp (20162).
    """
    d = tmp_path / "basic"
    d.mkdir()
    outdir = d / "out"

    nir, red, blue, green, swir_t0, swir_hi = _make_grid_values()

    # T0: 2016_1 (forest at (1,1))
    _write_scene(d / "scene_2016_1.tif", swir_t0, nir, red, blue, green)

    # T1: 2016_2 (turn (1,1) to very non-forest)
    swir_t1 = swir_t0.copy()
    swir_t1[1, 1] = swir_hi[1, 1]
    _write_scene(d / "scene_2016_2.tif", swir_t1, nir, red, blue, green)

    _run_cli(d, "scene_*.tif", outdir, extra_args=["--persistence", "0"])

    with rasterio.open(outdir / "deforestation_mask.tif") as ds:
        mask = ds.read(1)
        assert mask.dtype == np.uint8
        # only (1,1) deforested
        expected = np.zeros_like(mask, dtype=np.uint8)
        expected[1, 1] = 1
        np.testing.assert_array_equal(mask, expected)

    with rasterio.open(outdir / "deforestation_time.tif") as ds:
        tarr = ds.read(1)
        assert tarr.dtype == np.int32
        assert tarr[1, 1] == 20162  # year*10 + half
        assert int(tarr.sum()) == 20162  # only one pixel set


def test_cli_persistence1_requires_next_nonforest(tmp_path: Path):
    """
    With three time steps and persistence=1, change at T=1 should only be confirmed
    if T=2 remains non-forest.
    """
    d = tmp_path / "persists"
    d.mkdir()
    outdir = d / "out"

    nir, red, blue, green, swir_t0, swir_hi = _make_grid_values()

    # T0: 2016_1 (forest at (1,1))
    _write_scene(d / "scene_2016_1.tif", swir_t0, nir, red, blue, green)

    # T1: 2016_2 (deforest at (1,1))
    swir_t1 = swir_t0.copy()
    swir_t1[1, 1] = swir_hi[1, 1]
    _write_scene(d / "scene_2016_2.tif", swir_t1, nir, red, blue, green)

    # T2: 2017_1 (still non-forest at (1,1))
    swir_t2 = swir_t1.copy()
    _write_scene(d / "scene_2017_1.tif", swir_t2, nir, red, blue, green)

    _run_cli(d, "scene_*.tif", outdir, extra_args=["--persistence", "1"])

    with rasterio.open(outdir / "deforestation_mask.tif") as ds:
        mask = ds.read(1)
        expected = np.zeros_like(mask, dtype=np.uint8)
        expected[1, 1] = 1
        np.testing.assert_array_equal(mask, expected)

    with rasterio.open(outdir / "deforestation_time.tif") as ds:
        tarr = ds.read(1)
        assert tarr[1, 1] == 20162  # detected at the second scene
        # ensure only one detection
        assert np.count_nonzero(tarr) == 1


def test_cli_require_nonforest_until_end_blocks_regrowth(tmp_path: Path):
    """
    If the pixel regrows to forest later, '--require-nonforest-until-end'
    should prevent a detection.
    """
    d = tmp_path / "regrowth"
    d.mkdir()
    outdir = d / "out"

    nir, red, blue, green, swir_t0, swir_hi = _make_grid_values()

    # T0: 2016_1 (forest at (1,1))
    _write_scene(d / "scene_2016_1.tif", swir_t0, nir, red, blue, green)

    # T1: 2016_2 (deforest at (1,1))
    swir_t1 = swir_t0.copy()
    swir_t1[1, 1] = swir_hi[1, 1]
    _write_scene(d / "scene_2016_2.tif", swir_t1, nir, red, blue, green)

    # T2: 2017_1 (regrow: back to forest at (1,1))
    swir_t2 = swir_t0.copy()  # forest again
    _write_scene(d / "scene_2017_1.tif", swir_t2, nir, red, blue, green)

    _run_cli(
        d, "scene_*.tif", outdir,
        extra_args=["--persistence", "0", "--require-nonforest-until-end"]
    )

    with rasterio.open(outdir / "deforestation_mask.tif") as ds:
        mask = ds.read(1)
        assert np.count_nonzero(mask) == 0

    with rasterio.open(outdir / "deforestation_time.tif") as ds:
        tarr = ds.read(1)
        assert np.count_nonzero(tarr) == 0


def test_cli_writes_debug_forest_masks(tmp_path: Path):
    """
    When '--dump-forest-masks' is set, per-timestep masks should be written.
    """
    d = tmp_path / "debug"
    d.mkdir()
    outdir = d / "out"

    nir, red, blue, green, swir_t0, swir_hi = _make_grid_values()

    _write_scene(d / "scene_2016_1.tif", swir_t0, nir, red, blue, green)
    _write_scene(d / "scene_2016_2.tif", swir_t0, nir, red, blue, green)

    _run_cli(d, "scene_*.tif", outdir, extra_args=["--persistence", "0", "--dump-forest-masks"])

    m1 = outdir / "forest_mask_2016_1.tif"
    m2 = outdir / "forest_mask_2016_2.tif"
    assert m1.exists() and m2.exists()

    with rasterio.open(m1) as ds:
        arr = ds.read(1)
        assert arr.dtype == np.uint8
        # There must be at least one forest pixel (value 1)
        assert np.count_nonzero(arr) > 0

def test_cli_skips_invalid_middle_still_detects(tmp_path: Path):
    """
    t0=forest (valid), t1=invalid, t2=non-forest (valid) -> should detect at t2.
    """
    d = tmp_path / "skip_invalid"
    d.mkdir()
    outdir = d / "out"

    nir, red, blue, green, swir_t0, swir_hi = _make_grid_values()

    # T0: 2016_1 (forest at (1,1))
    _write_scene(d / "scene_2016_1.tif", swir_t0, nir, red, blue, green)

    # T1: 2016_2 (INVALID at (1,1): set NIR to NaN so valid mask is False there)
    nir_t1 = nir.copy()
    nir_t1[1, 1] = np.nan
    _write_scene(d / "scene_2016_2.tif", swir_t0, nir_t1, red, blue, green)

    # T2: 2017_1 (non-forest at (1,1))
    swir_t2 = swir_t0.copy()
    swir_t2[1, 1] = swir_hi[1, 1]
    _write_scene(d / "scene_2017_1.tif", swir_t2, nir, red, blue, green)

    _run_cli(d, "scene_*.tif", outdir, extra_args=["--persistence", "0"])

    with rasterio.open(outdir / "deforestation_mask.tif") as ds:
        mask = ds.read(1)
        assert mask[1, 1] == 1

    with rasterio.open(outdir / "deforestation_time.tif") as ds:
        tarr = ds.read(1)
        assert tarr[1, 1] == 20171  # detects at the third scene (2017_1)


def test_cli_persistence_counts_valid_only(tmp_path: Path):
    """
    t0=forest, t1=invalid, t2=non-forest, t3=non-forest; persistence=1 -> confirm at t2,
    because the next *valid* observation (t3) is also non-forest.
    """
    d = tmp_path / "persistence_valid_only"
    d.mkdir()
    outdir = d / "out"

    nir, red, blue, green, swir_t0, swir_hi = _make_grid_values()

    _write_scene(d / "scene_2016_1.tif", swir_t0, nir, red, blue, green)

    nir_t1 = nir.copy()
    nir_t1[1, 1] = np.nan  # invalid at t1
    _write_scene(d / "scene_2016_2.tif", swir_t0, nir_t1, red, blue, green)

    swir_t2 = swir_t0.copy()
    swir_t2[1, 1] = swir_hi[1, 1]  # non-forest
    _write_scene(d / "scene_2017_1.tif", swir_t2, nir, red, blue, green)

    swir_t3 = swir_t2.copy()  # stays non-forest
    _write_scene(d / "scene_2017_2.tif", swir_t3, nir, red, blue, green)

    _run_cli(d, "scene_*.tif", outdir, extra_args=["--persistence", "1"])

    with rasterio.open(outdir / "deforestation_mask.tif") as ds:
        mask = ds.read(1)
        assert mask[1, 1] == 1

    with rasterio.open(outdir / "deforestation_time.tif") as ds:
        tarr = ds.read(1)
        assert tarr[1, 1] == 20171  # detection anchored at the first valid non-forest

