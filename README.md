# deforestation

Simple deforestation detection using swir/nir thresholding.

# Add badges showing the success of CI, Docker, and PyPI publish
[![CI](https://github.com/YanCheng-go/litmus-test-yan/actions/workflows/ci.yml/badge.svg)](https://github.com/YanCheng-go/litmus-test-yan/actions/workflows/ci.yml)
[![Docker](https://github.com/YanCheng-go/litmus-test-yan/actions/workflows/docker.yml/badge.svg)](https://github.com/YanCheng-go/litmus-test-yan/actions/workflows/docker.yml)
[![PyPI version](https://img.shields.io/pypi/v/litmus-test-yan.svg)](https://pypi.org/project/litmus-test-yan/)
[![PyPI publish](https://github.com/YanCheng-go/litmus-test-yan/actions/workflows/publish.yml/badge.svg)](https://github.com/YanCheng-go/litmus-test-yan/actions/workflows/publish.yml)

## Overview
- Load one or more satellite images (GeoTIFF format).
- Compute index = 1.5 * swir / nir (high = bare, low = forest).
- Thresholds per image to produce forest masks.
- Create cloud/water masks if requested.
- Detects first time a pixel transitions from forest to non-forest (with persistence).
- Post-processes the deforestation mask to remove small isolated areas (TO BE ADDED).
- Saves deforestation mask, time of deforestation, and gaps years (uncertainty layer).
- Outputs:
    - deforestation_mask.tif (unit8: 1 deforested, 0 otherwise)
    - deforestation_time.tif (int32: yyyyi half (1/2) encoded as 20161; 0=none)
    - deforestation_uncertainty_years.tif (float32: years of invalid-gap uncertainty)
    - forest_mask_<stamp>.tif for debugging when --dump-forest-masks is used
    - Logs to logs/deforestation.log if run from command line

## Requirements
- Python 3.9+
- Packages: numpy, rasterio, scikit-image, click, tqdm, matplotlib
- Optional: Docker
- Tested on Linux and MacOS (Windows untested)

## Install package

```bash
pip install .
```

## Download data
```bash
mkdir data
cd data
wget https://storage.googleapis.com/s11-litmustest/Peninsular_Malaysia_{2016..2017}_{1..2}_Landsat8.tif
```

## Test
```bash
pip install -e .[dev]
pytest tests
```

## Usage

```bash
deforest \
  --inputs "data/Peninsular_Malaysia_*.tif" \
  --nir-band 5 --swir-band 6 \
  --blue-band 2 --green-band 3 --red-band 4 \
  --outdir outputs \
  --persistence 0 \
  --dump-forest-masks \
  --percentile-thr 50
```

Command line arguments:

| Argument                        | Type                  | Default      | Description                                                                                  |
| ------------------------------- | --------------------- | ------------ | -------------------------------------------------------------------------------------------- |
| `--inputs`                      | list (glob supported) | **required** | Input satellite images (GeoTIFF format). Supports glob patterns like `data/*.tif`.           |
| `--nir-band`                    | int                   | 5            | 1-based band index for NIR (e.g., Landsat 8 Band 5).                                         |
| `--swir-band`                   | int                   | 6            | 1-based band index for SWIR (e.g., Landsat 8 Band 6).                                        |
| `--blue-band`                   | int                   | 2            | 1-based band index for Blue (e.g., Landsat 8 Band 2).                                        |
| `--green-band`                  | int                   | 3            | 1-based band index for Green (e.g., Landsat 8 Band 3).                                       |
| `--red-band`                    | int                   | 4            | 1-based band index for Red (e.g., Landsat 8 Band 4).                                         |
| `--outdir`                      | str                   | `outputs`    | Output directory where results will be saved.                                                |
| `--persistence`                 | int                   | 0            | Number of consecutive non-forest detections required to confirm deforestation.               |
| `--dump-forest-masks`           | flag                  | off          | If set, dumps per-timestep forest masks to files.                                            |
| `--percentile-thr`              | float                 | 50.0         | Fallback percentile threshold (0–100) if Otsu’s method fails.                                |
| `--require-nonforest-until-end` | flag                  | off          | Only mark deforestation if a pixel remains non-forest until the final image.                 |
| `--use-water-mask`              | flag                  | off          | Apply water masking at each timestep.                                                        |
| `--use-cloud-mask`              | flag                  | off          | Apply cloud/shadow masking at each timestep.                                                 |
| `--index`                       | choice                | `swir_nir`   | Index formula to use. Options: `swir_nir` (1.5 × SWIR / NIR) or `nir_minus_red` (NIR − Red). |

Outputs:
- `outputs/deforestation_mask.tif` (`uint8`: 1 deforested, 0 otherwise)
- `outputs/deforestation_time.tif` (`int32`: encoded as `year*10 + half`, e.g., `20161`)
- `outputs/deforestation_uncertainty_years.tif` (`float32` years of invalid-gap uncertainty)
- Optional `forest_mask_<stamp>.tif` when `--dump-forest-masks` is set

Notes:
- Supports two indices: `swir_nir` (1.5 * SWIR / NIR) and `nir_minus_red` (NIR - Red)
- Optional crude water/cloud/shadow masking per timestep

## Docker

Build the image:

```bash
docker build -t deforestation:latest .
```

Run the CLI (mount data & outputs):

```bash
docker run --rm \
  -v $PWD/data:/data \
  -v $PWD/outputs:/outputs \
  deforestation:latest \
  --inputs "/data/Peninsular_Malaysia_*.tif" \
  --nir-band 5 --swir-band 6 \
  --blue-band 2 --green-band 3 --red-band 4 \
  --outdir /outputs \
  --persistence 0 \
  --percentile-thr 50
```

## GitHub Actions

- **CI** runs tests on pushes/PRs to `main`/`master`.
- **Docker** images are pushed to GHCR when you push to `main` or tag (prefix `v`).
- **PyPI** publish runs on tags like `v0.1.0` if `PYPI_API_TOKEN` is set in repo secrets.

## TODOS
- [ ] Post-processing to remove small deforested patches and fill small holes, and remove non-forested areas using state-of-art land cover maps
- [ ] Add more robust cloud/water/shadow masking
- [ ] Scalability improvements
  - [ ] Parallel processing
  - [ ] Tiling for large images
- [ ] Edge cases
  - [ ] Handle missing timesteps
  - [ ] Handle different CRS/resolutions
- [ ] Test different indices or machine learning approaches