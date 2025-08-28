# deforestation

Simple deforestation detection using NDVI-like thresholding.

## Install

```bash
pip install -e .
```

## Test
```bash
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