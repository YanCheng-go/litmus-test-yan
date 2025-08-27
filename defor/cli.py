
from __future__ import annotations
from pathlib import Path
from typing import List
import logging
import typer

from .types import Bands, Options
from .pipeline import run_deforestation

logger = logging.getLogger("deforestation")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def main(
    inputs: List[str] = typer.Argument(..., help="One or more glob patterns for GeoTIFFs"),
    outdir: Path = typer.Option(Path("outputs"), help="Output directory"),
    # bands
    nir_band: int = typer.Option(5, help="1-based band index for NIR"),
    swir_band: int = typer.Option(6, help="1-based band index for SWIR1"),
    blue_band: int = typer.Option(2, help="1-based band index for Blue"),
    green_band: int = typer.Option(3, help="1-based band index for Green"),
    red_band: int = typer.Option(4, help="1-based band index for Red"),
    # algo
    index: str = typer.Option("swir_nir", help='Index choice: "swir_nir" or "nir_minus_red"'),
    percentile_thr: float = typer.Option(50.0, help="Fallback percentile if Otsu fails (0–100)"),
    persistence: int = typer.Option(0, help="Consecutive non-forest steps to confirm deforestation"),
    require_nonforest_until_end: bool = typer.Option(False, help="Only mark if stays non-forest until the last image"),
    use_water_mask: bool = typer.Option(False, help="Mask water per time step"),
    use_cloud_mask: bool = typer.Option(False, help="Mask clouds/shadows per time step"),
    dump_forest_masks: bool = typer.Option(False, help="Write forest masks per image for debugging"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    bands = Bands(nir=nir_band, swir1=swir_band, blue=blue_band, green=green_band, red=red_band)
    opts = Options(
        index=index,
        percentile_thr=percentile_thr,
        persistence=persistence,
        require_nonforest_until_end=require_nonforest_until_end,
        use_water_mask=use_water_mask,
        use_cloud_mask=use_cloud_mask,
    )
    stats = run_deforestation(inputs, outdir, bands, opts, dump_forest_masks)
    total = stats["total_pixels"]
    changed = stats["deforested_pixels"]
    pct = 100 * changed / max(total, 1)
    logger.info(f"Deforested pixels: {changed:,} / {total:,} ({pct:.3f}%)")
    logger.info(f"Outputs in: {stats['outdir']}")

def entrypoint():
    app()

if __name__ == "__main__":
    entrypoint()
