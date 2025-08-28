from __future__ import annotations
import argparse
from .logging_utils import setup_logger
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Basic deforestation detector (SWIR/NIR or NIR-Red).")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input satellite images (GeoTIFF). Supports globs.")
    ap.add_argument("--nir-band", type=int, default=5, help="1-based band index for NIR (L8 B5). Default: 5")
    ap.add_argument("--swir-band", type=int, default=6, help="1-based band index for SWIR (L8 B6). Default: 6")
    ap.add_argument("--blue-band", type=int, default=2, help="1-based band index for Blue (L8 B2). Default: 2")
    ap.add_argument("--green-band", type=int, default=3, help="1-based band index for Green (L8 B3). Default: 3")
    ap.add_argument("--red-band", type=int, default=4, help="1-based band index for Red (L8 B4). Default: 4")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory. Default: outputs")
    ap.add_argument("--persistence", type=int, default=0, help="Consecutive non-forest steps required to confirm. Default: 0")
    ap.add_argument("--dump-forest-masks", action="store_true", help="If set, dumps forest masks per time step.")
    ap.add_argument("--percentile-thr", type=float, default=50.0, help="Fallback percentile threshold if Otsu fails. Default: 50")
    ap.add_argument("--require-nonforest-until-end", action="store_true", help="Only mark deforestation if stays non-forest until last image.")
    ap.add_argument("--use-water-mask", action="store_true", help="Mask water per time step.")
    ap.add_argument("--use-cloud-mask", action="store_true", help="Mask clouds/shadows per time step.")
    ap.add_argument("--index", choices=["swir_nir", "nir_minus_red"], default="swir_nir", help="Index choice.")
    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logger()
    mask_path, time_path, unc_path = run_pipeline(
        inputs=args.inputs,
        nir_band=args.nir_band,
        swir_band=args.swir_band,
        blue_band=args.blue_band,
        green_band=args.green_band,
        red_band=args.red_band,
        outdir=args.outdir,
        persistence=args.persistence,
        dump_forest_masks=args.dump_forest_masks,
        percentile_thr=args.percentile_thr,
        require_nonforest_until_end=args.require_nonforest_until_end,
        use_water_mask=args.use_water_mask,
        use_cloud_mask=args.use_cloud_mask,
        index_kind=args.index,
    )
    logger.info("[OK] Outputs in: %s", args.outdir)
    logger.info("Mask: %s | Time: %s | Uncertainty: %s", mask_path, time_path, unc_path)


if __name__ == "__main__":
    main()
