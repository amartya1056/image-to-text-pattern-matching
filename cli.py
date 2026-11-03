"""Command-line interface for the Image-to-Text Pattern Matching Pipeline.

Examples
  python cli.py --input data/samples
  python cli.py --input data/samples/02_invoice.png --engine easyocr
  python cli.py --input data/samples --no-preprocess --modes exact regex
  python cli.py --input scan.jpg --threshold adaptive --resize 2 --fuzzy-threshold 85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ocrmatch import output
from ocrmatch.config import ConfigError, config_from_dict, load_config
from ocrmatch.io_utils import ImageLoadError
from ocrmatch.matching import MODES
from ocrmatch.ocr_engine import OCRError
from ocrmatch.pipeline import Pipeline, PipelineOptions
from ocrmatch.preprocessing import PreprocessOptions

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Extract text from images and locate predefined keywords, phrases and patterns.",
    )
    p.add_argument("-i", "--input", required=True, help="image file or folder of images")
    p.add_argument("-c", "--config", default=str(ROOT / "config" / "keywords.json"),
                   help="keywords/patterns file (.json or .yaml)")
    p.add_argument("-o", "--output", default=str(ROOT / "outputs"), help="output folder")
    p.add_argument("-r", "--recursive", action="store_true", help="search sub-folders too")
    p.add_argument("--keywords", help="extra comma-separated keywords to add to the config")

    g = p.add_argument_group("OCR")
    g.add_argument("--engine", choices=["tesseract", "easyocr"], default="tesseract")
    g.add_argument("--lang", default="eng", help="Tesseract language code(s), e.g. eng+hin")
    g.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
    g.add_argument("--no-retry", action="store_true",
                   help="do not retry when OCR confidence or text coverage is low")

    g = p.add_argument_group("Preprocessing (each step can be toggled)")
    g.add_argument("--no-preprocess", action="store_true", help="disable all preprocessing")
    g.add_argument("--no-grayscale", action="store_true")
    g.add_argument("--denoise", choices=["none", "median", "gaussian"], default="none")
    g.add_argument("--kernel", type=int, default=3, help="blur kernel size (odd)")
    g.add_argument("--threshold", choices=["none", "otsu", "adaptive"], default="none")
    g.add_argument("--no-deskew", action="store_true")
    g.add_argument("--resize", type=float, default=None, metavar="SCALE", help="upscale factor, e.g. 2")
    g.add_argument("--clahe", action="store_true", help="local contrast enhancement")

    g = p.add_argument_group("Matching")
    g.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    g.add_argument("--fuzzy-threshold", type=float, default=None, help="0-100, default from config")
    g.add_argument("--no-clean", action="store_true", help="skip OCR text cleaning")

    g = p.add_argument_group("Output")
    g.add_argument("--no-annotate", action="store_true", help="do not write annotated images")
    g.add_argument("--quiet", action="store_true", help="only print the batch summary")
    return p


def options_from_args(a) -> PipelineOptions:
    if a.no_preprocess:
        pre = PreprocessOptions.disabled()
    else:
        pre = PreprocessOptions(
            resize=a.resize is not None, scale=a.resize or 2.0,
            grayscale=not a.no_grayscale, contrast=a.clahe,
            denoise=a.denoise, kernel_size=a.kernel,
            deskew=not a.no_deskew, threshold=a.threshold,
        )
    return PipelineOptions(preprocess=pre, engine=a.engine, lang=a.lang, psm=a.psm,
                           clean=not a.no_clean, modes=tuple(a.modes),
                           fuzzy_threshold=a.fuzzy_threshold, auto_retry=not a.no_retry)


def print_result(r) -> None:
    print(f"\n=== {r.image_name}  ({r.image_size[0]}x{r.image_size[1]}) ===")
    print(f"  words: {r.word_count:<5} mean OCR confidence: {r.mean_confidence:5.1f}   "
          f"skew: {r.skew_angle:+.1f} deg   time: {r.timings['total']:.2f}s")
    for w in r.warnings:
        print(f"  ! {w}")
    if not r.matches:
        print("  (no matches)")
        return
    print(f"  {'TYPE':<6} {'KEYWORD / PATTERN':<26} {'MATCHED TEXT':<40} {'SCORE':>5} {'CONF':>5}")
    for m in r.matches:
        text = m.matched_text.replace("\n", " ")
        text = text if len(text) <= 40 else text[:37] + "..."
        conf = f"{m.ocr_confidence:.0f}" if m.ocr_confidence is not None else "-"
        print(f"  {m.match_type:<6} {m.keyword[:26]:<26} {text:<40} {m.score:>5.0f} {conf:>5}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.keywords:
            data = config.to_dict()
            data["keywords"] += [k.strip() for k in args.keywords.split(",") if k.strip()]
            config = config_from_dict(data)
    except (ConfigError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    pipeline = Pipeline(config, options_from_args(args))
    out_dir = Path(args.output)
    print(f"Pipeline: engine={args.engine}  preprocessing={'off' if args.no_preprocess else 'on'}  "
          f"modes={','.join(args.modes)}  terms={len(config.terms)}  patterns={len(config.patterns)}")

    def save(r):
        stem = Path(r.image_name).stem
        output.write_json(out_dir / "json" / f"{stem}.json", r.to_dict(include_words=True))
        if not args.no_annotate:
            output.save_image(out_dir / "annotated" / f"{stem}_annotated.png", r.annotated_image())
        if not args.quiet:
            print_result(r)

    try:
        results, failures = pipeline.process_batch(args.input, args.recursive, on_result=save)
    except (ImageLoadError, OCRError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = output.batch_summary(results, failures)
    output.write_csv(out_dir / "matches.csv", results)
    output.write_json(out_dir / "summary.json", summary)

    print("\n" + "=" * 72)
    print(f"BATCH SUMMARY: {summary['images_processed']} processed, {summary['images_failed']} failed, "
          f"{summary['total_matches']} matches")
    print("  by type: " + ", ".join(f"{k}={v}" for k, v in summary["matches_by_type"].items()))
    print(f"  mean OCR confidence: {summary['mean_ocr_confidence']:.1f}   "
          f"mean time/image: {summary['mean_time_per_image_s']:.2f}s")
    for f in failures:
        print(f"  skipped {f['file']}: {f['error']}")
    print(f"  results -> {out_dir / 'matches.csv'}, {out_dir / 'summary.json'}, {out_dir / 'json'}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
