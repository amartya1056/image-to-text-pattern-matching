"""Evaluation of the pipeline against the ground truth in data/ground_truth.json.

Measures
  * OCR quality      - character accuracy and word accuracy (1 - normalised
                       edit distance), per image and per configuration
  * keyword detection - precision / recall / F1 of keyword & phrase hits
                       for exact vs. exact+fuzzy matching, and a sweep of the
                       fuzzy threshold
  * regex extraction - precision / recall / F1 of extracted pattern values
  * cleaning         - accuracy and F1 with and without the cleaning stage
  * speed            - mean processing time per image

Ground-truth detections are obtained by running the exact matcher and the
regex patterns on the ground-truth text itself, i.e. "what a perfect OCR
engine would have let the matcher find".

Usage:  python evaluation/evaluate.py [--no-easyocr]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocrmatch.cleaning import clean_text  # noqa: E402
from ocrmatch.config import load_config  # noqa: E402
from ocrmatch.matching import find_matches  # noqa: E402
from ocrmatch.ocr_engine import OCRWord, assemble_text, easyocr_available  # noqa: E402
from ocrmatch.pipeline import Pipeline, PipelineOptions  # noqa: E402
from ocrmatch.preprocessing import PreprocessOptions  # noqa: E402

RESULTS = ROOT / "evaluation" / "results"
SAMPLES = ROOT / "data" / "samples"

# name -> (engine, preprocessing, auto-retry)
OCR_CONFIGS = {
    "Tesseract - no preprocessing": ("tesseract", PreprocessOptions.disabled(), False),
    "Tesseract - grayscale + deskew": ("tesseract", PreprocessOptions(), False),
    "Tesseract - full + Otsu": ("tesseract", PreprocessOptions.standard(), False),
    "Tesseract - full + adaptive": ("tesseract", PreprocessOptions(denoise="median", threshold="adaptive"), False),
    "Proposed pipeline (Tesseract)": ("tesseract", PreprocessOptions(), True),
    "EasyOCR - no preprocessing": ("easyocr", PreprocessOptions.disabled(), False),
    "EasyOCR - grayscale + deskew": ("easyocr", PreprocessOptions(), False),
}
PROPOSED = "Proposed pipeline (Tesseract)"
THRESHOLDS = list(range(50, 101, 5))


# ------------------------------------------------------------------ metrics
def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def char_accuracy(pred: str, truth: str) -> float:
    p, t = _norm(pred), _norm(truth)
    return max(0.0, 1 - Levenshtein.distance(p, t) / max(1, len(t)))


def word_accuracy(pred: str, truth: str) -> float:
    p, t = _norm(pred).split(), _norm(truth).split()
    return max(0.0, 1 - Levenshtein.distance(p, t) / max(1, len(t)))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def compare(pred: Counter, truth: Counter) -> tuple[int, int, int]:
    """Occurrence-level comparison of two multisets."""
    tp = sum((pred & truth).values())
    return tp, sum((pred - truth).values()), sum((truth - pred).values())


def words_from_text(text: str) -> list[OCRWord]:
    words = []
    for line_id, line in enumerate(text.splitlines()):
        for tok in line.split():
            words.append(OCRWord(tok, 100.0, (0, 0, 1, 1), line_id))
    return words


def keyword_counter(matches) -> Counter:
    return Counter(m.keyword.lower() for m in matches if m.match_type in ("exact", "fuzzy"))


def pattern_counter(matches) -> Counter:
    return Counter((m.keyword, "".join(m.matched_text.lower().split()))
                   for m in matches if m.match_type == "regex")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-easyocr", action="store_true", help="skip the EasyOCR configurations")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "config" / "keywords.json")
    truth = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))
    images = [name for name in truth if (SAMPLES / name).exists()]

    # ground-truth detections from the clean ground-truth text
    gt_kw, gt_pat = {}, {}
    for name in images:
        gt_text = clean_text(truth[name]["text"])
        words = words_from_text(gt_text)
        text, spans = assemble_text(words)
        matches = find_matches(text, spans, words, config, modes=("exact", "regex"))
        gt_kw[name] = keyword_counter(matches)
        gt_pat[name] = pattern_counter(matches)
    print(f"Ground truth: {len(images)} images, {sum(sum(c.values()) for c in gt_kw.values())} keyword "
          f"occurrences, {sum(sum(c.values()) for c in gt_pat.values())} pattern values")

    configs = {k: v for k, v in OCR_CONFIGS.items()
               if v[0] == "tesseract" or (easyocr_available() and not args.no_easyocr)}

    per_image, stored = [], {}
    for cname, (engine, pre, retry) in configs.items():
        opts = PipelineOptions(preprocess=pre, engine=engine, auto_retry=retry, modes=("exact", "regex"))
        pipe = Pipeline(config, opts)
        pipe.process_path(SAMPLES / images[0])                     # warm-up (model loading)
        for name in images:
            r = pipe.process_path(SAMPLES / name)
            gt = truth[name]["text"]
            kw_tp, kw_fp, kw_fn = compare(keyword_counter(r.matches), gt_kw[name])
            per_image.append({
                "config": cname, "engine": engine, "image": name, "category": truth[name]["category"],
                "char_acc_raw": round(char_accuracy(r.raw_text, gt), 4),
                "char_acc": round(char_accuracy(r.cleaned_text, clean_text(gt)), 4),
                "word_acc": round(word_accuracy(r.cleaned_text, clean_text(gt)), 4),
                "mean_conf": r.mean_confidence, "coverage": r.text_coverage,
                "retried": any("Retried" in w for w in r.warnings), "time_s": r.timings["total"],
                "ocr_time_s": r.timings["ocr"], "words": r.word_count,
                "kw_tp": kw_tp, "kw_fp": kw_fp, "kw_fn": kw_fn,
            })
            stored[(cname, name)] = r
        print(f"  {cname:<34} done")

    df = pd.DataFrame(per_image)
    df.to_csv(RESULTS / "ocr_per_image.csv", index=False)

    summary = df.groupby("config", sort=False).agg(
        char_accuracy=("char_acc", "mean"), char_accuracy_before_cleaning=("char_acc_raw", "mean"),
        word_accuracy=("word_acc", "mean"), mean_confidence=("mean_conf", "mean"),
        time_per_image_s=("time_s", "mean"), ocr_time_per_image_s=("ocr_time_s", "mean"),
        kw_tp=("kw_tp", "sum"), kw_fp=("kw_fp", "sum"), kw_fn=("kw_fn", "sum"),
    ).reset_index()
    summary["exact_keyword_f1"] = [prf(r.kw_tp, r.kw_fp, r.kw_fn)["f1"] for r in summary.itertuples()]
    summary = summary.drop(columns=["kw_tp", "kw_fp", "kw_fn"]).round(4)
    summary.to_csv(RESULTS / "ocr_summary.csv", index=False)

    by_cat = df.pivot_table(index="category", columns="config", values="char_acc", sort=False).round(4)
    by_cat = by_cat[[c for c in configs if c in by_cat.columns]]
    by_cat.to_csv(RESULTS / "char_accuracy_by_category.csv")

    best = PROPOSED

    def detection(cname, modes, threshold):
        kw, pat = [0, 0, 0], [0, 0, 0]
        for name in images:
            r = stored[(cname, name)]
            text, spans = assemble_text(r.words)
            m = find_matches(text, spans, r.words, config, modes=modes, fuzzy_threshold=threshold)
            for acc, got in ((kw, compare(keyword_counter(m), gt_kw[name])),
                             (pat, compare(pattern_counter(m), gt_pat[name]))):
                for i in range(3):
                    acc[i] += got[i]
        return prf(*kw), prf(*pat)

    rows = []
    for cname in configs:
        for label, modes in (("Exact only", ("exact",)), ("Fuzzy only", ("fuzzy",)),
                             ("Exact + fuzzy", ("exact", "fuzzy"))):
            kw, _ = detection(cname, modes + ("regex",), config.fuzzy_threshold)
            rows.append({"config": cname, "mode": label, **kw})
    modes_df = pd.DataFrame(rows)
    modes_df.to_csv(RESULTS / "matching_modes.csv", index=False)

    sweep = []
    for cname in configs:
        for t in THRESHOLDS:
            kw, _ = detection(cname, ("exact", "fuzzy"), t)
            sweep.append({"config": cname, "threshold": t, **kw})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(RESULTS / "fuzzy_threshold_sweep.csv", index=False)

    regex_rows = []
    for cname in configs:
        _, pat = detection(cname, ("regex",), None)
        regex_rows.append({"config": cname, **pat})
    regex_overall = pd.DataFrame(regex_rows)
    regex_overall.to_csv(RESULTS / "regex_extraction.csv", index=False)

    # per-pattern breakdown for the best configuration
    per_pat = {}
    for name in images:
        r = stored[(best, name)]
        text, spans = assemble_text(r.words)
        pred = pattern_counter(find_matches(text, spans, r.words, config, modes=("regex",)))
        for p in config.patterns:
            pp = Counter({k: v for k, v in pred.items() if k[0] == p.name})
            gg = Counter({k: v for k, v in gt_pat[name].items() if k[0] == p.name})
            acc = per_pat.setdefault(p.name, [0, 0, 0])
            for i, v in enumerate(compare(pp, gg)):
                acc[i] += v
    per_pattern = pd.DataFrame([{"pattern": k, **prf(*v)} for k, v in per_pat.items()])
    per_pattern.to_csv(RESULTS / "regex_per_pattern.csv", index=False)

    # cleaning ablation on the proposed pipeline
    engine, pre, retry = configs[best]
    ablation = []
    for clean in (False, True):
        pipe = Pipeline(config, PipelineOptions(preprocess=pre, engine=engine, auto_retry=retry, clean=clean,
                                                modes=("exact", "regex", "fuzzy")))
        acc, kw, pat = [], [0, 0, 0], [0, 0, 0]
        for name in images:
            r = pipe.process_path(SAMPLES / name)
            acc.append(char_accuracy(r.cleaned_text, clean_text(truth[name]["text"]) if clean else truth[name]["text"]))
            for tot, got in ((kw, compare(keyword_counter(r.matches), gt_kw[name])),
                             (pat, compare(pattern_counter(r.matches), gt_pat[name]))):
                for i in range(3):
                    tot[i] += got[i]
        ablation.append({"cleaning": "on" if clean else "off", "char_accuracy": round(sum(acc) / len(acc), 4),
                         "keyword_f1": prf(*kw)["f1"], "pattern_f1": prf(*pat)["f1"]})
    ablation_df = pd.DataFrame(ablation)
    ablation_df.to_csv(RESULTS / "cleaning_ablation.csv", index=False)

    # decision-point log: which images triggered the retry in the proposed pipeline
    retry_log = df[df.config == PROPOSED][["image", "mean_conf", "coverage", "retried", "char_acc"]]
    retry_log.to_csv(RESULTS / "auto_retry.csv", index=False)

    results = {
        "images": len(images),
        "proposed_config": best,
        "ocr_summary": summary.to_dict(orient="records"),
        "matching_modes": modes_df.to_dict(orient="records"),
        "regex_extraction": regex_overall.to_dict(orient="records"),
        "regex_per_pattern": per_pattern.to_dict(orient="records"),
        "cleaning_ablation": ablation_df.to_dict(orient="records"),
        "fuzzy_threshold_sweep": sweep_df.to_dict(orient="records"),
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # store OCR text of each config for inspection
    dump = {f"{c} | {n}": stored[(c, n)].cleaned_text for (c, n) in stored}
    (RESULTS / "ocr_text_dump.json").write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nOCR summary\n", summary.to_string(index=False))
    print("\nCharacter accuracy by category\n", by_cat.to_string())
    print("\nMatching modes\n", modes_df.to_string(index=False))
    print("\nRegex extraction\n", regex_overall.to_string(index=False))
    print("\nRegex per pattern (proposed pipeline)\n", per_pattern.to_string(index=False))
    print("\nDecision point (proposed pipeline)\n", retry_log.to_string(index=False))
    print("\nCleaning ablation\n", ablation_df.to_string(index=False))
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
