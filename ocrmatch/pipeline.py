"""Orchestrator that chains the six stages and times each one."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import output
from .cleaning import CleaningOptions, clean_words
from .config import MatchConfig
from .io_utils import ImageLoadError, collect_images, load_image
from .matching import MODES, Match, find_matches
from .ocr_engine import OCRWord, assemble_text, mean_confidence, run_ocr
from .preprocessing import PreprocessOptions, PreprocessResult, preprocess, text_coverage


@dataclass
class PipelineOptions:
    preprocess: PreprocessOptions = field(default_factory=PreprocessOptions)
    engine: str = "tesseract"
    lang: str = "eng"
    psm: int = 3
    clean: bool = True
    modes: tuple[str, ...] = MODES
    fuzzy_threshold: float | None = None       # None -> value from config file
    auto_retry: bool = True                    # re-run OCR when confidence is low


@dataclass
class PipelineResult:
    image_name: str
    image_size: tuple[int, int]
    engine: str
    raw_text: str
    cleaned_text: str
    words: list[OCRWord]
    matches: list[Match]
    mean_confidence: float
    text_coverage: float
    skew_angle: float
    preprocessing: dict
    timings: dict
    warnings: list[str] = field(default_factory=list)
    original: np.ndarray | None = None
    processed: np.ndarray | None = None
    transform: np.ndarray | None = None
    steps: list = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.words)

    def annotated_image(self, labels: bool = True) -> np.ndarray:
        return output.annotate(self.original, self.matches, self.transform, show_labels=labels)

    def to_dict(self, include_words: bool = False) -> dict:
        d = {
            "image": self.image_name,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "ocr_engine": self.engine,
            "preprocessing": self.preprocessing,
            "skew_angle_deg": self.skew_angle,
            "mean_ocr_confidence": self.mean_confidence,
            "text_coverage": self.text_coverage,
            "word_count": self.word_count,
            "extracted_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "match_count": len(self.matches),
            "matches": [m.to_dict() for m in self.matches],
            "processing_time_s": self.timings,
            "warnings": self.warnings,
        }
        if include_words:
            d["words"] = [w.to_dict() for w in self.words]
        return d


class Pipeline:
    def __init__(self, config: MatchConfig, options: PipelineOptions | None = None):
        self.config = config
        self.options = options or PipelineOptions()

    # -- one OCR pass -------------------------------------------------------
    def _ocr_pass(self, image: np.ndarray, popts: PreprocessOptions):
        t0 = time.perf_counter()
        pre: PreprocessResult = preprocess(image, popts)
        t1 = time.perf_counter()
        words = run_ocr(pre.image, self.options.engine, self.options.lang, self.options.psm)
        t2 = time.perf_counter()
        min_conf = self.config.min_word_confidence
        if min_conf > 0:
            words = [w for w in words if w.confidence >= min_conf]
        return pre, words, t1 - t0, t2 - t1

    # -- full pipeline for one image ---------------------------------------
    def process_image(self, image: np.ndarray, name: str = "image") -> PipelineResult:
        opts = self.options
        warnings: list[str] = []
        start = time.perf_counter()

        popts = opts.preprocess
        pre, words, t_pre, t_ocr = self._ocr_pass(image, popts)
        conf = mean_confidence(words)
        coverage = text_coverage(pre.image, [w.bbox for w in words])

        # Decision point: does the OCR pass look poor?
        #   low mean confidence  -> characters were misread
        #   low text coverage    -> text was missed entirely
        cfg = self.config
        low_conf = conf < cfg.low_confidence_threshold
        low_cov = coverage < cfg.min_text_coverage
        if low_conf:
            warnings.append(f"Low mean OCR confidence ({conf:.1f} < {cfg.low_confidence_threshold:.0f}).")
        if low_cov:
            warnings.append(f"OCR boxes cover only {coverage:.0%} of the ink; some text may be missed.")
        if (low_conf or low_cov) and opts.auto_retry:
            rescue = PreprocessOptions.rescue()
            pre2, words2, tp2, to2 = self._ocr_pass(image, rescue)
            conf2 = mean_confidence(words2)
            coverage2 = text_coverage(pre2.image, [w.bbox for w in words2])
            t_pre += tp2
            t_ocr += to2
            # keep the pass with the better quality score = coverage x confidence
            if coverage2 * conf2 > coverage * conf:
                warnings.append(f"Retried with rescue preprocessing (x2, CLAHE, adaptive): confidence "
                                f"{conf:.1f} -> {conf2:.1f}, coverage {coverage:.0%} -> {coverage2:.0%}.")
                pre, words, conf, coverage, popts = pre2, words2, conf2, coverage2, rescue
            else:
                warnings.append("Rescue preprocessing was tried but did not improve the result; kept first pass.")
        if not words:
            warnings.append("No text was detected in the image.")

        raw_text, _ = assemble_text(words)
        t2 = time.perf_counter()
        cleaned = clean_words(words, CleaningOptions()) if opts.clean else list(words)
        text, spans = assemble_text(cleaned)
        t3 = time.perf_counter()
        matches = find_matches(text, spans, cleaned, self.config, opts.modes, opts.fuzzy_threshold)
        t4 = time.perf_counter()
        if words and not matches:
            warnings.append("No keyword or pattern matches were found.")

        timings = {
            "preprocessing": round(t_pre, 4),
            "ocr": round(t_ocr, 4),
            "cleaning": round(t3 - t2, 4),
            "matching": round(t4 - t3, 4),
            "total": round(t4 - start, 4),
        }
        return PipelineResult(
            image_name=name,
            image_size=(image.shape[1], image.shape[0]),
            engine=opts.engine,
            raw_text=raw_text,
            cleaned_text=text,
            words=cleaned,
            matches=matches,
            mean_confidence=conf,
            text_coverage=coverage,
            skew_angle=pre.skew_angle,
            preprocessing=popts.to_dict() | {"enabled": not popts.is_disabled},
            timings=timings,
            warnings=warnings,
            original=image,
            processed=pre.image,
            transform=pre.transform,
            steps=pre.steps,
        )

    def process_path(self, path: str | Path) -> PipelineResult:
        path = Path(path)
        return self.process_image(load_image(path), path.name)

    # -- batch ---------------------------------------------------------------
    def process_batch(self, input_path: str | Path, recursive: bool = False, on_result=None):
        """Process a file or folder. Unreadable files are reported, not fatal."""
        paths, skipped = collect_images(input_path, recursive)
        results, failures = [], [{"file": s.split(":")[0], "error": s} for s in skipped]
        for p in paths:
            try:
                r = self.process_path(p)
            except ImageLoadError as exc:
                failures.append({"file": p.name, "error": str(exc)})
                continue
            results.append(r)
            if on_result:
                on_result(r)
        return results, failures
