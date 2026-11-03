"""Stage 3 - text extraction (OCR) with word-level boxes and confidences.

Two engines share one output format (a list of OCRWord):
  * Tesseract (primary) via pytesseract.image_to_data
  * EasyOCR (optional, for comparison) - CRAFT detector + CRNN recogniser
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass
class OCRWord:
    text: str
    confidence: float                    # 0-100
    bbox: tuple[int, int, int, int]      # x, y, w, h in processed-image pixels
    line_id: int

    def to_dict(self) -> dict:
        return asdict(self)


class OCRError(RuntimeError):
    pass


# ---------------------------------------------------------------- Tesseract
def configure_tesseract() -> str:
    """Locate the tesseract binary (env var, PATH, then the default Windows path)."""
    import pytesseract

    candidates = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), str(WINDOWS_TESSERACT)]
    for cmd in candidates:
        if cmd and Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = cmd
            return cmd
    raise OCRError(
        "Tesseract binary not found. Install it (see README) or set the TESSERACT_CMD "
        "environment variable to the full path of tesseract(.exe)."
    )


def tesseract_version() -> str:
    import pytesseract

    configure_tesseract()
    return str(pytesseract.get_tesseract_version())


def run_tesseract(image: np.ndarray, lang: str = "eng", psm: int = 3) -> list[OCRWord]:
    import pytesseract

    configure_tesseract()
    rgb = image if image.ndim == 2 else image[:, :, ::-1]   # OpenCV BGR -> RGB
    data = pytesseract.image_to_data(rgb, lang=lang, config=f"--oem 3 --psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    words, line_ids = [], {}
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:            # level 1-4 rows (page/block/para/line) have conf -1
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_id = line_ids.setdefault(key, len(line_ids))
        bbox = (int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i]))
        words.append(OCRWord(text, conf, bbox, line_id))
    return words


# ---------------------------------------------------------------- EasyOCR
_easyocr_readers: dict[tuple, object] = {}


def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_easyocr_reader(langs: tuple[str, ...]):
    """One reader per language set; uses the GPU when one is available."""
    gpu = _cuda_available()
    key = langs + (("gpu",) if gpu else ())
    if key not in _easyocr_readers:
        try:
            import easyocr
        except ImportError as exc:
            raise OCRError("EasyOCR is not installed: pip install easyocr") from exc
        _easyocr_readers[key] = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
    return _easyocr_readers[key]


def run_easyocr(image: np.ndarray, langs: tuple[str, ...] = ("en",)) -> list[OCRWord]:
    """EasyOCR returns one box per detected text segment; split it into word boxes.

    Word boxes are approximated by dividing the segment width in proportion to
    the number of characters in each word (spaces included).
    """
    reader = _get_easyocr_reader(langs)
    detections = reader.readtext(image, detail=1, paragraph=False)
    # reading order: top-to-bottom, then left-to-right
    detections.sort(key=lambda d: (min(p[1] for p in d[0]) // 15, min(p[0] for p in d[0])))

    words: list[OCRWord] = []
    for line_id, (points, text, conf) in enumerate(detections):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        tokens = text.split()
        if not tokens:
            continue
        total_chars = max(1, len(text))
        cursor = 0
        for tok in tokens:
            start = text.find(tok, cursor)
            end = start + len(tok)
            cursor = end
            wx0 = x0 + int((x1 - x0) * start / total_chars)
            wx1 = x0 + int((x1 - x0) * end / total_chars)
            words.append(OCRWord(tok, round(float(conf) * 100, 2), (wx0, y0, max(1, wx1 - wx0), y1 - y0), line_id))
    return words


# ---------------------------------------------------------------- common
def run_ocr(image: np.ndarray, engine: str = "tesseract", lang: str = "eng", psm: int = 3) -> list[OCRWord]:
    engine = engine.lower()
    if engine == "tesseract":
        return run_tesseract(image, lang=lang, psm=psm)
    if engine == "easyocr":
        return run_easyocr(image)
    raise OCRError(f"Unknown OCR engine '{engine}' (use 'tesseract' or 'easyocr')")


def assemble_text(words: list[OCRWord]) -> tuple[str, list[tuple[int, int]]]:
    """Join words into one string (space within a line, newline between lines).

    Also returns the (start, end) character span of every word so that a regex
    match on the text can be traced back to the words - and their boxes.
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    prev_line = None
    for w in words:
        if prev_line is not None:
            sep = " " if w.line_id == prev_line else "\n"
            parts.append(sep)
            pos += 1
        spans.append((pos, pos + len(w.text)))
        parts.append(w.text)
        pos += len(w.text)
        prev_line = w.line_id
    return "".join(parts), spans


def mean_confidence(words: list[OCRWord]) -> float:
    return round(float(np.mean([w.confidence for w in words])), 2) if words else 0.0
