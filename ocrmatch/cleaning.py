"""Stage 4 - text cleaning: normalise punctuation, whitespace and case, and
repair the most common OCR character confusions (0/O, 1/l/I, 5/S).

The confusion repair is context-based. A token is split into alphanumeric
runs (so "INV-2O24" gives "INV" and "2O24"), and each run of length >= 3 is
classified:
  * digit context (mostly digits, or every letter is a look-alike such as
    the O's in "5OO") -> O/o, l/I/|, S/s become 0, 1, 5 unless they touch a
    real letter
  * letter context (mostly letters) -> a lone 0, 1, 5 between letters becomes
    O, l/I, S; a leading digit is kept because it is usually a quantity
Genuine mixes such as "COVID19", "user123" or "5kg" are left untouched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from .ocr_engine import OCRWord

PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...", " ": " ", "«": '"', "»": '"',
}
EDGE_JUNK = "|~`_^"
KEEP_SYMBOL_TOKENS = {"&", "@", "%", "$", "#", "+", "-", "/", "="}

LETTER_TO_DIGIT = {"O": "0", "o": "0",
                   "l": "1", "I": "1", "|": "1",
                   "S": "5", "s": "5"}
DIGIT_TO_UPPER = {"0": "O", "1": "I", "5": "S"}
DIGIT_TO_LOWER = {"0": "o", "1": "l", "5": "s"}

_RUN = re.compile(r"[A-Za-z0-9|]+")


@dataclass
class CleaningOptions:
    normalize_punctuation: bool = True
    fix_confusions: bool = True
    lowercase: bool = False          # matching is case-insensitive anyway


def normalize_punctuation(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(PUNCT_MAP.get(ch, ch) for ch in text)


def _fix_run(run: str) -> str:
    if len(run) < 3:
        return run
    n_digits = sum(ch.isdigit() for ch in run)
    n_letters = sum(ch.isalpha() for ch in run)
    if n_digits == 0 or n_letters == 0:
        return run.replace("|", "1") if n_digits and "|" in run else run

    chars = list(run)
    letters = [c for c in run if c.isalpha()]
    all_lookalikes = all(c in LETTER_TO_DIGIT for c in letters)
    if n_digits >= n_letters or all_lookalikes:
        # digit context ("2O24", "5OO"): convert look-alike letters into digits
        # unless they touch a real letter such as the V in "INV2O24"
        def real_letter(c):
            return c.isalpha() and c not in LETTER_TO_DIGIT

        for i, ch in enumerate(run):
            if ch in LETTER_TO_DIGIT:
                left = run[i - 1] if i > 0 else ""
                right = run[i + 1] if i + 1 < len(run) else ""
                if not real_letter(left) and not real_letter(right):
                    chars[i] = LETTER_TO_DIGIT[ch]
    else:
        # letter context ("Tota1"): convert isolated look-alike digits into
        # letters. A leading digit is kept, as it is usually a quantity ("5kg").
        upper = sum(c.isupper() for c in letters) > len(letters) / 2
        table = DIGIT_TO_UPPER if upper else DIGIT_TO_LOWER
        for i, ch in enumerate(run):
            if ch in table and i > 0:
                left = run[i - 1] if i > 0 else ""
                right = run[i + 1] if i + 1 < len(run) else ""
                if not left.isdigit() and not right.isdigit():
                    chars[i] = table[ch]
    return "".join(chars)


def fix_confusions(token: str) -> str:
    return _RUN.sub(lambda m: _fix_run(m.group(0)), token)


def clean_token(token: str, opts: CleaningOptions | None = None) -> str:
    """Clean one OCR word. Returns '' if the token is pure noise."""
    opts = opts or CleaningOptions()
    if opts.normalize_punctuation:
        token = normalize_punctuation(token)
    token = token.strip().strip(EDGE_JUNK)
    if not token:
        return ""
    if not any(ch.isalnum() for ch in token) and token not in KEEP_SYMBOL_TOKENS:
        # stray marks such as '.', ',', '»' or '--' produced by specks
        if not re.fullmatch(r"[$%&@#+/=-]+", token):
            return ""
    if opts.fix_confusions:
        token = fix_confusions(token)
    if opts.lowercase:
        token = token.lower()
    return token


def _should_join(a: str, b: str) -> bool:
    """OCR often inserts a space inside e-mail addresses and web addresses,
    e.g. 'legal@ nimbus.com' or 'www. example.com'."""
    return (a.endswith("@") or b.startswith("@")
            or a.lower() in {"www.", "http://", "https://"}
            or (a.endswith(".") and b[:1].isalnum() and "@" in a))


def clean_words(words: list[OCRWord], opts: CleaningOptions | None = None) -> list[OCRWord]:
    """Clean every word, drop the ones that turn out to be noise, and rejoin
    e-mail / URL fragments that OCR split apart (boxes are merged too)."""
    out: list[OCRWord] = []
    for w in words:
        text = clean_token(w.text, opts)
        if not text:
            continue
        text = re.sub(r"[(\[{]@", "@", text)          # 'legal(@' -> 'legal@'
        prev = out[-1] if out else None
        if prev and prev.line_id == w.line_id and _should_join(prev.text, text):
            x0, y0 = min(prev.bbox[0], w.bbox[0]), min(prev.bbox[1], w.bbox[1])
            x1 = max(prev.bbox[0] + prev.bbox[2], w.bbox[0] + w.bbox[2])
            y1 = max(prev.bbox[1] + prev.bbox[3], w.bbox[1] + w.bbox[3])
            out[-1] = replace(prev, text=prev.text + text, bbox=(x0, y0, x1 - x0, y1 - y0),
                              confidence=round((prev.confidence + w.confidence) / 2, 2))
            continue
        out.append(replace(w, text=text))
    return out


def clean_text(text: str, opts: CleaningOptions | None = None) -> str:
    """Clean free text line by line (used for ground truth and plain strings)."""
    lines = []
    for line in text.splitlines():
        tokens = [clean_token(t, opts) for t in line.split()]
        tokens = [t for t in tokens if t]
        if tokens:
            lines.append(" ".join(tokens))
    return "\n".join(lines)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
