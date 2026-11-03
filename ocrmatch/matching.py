"""Stage 5 - pattern matching on the cleaned OCR text.

Three modes:
  exact - case-insensitive whole-word search for each keyword / phrase
  regex - user-defined regular expressions (e-mail, phone, date, amount, ...)
  fuzzy - Levenshtein similarity between each keyword and every window of
          OCR words, tolerating OCR errors such as "Tota1" or "Invoce"

All matches are expressed as character spans of the assembled text and are
mapped back to the OCR words, which gives their bounding boxes and confidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from rapidfuzz.distance import Levenshtein
from rapidfuzz.utils import default_process

from .config import MatchConfig
from .ocr_engine import OCRWord

MODES = ("exact", "regex", "fuzzy")


@dataclass
class Match:
    match_type: str                 # exact | regex | fuzzy
    keyword: str                    # keyword / phrase, or regex pattern name
    matched_text: str
    score: float                    # similarity 0-100 (100 for exact and regex)
    ocr_confidence: float | None
    bbox: tuple[int, int, int, int] | None          # union box (processed image)
    line_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    word_indices: list[int] = field(default_factory=list)
    start: int = -1
    end: int = -1

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("line_boxes")
        return d


# ------------------------------------------------------------ helpers
def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _locate(indices: list[int], words: list[OCRWord]):
    """Union box, per-line boxes and mean OCR confidence for a group of words."""
    if not indices:
        return None, [], None
    by_line: dict[int, list] = {}
    for i in indices:
        by_line.setdefault(words[i].line_id, []).append(words[i].bbox)
    line_boxes = [_union(b) for b in by_line.values()]
    conf = round(sum(words[i].confidence for i in indices) / len(indices), 2)
    return _union(line_boxes), line_boxes, conf


def words_in_span(spans: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [i for i, (s, e) in enumerate(spans) if s < end and e > start]


def _make_match(kind, keyword, text, start, end, score, words, spans) -> Match:
    idx = words_in_span(spans, start, end)
    bbox, line_boxes, conf = _locate(idx, words)
    return Match(kind, keyword, text[start:end], round(score, 2), conf, bbox, line_boxes, idx, start, end)


def keyword_regex(term: str) -> re.Pattern:
    """Whole-word, case-insensitive pattern; any run of whitespace between words."""
    parts = [re.escape(p) for p in term.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


def similarity(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in percent: 100 * (1 - d / max(len))."""
    return Levenshtein.normalized_similarity(a, b) * 100.0


# ------------------------------------------------------------ the three modes
def exact_matches(text, spans, words, terms) -> list[Match]:
    out = []
    for term in terms:
        for m in keyword_regex(term).finditer(text):
            out.append(_make_match("exact", term, text, m.start(), m.end(), 100.0, words, spans))
    return out


def regex_matches(text, spans, words, patterns) -> list[Match]:
    out = []
    for p in patterns:
        for m in p.compiled.finditer(text):
            if m.end() > m.start():
                out.append(_make_match("regex", p.name, text, m.start(), m.end(), 100.0, words, spans))
    return out


def fuzzy_matches(text, spans, words, terms, threshold=80.0, min_length=4) -> list[Match]:
    """Slide a window of n-1, n and n+1 words over the OCR output (n = words in
    the keyword) so that split ("in voice") and merged ("duedate") words are
    also caught. Keep windows scoring >= threshold, then suppress overlaps so
    each region is reported once with its best score."""
    tokens = [default_process(w.text) for w in words]
    out = []
    for term in terms:
        target = default_process(term)
        if len(target.replace(" ", "")) < min_length:
            continue                      # very short keywords give false hits
        n = len(target.split())
        candidates = []
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(0, len(words) - size + 1):
                window = " ".join(t for t in tokens[i:i + size] if t)
                if not window or abs(len(window) - len(target)) > len(target) * 0.5:
                    continue              # cheap length filter before scoring
                score = similarity(target, window)
                if score >= threshold:
                    candidates.append((score, i, i + size))
        # non-maximum suppression over overlapping windows
        candidates.sort(key=lambda c: (-c[0], c[2] - c[1]))
        taken: set[int] = set()
        for score, i, j in candidates:
            if taken.intersection(range(i, j)):
                continue
            taken.update(range(i, j))
            start, end = spans[i][0], spans[j - 1][1]
            out.append(_make_match("fuzzy", term, text, start, end, score, words, spans))
    return out


def find_matches(text: str, spans, words, config: MatchConfig,
                 modes=MODES, fuzzy_threshold: float | None = None) -> list[Match]:
    """Run the selected modes and merge their results.

    A fuzzy hit that overlaps an exact hit of the same keyword is dropped -
    it would only repeat what the exact matcher already found.
    """
    threshold = config.fuzzy_threshold if fuzzy_threshold is None else fuzzy_threshold
    terms = config.terms
    results: list[Match] = []
    if "exact" in modes:
        results += exact_matches(text, spans, words, terms)
    if "regex" in modes:
        results += regex_matches(text, spans, words, config.patterns)
    if "fuzzy" in modes:
        exact_spans = {}
        for m in results:
            if m.match_type == "exact":
                exact_spans.setdefault(m.keyword.lower(), []).append((m.start, m.end))
        for m in fuzzy_matches(text, spans, words, terms, threshold, config.min_fuzzy_length):
            covered = exact_spans.get(m.keyword.lower(), [])
            if any(s < m.end and e > m.start for s, e in covered):
                continue
            results.append(m)
    results.sort(key=lambda m: (m.start, MODES.index(m.match_type)))
    return results
