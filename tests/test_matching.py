import pytest

from ocrmatch.config import ConfigError, config_from_dict
from ocrmatch.matching import find_matches, keyword_regex, similarity
from ocrmatch.ocr_engine import OCRWord, assemble_text

CONFIG = config_from_dict({
    "keywords": ["invoice", "total", "urgent", "tax"],
    "phrases": ["due date", "account number"],
    "patterns": {
        "email": {"regex": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"},
        "date": {"regex": r"\b\d{1,2}/\d{1,2}/\d{4}\b"},
        "amount": {"regex": r"(?:Rs\.?|\$)\s?\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?"},
    },
    "settings": {"fuzzy_threshold": 80, "min_fuzzy_length": 4},
})


def make_words(lines):
    """Build OCR words with fake boxes from a list of lines."""
    words = []
    for line_id, line in enumerate(lines):
        x = 0
        for tok in line.split():
            words.append(OCRWord(tok, 90.0, (x, line_id * 30, len(tok) * 10, 20), line_id))
            x += len(tok) * 10 + 10
    return words


def run(lines, modes=("exact", "regex", "fuzzy"), threshold=None):
    words = make_words(lines)
    text, spans = assemble_text(words)
    return find_matches(text, spans, words, CONFIG, modes, threshold)


def kinds(matches, kind):
    return [(m.keyword, m.matched_text) for m in matches if m.match_type == kind]


# -------------------------------------------------------------- exact
def test_exact_is_case_insensitive():
    assert kinds(run(["INVOICE no 12"], ["exact"]), "exact") == [("invoice", "INVOICE")]


def test_exact_is_whole_word_only():
    found = kinds(run(["Subtotal and invoiced and taxes"], ["exact"]), "exact")
    assert found == []


def test_exact_phrase_may_span_lines():
    found = kinds(run(["Payment due", "date is Friday"], ["exact"]), "exact")
    assert found == [("due date", "due\ndate")]


def test_exact_ignores_trailing_punctuation():
    assert kinds(run(["Total: Rs. 50"], ["exact"]), "exact") == [("total", "Total")]


# -------------------------------------------------------------- regex
def test_regex_patterns_extract_values():
    m = run(["Mail a.b@example.com by 21/03/2026 amount Rs. 1,250.00"], ["regex"])
    assert kinds(m, "regex") == [("email", "a.b@example.com"), ("date", "21/03/2026"),
                                 ("amount", "Rs. 1,250.00")]


def test_regex_match_is_mapped_to_word_boxes():
    m = run(["Pay Rs. 500 now"], ["regex"])[0]
    assert m.word_indices == [1, 2]                  # "Rs." and "500"
    assert m.bbox == (40, 0, 70, 20)                 # union of both word boxes


def test_invalid_regex_in_config_is_reported():
    with pytest.raises(ConfigError):
        config_from_dict({"patterns": {"bad": {"regex": "([unclosed"}}})


# -------------------------------------------------------------- fuzzy
def test_similarity_is_normalised_levenshtein():
    assert similarity("invoice", "invoice") == 100
    assert similarity("total", "tota") == pytest.approx(80.0)        # 1 edit / 5 chars
    assert similarity("invoice", "lnvoice") == pytest.approx(100 * 6 / 7)


def test_fuzzy_tolerates_ocr_errors():
    found = kinds(run(["lnvoice Totai URGEMT"], ["fuzzy"]), "fuzzy")
    assert ("invoice", "lnvoice") in found
    assert ("total", "Totai") in found
    assert ("urgent", "URGEMT") in found


def test_fuzzy_threshold_controls_strictness():
    assert kinds(run(["lnvoice"], ["fuzzy"], threshold=95), "fuzzy") == []
    assert kinds(run(["lnvoice"], ["fuzzy"], threshold=80), "fuzzy") == [("invoice", "lnvoice")]


def test_fuzzy_handles_split_and_merged_words():
    found = kinds(run(["in voice", "accountnumber"], ["fuzzy"]), "fuzzy")
    assert ("invoice", "in voice") in found
    assert ("account number", "accountnumber") in found


def test_fuzzy_skips_short_keywords():
    # "tax" is shorter than min_fuzzy_length=4, so "fax" must not match it
    assert kinds(run(["fax"], ["fuzzy"]), "fuzzy") == []


def test_fuzzy_does_not_repeat_exact_hits():
    m = run(["Invoice"], ["exact", "fuzzy"])
    assert [x.match_type for x in m] == ["exact"]


def test_scores_are_reported():
    m = run(["Totai"], ["fuzzy"])[0]
    assert m.score == pytest.approx(80.0)
    assert m.ocr_confidence == 90.0


def test_keyword_regex_escapes_special_characters():
    assert keyword_regex("non-disclosure agreement").search("A Non-Disclosure  Agreement")


def test_url_pattern_from_config_excludes_trailing_full_stop():
    from pathlib import Path
    from ocrmatch.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "keywords.json")
    url = next(p for p in cfg.patterns if p.name == "url").compiled
    assert url.search("see www.example.com/legal.").group() == "www.example.com/legal"
    assert url.search("go to https://a.example.org/verify now").group() == "https://a.example.org/verify"
