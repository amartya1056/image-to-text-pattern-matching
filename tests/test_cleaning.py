import pytest

from ocrmatch.cleaning import (CleaningOptions, clean_text, clean_token, clean_words,
                               fix_confusions, normalize_punctuation)
from ocrmatch.ocr_engine import OCRWord


@pytest.mark.parametrize("raw, expected", [
    ("2O24", "2024"),            # letter O inside a number
    ("1O0", "100"),
    ("4321O", "43210"),
    ("20l6", "2016"),            # lower-case L inside a number
    ("RS.5OO", "RS.500"),        # only the numeric run is changed
    ("Tota1", "Total"),          # digit 1 inside a lower-case word
    ("C0NFIDENTIAL", "CONFIDENTIAL"),
    ("INV0ICE", "INVOICE"),
    ("gmai1", "gmail"),
    ("Ca5h", "Cash"),            # 5 -> s in a lower-case word
    ("1OO", "100"),              # every letter is a look-alike -> number
])
def test_fix_confusions_repairs_lookalikes(raw, expected):
    assert fix_confusions(raw) == expected


@pytest.mark.parametrize("token", ["COVID19", "user123", "A4", "B2B", "29ABCDE1234F1Z5", "5kg", "x2"])
def test_fix_confusions_keeps_genuine_mixed_tokens(token):
    assert fix_confusions(token) == token


def test_fix_confusions_works_per_segment_of_a_token():
    assert fix_confusions("INV-2O26-O147") == "INV-2026-0147"
    assert fix_confusions("j0hn.d0e@gmai1.com") == "john.doe@gmail.com"


def test_normalize_punctuation():
    assert normalize_punctuation("“Quoted” – it’s…") == '"Quoted" - it\'s...'


def test_clean_token_strips_edge_junk_and_drops_noise():
    assert clean_token("|Invoice|") == "Invoice"
    assert clean_token("~~") == ""
    assert clean_token(".") == ""
    assert clean_token("»") == ""
    assert clean_token("&") == "&"


def test_clean_token_lowercase_option():
    assert clean_token("TOTAL", CleaningOptions(lowercase=True)) == "total"


def test_clean_token_can_disable_confusion_fixing():
    assert clean_token("2O24", CleaningOptions(fix_confusions=False)) == "2O24"


def test_clean_text_collapses_whitespace_and_keeps_lines():
    raw = "  Grand   Tota1 :  Rs. 1,25O.00 \n\n  |  \nDue  date"
    # the stray ':' and '|' tokens are OCR noise and are removed
    assert clean_text(raw) == "Grand Total Rs. 1,250.00\nDue date"


def test_clean_words_rejoins_split_email_and_url():
    words = [OCRWord("legal(@", 80, (0, 0, 60, 20), 0), OCRWord("nimbus.example.com", 90, (70, 0, 150, 20), 0),
             OCRWord("www.", 85, (0, 30, 40, 20), 1), OCRWord("example.com", 95, (50, 30, 100, 20), 1)]
    cleaned = clean_words(words)
    assert [w.text for w in cleaned] == ["legal@nimbus.example.com", "www.example.com"]
    assert cleaned[0].bbox == (0, 0, 220, 20)
    assert cleaned[0].confidence == 85


def test_clean_words_does_not_join_across_lines():
    words = [OCRWord("mail@", 80, (0, 0, 60, 20), 0), OCRWord("next", 90, (0, 30, 40, 20), 1)]
    assert [w.text for w in clean_words(words)] == ["mail@", "next"]


def test_clean_words_drops_empty_words_and_keeps_geometry():
    words = [OCRWord("Tota1", 80, (10, 10, 50, 20), 0), OCRWord("|", 30, (70, 10, 5, 20), 0)]
    cleaned = clean_words(words)
    assert len(cleaned) == 1
    assert cleaned[0].text == "Total"
    assert cleaned[0].bbox == (10, 10, 50, 20)
    assert cleaned[0].confidence == 80
