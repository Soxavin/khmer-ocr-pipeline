from __future__ import annotations
from contextlib import nullcontext
from unittest.mock import patch
import khmer_pipeline.postprocess as pp
from khmer_pipeline.models import SuryaResult, SuryaPageResult, PostprocessResult

# Unambiguous character constants — chr() resolves at module load time
_SINHALA_KA = chr(0x0D9A)   # Sinhala letter KA, range U+0D80-U+0DFF
_LAO_KO = chr(0x0E81)       # Lao letter KO, range U+0E80-U+0EFF
_KHMER_NUM_3 = chr(0x17E3)  # Khmer numeral 3 (U+17E0-U+17E9)
_KHMER_NUM_4 = chr(0x17E4)  # Khmer numeral 4
_KHMER_NUM_5 = chr(0x17E5)  # Khmer numeral 5


def _make_surya_result(ocr_text: str = "ខ្មែរ", text_blocks: list | None = None) -> SuryaResult:
    page = SuryaPageResult(
        page_index=0,
        text_blocks=text_blocks if text_blocks is not None else [],
        tables=[],
        ocr_text=ocr_text,
    )
    return SuryaResult(source_name="test.pdf", pages=[page])


def _mock_qwen():
    """No-op context: the Qwen LLM corrector was retired (§2.102); the deterministic
    Stage-4 layer these tests exercise runs regardless. Kept so existing call sites
    (`with _mock_qwen():`) stay unchanged."""
    return nullcontext()


# --- Contract tests ---

def test_postprocess_returns_postprocess_result():
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result())
    assert isinstance(r, PostprocessResult)


def test_raw_ocr_text_never_modified():
    original = "ខ្មែរ original"
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result(ocr_text=original))
    assert r.pages[0].raw_ocr_text == original


def test_rules_apply_correctly():
    with patch.dict(pp.RULE_BASED_CORRECTIONS, {"WRONG": "RIGHT"}):
        result = pp._apply_rules("some WRONG text")
    assert result == "some RIGHT text"


def test_nfc_normalization_applied():
    import unicodedata
    nfd_text = "កា"
    assert pp._apply_rules(nfd_text) == unicodedata.normalize("NFC", nfd_text)


# --- _anomaly_score / _detect_errors ---

def test_anomaly_score_zero_for_empty():
    assert pp._anomaly_score("") == 0.0
    assert pp._anomaly_score("   ") == 0.0


def test_anomaly_score_between_0_and_1():
    for text in ["clean text", "ខ្មែរ", _SINHALA_KA * 5, "CP 03-06-26 0.00%"]:
        assert 0.0 <= pp._anomaly_score(text) <= 1.0


def test_anomaly_score_high_for_sinhala():
    assert pp._anomaly_score(_SINHALA_KA * 10) >= pp.ANOMALY_THRESHOLD


def test_anomaly_score_low_for_latin():
    # Latin text including numbers should not trigger — mixed numerals are normal
    assert pp._anomaly_score("CP ARDB 03-06-26 12,000 0.00%") < pp.ANOMALY_THRESHOLD


def test_anomaly_score_denominator_excludes_whitespace():
    # 1 foreign char among 3 non-whitespace chars ("<sinhala>ok") = 1/3, and the
    # surrounding whitespace must NOT dilute that ratio (new denominator).
    text = _SINHALA_KA + "        " + "ok"
    assert abs(pp._anomaly_score(text) - 1 / 3) < 1e-9


def test_anomaly_score_ignores_whitespace_padding():
    dense = _SINHALA_KA + "ok"            # 1 / 3
    padded = _SINHALA_KA + "   \n  " + "ok"  # same 3 non-ws chars → same score
    assert pp._anomaly_score(dense) == pp._anomaly_score(padded)


def test_mixed_khmer_arabic_numerals_not_anomalous():
    # Khmer row numbers + Arabic prices — normal in financial docs, must not trigger
    mixed = "៩ សាច់ជ្រូករស់ 12,000 13,000 8.33%"
    assert pp._anomaly_score(mixed) < pp.ANOMALY_THRESHOLD


def test_foreign_script_sinhala_triggers():
    # 3 Sinhala chars in a 13-char string: 3/13 ≈ 0.23 >= 0.15
    text = "text " + _SINHALA_KA * 3 + " more"
    assert pp._detect_errors(text) is True


def test_foreign_script_lao_triggers():
    # 3 Lao chars in a 13-char string: 3/13 ≈ 0.23 >= 0.15
    text = "text " + _LAO_KO * 3 + " more"
    assert pp._detect_errors(text) is True


def test_latin_does_not_trigger():
    assert pp._detect_errors("CP ARDB 03-06-26 0.00%") is False


# --- qwen_used always False (LLM corrector retired §2.102) ---

def test_qwen_used_always_false():
    # The Qwen corrector was retired; the field is kept (always False) for API stability.
    # Even a foreign-script-heavy block only gets the deterministic scrub, never an LLM.
    sinhala_text = "text " + _SINHALA_KA * 3 + " more"
    r = pp.postprocess(_make_surya_result(
        ocr_text=sinhala_text, text_blocks=[{"text": sinhala_text}]
    ), skip_qwen=False)  # skip_qwen is inert now
    assert r.pages[0].qwen_used is False
    # The deterministic foreign-script scrub still cleans the block.
    assert r.pages[0].corrected_text == pp._strip_foreign_scripts(pp._apply_rules(sinhala_text))[0]


# --- A6: Stage 4 normalizes table cell text, copy-on-write ---

def _table_with_cell_text(text: str) -> dict:
    cell = {"row_id": 0, "col_id": 0,
            "text_lines": [{"text": text, "bbox": [0, 0, 10, 10]}], "bbox": [0, 0, 10, 10]}
    return {"rows": [{"row_id": 0}], "cols": [{"col_id": 0}], "cells": [cell],
            "image_bbox": [0, 0, 100, 100]}


def _cell_text_of(page_result, t=0, c=0) -> str:
    return page_result.tables[t]["cells"][c]["text_lines"][0]["text"]


def test_postprocess_strips_zwsp_in_table_cell():
    zwsp = chr(0x200B)
    table = _table_with_cell_text("12" + zwsp + "000")
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[table], ocr_text="")
    with _mock_qwen():
        r = pp.postprocess(SuryaResult(source_name="t.pdf", pages=[page]))
    out = _cell_text_of(r.pages[0])
    assert zwsp not in out
    assert out == "12000"


def test_postprocess_collapses_duplicate_diacritic_in_cell():
    base = chr(0x1780)   # Khmer KA
    mark = chr(0x17B6)   # dependent vowel (combining)
    table = _table_with_cell_text(base + mark + mark)
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[table], ocr_text="")
    with _mock_qwen():
        r = pp.postprocess(SuryaResult(source_name="t.pdf", pages=[page]))
    assert _cell_text_of(r.pages[0]) == base + mark  # duplicate mark collapsed


def test_postprocess_does_not_mutate_surya_tables():
    """CRITICAL INVARIANT: the input SuryaResult.tables must stay byte-identical
    (postprocess owns copies; no aliasing)."""
    import copy
    zwsp = chr(0x200B)
    table = _table_with_cell_text("a" + zwsp + "b")
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[table], ocr_text="")
    surya = SuryaResult(source_name="t.pdf", pages=[page])
    before = copy.deepcopy(surya.pages[0].tables)
    with _mock_qwen():
        r = pp.postprocess(surya)
    # original untouched (still contains the ZWSP), corrected copy normalized
    assert surya.pages[0].tables == before
    assert zwsp in _cell_text_of(surya.pages[0])
    assert r.pages[0].tables is not surya.pages[0].tables
    assert r.pages[0].tables[0]["cells"] is not surya.pages[0].tables[0]["cells"]
    assert zwsp not in _cell_text_of(r.pages[0])


# --- Step 2a: GDDE-domain cell rules (PROJECT_LOG §2.33/§2.34) ---

def test_riel_prefix_rules_fix_measured_confusions():
    cases = {
        "អគ.ក": "៛/គ.ក", "#គ.ក": "៛/គ.ក", "វ/គ.ក": "៛/គ.ក", "អ/គ.ក": "៛/គ.ក",
        "អគៈក": "៛/គ.ក", "អគ:ក": "៛/គ.ក",
        "អគ្រាប់": "៛/គ្រាប់", "#ផ្លែ": "៛/ផ្លែ", "អផ្លែ": "៛/ផ្លែ",
    }
    for corrupt, fixed in cases.items():
        assert pp._apply_cell_rules(corrupt) == fixed, corrupt


def test_cell_rules_identity_on_clean_and_generic_text():
    # Anti-overfit contract: rules fire ONLY on full-cell corrupt forms.
    for s in ["៛/គ.ក", "៛/គ្រាប់", "គ.ក", "សាច់ជ្រូក", "អង្ករ", "អគរ",
              "12,000", "8.33%", "", "ABA", "អគ.ក extra", "x អគ.ក"]:
        assert pp._apply_cell_rules(s) == s, s


def test_percent_khmer_zero_folded():
    assert pp._apply_cell_rules("០.00%") == "0.00%"
    assert pp._apply_cell_rules("០.០០%") == "0.00%"
    assert pp._apply_cell_rules("-០.៥%") == "-0.5%"


def test_khmer_row_index_cells_not_folded():
    # Legitimately-Khmer numerals without % (the row-index column) must pass through.
    for s in ["១", "២៣", "៩"]:
        assert pp._apply_cell_rules(s) == s


def test_cell_rules_applied_through_postprocess():
    table = _table_with_cell_text("អគ.ក")
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[table], ocr_text="")
    with _mock_qwen():
        r = pp.postprocess(SuryaResult(source_name="t.pdf", pages=[page]))
    assert _cell_text_of(r.pages[0]) == "៛/គ.ក"


# --- Step 2b: malformed-number flag + Stage-4 warnings channel ---

def _postprocess_single_cell(text: str, confidence: float | None = None):
    table = _table_with_cell_text(text)
    if confidence is not None:
        table["cells"][0]["confidence"] = confidence
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[table], ocr_text="")
    with _mock_qwen():
        return pp.postprocess(SuryaResult(source_name="t.pdf", pages=[page]))


def test_malformed_comma_number_flagged_not_rewritten():
    r = _postprocess_single_cell("7,8000", confidence=0.97)
    cell = r.pages[0].tables[0]["cells"][0]
    assert cell["text_lines"][0]["text"] == "7,8000"      # digits NEVER rewritten
    assert cell["confidence"] < pp.CONFIDENCE_LOW          # capped → red in UI
    assert any("7,8000" in w for w in r.warnings)


def test_malformed_percent_flagged_sets_confidence_when_absent():
    r = _postprocess_single_cell("8333%")                  # no confidence on input
    cell = r.pages[0].tables[0]["cells"][0]
    assert cell["text_lines"][0]["text"] == "8333%"
    assert cell["confidence"] < pp.CONFIDENCE_LOW
    assert len(r.warnings) == 1


def test_wellformed_numbers_not_flagged():
    # "150%"-style ≥2-digit integer percents are now deliberately flagged (§2.35).
    for s in ["4,000", "8.33%", "-13.33%", "5%", "12,000", "0.00%"]:
        r = _postprocess_single_cell(s, confidence=0.97)
        cell = r.pages[0].tables[0]["cells"][0]
        assert r.warnings == [], s
        assert cell["confidence"] == 0.97, s


# --- Step 3-pre: gridline-noise strip, foreign-script scrub, widened % flag ---

def test_pipe_noise_cells_emptied():
    for noise in ["|", "-|", "|-", "||", "— |"]:
        r = _postprocess_single_cell(noise)
        assert _cell_text_of(r.pages[0]) == "", noise


def test_bare_dash_placeholder_survives():
    # Conservative rule: only pipe-bearing junk is stripped; a lone dash may be a
    # legitimate "no data" marker in other document types.
    for keep in ["-", "—", "_"]:
        r = _postprocess_single_cell(keep)
        assert _cell_text_of(r.pages[0]) == keep, keep


def test_foreign_script_scrubbed_from_cell_with_warning():
    r = _postprocess_single_cell("4,000" + _SINHALA_KA * 2)
    assert _cell_text_of(r.pages[0]) == "4,000"
    assert any("foreign-script" in w for w in r.warnings)


def test_foreign_script_scrubbed_from_narrative_with_warning():
    dirty = "តម្លៃ " + _LAO_KO * 3 + " 12,000"
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result(text_blocks=[{"text": dirty}]))
    assert _LAO_KO not in r.pages[0].corrected_text
    assert "តម្លៃ" in r.pages[0].corrected_text and "12,000" in r.pages[0].corrected_text
    assert any("foreign-script" in w for w in r.warnings)


def test_khmer_english_digits_never_scrubbed():
    clean = "៩ សាច់ជ្រូក Pork 12,000 ៛/គ.ក 8.33%"
    assert pp._strip_foreign_scripts(clean) == (clean, 0)


def test_integer_percent_flagged_decimal_percent_not():
    # ≥2-digit integer percents = likely dot-drops (2.94%→294%) → flagged
    for bad in ["294%", "-476%", "8333%", "25%"]:
        r = _postprocess_single_cell(bad, confidence=0.97)
        assert r.pages[0].tables[0]["cells"][0]["confidence"] < pp.CONFIDENCE_LOW, bad
        assert len(r.warnings) == 1, bad
    for ok in ["8.33%", "-13.33%", "0.00%", "5%"]:
        r = _postprocess_single_cell(ok, confidence=0.97)
        assert r.warnings == [], ok


def test_postprocess_warnings_empty_by_default():
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result())
    assert r.warnings == []


def test_correction_diff_populated():
    # After any correction, correction_diff must be a string (may be empty if no change)
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result())
    assert isinstance(r.pages[0].correction_diff, str)


def test_page_with_no_text_blocks_falls_back_to_ocr_text():
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[], ocr_text="raw text")
    surya_result = SuryaResult(source_name="test.pdf", pages=[page])
    with _mock_qwen():
        result = pp.postprocess(surya_result)
    assert result.pages[0].corrected_text == pp._apply_rules("raw text")
    assert result.pages[0].qwen_used is False


def test_multi_page_each_page_corrected_independently():
    # page 0: clean Khmer block; page 1: Sinhala-dense block (deterministic scrub only).
    clean_text = "ចំណូល " + _KHMER_NUM_3
    dirty_text = "text " + _SINHALA_KA * 3 + " here"

    def _make_page(idx, text):
        return SuryaPageResult(
            page_index=idx, text_blocks=[{"text": text}], tables=[], ocr_text=text
        )

    multi = SuryaResult(
        source_name="multi.pdf",
        pages=[_make_page(0, clean_text), _make_page(1, dirty_text)],
    )
    r = pp.postprocess(multi)

    assert len(r.pages) == 2
    assert all(p.qwen_used is False for p in r.pages)
    # Page 0 unchanged; page 1's Sinhala is scrubbed by the deterministic layer.
    assert r.pages[0].corrected_text == pp._strip_foreign_scripts(pp._apply_rules(clean_text))[0]
    assert _SINHALA_KA not in r.pages[1].corrected_text
