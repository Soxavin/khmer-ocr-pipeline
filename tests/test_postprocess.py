from __future__ import annotations
import warnings
from unittest.mock import MagicMock, patch
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
    """Returns a patch context that makes _get_qwen return dummy model/tokenizer."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    return patch("khmer_pipeline.postprocess._get_qwen", return_value=(mock_model, mock_tokenizer))


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


# --- qwen_used flag / region-level routing ---

def test_qwen_used_false_when_no_errors():
    # Khmer text with Khmer numerals — no foreign scripts, no text_blocks (fallback path)
    clean = "ចំណូល " + _KHMER_NUM_3 + " " + _KHMER_NUM_4 + " " + _KHMER_NUM_5
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result(ocr_text=clean))
    assert r.pages[0].qwen_used is False


def test_qwen_used_true_when_errors():
    # A text block with Sinhala density >= ANOMALY_THRESHOLD forces the Qwen path
    sinhala_text = "text " + _SINHALA_KA * 3 + " more"
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate", return_value="corrected") as _:
        mock_get.return_value = (MagicMock(), MagicMock())
        r = pp.postprocess(_make_surya_result(
            ocr_text=sinhala_text, text_blocks=[{"text": sinhala_text}]
        ))
    assert r.pages[0].qwen_used is True


def test_qwen_failure_falls_back_gracefully():
    # When generate raises, corrected_text should equal rule-based output, no crash
    sinhala_text = "text " + _SINHALA_KA * 3 + " more"
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate", side_effect=RuntimeError("GPU OOM")):
        mock_get.return_value = (MagicMock(), MagicMock())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = pp.postprocess(_make_surya_result(
                ocr_text=sinhala_text, text_blocks=[{"text": sinhala_text}]
            ))
        assert len(w) == 1
        assert "Qwen correction failed" in str(w[0].message)
    # corrected_text equals rule-applied block text (Qwen failed, returned input unchanged)
    assert r.pages[0].corrected_text == pp._apply_rules(sinhala_text)
    assert r.pages[0].qwen_used is True  # Qwen was attempted


def test_correction_diff_populated():
    # After any correction, correction_diff must be a string (may be empty if no change)
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result())
    assert isinstance(r.pages[0].correction_diff, str)


def test_qwen_not_called_when_no_errors():
    # generate should never be called when no text block is anomalous
    clean = "ចំណូល " + _KHMER_NUM_3 + " " + _KHMER_NUM_4 + " " + _KHMER_NUM_5
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate") as mock_gen:
        mock_get.return_value = (MagicMock(), MagicMock())
        pp.postprocess(_make_surya_result(ocr_text=clean))
    mock_gen.assert_not_called()


def test_page_with_no_text_blocks_falls_back_to_ocr_text():
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[], ocr_text="raw text")
    surya_result = SuryaResult(source_name="test.pdf", pages=[page])
    with _mock_qwen():
        result = pp.postprocess(surya_result)
    assert result.pages[0].corrected_text == pp._apply_rules("raw text")
    assert result.pages[0].qwen_used is False


def test_multi_page_each_page_corrected_independently():
    # page 0: clean Khmer block; page 1: Sinhala-dense block triggers Qwen
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
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate", return_value="fixed") as mock_gen:
        mock_get.return_value = (MagicMock(), MagicMock())
        r = pp.postprocess(multi)

    assert len(r.pages) == 2
    assert r.pages[0].qwen_used is False
    assert r.pages[1].qwen_used is True
    mock_gen.assert_called_once()
