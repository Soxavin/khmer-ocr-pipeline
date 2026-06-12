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


def _make_surya_result(ocr_text: str = "ខ្មែរ") -> SuryaResult:
    page = SuryaPageResult(
        page_index=0,
        text_blocks=[],
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


# --- _detect_errors: foreign script checks ---

def test_foreign_script_sinhala_triggers():
    assert pp._detect_errors("normal text " + _SINHALA_KA + " more") is True


def test_foreign_script_lao_triggers():
    assert pp._detect_errors("text " + _LAO_KO + " here") is True


def test_latin_does_not_trigger():
    assert pp._detect_errors("CP ARDB 03-06-26 0.00%") is False


# --- _detect_errors: Khmer numeral check ---

def test_khmer_numeral_check_triggers():
    # 6 Arabic numerals, 0 Khmer numerals → should trigger
    text = "ទំនិញ 1 2 3 4 5 6 នៅ"
    assert pp._detect_errors(text) is True


def test_khmer_numeral_check_does_not_trigger_below_threshold():
    # 3 Arabic numerals (≤5), 0 Khmer numerals → should NOT trigger
    text = "ទំនិញ 1 2 3 នៅ"
    assert pp._detect_errors(text) is False


# --- qwen_used flag ---

def test_qwen_used_false_when_no_errors():
    # Khmer text with Khmer numerals — no foreign scripts, numeral check satisfied
    clean = "ចំណូល " + _KHMER_NUM_3 + " " + _KHMER_NUM_4 + " " + _KHMER_NUM_5
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result(ocr_text=clean))
    assert r.pages[0].qwen_used is False


def test_qwen_used_true_when_errors():
    # Sinhala character (U+0D9A) forces Qwen path
    sinhala_text = "text " + _SINHALA_KA + " more"
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate", return_value="corrected") as _:
        mock_get.return_value = (MagicMock(), MagicMock())
        r = pp.postprocess(_make_surya_result(ocr_text=sinhala_text))
    assert r.pages[0].qwen_used is True


def test_qwen_failure_falls_back_gracefully():
    # When generate raises, corrected_text should equal rule-based output, no crash
    sinhala_text = "text " + _SINHALA_KA + " more"
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate", side_effect=RuntimeError("GPU OOM")):
        mock_get.return_value = (MagicMock(), MagicMock())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = pp.postprocess(_make_surya_result(ocr_text=sinhala_text))
        assert len(w) == 1
        assert "Qwen correction failed" in str(w[0].message)
    # corrected_text equals rule-applied input (Qwen failed, returned input unchanged)
    assert r.pages[0].corrected_text == pp._apply_rules(sinhala_text)
    assert r.pages[0].qwen_used is True  # Qwen was attempted


def test_correction_diff_populated():
    # After any correction, correction_diff must be a string (may be empty if no change)
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result())
    assert isinstance(r.pages[0].correction_diff, str)


def test_qwen_not_called_when_no_errors():
    # generate should never be called when _detect_errors returns False
    clean = "ចំណូល " + _KHMER_NUM_3 + " " + _KHMER_NUM_4 + " " + _KHMER_NUM_5
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate") as mock_gen:
        mock_get.return_value = (MagicMock(), MagicMock())
        pp.postprocess(_make_surya_result(ocr_text=clean))
    mock_gen.assert_not_called()


def test_multi_page_each_page_corrected_independently():
    # page 0: clean Khmer; page 1: has Sinhala → triggers Qwen; verify per-page independence
    from khmer_pipeline.models import SuryaPageResult

    def _make_page(idx, text):
        return SuryaPageResult(
            page_index=idx, text_blocks=[], tables=[], ocr_text=text
        )

    clean_text = "ចំណូល " + _KHMER_NUM_3
    dirty_text = "text " + _SINHALA_KA + " here"
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


def test_nfc_normalization_applied():
    import unicodedata
    nfd_text = "កា"
    assert pp._apply_rules(nfd_text) == unicodedata.normalize("NFC", nfd_text)
