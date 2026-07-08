from __future__ import annotations
import pytest
from khmer_pipeline.engines.engine_registry import get_ocr_engine
from khmer_pipeline.engines.surya import run_surya
from khmer_pipeline.engines.surya_kiri_engine import run_surya_kiri


def test_get_ocr_engine_returns_surya_kiri():
    assert get_ocr_engine("surya_kiri") is run_surya_kiri


def test_get_ocr_engine_returns_surya():
    assert get_ocr_engine("surya") is run_surya


def test_get_ocr_engine_unknown_raises_value_error():
    """B2: a typo'd engine name must fail loudly, not silently run Surya."""
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        get_ocr_engine("not_a_real_engine")


def test_get_ocr_engine_error_lists_valid_names():
    with pytest.raises(ValueError, match="surya_kiri"):
        get_ocr_engine("surya-kiri")  # hyphen typo


def test_unknown_ocr_engine_env_raises_at_import(monkeypatch):
    """B2: an unknown OCR_ENGINE env value must raise at import (fail loudly),
    so a mistyped benchmark run cannot silently test the wrong engine."""
    import importlib
    import khmer_pipeline.engines.engine_registry as reg
    monkeypatch.setenv("OCR_ENGINE", "surya-kiri")  # hyphen typo
    try:
        with pytest.raises(ValueError, match="Unknown OCR engine"):
            importlib.reload(reg)
    finally:
        # Restore a valid module state for the rest of the suite.
        monkeypatch.setenv("OCR_ENGINE", "surya")
        importlib.reload(reg)
