from __future__ import annotations
import pytest

from khmer_pipeline.evaluation.gt_provenance import (
    drafting_model_family,
    engine_model_family,
    is_circular,
    circularity_note,
)

# --- drafting_model_family ---

def test_drafted_by_field_is_authoritative():
    assert drafting_model_family({"gt_drafted_by": "gemini"}) == "gemini"


def test_drafted_by_is_normalised():
    # Version/variant suffixes must collapse to the FAMILY — the circularity is
    # shared weights, not an exact checkpoint string.
    assert drafting_model_family({"gt_drafted_by": "Gemini-2.5-Pro"}) == "gemini"
    assert drafting_model_family({"gt_drafted_by": "GPT-4o"}) == "openai"


def test_falls_back_to_sniffing_gt_source():
    # The real moc_gas GT predates gt_drafted_by and encodes provenance in gt_source.
    gt = {"gt_source": "gemini_draft_human_verified_TABLE_ONLY"}
    assert drafting_model_family(gt) == "gemini"


def test_drafted_by_wins_over_gt_source():
    gt = {"gt_drafted_by": "mistral", "gt_source": "gemini_draft_human_verified"}
    assert drafting_model_family(gt) == "mistral"


def test_human_transcribed_gt_has_no_model_family():
    assert drafting_model_family({"gt_source": "hand_transcribed"}) is None
    assert drafting_model_family({}) is None


def test_text_layer_gt_has_no_model_family():
    # Born-digital text-layer GT (PyMuPDF) is model-free by construction.
    assert drafting_model_family({"gt_source": "pdf_text_layer"}) is None


# --- engine_model_family ---

def test_local_engines_have_no_model_family():
    for key in ("surya", "surya_kiri", "surya_kiri_vlm", "auto", "hybrid", "tesseract"):
        assert engine_model_family(key) is None


def test_api_engines_map_to_their_family():
    assert engine_model_family("gemini") == "gemini"
    assert engine_model_family("mistral_ocr") == "mistral"


def test_unknown_engine_has_no_family():
    # An unregistered engine must not silently claim a family (would mis-fire the guard).
    assert engine_model_family("some_new_engine") is None


# --- is_circular / circularity_note ---

def test_circular_when_engine_matches_drafter():
    gt = {"gt_source": "gemini_draft_human_verified"}
    assert is_circular("gemini", gt) is True


def test_not_circular_for_a_different_model():
    gt = {"gt_source": "gemini_draft_human_verified"}
    assert is_circular("mistral_ocr", gt) is False


def test_not_circular_for_local_engines():
    # The whole local field is safe against LLM-drafted GT.
    gt = {"gt_source": "gemini_draft_human_verified"}
    assert is_circular("surya", gt) is False
    assert is_circular("surya_kiri", gt) is False


def test_not_circular_when_gt_is_human_or_text_layer():
    assert is_circular("gemini", {"gt_source": "hand_transcribed"}) is False
    assert is_circular("gemini", {}) is False


def test_circularity_note_names_both_sides():
    note = circularity_note("gemini", {"gt_source": "gemini_draft_human_verified"})
    assert note is not None
    assert "gemini" in note.lower()


def test_circularity_note_is_none_when_safe():
    assert circularity_note("surya", {"gt_source": "gemini_draft_human_verified"}) is None
