"""Tests for the Qwen3.5-0.8B ARDB fine-tune engine (qwen_ardb_engine.py).

Now exposed in the UI (apps/api/api.py's _ENGINES) specifically BECAUSE it's
unreliable: every tested config fails on most real documents, and the point is
to show that failure live, not hide it. Contract differs from
gemma_ardb_engine.py in one deliberate way: on an unparseable-JSON page, the
raw model output survives in `ocr_text` (not discarded) so the review UI's
raw-output fallback can show it — see PageTextPanel.tsx's empty-state branch.
The isolated subprocess is monkeypatched here — no real uv/model download."""
from __future__ import annotations

import json

import numpy as np
import pytest

from khmer_pipeline.models import PreprocessResult
import khmer_pipeline.engines.qwen_ardb_engine as qe


def _pre(n_pages: int = 1) -> PreprocessResult:
    imgs = [np.full((40, 60, 3), 255, dtype=np.uint8) for _ in range(n_pages)]
    return PreprocessResult(source_name="doc", page_images=imgs, dpi=200, page_count=n_pages)


def _regions(*objs) -> str:
    return json.dumps(list(objs), ensure_ascii=False)


def _install_fake(monkeypatch, outputs):
    """outputs: list of str (stdout) or Exception, one per page call."""
    calls = {"n": 0}

    def fake_run(page_image, config, **kwargs):
        out = outputs[calls["n"]]
        calls["n"] += 1
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(qe, "run_isolated_inference", fake_run)
    return calls


# --- happy path (rare, but the config exists so it must work when it does) ---

def test_table_region_becomes_a_table_markdown(monkeypatch):
    md = "| a | b |\n|---|---|\n| 1 | 2 |"
    regions = _regions({"box_2d": [0, 0, 500, 1000], "label": "Table", "text": md})
    _install_fake(monkeypatch, [regions])
    result = qe.run_qwen_ardb(_pre(1))
    page = result.pages[0]
    assert len(page.tables) == 1
    cells = page.tables[0]["cells"]
    texts = {(c["row_id"], c["col_id"]): c["text_lines"][0]["text"] for c in cells if c["text_lines"]}
    assert texts[(1, 0)] == "1"
    assert texts[(1, 1)] == "2"


def test_text_region_becomes_a_text_block(monkeypatch):
    regions = _regions({"box_2d": [0, 0, 100, 100], "label": "Text", "text": "hello"})
    _install_fake(monkeypatch, [regions])
    result = qe.run_qwen_ardb(_pre(1))
    assert any(b["text"] == "hello" for b in result.pages[0].text_blocks)


# --- the expected case: unparseable output, raw text preserved (NOT discarded) ---

def test_unparseable_output_preserves_raw_text_in_ocr_text(monkeypatch):
    garbled = "{label: Table, text: <not real json at all, cut off mid-fiel"
    _install_fake(monkeypatch, [garbled])
    result = qe.run_qwen_ardb(_pre(1))
    page = result.pages[0]
    assert page.tables == [] and page.text_blocks == []
    # The raw model output must survive verbatim in ocr_text — this is the
    # field the frontend's raw-output fallback reads (see PageTextPanel.tsx).
    assert page.ocr_text == garbled
    assert any("pars" in w.lower() for w in result.warnings)


def test_repairable_truncated_json_still_recovers_normally(monkeypatch):
    # When the shared repair logic CAN recover it, behave like Gemma: structured
    # output wins over the raw-text fallback.
    truncated = '[{"box_2d": [0,0,10,10], "label": "Text", "text": "recovered"}'
    _install_fake(monkeypatch, [truncated])
    result = qe.run_qwen_ardb(_pre(1))
    assert any(b["text"] == "recovered" for b in result.pages[0].text_blocks)


def test_subprocess_failure_has_no_raw_text_to_preserve(monkeypatch):
    # No model output exists in this branch (the subprocess itself errored,
    # e.g. dependency failure) — ocr_text stays empty, nothing to show raw.
    boom = RuntimeError("qwen_ardb_infer.py exited 1: ModuleNotFoundError: peft")
    _install_fake(monkeypatch, [boom])
    result = qe.run_qwen_ardb(_pre(1))
    page = result.pages[0]
    assert page.ocr_text == ""
    assert any("page 1" in w.lower() for w in result.warnings)


def test_one_bad_page_leaves_others_intact(monkeypatch):
    good = _regions({"box_2d": [0, 0, 10, 10], "label": "Text", "text": "ok"})
    garbled = "not json, no braces at all"
    _install_fake(monkeypatch, [good, garbled])
    result = qe.run_qwen_ardb(_pre(2))
    assert "ok" in result.pages[0].ocr_text
    assert result.pages[1].ocr_text == garbled


# --- audit + telemetry contract ---

def test_audit_warning_names_the_model(monkeypatch):
    regions = _regions({"box_2d": [0, 0, 1, 1], "label": "Text", "text": "x"})
    _install_fake(monkeypatch, [regions])
    result = qe.run_qwen_ardb(_pre(1))
    assert any("qwen" in w.lower() for w in result.warnings)


def test_accepts_on_page_and_on_step(monkeypatch):
    regions = _regions({"box_2d": [0, 0, 1, 1], "label": "Text", "text": "x"})
    _install_fake(monkeypatch, [regions])
    seen = []
    qe.run_qwen_ardb(_pre(1), on_page=lambda i, t: seen.append((i, t)), on_step=lambda s: None)
    assert seen == [(0, 1)]


def test_on_step_emits_a_distinct_slow_finetune_marker(monkeypatch):
    # The frontend's SUB_STEPS lookup needs a value distinct from Surya's
    # layout/text/tables steps to show engine-appropriate "may take minutes" copy.
    regions = _regions({"box_2d": [0, 0, 1, 1], "label": "Text", "text": "x"})
    _install_fake(monkeypatch, [regions])
    steps = []
    qe.run_qwen_ardb(_pre(1), on_step=steps.append)
    assert "finetune_slow" in steps


def test_forwards_is_cancelled_to_the_subprocess_runner(monkeypatch):
    # Qwen's generations run up to 1200s — mid-subprocess cancellation matters
    # here even more than for Gemma's shorter, faster subprocess calls.
    captured = {}

    def fake_run(page_image, config, is_cancelled=None):
        captured["is_cancelled"] = is_cancelled
        return "[]"

    monkeypatch.setattr(qe, "run_isolated_inference", fake_run)
    sentinel = lambda: False
    qe.run_qwen_ardb(_pre(1), is_cancelled=sentinel)
    assert captured["is_cancelled"] is sentinel


def test_inference_cancelled_propagates_uncaught_not_failed_soft(monkeypatch):
    """A user Stop mid-subprocess must abort the whole run — unlike a normal
    unparseable-output page (Qwen's expected failure mode), this is not
    something to fail soft on and continue past."""
    from khmer_pipeline.engines.finetune_ardb.subprocess_runner import InferenceCancelled

    def fake_run(page_image, config, is_cancelled=None):
        raise InferenceCancelled("cancelled by user")

    monkeypatch.setattr(qe, "run_isolated_inference", fake_run)
    with pytest.raises(InferenceCancelled):
        qe.run_qwen_ardb(_pre(2), is_cancelled=lambda: True)


# --- registry + API wiring ---

def test_qwen_ardb_is_registered():
    from khmer_pipeline.engines.engine_registry import get_ocr_engine
    assert get_ocr_engine("qwen_ardb") is qe.run_qwen_ardb


def test_qwen_ardb_present_in_api_engines_as_trial():
    from apps.api.api import _ENGINES
    entry = next(e for e in _ENGINES if e["key"] == "qwen_ardb")
    assert entry["group"] == "local"
    assert entry.get("experimental") is True
    assert entry.get("trial") is True


def test_adapter_repo_id_is_the_real_confirmed_adapter():
    # Was a deliberate placeholder while unexposed; must be the real one now
    # that this engine is selectable in the UI.
    assert qe._CONFIG.adapter_repo_id == "Soxavin/qwen35-ardb-lora-v5-e3-ga2"


def test_timeout_is_longer_than_gemmas_default():
    import khmer_pipeline.engines.gemma_ardb_engine as ge
    assert qe._CONFIG.timeout_s > ge._CONFIG.timeout_s
