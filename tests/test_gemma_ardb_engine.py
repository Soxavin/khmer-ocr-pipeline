"""Tests for the Gemma 4 E2B ARDB fine-tune engine (gemma_ardb_engine.py).

The isolated subprocess is monkeypatched here — no real uv/model download.
Contract mirrors gemini_engine.py: fail SOFT per page (a subprocess/parse
error leaves that page empty and the run continues), always emit an audit
warning, reuse the shared finetune_ardb transform (HTML table format)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from khmer_pipeline.models import PreprocessResult
import khmer_pipeline.engines.gemma_ardb_engine as ge


def _pre(n_pages: int = 1) -> PreprocessResult:
    imgs = [np.full((40, 60, 3), 255, dtype=np.uint8) for _ in range(n_pages)]
    return PreprocessResult(source_name="doc", page_images=imgs, dpi=200, page_count=n_pages)


def _regions(*objs) -> str:
    return json.dumps(list(objs), ensure_ascii=False)


def _install_fake(monkeypatch, outputs):
    """outputs: list of str (stdout) or Exception, one per page call."""
    calls = {"n": 0}

    def fake_run(page_image, config):
        out = outputs[calls["n"]]
        calls["n"] += 1
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(ge, "run_isolated_inference", fake_run)
    return calls


# --- happy path ---

def test_table_region_becomes_a_table(monkeypatch):
    regions = _regions(
        {"box_2d": [0, 0, 500, 1000], "label": "Table",
         "text": "<table><tr><td>a</td><td>1</td></tr></table>"},
    )
    _install_fake(monkeypatch, [regions])
    result = ge.run_gemma_ardb(_pre(1))
    page = result.pages[0]
    assert len(page.tables) == 1
    cells = page.tables[0]["cells"]
    assert any(c["text_lines"] and c["text_lines"][0]["text"] == "a" for c in cells)


def test_text_region_becomes_a_text_block(monkeypatch):
    regions = _regions({"box_2d": [0, 0, 100, 100], "label": "Text", "text": "hello"})
    _install_fake(monkeypatch, [regions])
    result = ge.run_gemma_ardb(_pre(1))
    assert any(b["text"] == "hello" for b in result.pages[0].text_blocks)
    assert "hello" in result.pages[0].ocr_text


# --- fail soft per page ---

def test_subprocess_failure_leaves_page_empty(monkeypatch):
    good = _regions({"box_2d": [0, 0, 10, 10], "label": "Text", "text": "ok"})
    boom = RuntimeError("gemma_ardb_infer.py exited 1: ModuleNotFoundError: peft")
    _install_fake(monkeypatch, [good, boom])
    result = ge.run_gemma_ardb(_pre(2))
    assert "ok" in result.pages[0].ocr_text
    assert result.pages[1].tables == [] and result.pages[1].text_blocks == []
    assert any("page 2" in w.lower() for w in result.warnings)


def test_unparseable_output_is_soft(monkeypatch):
    _install_fake(monkeypatch, ["not json at all, and no repairable braces"])
    result = ge.run_gemma_ardb(_pre(1))  # must not raise
    assert result.pages[0].text_blocks == []
    assert any("pars" in w.lower() for w in result.warnings)


def test_repairable_truncated_json_recovers(monkeypatch):
    # missing closing ']' — the shared repair path should recover it.
    truncated = '[{"box_2d": [0,0,10,10], "label": "Text", "text": "recovered"}'
    _install_fake(monkeypatch, [truncated])
    result = ge.run_gemma_ardb(_pre(1))
    assert "recovered" in result.pages[0].ocr_text


# --- audit + telemetry contract ---

def test_audit_warning_names_the_model(monkeypatch):
    regions = _regions({"box_2d": [0, 0, 1, 1], "label": "Text", "text": "x"})
    _install_fake(monkeypatch, [regions])
    result = ge.run_gemma_ardb(_pre(1))
    assert any("gemma" in w.lower() for w in result.warnings)


def test_accepts_on_page_and_on_step(monkeypatch):
    regions = _regions({"box_2d": [0, 0, 1, 1], "label": "Text", "text": "x"})
    _install_fake(monkeypatch, [regions])
    seen = []
    ge.run_gemma_ardb(_pre(1), on_page=lambda i, t: seen.append((i, t)), on_step=lambda s: None)
    assert seen == [(0, 1)]


# --- registry + API wiring ---

def test_gemma_ardb_is_registered():
    from khmer_pipeline.engines.engine_registry import get_ocr_engine
    assert get_ocr_engine("gemma_ardb") is ge.run_gemma_ardb


def test_gemma_ardb_present_in_api_engines_as_experimental():
    from webapp.api import _ENGINES
    entry = next(e for e in _ENGINES if e["key"] == "gemma_ardb")
    assert entry["group"] == "local"
    assert entry.get("experimental") is True


