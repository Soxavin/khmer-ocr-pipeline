"""Tests for the Gemini cloud OCR engine (gemini_engine.py).

Gemini is called over the network, so every test here monkeypatches the client
with a canned response — no key, no network. The engine's contract:
  - fail FAST on setup (no key / SDK) with an actionable error, before any call;
  - fail SOFT per page (one page's API/parse error leaves that page empty and
    the run continues);
  - always emit an audit warning naming what was sent to Google;
  - reuse the production HTML-table parser so colspan/rowspan survive.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from khmer_pipeline.models import PreprocessResult
import khmer_pipeline.engines.gemini_engine as ge


def _pre(n_pages: int = 1) -> PreprocessResult:
    imgs = [np.full((40, 60, 3), 255, dtype=np.uint8) for _ in range(n_pages)]
    return PreprocessResult(source_name="doc", page_images=imgs, dpi=200, page_count=n_pages)


def _layout(*elements: dict) -> str:
    return json.dumps(list(elements), ensure_ascii=False)


class _FakeResp:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    """Stands in for client.models; scripted per-call outputs or an exception."""
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def generate_content(self, **kwargs):
        out = self._outputs[self.calls]
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return _FakeResp(out)


class _FakeClient:
    def __init__(self, outputs):
        self.models = _FakeModels(outputs)


def _install_fake(monkeypatch, outputs, key="test-key"):
    """Wire a fake genai client returning `outputs` (one per page)."""
    if key is not None:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    created = {}
    monkeypatch.setattr(ge, "_make_client", lambda k: _FakeClient(outputs))
    return created


# --- happy path: layout JSON -> tables + text blocks ---

def test_table_html_becomes_a_table_with_spans(monkeypatch):
    layout = _layout(
        {"bbox": [0, 0, 10, 10], "category": "Title", "text": "T"},
        {"bbox": [0, 10, 100, 200], "category": "Table",
         "text": "<table><tr><td colspan=2>H</td></tr><tr><td>a</td><td>1</td></tr></table>"},
    )
    _install_fake(monkeypatch, [layout])
    result = ge.run_gemini(_pre(1))

    page = result.pages[0]
    assert len(page.tables) == 1
    cells = page.tables[0]["cells"]
    # 2 columns inferred from the data row; header spans both.
    assert max(c["col_id"] for c in cells) == 1
    assert any(c.get("col_span", 1) == 2 for c in cells)  # span survived the HTML parse
    assert any(b["text"] == "T" for b in page.text_blocks)


def test_ocr_text_is_populated(monkeypatch):
    layout = _layout({"bbox": [0, 0, 5, 5], "category": "Text", "text": "hello"})
    _install_fake(monkeypatch, [layout])
    result = ge.run_gemini(_pre(1))
    assert "hello" in result.pages[0].ocr_text


def test_nested_polygon_bbox_does_not_crash(monkeypatch):
    # Live Gemini returned bbox as a nested polygon [[x,y],...], not a flat box.
    # The engine must coerce it, not crash (the failure the first real call hit).
    layout = _layout(
        {"bbox": [[10, 20], [90, 20], [90, 60], [10, 60]], "category": "Text", "text": "poly"},
        {"bbox": [[0, 0], [100, 200]], "category": "Table",
         "text": "<table><tr><td>a</td><td>1</td></tr></table>"},
    )
    _install_fake(monkeypatch, [layout])
    page = ge.run_gemini(_pre(1)).pages[0]
    block = next(b for b in page.text_blocks if b["text"] == "poly")
    assert block["bbox"] == [10.0, 20.0, 90.0, 60.0]  # bounding rect of the polygon
    assert page.tables and page.tables[0]["image_bbox"] == [0.0, 0.0, 100.0, 200.0]


def test_malformed_bbox_degrades_to_empty(monkeypatch):
    layout = _layout({"bbox": "garbage", "category": "Text", "text": "x"})
    _install_fake(monkeypatch, [layout])
    page = ge.run_gemini(_pre(1)).pages[0]
    assert page.text_blocks[0]["bbox"] == []  # unusable bbox -> [], not a crash


def test_non_table_categories_become_text_blocks(monkeypatch):
    layout = _layout(
        {"bbox": [0, 0, 5, 5], "category": "Section-header", "text": "S"},
        {"bbox": [0, 6, 5, 9], "category": "Page-footer", "text": "F"},
    )
    _install_fake(monkeypatch, [layout])
    page = ge.run_gemini(_pre(1)).pages[0]
    assert {b["text"] for b in page.text_blocks} == {"S", "F"}
    assert page.tables == []


# --- fail-fast on setup ---

def test_missing_key_raises_before_any_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        ge.run_gemini(_pre(1))


def test_sdk_absent_raises_actionable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    # Simulate the SDK not being importable.
    def _boom(key):
        raise ImportError("No module named 'google.genai'")
    monkeypatch.setattr(ge, "_make_client", _boom)
    with pytest.raises(RuntimeError, match="google-genai"):
        ge.run_gemini(_pre(1))


# --- fail-soft per page ---

def test_one_bad_page_leaves_others_intact(monkeypatch):
    good = _layout({"bbox": [0, 0, 5, 5], "category": "Text", "text": "ok"})
    # page 1 ok, page 2 raises on every attempt (incl. the retry).
    boom = RuntimeError("500 backend error")
    _install_fake(monkeypatch, [good, boom, boom])
    result = ge.run_gemini(_pre(2))
    assert "ok" in result.pages[0].ocr_text
    assert result.pages[1].tables == [] and result.pages[1].text_blocks == []
    assert any("page 2" in w.lower() or "page 2" in w for w in result.warnings)


def test_unparseable_output_is_soft(monkeypatch):
    _install_fake(monkeypatch, ["not json at all"])
    result = ge.run_gemini(_pre(1))  # must not raise
    assert result.pages[0].text_blocks == []
    assert any("pars" in w.lower() for w in result.warnings)


def test_rate_limit_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setattr(ge.time, "sleep", lambda *_: None)  # no real backoff wait
    good = _layout({"bbox": [0, 0, 5, 5], "category": "Text", "text": "recovered"})
    _install_fake(monkeypatch, [RuntimeError("429 RESOURCE_EXHAUSTED"), good])
    result = ge.run_gemini(_pre(1))
    assert "recovered" in result.pages[0].ocr_text


# --- audit + telemetry contract ---

def test_audit_warning_names_google_and_model(monkeypatch):
    _install_fake(monkeypatch, [_layout({"category": "Text", "text": "x", "bbox": [0, 0, 1, 1]})])
    result = ge.run_gemini(_pre(1))
    assert any("Google" in w and "Gemini" in w for w in result.warnings)


def test_accepts_on_page_and_on_step(monkeypatch):
    _install_fake(monkeypatch, [_layout({"category": "Text", "text": "x", "bbox": [0, 0, 1, 1]})])
    seen = []
    ge.run_gemini(_pre(1), on_page=lambda i, t: seen.append((i, t)), on_step=lambda s: None)
    assert seen == [(0, 1)]


# --- registry + API wiring ---

def test_gemini_is_registered():
    from khmer_pipeline.engines.engine_registry import get_ocr_engine
    assert get_ocr_engine("gemini") is ge.run_gemini


def test_api_engines_are_grouped_and_gemini_is_cloud():
    from webapp.api import _ENGINES
    assert all("group" in e for e in _ENGINES)
    gemini = next(e for e in _ENGINES if e["key"] == "gemini")
    assert gemini["group"] == "cloud"
    assert all(e["group"] == "local" for e in _ENGINES if e["key"] != "gemini")
