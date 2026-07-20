"""Tests for webapp.runner — pipeline driving and run cancellation.

Cancellation contract: setting `doc.progress.cancel_requested` stops the run at
the next stage boundary AND mid-OCR (the per-page callback aborts), partial
stage results are cleared (no half-populated document), and the run reports
"cancelled" rather than a failure.
"""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
from unittest.mock import patch

from webapp.runner import run_pipeline
from webapp.settings import Settings
from webapp.state import Document


def _doc() -> Document:
    return Document("doc.pdf", b"%PDF", "id1", 3)


def _run_with(stubs: dict, doc: Document, s: Settings | None = None) -> bool:
    """Run the pipeline with named runner-module attributes stubbed."""
    async def main():
        with ExitStack() as stack:
            for name, fn in stubs.items():
                stack.enter_context(patch(f"webapp.runner.{name}", fn))
            stack.enter_context(patch("webapp.runner.clear_device_cache"))
            return await run_pipeline(doc, s or Settings())
    return asyncio.run(main())


def test_cancel_before_next_stage_stops_run_and_clears_partials():
    doc = _doc()

    def fake_ingest(*a, **kw):
        doc.progress.cancel_requested = True  # user hits Stop during stage 1
        return "INGEST"

    def fail(*a, **kw):  # stages after the cancel must never run
        raise AssertionError("stage ran after cancellation")

    ok = _run_with({"ingest": fake_ingest, "preprocess": fail}, doc)
    assert ok is False
    assert "cancelled" in (doc.run_error or "").lower()
    assert doc.ingest_result is None       # partial results cleared
    assert doc.progress.active is False
    assert doc.progress.cancel_requested is False  # flag reset for the next run


def test_cancel_mid_ocr_aborts_via_on_page():
    doc = _doc()

    def fake_engine(pre, on_page=None):
        # engine reports page 1, user cancels, page 2's callback aborts the loop
        on_page(0, 3)
        doc.progress.cancel_requested = True
        on_page(1, 3)  # must raise inside the engine
        raise AssertionError("OCR continued past cancellation")

    ok = _run_with({
        "ingest": lambda *a, **kw: "INGEST",
        "preprocess": lambda *a, **kw: "PRE",
        "get_ocr_engine": lambda key: fake_engine,
    }, doc)
    assert ok is False
    assert "cancelled" in (doc.run_error or "").lower()
    assert doc.surya_result is None and doc.ingest_result is None


def test_uncancelled_run_reports_stage_failure_not_cancel():
    doc = _doc()

    def boom(*a, **kw):
        raise RuntimeError("boom")

    ok = _run_with({"ingest": boom}, doc)
    assert ok is False
    assert "failed" in (doc.run_error or "")
    assert "cancelled" not in (doc.run_error or "").lower()


def test_provenance_includes_preprocess_scores():
    import numpy as np
    from types import SimpleNamespace

    doc = _doc()
    captured = {}

    def fake_export(pp, provenance=None, **kw):
        captured["provenance"] = provenance
        return "EXPORT"

    flat = np.full((50, 50, 3), 128, dtype=np.uint8)
    ok = _run_with({
        "ingest": lambda *a, **kw: SimpleNamespace(page_images=[flat]),
        "preprocess": lambda *a, **kw: "PRE",
        "get_ocr_engine": lambda key: lambda pre, on_page=None: "OCR",
        "ACTIVE_CORRECTION_ENGINE": lambda *a, **kw: "POST",
        "export": fake_export,
    }, doc)
    assert ok is True
    scores = captured["provenance"]["preprocess_scores"]
    assert set(scores) == {"laplacian_var", "contrast_std", "skew_deg", "stamp_ink_ratio"}
    assert all(isinstance(v, float) for v in scores.values())


def _flat_ingest():
    """Ingest stub with real page images — provenance scoring reads them."""
    import numpy as np
    from types import SimpleNamespace
    return lambda *a, **kw: SimpleNamespace(page_images=[np.full((50, 50, 3), 128, dtype=np.uint8)])


def test_engine_receives_on_step_when_it_accepts_one():
    """Engines that expose an `on_step` hook get sub-stage telemetry wired in, so
    the long OCR stage reports what it is doing inside each page."""
    doc = _doc()
    seen: list[str] = []

    def engine_with_steps(result, on_page=None, on_step=None):
        assert on_step is not None, "runner must pass on_step to a willing engine"
        on_step("layout")
        seen.append(doc.progress.step)
        on_step("tables")
        seen.append(doc.progress.step)
        return "OCR"

    ok = _run_with({
        "ingest": _flat_ingest(),
        "preprocess": lambda *a, **kw: "PRE",
        "get_ocr_engine": lambda key: engine_with_steps,
        "ACTIVE_CORRECTION_ENGINE": lambda *a, **kw: "POST",
        "export": lambda *a, **kw: "EXPORT",
    }, doc)

    assert ok is True
    assert seen == ["layout", "tables"]
    # The step is transient: it must not linger once the run is over.
    assert doc.progress.step == ""


def test_engine_without_on_step_is_called_unchanged():
    """Engines predating sub-stage telemetry keep working — the runner must not
    pass a keyword they cannot accept."""
    doc = _doc()

    def legacy_engine(result, on_page=None):
        return "OCR"

    ok = _run_with({
        "ingest": _flat_ingest(),
        "preprocess": lambda *a, **kw: "PRE",
        "get_ocr_engine": lambda key: legacy_engine,
        "ACTIVE_CORRECTION_ENGINE": lambda *a, **kw: "POST",
        "export": lambda *a, **kw: "EXPORT",
    }, doc)

    assert ok is True
