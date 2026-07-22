"""Drives the 5-stage pipeline off the NiceGUI event loop.

Each blocking stage runs via `run.io_bound` (a thread — same process, so the multi-GB
Surya/Qwen models stay loaded, unlike `run.cpu_bound` which would fork and reload them).
`clear_device_cache()` runs between stages exactly as `app.py`/`pipeline.py` do. Stage
progress and the per-page OCR callback are written to `state.progress` for a UI timer to
reflect. Mirrors the run block of the Streamlit `app.py`.
"""
from __future__ import annotations

import importlib.metadata
from inspect import signature
from pathlib import Path
from time import perf_counter
from typing import Callable

from nicegui import run

from khmer_pipeline.ingest import ingest, resolve_auto_dpi
from khmer_pipeline.preprocess import preprocess, suggest_preprocess_settings, PreprocessConfig
from khmer_pipeline.engines.engine_registry import get_ocr_engine, ACTIVE_CORRECTION_ENGINE
from khmer_pipeline.export import export
from khmer_pipeline.utils.memory import clear_device_cache

from .settings import Settings
from .state import Document


class _RunCancelled(Exception):
    """Raised (from the on_page callback or a stage boundary) to abort a run the
    user cancelled — distinct from a stage failure."""


async def run_pipeline(doc: Document, s: Settings, on_stage: Callable[[str], None] | None = None) -> bool:
    """Run ingest → preprocess → OCR → postprocess → export, filling `doc` stage by
    stage using shared settings `s`. Returns True on success; on failure sets
    `doc.run_error` and returns False. `on_stage(label)` runs on the event loop before
    each stage for UI updates."""
    state = doc  # local alias: the rest of the function fills the document's fields
    is_pdf = Path(state.upload_name or "").suffix.lower() == ".pdf"
    page_indices = s.page_indices(state.doc_page_count) if is_pdf else None
    # Recorded so the UI can map a result index back to its document page while the
    # run is still in flight (the grid upgrades thumbnails as soon as stage 2 lands).
    state.run_page_indices = page_indices
    times: dict[str, float] = {}
    state.run_error = None
    state.progress.active = True
    # "auto" DPI is resolved once here, from the actual document, into a concrete
    # render DPI that ingest and the provenance record both see.
    dpi = resolve_auto_dpi(state.upload_bytes, state.upload_name) if s.dpi == "auto" else int(s.dpi)
    # NOTE: cancel_requested is deliberately NOT cleared here. `reset_run` already
    # provides a fresh progress object; clearing again would silently swallow a
    # cancel that lands in the reset→start window (§2.56 race).

    def _mark(stage: str) -> None:
        state.progress.stage = stage
        if on_stage is not None:
            on_stage(stage)

    async def _stage(label: str, key: str, fn, *args, **kwargs):
        if state.progress.cancel_requested:
            raise _RunCancelled()
        _mark(label)
        t0 = perf_counter()
        result = await run.io_bound(fn, *args, **kwargs)
        if state.progress.cancel_requested:
            raise _RunCancelled()
        times[key] = perf_counter() - t0
        clear_device_cache()
        return result

    try:
        state.ingest_result = await _stage(
            "Reading the document…", "Stage 1 — Ingest",
            ingest, state.upload_bytes, state.upload_name, dpi=dpi, page_indices=page_indices,
        )

        config = PreprocessConfig(
            remove_stamps=s.remove_stamps, sharpen=s.sharpen, normalise=s.normalise,
            deskew=s.deskew, normalise_table_backgrounds=s.normalise_table_backgrounds,
            # `auto` may route to surya_kiri, which reads the geometric-only frame;
            # omitting it silently costs accuracy (§2.30), so both keys request it.
            with_recognition_images=(s.ocr_engine_key in ("surya_kiri", "auto")),
        )
        state.preprocess_result = await _stage(
            "Cleaning the pages…", "Stage 2 — Preprocess",
            preprocess, state.ingest_result, config,
        )

        def _on_page(idx: int, total: int) -> None:
            # Called from the OCR worker thread; scalar writes are GIL-atomic.
            # Raising here aborts the engine's page loop — cancellation is
            # page-granular during the long OCR stage, not just at stage ends.
            if state.progress.cancel_requested:
                raise _RunCancelled()
            state.progress.page = idx + 1
            state.progress.total = total
            state.progress.fraction = (idx + 1) / total if total else 0.0

        def _on_step(step: str) -> None:
            # Same worker-thread contract as _on_page: one scalar write.
            state.progress.step = step

        engine = get_ocr_engine(s.ocr_engine_key)
        # Sub-stage telemetry is opt-in per engine: only pass `on_step` to engines
        # whose signature accepts it, so older engines keep working untouched.
        extra = {}
        try:
            if "on_step" in signature(engine).parameters:
                extra["on_step"] = _on_step
        except (TypeError, ValueError):
            pass  # builtin/C callable with no introspectable signature
        state.surya_result = await _stage(
            "Finding text & tables…", "Stage 3 — OCR",
            engine, state.preprocess_result, on_page=_on_page, **extra,
        )
        state.progress.step = ""

        state.postprocess_result = await _stage(
            "Tidying the text…", "Stage 4 — Post-process",
            ACTIVE_CORRECTION_ENGINE, state.surya_result,
            skip_qwen=not s.enable_qwen, anomaly_threshold=s.anomaly_threshold,
        )

        try:
            surya_ver = importlib.metadata.version("surya-ocr")
        except Exception:
            surya_ver = "unknown"
        provenance = {
            "engine": s.ocr_engine_key,
            "surya_ocr_version": surya_ver,
            "dpi": dpi,
            "dpi_auto": s.dpi == "auto",
            "preprocess": {
                "remove_stamps": s.remove_stamps, "sharpen": s.sharpen,
                "normalise": s.normalise, "deskew": s.deskew,
                "normalise_table_backgrounds": s.normalise_table_backgrounds,
            },
            # Raw quality scores of the pages this run actually ingested — the
            # basis of the Auto suggestions, logged for the project report.
            "preprocess_scores": suggest_preprocess_settings(
                state.ingest_result.page_images)["scores"],
            "stitch_pages": s.stitch_pages,
            "convert_numerals": s.convert_numerals,
            "repair_tables": s.repair_tables,
        }
        state.export_result = await _stage(
            "Preparing your files…", "Stage 5 — Export",
            export, state.postprocess_result,
            convert_numerals=s.convert_numerals, repair_tables=s.repair_tables,
            stitch_pages=s.stitch_pages, provenance=provenance,
        )
    except _RunCancelled:
        # No half-populated document: clear partial stage results (reset_run also
        # replaces `progress`, resetting active + cancel_requested for the next run),
        # then record why there are no results.
        state.reset_run()
        state.run_error = "Extraction cancelled."
        clear_device_cache()
        return False
    except Exception as e:  # surface the failing stage, same shape as app.py
        state.run_error = f"{state.progress.stage.rstrip('…')} failed: {e}"
        state.progress.active = False
        return False

    state.stage_times = times
    state.last_key = s.settings_key(state.upload_id or "")
    state.progress.active = False
    return True
