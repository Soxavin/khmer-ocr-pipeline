"""REST API for the React frontend, registered on `nicegui.app` (a FastAPI
subclass) — one process, models loaded once, thin handlers over the tested
webapp modules (`runner`, `state`, `settings`, `tables`, `downloads`, `edits`).

Import this module BEFORE `ui.run` (done in `webapp.main`). The built React
bundle, when present at `frontend/dist`, is mounted at `/app`.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import urllib.parse
import zipfile
from dataclasses import asdict, fields
from pathlib import Path

import fitz
from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from nicegui import app

from khmer_pipeline.export import grid_to_csv, tables_to_xlsx
from khmer_pipeline.ingest import ingest
from khmer_pipeline.preprocess import suggest_preprocess_settings
from khmer_pipeline.model_config import CELL_CONF_LOW
from khmer_pipeline.utils.backend_status import llama_server_running

from . import components, downloads, edits, registry, tables
from .runner import run_pipeline
from .settings import Settings
from .state import Document


class ApiError(Exception):
    """API error with an HTTP status — handled as JSON. (FastAPI's HTTPException
    is intercepted by NiceGUI's HTML error-page handler, which is wrong for API
    clients; a dedicated exception type keeps API errors as JSON.)"""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

# Task-language engine labels (no jargon in the UI; keys map to the registry).
_ENGINES = [
    {"key": "surya", "label": "Standard",
     "guidance": "Best all-round, fastest. Use for number-heavy or wide tables."},
    {"key": "surya_kiri", "label": "Khmer-text specialist",
     "guidance": "Strongest on Khmer-text-heavy narrow tables (ARDB bulletins). Slower."},
    {"key": "surya_kiri_vlm", "label": "Best structure (slow)",
     "guidance": "Keeps spanning headers intact and upgrades Khmer cells when safe. Slowest."},
]


def _probe_pages(name: str, data: bytes) -> int:
    """Page count for PDFs (0 = unreadable), 1 for images."""
    if Path(name).suffix.lower() != ".pdf":
        return 1
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return len(doc)
    except Exception:
        return 0


def _doc_or_404(doc_id: str) -> Document:
    doc = registry.get(doc_id)
    if doc is None:
        raise ApiError(404, f"Unknown document {doc_id!r}")
    return doc


def _doc_summary(doc: Document) -> dict:
    if doc.progress.active:
        status = "running"
    elif doc.has_results:
        status = "done"
    elif doc.run_error and "cancelled" in doc.run_error:
        status = "stopped"  # user asked for the stop: neutral state, not a failure
    elif doc.run_error:
        status = "error"
    else:
        status = "queued"
    total_tables = reviewed = 0
    if doc.has_results:
        tids = [tid for tid, _g, _c in tables.all_export_tables(doc.export_result.document_json)]
        total_tables = len(tids)
        reviewed = sum(1 for tid in tids if doc.reviewed.get(tid))
    return {
        "id": doc.upload_id,
        "name": doc.upload_name,
        "pages": doc.doc_page_count,
        "size_kb": round(len(doc.upload_bytes) / 1024, 1),
        "status": status,
        "total_tables": total_tables,
        "reviewed_tables": reviewed,
    }


@app.get("/api/meta")
def api_meta() -> dict:
    return {
        "engines": _ENGINES,
        "defaults": asdict(Settings()),
        "setting_fields": [f.name for f in fields(Settings)],
        "backend_ready": llama_server_running(),
    }


@app.post("/api/documents")
async def api_upload(files: list[UploadFile]) -> dict:
    added = []
    for f in files:
        data = await f.read()
        name = f.filename or "document"
        doc = registry.add(Document(
            upload_name=name,
            upload_bytes=data,
            upload_id=hashlib.md5(data).hexdigest()[:12],
            doc_page_count=_probe_pages(name, data),
        ))
        added.append(_doc_summary(doc))
    return {"documents": added}


@app.get("/api/documents")
def api_list() -> dict:
    return {"documents": [_doc_summary(d) for d in registry.all_documents()]}


@app.delete("/api/documents")
def api_clear() -> dict:
    registry.clear()
    return {"ok": True}


@app.delete("/api/documents/{doc_id}")
def api_delete(doc_id: str) -> dict:
    if not registry.remove(doc_id):
        raise ApiError(404, f"Unknown document {doc_id!r}")
    return {"ok": True}


@app.get("/api/documents/{doc_id}/status")
def api_status(doc_id: str) -> dict:
    doc = _doc_or_404(doc_id)
    p = doc.progress
    return {
        "active": p.active,
        "stage": p.stage,
        "step": p.step,
        "page": p.page,
        "total": p.total,
        "fraction": p.fraction,
        "has_results": doc.has_results,
        "run_error": doc.run_error,
        "last_run_settings": registry.last_run_settings(doc_id),
    }


@app.get("/api/documents/{doc_id}/suggest")
def api_suggest(doc_id: str) -> dict:
    """Auto-preprocess suggestion for a document: cheap image-quality scores plus
    suggested toggle values (advisory — the UI pre-fills, the user decides).
    Rasterizes the upload lazily on first call (upload stores only bytes; ingest
    normally happens at run time) and caches the result on the doc record."""
    doc = _doc_or_404(doc_id)
    if doc.preprocess_suggestion is None:
        try:
            pages = ingest(doc.upload_bytes, doc.upload_name).page_images
        except Exception:
            # Unreadable upload: the suggestion is advisory, so degrade to the
            # empty (all-defaults) shape — the run itself will surface the error.
            pages = []
        doc.preprocess_suggestion = suggest_preprocess_settings(pages)
    return doc.preprocess_suggestion


# ---------------------------------------------------------------------------
# Run + cancel
# ---------------------------------------------------------------------------

def _settings_from(payload: dict) -> Settings:
    """Validate a settings JSON body into a `Settings`, 400 on anything off."""
    valid = {f.name for f in fields(Settings)}
    unknown = set(payload) - valid
    if unknown:
        raise ApiError(400, f"Unknown settings: {sorted(unknown)}")
    s = Settings(**payload)
    if s.ocr_engine_key not in {e["key"] for e in _ENGINES}:
        raise ApiError(400, f"Unknown engine {s.ocr_engine_key!r}")
    if s.invalid_range:
        raise ApiError(400, "Page range end is before start.")
    return s


async def _execute_run(doc: Document, s: Settings) -> None:
    """Run the pipeline holding the (already acquired) global run lock. The
    `finally` release guarantees a cancelled or crashed run never leaves the
    registry locked."""
    try:
        doc.reset_run()
        ok = await run_pipeline(doc, s)
        if ok:
            registry.set_last_run_settings(doc.upload_id, asdict(s))
    except Exception as e:  # never propagate into the event loop: surface on the doc
        doc.run_error = f"Extraction failed: {e}"
        doc.progress = type(doc.progress)()
    finally:
        registry.run_lock.release()


@app.post("/api/documents/{doc_id}/run", status_code=202)
async def api_run(doc_id: str, payload: dict) -> dict:
    doc = _doc_or_404(doc_id)
    s = _settings_from(payload)
    if registry.run_lock.locked():
        raise ApiError(409, "Another extraction is already running.")
    # Acquire synchronously within the request (uncontended, so no await gap a
    # second request could slip through), then hand off to a background task —
    # the run must survive client refresh/disconnect; ■ Stop is the cancel path.
    await registry.run_lock.acquire()
    asyncio.create_task(_execute_run(doc, s))
    return {"started": True}


@app.post("/api/documents/{doc_id}/cancel")
def api_cancel(doc_id: str) -> dict:
    doc = _doc_or_404(doc_id)
    if doc.progress.active:
        doc.progress.cancel_requested = True
        return {"cancelling": True}
    return {"cancelling": False}


# ---------------------------------------------------------------------------
# Results: overview / page / image / export
# ---------------------------------------------------------------------------

def _doc_with_results(doc_id: str) -> Document:
    doc = _doc_or_404(doc_id)
    if not doc.has_results:
        raise ApiError(409, "No results yet — run the extraction first.")
    return doc


@app.get("/api/documents/{doc_id}/overview")
def api_overview(doc_id: str) -> dict:
    doc = _doc_with_results(doc_id)
    doc_json = doc.export_result.document_json
    warnings = list(doc.surya_result.warnings) + list(getattr(doc.postprocess_result, "warnings", []))
    return {
        "pages": len(doc.preprocess_result.page_images),
        "total_tables": sum(len(p.tables) for p in doc.surya_result.pages),
        "warnings": warnings,
        "stitched": tables.is_stitched(doc_json),
        "stage_times": doc.stage_times,
    }


@app.get("/api/documents/{doc_id}/pages/{n}")
def api_page(doc_id: str, n: int) -> dict:
    doc = _doc_with_results(doc_id)
    if not 0 <= n < len(doc.preprocess_result.page_images):
        raise ApiError(404, f"No page {n}")
    doc_json = doc.export_result.document_json
    surya_page = doc.surya_result.pages[n]
    page_tables = []
    for tid, grid, conf in tables.page_export_tables(doc_json, n):
        edited = tid in doc.edited_tables
        page_tables.append({
            "table_id": tid,
            "grid": doc.edited_tables.get(tid, grid),
            "original_grid": grid,   # diff view baseline
            "confidence": conf,
            "edited": edited,
            "verified": bool(doc.reviewed.get(tid)),
        })
    pages_json = doc_json.get("pages", [])
    corrected = doc.edited_text.get(n)
    if corrected is None and n < len(pages_json):
        corrected = pages_json[n].get("corrected_text", "")
    # table_id → region bbox for image↔table linking (table-level: the pipeline
    # exposes no per-cell geometry). Empty when stitched — doc-tables span pages.
    page_blocks = tables.page_table_blocks(doc_json, n)
    bbox_index = ({} if tables.is_stitched(doc_json)
                  else components.table_bbox_index(surya_page.tables, page_blocks))
    return {
        "corrected_text": corrected or "",
        "tables": page_tables,
        "text_blocks": surya_page.text_blocks,
        "table_bboxes": [t.get("bbox") for t in surya_page.tables],
        "table_bbox_index": bbox_index,
        "qwen_used": bool(doc.postprocess_result.pages[n].qwen_used),
    }


@app.put("/api/documents/{doc_id}/tables/{table_id}")
def api_put_table(doc_id: str, table_id: str, payload: dict) -> dict:
    doc = _doc_with_results(doc_id)
    known = {tid for tid, _g, _c in tables.all_export_tables(doc.export_result.document_json)}
    if table_id not in known:
        raise ApiError(404, f"Unknown table {table_id!r}")
    grid = payload.get("grid")
    if (not isinstance(grid, list) or not grid
            or not all(isinstance(row, list) and all(isinstance(c, str) for c in row) for row in grid)
            or len({len(row) for row in grid}) != 1):
        raise ApiError(400, "grid must be a non-empty rectangular list of string rows")
    doc.edited_tables[table_id] = grid
    return {"ok": True, "edited": True}


@app.delete("/api/documents/{doc_id}/tables/{table_id}")
def api_reset_table(doc_id: str, table_id: str) -> dict:
    doc = _doc_with_results(doc_id)
    doc.edited_tables.pop(table_id, None)
    return {"ok": True, "edited": False}


# Failure-mode reason priority (highest first). A cell with several flags is
# reported as one issue keyed on its most severe reason; `reasons` keeps them all.
_REASON_PRIORITY = [
    "numeric_mismatch", "sequence_illegal", "digit_mixed",
    "numeric_unparseable", "structure_ragged", "low_conf",
]
_REASON_RANK = {r: i for i, r in enumerate(_REASON_PRIORITY)}


@app.get("/api/documents/{doc_id}/lowconf")
def api_lowconf(doc_id: str) -> dict:
    """Triage index: every table cell carrying a validator flag or under the
    calibrated CELL_CONF_LOW confidence bucket. One issue per cell, keyed on its
    highest-priority reason; validator issues first (priority order), then
    low_conf worst-first. `page` is None when stitched; `conf` may be null for
    cells flagged only by validators."""
    doc = _doc_with_results(doc_id)
    doc_json = doc.export_result.document_json
    if tables.is_stitched(doc_json):
        page_iter: list[tuple[int | None, list[dict]]] = [(None, doc_json["document_tables"])]
    else:
        page_iter = [(i, pd.get("tables", [])) for i, pd in enumerate(doc_json.get("pages", []))]
    issues = []
    for page_idx, blocks in page_iter:
        for block in blocks:
            tid, grid, conf = tables.block_to_table(block)
            flags = tables.block_flags(block)
            grid = doc.edited_tables.get(tid, grid)
            rows = max(len(grid), len(conf))
            for r in range(rows):
                width = max(len(grid[r]) if r < len(grid) else 0,
                            len(conf[r]) if r < len(conf) else 0)
                for c in range(width):
                    text = grid[r][c] if r < len(grid) and c < len(grid[r]) else ""
                    # Blank cells are usually intentional table structure, not
                    # OCR errors — flagging them floods triage with noise.
                    if not text.strip():
                        continue
                    v = conf[r][c] if r < len(conf) and c < len(conf[r]) else None
                    reasons = list(flags.get((r, c), []))
                    if v is not None and v < CELL_CONF_LOW and "low_conf" not in reasons:
                        reasons.append("low_conf")
                    if not reasons:
                        continue
                    reasons.sort(key=lambda x: _REASON_RANK.get(x, len(_REASON_PRIORITY)))
                    issues.append({"page": page_idx, "table_id": tid, "row": r, "col": c,
                                   "conf": v, "text": text,
                                   "reason": reasons[0], "reasons": reasons})
    # Validator-flagged issues first (by reason priority); low_conf-only issues
    # after, worst confidence first. Deterministic tie-break on position.
    def _key(i: dict):
        rank = _REASON_RANK.get(i["reason"], len(_REASON_PRIORITY))
        is_lowconf_only = i["reason"] == "low_conf"
        conf = i["conf"] if i["conf"] is not None else 1.0
        return (is_lowconf_only, rank, conf, i["table_id"], i["row"], i["col"])
    issues.sort(key=_key)
    return {"issues": issues}


@app.put("/api/documents/{doc_id}/review/{table_id}")
def api_review(doc_id: str, table_id: str, payload: dict) -> dict:
    doc = _doc_with_results(doc_id)
    known = {tid for tid, _g, _c in tables.all_export_tables(doc.export_result.document_json)}
    if table_id not in known:
        raise ApiError(404, f"Unknown table {table_id!r}")
    doc.reviewed[table_id] = bool(payload.get("verified"))
    return {"ok": True, "verified": doc.reviewed[table_id]}


# Pre-replace snapshots of edited_tables, per doc — makes replace-all undoable
# (it is the only bulk mutation; everything else has per-table undo).
_replace_backup: dict[str, dict[str, list[list[str]]]] = {}


@app.post("/api/documents/{doc_id}/replace")
def api_replace(doc_id: str, payload: dict) -> dict:
    doc = _doc_with_results(doc_id)
    find = str(payload.get("find", ""))
    if not find:
        raise ApiError(400, "find must be non-empty")
    changed, total = edits.replace_across(downloads.final_tables(doc), find, str(payload.get("replace", "")))
    if changed:
        _replace_backup[doc_id] = {tid: [row[:] for row in g] for tid, g in doc.edited_tables.items()}
    doc.edited_tables.update(changed)
    return {"total": total, "tables_changed": len(changed)}


@app.post("/api/documents/{doc_id}/replace/undo")
def api_replace_undo(doc_id: str) -> dict:
    doc = _doc_with_results(doc_id)
    backup = _replace_backup.pop(doc_id, None)
    if backup is None:
        raise ApiError(409, "Nothing to undo — no replace has been made.")
    doc.edited_tables.clear()
    doc.edited_tables.update(backup)
    return {"ok": True}


@app.put("/api/documents/{doc_id}/pages/{n}/text")
def api_put_page_text(doc_id: str, n: int, payload: dict) -> dict:
    doc = _doc_with_results(doc_id)
    if not 0 <= n < len(doc.preprocess_result.page_images):
        raise ApiError(404, f"No page {n}")
    doc.edited_text[n] = str(payload.get("text", ""))
    return {"ok": True}


@app.get("/api/documents/{doc_id}/preview/{n}")
def api_preview_image(doc_id: str, n: int) -> Response:
    """Raw page image BEFORE any run: lets the analyst see the document (and pick a
    page range) pre-extraction. Rasterizes lazily on first call and caches the ingest
    on the doc record; a later run simply replaces it."""
    doc = _doc_or_404(doc_id)
    if doc.ingest_result is None:
        try:
            doc.ingest_result = ingest(doc.upload_bytes, doc.upload_name)
        except Exception as e:
            raise ApiError(422, f"Cannot render a preview of this file: {e}")
    imgs = doc.ingest_result.page_images
    if not 0 <= n < len(imgs):
        raise ApiError(404, f"No page {n}")
    buf = io.BytesIO()
    Image.fromarray(imgs[n]).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/documents/{doc_id}/pages/{n}/image")
def api_page_image(doc_id: str, n: int, variant: str = "processed") -> Response:
    doc = _doc_with_results(doc_id)
    if variant not in ("processed", "original"):
        raise ApiError(400, f"Unknown variant {variant!r}")
    imgs = (doc.preprocess_result if variant == "processed" else doc.ingest_result).page_images
    if not 0 <= n < len(imgs):
        raise ApiError(404, f"No page {n}")
    buf = io.BytesIO()
    Image.fromarray(imgs[n]).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


def _attachment(filename: str, fallback: str) -> dict[str, str]:
    """Headers are latin-1: real documents have Khmer filenames, so send an ASCII
    fallback plus the RFC 5987 UTF-8 form browsers actually use."""
    quoted = urllib.parse.quote(filename)
    return {"Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quoted}"}


def _export_parts(doc: Document, combine: bool = True) -> tuple[Settings, str, dict, str, list]:
    """(settings, stem, patched_json, text_report, final_tables) for one done doc.

    `combine` joins continuation tables across pages into one table each — the
    export-time stitch. Review always runs per-page (so page↔image linking works),
    so this is where an analyst's "one table for Excel" is produced, from the
    grids their edits already live in. The JSON keeps its per-page structure: it
    is the faithful record, the CSV/XLSX are the working artifacts.
    """
    s = Settings(**(registry.last_run_settings(doc.upload_id) or {}))
    doc_json = downloads.patched_document_json(doc)
    stem = Path(doc.upload_name).stem
    ft = downloads.final_tables(doc)
    if combine and not tables.is_stitched(doc_json):
        ft = tables.stitch_grids(ft, stem)
    all_text = downloads.text_report(doc, s, doc_json.get("pages", []))
    return s, stem, doc_json, all_text, ft


@app.get("/api/documents/{doc_id}/export/zip")
def api_export_zip(doc_id: str, combine: bool = True) -> Response:
    doc = _doc_with_results(doc_id)
    s, stem, doc_json, all_text, ft = _export_parts(doc, combine)
    data = downloads.zip_bundle(doc, s, stem, doc_json, all_text, ft)
    return Response(data, media_type="application/zip",
                    headers=_attachment(f"{stem}_extracted.zip", "extracted.zip"))


@app.get("/api/documents/{doc_id}/export/csv/{table_id}")
def api_export_csv(doc_id: str, table_id: str) -> Response:
    doc = _doc_with_results(doc_id)
    s = Settings(**(registry.last_run_settings(doc_id) or {}))
    grids = dict(downloads.final_tables(doc))
    if table_id not in grids:
        raise ApiError(404, f"Unknown table {table_id!r}")
    data = grid_to_csv(grids[table_id], s.convert_numerals).encode("utf-8-sig")
    return Response(data, media_type="text/csv",
                    headers=_attachment(f"{table_id}.csv", "table.csv"))


@app.get("/api/documents/{doc_id}/export/flags.csv")
def api_export_flags(doc_id: str) -> Response:
    """Document-level failure-mode flags CSV (one row per flagged cell/reason).
    Header-only when nothing was flagged."""
    doc = _doc_with_results(doc_id)
    stem = Path(doc.upload_name).stem
    flags_csv = getattr(doc.export_result, "flags_csv", "") or ""
    data = flags_csv.encode("utf-8-sig")
    return Response(data, media_type="text/csv",
                    headers=_attachment(f"{stem}_flags.csv", "flags.csv"))


@app.get("/api/documents/{doc_id}/export/{fmt}")
def api_export_single(doc_id: str, fmt: str, combine: bool = True) -> Response:
    doc = _doc_with_results(doc_id)
    if fmt not in ("json", "txt", "xlsx"):
        raise ApiError(400, f"Unknown export format {fmt!r}")
    s, stem, doc_json, all_text, ft = _export_parts(doc, combine)
    if fmt == "json":
        return Response(downloads.json_bytes(doc_json), media_type="application/json",
                        headers=_attachment(f"{stem}_extracted.json", "extracted.json"))
    if fmt == "txt":
        return Response(all_text.encode("utf-8"), media_type="text/plain; charset=utf-8",
                        headers=_attachment(f"{stem}_extracted.txt", "extracted.txt"))
    return Response(tables_to_xlsx(ft, s.convert_numerals),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers=_attachment(f"{stem}_extracted.xlsx", "extracted.xlsx"))


@app.get("/api/export/all.zip")
def api_export_all(combine: bool = True) -> Response:
    """One zip with every finished document's bundle under `{stem}/`."""
    done = [d for d in registry.all_documents() if d.has_results]
    if not done:
        raise ApiError(409, "No finished documents to export.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc in done:
            s, stem, doc_json, all_text, ft = _export_parts(doc, combine)
            inner = downloads.zip_bundle(doc, s, stem, doc_json, all_text, ft)
            with zipfile.ZipFile(io.BytesIO(inner)) as inner_zf:
                for name in inner_zf.namelist():
                    zf.writestr(f"{stem}/{name}", inner_zf.read(name))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers=_attachment("all_documents.zip", "all_documents.zip"))


def mount_frontend() -> None:
    """Serve the built React bundle at /app when frontend/dist exists."""
    dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        class _FrontendFiles(StaticFiles):
            # index.html must never be served stale (it names the hashed bundle);
            # the content-hashed assets/* files are immutable and cache forever.
            async def get_response(self, path, scope):  # type: ignore[override]
                response = await super().get_response(path, scope)
                if path.startswith("assets/"):
                    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                else:
                    response.headers["Cache-Control"] = "no-cache"
                return response

        app.mount("/app", _FrontendFiles(directory=dist, html=True), name="react-frontend")
