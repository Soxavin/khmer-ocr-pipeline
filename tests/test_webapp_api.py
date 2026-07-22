"""Tests for webapp.api — the REST layer serving the React frontend.

Routes are registered on `nicegui.app` (a FastAPI subclass), so a plain
TestClient works without running the UI. Model-touching calls are stubbed;
the handlers themselves are thin translation over the tested webapp modules.
"""
from __future__ import annotations

import asyncio
import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from nicegui import app
    import webapp.api  # noqa: F401 — registers routes on import
    from webapp import registry
    registry.clear()
    registry.run_lock = asyncio.Lock()  # fresh lock: no leakage between tests
    # No context manager: NiceGUI's lifespan handlers need the full ui.run
    # runtime; plain-client usage skips startup/shutdown, which the pure API
    # routes don't need.
    yield TestClient(app)
    registry.clear()
    registry.run_lock = asyncio.Lock()


def _completed_doc(doc_id="abc123def456"):
    """A Document as it looks after a successful run — enough structure for the
    overview / page / image / export handlers, no real pipeline objects."""
    from webapp import registry
    from webapp.state import Document
    doc = Document(upload_name="report.pdf", upload_bytes=b"%PDF-x",
                   upload_id=doc_id, doc_page_count=1)
    doc_json = {
        "page_count": 1,
        "pages": [{
            "page": 1,
            "corrected_text": "hello",
            "tables": [{
                "table_id": "p1_t1", "bbox": [0, 0, 10, 10],
                "cells": [
                    {"row": 0, "col": 0, "text": "A", "confidence": 0.9},
                    {"row": 0, "col": 1, "text": "B", "confidence": 0.4},
                ],
            }],
        }],
    }
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    doc.export_result = SimpleNamespace(document_json=doc_json)
    doc.preprocess_result = SimpleNamespace(page_images=[img])
    doc.ingest_result = SimpleNamespace(page_images=[img])
    doc.surya_result = SimpleNamespace(
        pages=[SimpleNamespace(
            text_blocks=[{"bbox": [0, 0, 1, 1], "confidence": 0.9, "label": "Text"}],
            tables=[{"bbox": [0, 0, 10, 10]}],
        )],
        warnings=["w1"],
    )
    doc.postprocess_result = SimpleNamespace(pages=[SimpleNamespace(qwen_used=False)], warnings=[])
    doc.stage_times = {"Stage 3 — OCR": 1.5}
    return registry.add(doc)


def _upload(client, name="doc.pdf", data=b"%PDF-fake"):
    return client.post("/api/documents", files=[("files", (name, data, "application/pdf"))])


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def test_meta_lists_engines_and_defaults(client):
    with patch("webapp.api.llama_server_running", return_value=False):
        r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    engine_keys = [e["key"] for e in body["engines"]]
    assert {"surya", "surya_kiri", "surya_kiri_vlm"} <= set(engine_keys)
    assert all("label" in e and "guidance" in e for e in body["engines"])
    assert body["defaults"]["dpi"] == "auto"
    assert body["backend_ready"] is False


# ---------------------------------------------------------------------------
# Preprocess suggestions
# ---------------------------------------------------------------------------

def test_suggest_unknown_document_404s(client):
    assert client.get("/api/documents/nope/suggest").status_code == 404


def test_suggest_ingests_lazily_and_caches(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    flat = np.full((50, 50, 3), 128, dtype=np.uint8)
    fake_ingest = SimpleNamespace(page_images=[flat])
    with patch("webapp.api.ingest", return_value=fake_ingest) as m:
        r1 = client.get(f"/api/documents/{doc_id}/suggest")
        r2 = client.get(f"/api/documents/{doc_id}/suggest")
    assert r1.status_code == 200
    body = r1.json()
    assert set(body) == {"scores", "suggested", "rationale", "checks"}
    assert set(body["scores"]) == {"laplacian_var", "contrast_std", "skew_deg", "stamp_ink_ratio"}
    assert body["suggested"] == {}  # flat gray page: defaults stand
    assert r2.json() == body
    assert m.call_count == 1  # cached on the doc — no second rasterization


def test_suggest_returns_suggestions_for_high_contrast_pages(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    # Smooth 0→255 gradient: well contrasted (normalise off) but not sharp.
    row = np.arange(256, dtype=np.uint8).reshape(1, 256)
    img = np.stack([np.tile(row, (256, 1))] * 3, axis=2)
    with patch("webapp.api.ingest", return_value=SimpleNamespace(page_images=[img])):
        body = client.get(f"/api/documents/{doc_id}/suggest").json()
    assert body["suggested"] == {"normalise": False}
    assert list(body["rationale"]) == ["normalise"]


def test_suggest_unreadable_document_yields_empty_suggestion(client):
    with patch("webapp.api._probe_pages", return_value=0):
        doc_id = _upload(client).json()["documents"][0]["id"]
    with patch("webapp.api.ingest", side_effect=ValueError("bad pdf")):
        r = client.get(f"/api/documents/{doc_id}/suggest")
    assert r.status_code == 200
    assert r.json()["suggested"] == {}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def test_upload_registers_document(client):
    with patch("webapp.api._probe_pages", return_value=3):
        r = _upload(client)
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["name"] == "doc.pdf"
    assert docs[0]["pages"] == 3
    assert docs[0]["status"] == "queued"


def test_upload_dedupes_by_content(client):
    with patch("webapp.api._probe_pages", return_value=1):
        _upload(client)
        _upload(client)  # identical bytes → same id, no duplicate
    r = client.get("/api/documents")
    assert len(r.json()["documents"]) == 1


def test_list_and_delete(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get("/api/documents").json()["documents"] == []
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_clear_all(client):
    with patch("webapp.api._probe_pages", return_value=1):
        _upload(client, "a.pdf", b"%PDF-a")
        _upload(client, "b.pdf", b"%PDF-b")
    assert client.delete("/api/documents").status_code == 200
    assert client.get("/api/documents").json()["documents"] == []


def test_status_of_unknown_document_404s(client):
    assert client.get("/api/documents/nope/status").status_code == 404


def test_status_reports_idle_document(client):
    with patch("webapp.api._probe_pages", return_value=2):
        doc_id = _upload(client).json()["documents"][0]["id"]
    r = client.get(f"/api/documents/{doc_id}/status")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] is False
    assert body["has_results"] is False
    assert body["run_error"] is None
    assert body["last_run_settings"] is None


# ---------------------------------------------------------------------------
# Run + cancel
# ---------------------------------------------------------------------------

def test_run_unknown_document_404s(client):
    assert client.post("/api/documents/nope/run", json={}).status_code == 404


def test_run_rejects_bad_settings(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    assert client.post(f"/api/documents/{doc_id}/run",
                       json={"ocr_engine_key": "banana"}).status_code == 400
    assert client.post(f"/api/documents/{doc_id}/run",
                       json={"no_such_field": 1}).status_code == 400
    assert client.post(f"/api/documents/{doc_id}/run",
                       json={"page_scope": "range", "page_start": 5, "page_end": 2}).status_code == 400


def test_run_409_when_lock_held(client):
    from webapp import registry
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    asyncio.run(registry.run_lock.acquire())
    try:
        r = client.post(f"/api/documents/{doc_id}/run", json={})
        assert r.status_code == 409
    finally:
        registry.run_lock.release()


def test_run_accepted_when_free(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    with patch("webapp.api.run_pipeline", new=AsyncMock(return_value=True)):
        r = client.post(f"/api/documents/{doc_id}/run", json={"ocr_engine_key": "surya"})
    assert r.status_code == 202
    assert r.json()["started"] is True


def test_execute_run_releases_lock_and_stores_settings():
    """Lock is released in ALL outcomes (success / cancel / crash) — a stopped run
    must never leave the registry locked."""
    import webapp.api as api
    from webapp import registry
    from webapp.settings import Settings
    from webapp.state import Document
    registry.clear()
    registry.run_lock = asyncio.Lock()
    doc = Document("d.pdf", b"x", "id1", 1)
    registry.add(doc)

    async def go(result):
        await registry.run_lock.acquire()
        with patch("webapp.api.run_pipeline", new=AsyncMock(return_value=result)):
            await api._execute_run(doc, Settings())

    asyncio.run(go(True))
    assert not registry.run_lock.locked()
    assert registry.last_run_settings("id1") is not None

    registry.clear()
    registry.add(doc)
    asyncio.run(go(False))  # cancelled/failed run
    assert not registry.run_lock.locked()
    assert registry.last_run_settings("id1") is None

    async def crash():
        await registry.run_lock.acquire()
        with patch("webapp.api.run_pipeline", new=AsyncMock(side_effect=RuntimeError("boom"))):
            await api._execute_run(doc, Settings())

    asyncio.run(crash())
    assert not registry.run_lock.locked()
    assert "boom" in (doc.run_error or "")
    registry.clear()


def test_cancel_sets_flag_on_active_run(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    from webapp import registry
    doc = registry.get(doc_id)
    doc.progress.active = True
    r = client.post(f"/api/documents/{doc_id}/cancel")
    assert r.status_code == 200
    assert r.json()["cancelling"] is True
    assert doc.progress.cancel_requested is True


def test_cancel_idle_is_noop(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    r = client.post(f"/api/documents/{doc_id}/cancel")
    assert r.status_code == 200
    assert r.json()["cancelling"] is False
    assert client.post("/api/documents/nope/cancel").status_code == 404


# ---------------------------------------------------------------------------
# Results: overview / page / image / export
# ---------------------------------------------------------------------------

def test_overview_requires_results(client):
    with patch("webapp.api._probe_pages", return_value=1):
        doc_id = _upload(client).json()["documents"][0]["id"]
    assert client.get(f"/api/documents/{doc_id}/overview").status_code == 409


def test_overview_reports_counts_and_warnings(client):
    doc = _completed_doc()
    r = client.get(f"/api/documents/{doc.upload_id}/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["pages"] == 1
    assert body["total_tables"] == 1
    assert body["warnings"] == ["w1"]
    assert body["stitched"] is False
    assert body["stage_times"]["Stage 3 — OCR"] == 1.5


def test_page_returns_tables_with_confidence_and_edits(client):
    doc = _completed_doc()
    r = client.get(f"/api/documents/{doc.upload_id}/pages/0")
    assert r.status_code == 200
    body = r.json()
    assert body["corrected_text"] == "hello"
    [t] = body["tables"]
    assert t["table_id"] == "p1_t1"
    assert t["grid"] == [["A", "B"]]
    assert t["confidence"] == [[0.9, 0.4]]
    assert t["edited"] is False
    assert body["text_blocks"][0]["label"] == "Text"
    assert body["table_bboxes"] == [[0, 0, 10, 10]]
    # edits fold in
    doc.edited_tables["p1_t1"] = [["A2", "B2"]]
    body = client.get(f"/api/documents/{doc.upload_id}/pages/0").json()
    assert body["tables"][0]["grid"] == [["A2", "B2"]]
    assert body["tables"][0]["edited"] is True
    assert client.get(f"/api/documents/{doc.upload_id}/pages/9").status_code == 404


def test_page_image_serves_png_variants(client):
    doc = _completed_doc()
    for variant in ("processed", "original"):
        r = client.get(f"/api/documents/{doc.upload_id}/pages/0/image?variant={variant}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/api/documents/{doc.upload_id}/pages/9/image").status_code == 404
    assert client.get(f"/api/documents/{doc.upload_id}/pages/0/image?variant=weird").status_code == 400


def test_page_image_revalidates_with_etag(client):
    """Grid<->Single toggles remount every thumbnail. Without a validator the
    browser refetches and re-encodes every PNG, which is what starved the server
    during an active run and produced broken thumbnails."""
    doc = _completed_doc()
    url = f"/api/documents/{doc.upload_id}/pages/0/image?variant=processed"
    first = client.get(url)
    assert first.status_code == 200
    etag = first.headers.get("etag")
    assert etag, "page image must carry an ETag so the browser can revalidate"
    # no-cache (not max-age): a re-run replaces the image at this SAME url, so a
    # time-based cache would happily serve a stale page.
    assert "no-cache" in first.headers.get("cache-control", "")

    again = client.get(url, headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""  # 304 skips the PNG encode entirely


def test_preview_image_revalidates_with_etag(client):
    with patch("webapp.api._probe_pages", return_value=1):
        r = _upload(client, "raw.pdf", b"%PDF-raw")
    doc_id = r.json()["documents"][0]["id"]
    fake = SimpleNamespace(page_images=[np.zeros((8, 8, 3), dtype=np.uint8)])
    url = f"/api/documents/{doc_id}/preview/0"
    with patch("webapp.api.ingest", return_value=fake):
        first = client.get(url)
        assert first.status_code == 200
        etag = first.headers.get("etag")
        assert etag
        assert "no-cache" in first.headers.get("cache-control", "")
        assert client.get(url, headers={"If-None-Match": etag}).status_code == 304


def test_page_image_etag_changes_when_the_run_changes(client):
    """A re-run with different settings must invalidate the cached rendition."""
    doc = _completed_doc()
    url = f"/api/documents/{doc.upload_id}/pages/0/image?variant=processed"
    before = client.get(url).headers["etag"]
    doc.last_key = (doc.last_key or "") + "_rerun"
    assert client.get(url).headers["etag"] != before


def test_processed_image_available_mid_run_before_results(client):
    """Preprocessing finishes at stage 2, long before the run ends. Gating the
    processed rendition behind full results makes the grid show raw previews for the
    whole run even though the cleaned pages already exist."""
    from webapp import registry
    from webapp.state import Document
    doc = Document(upload_name="a.pdf", upload_bytes=b"%PDF", upload_id="midrun1", doc_page_count=2)
    doc.preprocess_result = SimpleNamespace(page_images=[np.zeros((4, 4, 3), dtype=np.uint8)])
    doc.run_page_indices = [1]  # a page-scoped run: result 0 IS document page 1
    registry.add(doc)

    r = client.get(f"/api/documents/midrun1/pages/0/image?variant=processed")
    assert r.status_code == 200, "processed pages must be servable as soon as they exist"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    # 'original' still needs the ingest, which a mid-run doc may not expose yet.
    assert client.get("/api/documents/midrun1/pages/5/image?variant=processed").status_code == 404


def test_status_reports_processed_page_mapping(client):
    """Result index != document page number for a page-scoped run, so the frontend
    needs the mapping to address the right rendition."""
    from webapp import registry
    from webapp.state import Document
    doc = Document(upload_name="a.pdf", upload_bytes=b"%PDF", upload_id="midrun2", doc_page_count=3)
    registry.add(doc)
    assert client.get("/api/documents/midrun2/status").json()["processed_pages"] == []

    doc.preprocess_result = SimpleNamespace(page_images=[np.zeros((4, 4, 3), dtype=np.uint8)] * 2)
    doc.run_page_indices = [1, 2]
    body = client.get("/api/documents/midrun2/status").json()
    assert body["processed_pages"] == [1, 2]

    # A whole-document run stores no explicit selection: every page was processed.
    doc.run_page_indices = None
    assert client.get("/api/documents/midrun2/status").json()["processed_pages"] == [0, 1]


def test_export_zip_reflects_edits(client):
    doc = _completed_doc()
    doc.edited_tables["p1_t1"] = [["X", "Y"]]
    r = client.get(f"/api/documents/{doc.upload_id}/export/zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        assert "report_extracted.json" in names
        assert "report_extracted.txt" in names
        # Combined export names tables `{stem}_tableN`, matching the pipeline's
        # own stitched CSV naming; per-page ids appear only with combine=false.
        assert "report_table1.csv" in names
        assert "X" in zf.read("report_table1.csv").decode("utf-8-sig")
    with patch("webapp.api._probe_pages", return_value=1):
        empty_id = _upload(client, "e.pdf", b"%PDF-e").json()["documents"][0]["id"]
    assert client.get(f"/api/documents/{empty_id}/export/zip").status_code == 409


def test_page_payload_supports_review(client):
    """P2 additions: original grid for the diff view, table_id-keyed bboxes for
    image↔table linking, and the per-page Qwen flag."""
    doc = _completed_doc()
    body = client.get(f"/api/documents/{doc.upload_id}/pages/0").json()
    assert body["tables"][0]["original_grid"] == [["A", "B"]]
    assert body["table_bbox_index"] == {"p1_t1": [0, 0, 10, 10]}
    assert body["qwen_used"] is False
    doc.edited_tables["p1_t1"] = [["A2", "B2"]]
    body = client.get(f"/api/documents/{doc.upload_id}/pages/0").json()
    assert body["tables"][0]["grid"] == [["A2", "B2"]]
    assert body["tables"][0]["original_grid"] == [["A", "B"]]


# ---------------------------------------------------------------------------
# Edits: tables + text
# ---------------------------------------------------------------------------

def test_put_table_sets_edit_and_delete_resets(client):
    doc = _completed_doc()
    url = f"/api/documents/{doc.upload_id}/tables/p1_t1"
    r = client.put(url, json={"grid": [["X", "Y"], ["", "42"]]})  # row added
    assert r.status_code == 200
    assert doc.edited_tables["p1_t1"] == [["X", "Y"], ["", "42"]]
    assert client.delete(url).status_code == 200
    assert "p1_t1" not in doc.edited_tables


def test_put_table_validates(client):
    doc = _completed_doc()
    base = f"/api/documents/{doc.upload_id}/tables"
    assert client.put(f"{base}/nope", json={"grid": [["X"]]}).status_code == 404
    assert client.put(f"{base}/p1_t1", json={"grid": []}).status_code == 400
    assert client.put(f"{base}/p1_t1", json={"grid": [["a"], ["b", "c"]]}).status_code == 400  # ragged
    assert client.put(f"{base}/p1_t1", json={"grid": [[1, 2]]}).status_code == 400  # non-string


def test_put_page_text(client):
    doc = _completed_doc()
    r = client.put(f"/api/documents/{doc.upload_id}/pages/0/text", json={"text": "fixed"})
    assert r.status_code == 200
    assert doc.edited_text[0] == "fixed"
    body = client.get(f"/api/documents/{doc.upload_id}/pages/0").json()
    assert body["corrected_text"] == "fixed"
    assert client.put(f"/api/documents/{doc.upload_id}/pages/9/text", json={"text": "x"}).status_code == 404


# ---------------------------------------------------------------------------
# Triage + review status + bulk replace (P3)
# ---------------------------------------------------------------------------

def test_lowconf_lists_cells_sorted_ascending(client):
    doc = _completed_doc()
    r = client.get(f"/api/documents/{doc.upload_id}/lowconf")
    assert r.status_code == 200
    issues = r.json()["issues"]
    # Only the 0.4 cell is below CELL_CONF_LOW (0.80); 0.9 is fine.
    assert issues == [{
        "page": 0, "table_id": "p1_t1", "row": 0, "col": 1,
        "conf": 0.4, "text": "B", "reason": "low_conf", "reasons": ["low_conf"],
    }]


def test_lowconf_skips_empty_cells(client):
    """Intentionally blank cells (spacer columns etc.) are not OCR errors — they
    must not flood the triage list or get tinted."""
    doc = _completed_doc()
    doc.export_result.document_json["pages"][0]["tables"][0]["cells"].append(
        {"row": 0, "col": 2, "text": "", "confidence": 0.0})
    issues = client.get(f"/api/documents/{doc.upload_id}/lowconf").json()["issues"]
    assert all(i["text"].strip() for i in issues)
    assert len(issues) == 1  # only the 0.4-confidence "B" cell


def test_lowconf_emits_validator_flags_with_reason(client):
    doc = _completed_doc()
    # Attach validator flags to the (high-confidence) "A" cell.
    cells = doc.export_result.document_json["pages"][0]["tables"][0]["cells"]
    cells[0]["flags"] = ["digit_mixed", "sequence_illegal"]
    issues = client.get(f"/api/documents/{doc.upload_id}/lowconf").json()["issues"]
    by_text = {i["text"]: i for i in issues}
    # "A" is high-confidence but validator-flagged → issue with null conf.
    a = by_text["A"]
    assert a["conf"] == 0.9
    assert a["reason"] == "sequence_illegal"  # higher priority than digit_mixed
    assert set(a["reasons"]) == {"digit_mixed", "sequence_illegal"}
    # Validator-flagged issue sorts before the low_conf-only "B" issue.
    assert issues[0]["text"] == "A"
    assert issues[-1]["reason"] == "low_conf"


def test_lowconf_null_conf_for_validator_only_cell(client):
    doc = _completed_doc()
    cells = doc.export_result.document_json["pages"][0]["tables"][0]["cells"]
    # A cell with no confidence key at all, only a validator flag.
    cells.append({"row": 1, "col": 0, "text": "X", "flags": ["structure_ragged"]})
    issues = client.get(f"/api/documents/{doc.upload_id}/lowconf").json()["issues"]
    x = next(i for i in issues if i["text"] == "X")
    assert x["conf"] is None
    assert x["reason"] == "structure_ragged"


def test_export_flags_csv_download(client):
    doc = _completed_doc()
    doc.export_result.flags_csv = "﻿table_id,page,row,col,text,confidence,flagged_reason\r\np1_t1,1,0,1,B,0.4,low_conf\r\n"
    r = client.get(f"/api/documents/{doc.upload_id}/export/flags.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "flags.csv" in r.headers["content-disposition"]
    assert "low_conf" in r.text


def test_replace_is_undoable(client):
    doc = _completed_doc()
    doc.edited_tables["p1_t1"] = [["A", "prior-edit"]]
    r = client.post(f"/api/documents/{doc.upload_id}/replace",
                    json={"find": "prior-edit", "replace": "Z"})
    assert r.json()["total"] == 1
    assert doc.edited_tables["p1_t1"] == [["A", "Z"]]
    r = client.post(f"/api/documents/{doc.upload_id}/replace/undo")
    assert r.status_code == 200
    assert doc.edited_tables["p1_t1"] == [["A", "prior-edit"]]  # snapshot restored
    # second undo: nothing to restore
    assert client.post(f"/api/documents/{doc.upload_id}/replace/undo").status_code == 409


def test_lowconf_reflects_edits_and_requires_results(client):
    doc = _completed_doc()
    doc.edited_tables["p1_t1"] = [["A", "fixed"]]
    issues = client.get(f"/api/documents/{doc.upload_id}/lowconf").json()["issues"]
    assert issues[0]["text"] == "fixed"
    with patch("webapp.api._probe_pages", return_value=1):
        empty_id = _upload(client, "n.pdf", b"%PDF-n").json()["documents"][0]["id"]
    assert client.get(f"/api/documents/{empty_id}/lowconf").status_code == 409


def test_review_toggle_and_progress(client):
    doc = _completed_doc()
    url = f"/api/documents/{doc.upload_id}/review/p1_t1"
    assert client.put(url, json={"verified": True}).status_code == 200
    assert doc.reviewed["p1_t1"] is True
    # progress surfaces in the queue summary and the page payload
    docs = client.get("/api/documents").json()["documents"]
    mine = next(d for d in docs if d["id"] == doc.upload_id)
    assert mine["reviewed_tables"] == 1 and mine["total_tables"] == 1
    body = client.get(f"/api/documents/{doc.upload_id}/pages/0").json()
    assert body["tables"][0]["verified"] is True
    assert client.put(url, json={"verified": False}).status_code == 200
    assert doc.reviewed["p1_t1"] is False
    assert client.put(f"/api/documents/{doc.upload_id}/review/nope",
                      json={"verified": True}).status_code == 404


def test_replace_across_tables(client):
    doc = _completed_doc()
    r = client.post(f"/api/documents/{doc.upload_id}/replace",
                    json={"find": "B", "replace": "Z"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1 and body["tables_changed"] == 1
    assert doc.edited_tables["p1_t1"] == [["A", "Z"]]
    r = client.post(f"/api/documents/{doc.upload_id}/replace",
                    json={"find": "", "replace": "x"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Export formats + batch export (P4)
# ---------------------------------------------------------------------------

def test_export_single_formats(client):
    doc = _completed_doc()
    base = f"/api/documents/{doc.upload_id}/export"
    r = client.get(f"{base}/json")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert json.loads(r.content)["pages"][0]["corrected_text"] == "hello"
    r = client.get(f"{base}/txt")
    assert r.status_code == 200 and "KHMER DOCUMENT EXTRACTION REPORT" in r.text
    r = client.get(f"{base}/xlsx")
    assert r.status_code == 200
    assert r.content[:2] == b"PK"  # xlsx is a zip container
    r = client.get(f"{base}/csv/p1_t1")
    assert r.status_code == 200 and "A" in r.text
    assert client.get(f"{base}/csv/nope").status_code == 404
    assert client.get(f"{base}/docx").status_code == 400


def _two_page_doc():
    """A doc whose one logical table is split across two pages — the ARDB shape."""
    from webapp import registry
    doc = _completed_doc("twopage00001")
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    header = [{"row": 0, "col": 0, "text": "ID"}, {"row": 0, "col": 1, "text": "Item"}]
    doc.export_result.document_json = {
        "page_count": 2,
        "pages": [
            {"page": 1, "corrected_text": "", "tables": [{
                "table_id": "report_page1_table1", "bbox": [0, 0, 10, 10],
                "cells": header + [{"row": 1, "col": 0, "text": "1"},
                                   {"row": 1, "col": 1, "text": "rice"}]}]},
            {"page": 2, "corrected_text": "", "tables": [{
                "table_id": "report_page2_table1", "bbox": [0, 0, 10, 10],
                "cells": header + [{"row": 1, "col": 0, "text": "2"},
                                   {"row": 1, "col": 1, "text": "beef"}]}]},
        ],
    }
    doc.preprocess_result = SimpleNamespace(page_images=[img, img])
    doc.ingest_result = SimpleNamespace(page_images=[img, img])
    doc.surya_result = SimpleNamespace(
        pages=[SimpleNamespace(text_blocks=[], tables=[{"bbox": [0, 0, 10, 10]}])] * 2, warnings=[])
    doc.postprocess_result = SimpleNamespace(
        pages=[SimpleNamespace(qwen_used=False)] * 2, warnings=[])
    return registry.add(doc)


def test_export_combines_pages_by_default(client):
    """Review is per-page, but the analyst's Excel wants ONE table — combining
    happens at export, on the edited grids."""
    doc = _two_page_doc()
    r = client.get(f"/api/documents/{doc.upload_id}/export/zip")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv")]
        assert csvs == ["report_table1.csv"]  # one combined table, not two pages
        body = zf.read("report_table1.csv").decode("utf-8-sig")
    assert "rice" in body and "beef" in body
    assert body.count("Item") == 1  # header de-duplicated across the page break


def test_export_per_page_when_combine_false(client):
    doc = _two_page_doc()
    r = client.get(f"/api/documents/{doc.upload_id}/export/zip?combine=false")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csvs = sorted(n for n in zf.namelist() if n.endswith(".csv"))
    assert csvs == ["report_page1_table1.csv", "report_page2_table1.csv"]


def test_export_combined_reflects_edits(client):
    """The risk named in the brief: combining must stitch EDITED grids."""
    doc = _two_page_doc()
    doc.edited_tables["report_page2_table1"] = [["ID", "Item"], ["2", "CORRECTED"]]
    with zipfile.ZipFile(io.BytesIO(client.get(f"/api/documents/{doc.upload_id}/export/zip").content)) as zf:
        body = zf.read("report_table1.csv").decode("utf-8-sig")
    assert "CORRECTED" in body and "beef" not in body


def test_export_all_zip_bundles_done_documents(client):
    doc = _completed_doc()
    with patch("webapp.api._probe_pages", return_value=1):
        _upload(client, "pending.pdf", b"%PDF-p")  # queued doc: excluded
    r = client.get("/api/export/all.zip")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
    assert any(n.startswith("report/") and n.endswith("_extracted.json") for n in names)
    assert not any("pending" in n for n in names)
    doc.export_result = None  # no done docs at all → 409
    assert client.get("/api/export/all.zip").status_code == 409


def test_export_zip_with_non_latin1_filename(client):
    """HTTP headers are latin-1: a non-ASCII upload name (the real documents have
    Khmer filenames) must not crash the Content-Disposition header."""
    doc = _completed_doc()
    doc.upload_name = "πρ-report.pdf"  # any non-latin-1 name reproduces the crash
    r = client.get(f"/api/documents/{doc.upload_id}/export/zip")
    assert r.status_code == 200
    disp = r.headers["content-disposition"]
    assert "filename*=UTF-8''" in disp
    disp.encode("latin-1")  # must be header-safe


def test_frontend_cache_headers(client):
    """index.html must revalidate on every load (it names the hashed bundle);
    the content-hashed assets are immutable and cache forever."""
    from pathlib import Path

    import webapp.api as api

    dist = Path(api.__file__).resolve().parent.parent / "frontend" / "dist"
    if not dist.is_dir():
        pytest.skip("frontend not built")
    api.mount_frontend()
    r = client.get("/app/")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    asset = next(p for p in (dist / "assets").iterdir() if p.suffix == ".js")
    r2 = client.get(f"/app/assets/{asset.name}")
    assert r2.status_code == 200
    assert "immutable" in r2.headers["cache-control"]


def test_preview_image_before_run(client):
    """A queued document's pages are viewable pre-run: /preview/{n} lazily ingests
    once, caches on the doc, and serves PNG; bad page index 404s."""
    with patch("webapp.api._probe_pages", return_value=1):
        r = _upload(client, "raw.pdf", b"%PDF-raw")
    doc_id = r.json()["documents"][0]["id"]
    fake = SimpleNamespace(page_images=[np.zeros((8, 8, 3), dtype=np.uint8)])
    with patch("webapp.api.ingest", return_value=fake) as ing:
        assert client.get(f"/api/documents/{doc_id}/preview/0").headers["content-type"] == "image/png"
        assert client.get(f"/api/documents/{doc_id}/preview/0").status_code == 200
        ing.assert_called_once()  # cached after the first render
    assert client.get(f"/api/documents/{doc_id}/preview/9").status_code == 404


def test_preview_unreadable_upload_422(client):
    with patch("webapp.api._probe_pages", return_value=1):
        r = _upload(client, "junk.pdf", b"not a pdf at all")
    doc_id = r.json()["documents"][0]["id"]
    with patch("webapp.api.ingest", side_effect=ValueError("boom")):
        assert client.get(f"/api/documents/{doc_id}/preview/0").status_code == 422


def test_settings_list_scope():
    """Grid page selection: 'list' scope carries disjoint 1-based pages; indices are
    sorted, deduped, 0-based, clamped; empty list defensively means all pages."""
    from webapp.settings import Settings
    s = Settings(page_scope="list", page_list=[5, 2, 3, 2])
    assert s.page_indices(10) == [1, 2, 4]
    assert s.page_indices(3) == [1, 2]  # page 5 clamped away
    assert Settings(page_scope="list", page_list=[]).page_indices(4) is None
    key = Settings(page_scope="list", page_list=[5, 2, 3]).settings_key("id0")
    assert "list_2_3_5" in key
    # A different selection must change the key (stale-run guard).
    assert key != Settings(page_scope="list", page_list=[2, 3]).settings_key("id0")


def test_run_payload_accepts_list_scope():
    from webapp.api import _settings_from
    s = _settings_from({"page_scope": "list", "page_list": [2, 4]})
    assert s.page_indices(9) == [1, 3]


def test_cancelled_doc_reports_stopped_status(client):
    """A user-cancelled run is not an error: the summary status must say 'stopped'
    so the UI can show a neutral state instead of a red failure tag."""
    doc = _completed_doc("stopcase00001")
    doc.export_result = None  # no results
    doc.run_error = "Extraction cancelled."
    body = client.get("/api/documents").json()["documents"]
    me = next(d for d in body if d["id"] == "stopcase00001")
    assert me["status"] == "stopped"


def test_failed_doc_still_reports_error_status(client):
    doc = _completed_doc("failcase00001")
    doc.export_result = None
    doc.run_error = "Stage 3 — OCR failed: boom"
    body = client.get("/api/documents").json()["documents"]
    me = next(d for d in body if d["id"] == "failcase00001")
    assert me["status"] == "error"


def test_status_exposes_sub_step(client):
    """The OCR sub-step reaches the UI so a long stage can narrate itself."""
    doc_id = _upload(client).json()["documents"][0]["id"]
    from webapp.api import registry
    registry.get(doc_id).progress.step = "tables"
    body = client.get(f"/api/documents/{doc_id}/status").json()
    assert body["step"] == "tables"


# ── HITL correction capture (§2.67) ──────────────────────────────────────────
# Verifying a table the analyst edited turns those fixes into training pairs.
# Capture is strictly a side effect: it must never alter or break the save.

def _doc_for_capture(client, tmp_path, monkeypatch):
    """A document with one page, one table whose cell carries recognizer geometry,
    plus an analyst edit of that cell. Returns (doc_id, table_id)."""
    import numpy as np
    from types import SimpleNamespace
    from webapp import api as api_mod
    from webapp.api import registry

    doc_id = _upload(client).json()["documents"][0]["id"]
    doc = registry.get(doc_id)
    img = np.full((40, 40, 3), 255, dtype=np.uint8)
    cell = {
        "row_id": 0, "col_id": 0, "bbox": [1, 1, 20, 20],
        "text_lines": [{"text": "១០០"}], "confidence": 0.4,
    }
    doc.surya_result = SimpleNamespace(pages=[SimpleNamespace(tables=[{"cells": [cell]}])])
    doc.preprocess_result = SimpleNamespace(recognition_page_images=[img], page_images=[img])
    doc.export_result = SimpleNamespace(document_json={
        "pages": [{"tables": [{"table_id": "t1", "cells": [{"row": 0, "col": 0, "text": "១០០"}]}]}]
    })
    doc.edited_tables["t1"] = [["២០០"]]  # the analyst's fix
    monkeypatch.setattr(api_mod, "_CORRECTIONS_DIR", tmp_path / "corrections")
    return doc_id, "t1"


def test_verifying_an_edited_table_captures_training_pairs(client, tmp_path, monkeypatch):
    doc_id, tid = _doc_for_capture(client, tmp_path, monkeypatch)
    r = client.put(f"/api/documents/{doc_id}/review/{tid}", json={"verified": True})
    assert r.status_code == 200 and r.json()["verified"] is True

    store = tmp_path / "corrections"
    lines = (store / "corrections.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json
    rec = json.loads(lines[0])
    assert rec["text"] == "២០០"
    assert (store / rec["image"]).exists()


def test_unverifying_captures_nothing(client, tmp_path, monkeypatch):
    doc_id, tid = _doc_for_capture(client, tmp_path, monkeypatch)
    client.put(f"/api/documents/{doc_id}/review/{tid}", json={"verified": False})
    assert not (tmp_path / "corrections" / "corrections.jsonl").exists()


def test_reverifying_the_same_table_does_not_duplicate(client, tmp_path, monkeypatch):
    doc_id, tid = _doc_for_capture(client, tmp_path, monkeypatch)
    for verified in (True, False, True):  # toggling must not append twice
        client.put(f"/api/documents/{doc_id}/review/{tid}", json={"verified": verified})
    lines = (tmp_path / "corrections" / "corrections.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_capture_failure_never_breaks_the_save(client, tmp_path, monkeypatch):
    """An analyst's verification must land even if capture explodes."""
    from webapp import api as api_mod
    doc_id, tid = _doc_for_capture(client, tmp_path, monkeypatch)

    def boom(**kwargs):
        raise OSError("disk on fire")
    monkeypatch.setattr(api_mod, "capture_corrections", boom)

    r = client.put(f"/api/documents/{doc_id}/review/{tid}", json={"verified": True})
    assert r.status_code == 200 and r.json()["verified"] is True
    from webapp.api import registry
    assert registry.get(doc_id).reviewed[tid] is True


def test_auto_engine_is_offered_and_accepted(client):
    """The validated `auto` router must be reachable from the UI (§2.57)."""
    engines = client.get("/api/meta").json()["engines"]
    assert "auto" in {e["key"] for e in engines}
    doc_id = _upload(client).json()["documents"][0]["id"]
    # A run payload naming it must pass validation (api.py `_settings_from`).
    from webapp.settings import Settings
    from webapp.api import _settings_from
    assert _settings_from({"ocr_engine_key": "auto"}).ocr_engine_key == "auto"
    assert Settings().ocr_engine_key == "auto"  # the out-of-the-box default
    assert doc_id


def test_auto_engine_is_first_in_the_picker(client):
    """`auto` is the recommended default, so it leads the Recognition engine list."""
    engines = client.get("/api/meta").json()["engines"]
    assert engines[0]["key"] == "auto"


def test_dpi_accepts_auto_and_rejects_garbage(client):
    """`dpi` may be "auto" or a positive int; anything else is a 400, not a crash
    deep in ingest (dpi/72)."""
    from webapp.api import _settings_from
    from webapp.state import Document
    assert _settings_from({"dpi": "auto"}).dpi == "auto"
    assert _settings_from({"dpi": 300}).dpi == 300
    doc_id = _upload(client).json()["documents"][0]["id"]
    assert doc_id
    r = client.post(f"/api/documents/{doc_id}/run", json={"dpi": "medium"})
    assert r.status_code == 400
