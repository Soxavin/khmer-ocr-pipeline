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
    assert body["defaults"]["dpi"] == 200
    assert body["backend_ready"] is False


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
        "conf": 0.4, "text": "B",
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
