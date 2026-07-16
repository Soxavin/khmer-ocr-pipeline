"""Unit tests for webapp.downloads — edit-folding into JSON/TXT/CSV/XLSX/zip exports."""
import io
import json
import zipfile
from types import SimpleNamespace

from webapp.downloads import (
    final_tables, patched_document_json, text_report, nonempty_tables,
    json_bytes, zip_bundle,
)
from webapp.settings import Settings
from webapp.state import Document


def _doc():
    doc_json = {
        "page_count": 1,
        "pages": [{
            "corrected_text": "orig text",
            "tables": [{
                "table_id": "doc_page1_table1",
                "cells": [{"row": 0, "col": 0, "text": "a"}, {"row": 0, "col": 1, "text": "b"}],
            }],
        }],
    }
    d = Document("doc.pdf", b"x", "id1", 1)
    d.export_result = SimpleNamespace(document_json=doc_json)
    d.stage_times = {"Stage 1 — Ingest": 1.2}
    return d


def test_final_tables_folds_edits():
    d = _doc()
    assert final_tables(d) == [("doc_page1_table1", [["a", "b"]])]
    d.edited_tables = {"doc_page1_table1": [["X", "b"]]}
    assert final_tables(d) == [("doc_page1_table1", [["X", "b"]])]


def test_patched_json_applies_text_and_cell_edits():
    d = _doc()
    d.edited_text = {0: "new text"}
    d.edited_tables = {"doc_page1_table1": [["X", "b"]]}
    dj = patched_document_json(d)
    assert dj["pages"][0]["corrected_text"] == "new text"
    assert dj["pages"][0]["tables"][0]["cells"][0] == {"row": 0, "col": 0, "text": "X"}
    # original cached object untouched
    assert d.export_result.document_json["pages"][0]["corrected_text"] == "orig text"


def test_text_report_uses_edited_text():
    d = _doc()
    d.edited_text = {0: "corrected!"}
    dj = patched_document_json(d)
    report = text_report(d, Settings(), dj["pages"])
    assert "corrected!" in report
    assert "KHMER DOCUMENT EXTRACTION REPORT" in report


def test_json_bytes_roundtrips():
    parsed = json.loads(json_bytes(patched_document_json(_doc())))
    assert parsed["pages"][0]["tables"][0]["table_id"] == "doc_page1_table1"


def test_nonempty_filters_blank_tables():
    assert nonempty_tables([("a", [["", " "]]), ("b", [["x"]])]) == [("b", [["x"]])]


def test_zip_bundle_contains_all_artifacts():
    d = _doc()
    s = Settings()
    dj = patched_document_json(d)
    data = zip_bundle(d, s, "doc", dj, text_report(d, s, dj["pages"]), final_tables(d))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
    assert {"doc_extracted.json", "doc_extracted.txt", "doc_extracted.xlsx", "doc_page1_table1.csv"} <= names
