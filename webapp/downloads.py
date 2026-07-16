"""Export builders for the NiceGUI review UI.

Pure functions (no NiceGUI imports) that fold a `Document`'s edits back into the exports —
patched JSON, a TXT report, XLSX, per-table CSV, and a zip bundle — reusing the pipeline's
`grid_to_csv` / `tables_to_xlsx`. Ports the Downloads section of the Streamlit `app.py`.
Bytes are built on demand (on the download click), so nothing is recomputed on page nav.
"""
from __future__ import annotations

import datetime
import io
import json
import zipfile

from khmer_pipeline.export import grid_to_csv, tables_to_xlsx

from . import tables
from .settings import Settings
from .state import Document


def preprocessing_info(s: Settings) -> str:
    steps = []
    if s.deskew:
        steps.append("Deskew")
    if s.remove_stamps:
        steps.append("Stamp removal")
    if s.sharpen:
        steps.append("Sharpen")
    if s.normalise:
        steps.append("Contrast enhancement")
    if s.normalise_table_backgrounds:
        steps.append("Background normalisation")
    return ", ".join(steps) if steps else "None"


def final_tables(doc: Document) -> list[tuple[str, list[list[str]]]]:
    """All document tables, folding in edits by table_id (edited grid wins)."""
    doc_json = doc.export_result.document_json
    return [
        (tid, doc.edited_tables.get(tid, grid))
        for tid, grid, _conf in tables.all_export_tables(doc_json)
    ]


def patched_document_json(doc: Document) -> dict:
    """Copy of the export JSON with edited corrected-text and edited table cells folded
    in, so the JSON matches the CSV/XLSX. Ported from app.py's download block."""
    doc_json = dict(doc.export_result.document_json)
    patched_pages = []
    for idx, page_data in enumerate(doc_json.get("pages", [])):
        edited_text = doc.edited_text.get(idx)
        if edited_text is not None:
            page_data = {**page_data, "corrected_text": edited_text}
        patched_pages.append(page_data)
    doc_json["pages"] = patched_pages

    grids_by_id = dict(final_tables(doc))
    if doc_json.get("document_tables") is not None:
        doc_json["document_tables"] = [
            tables.patch_table_block(b, grids_by_id) for b in doc_json["document_tables"]
        ]
    else:
        doc_json["pages"] = [
            {**pd, "tables": [tables.patch_table_block(t, grids_by_id) for t in pd.get("tables", [])]}
            for pd in doc_json["pages"]
        ]
    return doc_json


def text_report(doc: Document, s: Settings, patched_pages: list[dict]) -> str:
    div = "=" * 72
    timing = "\n".join(f"  {k:<28}: {v:.1f}s" for k, v in doc.stage_times.items())
    header = (
        f"{div}\nKHMER DOCUMENT EXTRACTION REPORT\n{div}\n"
        f"Source        : {doc.upload_name}\n"
        f"Extracted     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Pages         : {doc.export_result.document_json.get('page_count', len(patched_pages))}\n"
        f"DPI           : {s.dpi}\n"
        f"Preprocessing : {preprocessing_info(s)}\n"
        f"Mode          : {'Tables only' if s.tables_only else 'Full extraction (text + tables)'}\n"
        f"Qwen          : {'Enabled' if s.enable_qwen else 'Disabled'} (threshold: {s.anomaly_threshold:.2f})\n"
        + (f"{'-' * 72}\n{timing}\n" if timing else "")
        + div
    )
    sections = [
        f"--- Page {idx + 1} of {len(patched_pages)} ---\n\n{p.get('corrected_text', '')}"
        for idx, p in enumerate(patched_pages)
        if p.get("corrected_text")
    ]
    return header + "\n\n" + "\n\n".join(sections)


def nonempty_tables(ft: list[tuple[str, list[list[str]]]]) -> list[tuple[str, list[list[str]]]]:
    return [(tid, grid) for tid, grid in ft if any(cell.strip() for row in grid for cell in row)]


def json_bytes(doc_json: dict) -> bytes:
    return json.dumps(doc_json, ensure_ascii=False, indent=2).encode("utf-8")


def zip_bundle(doc: Document, s: Settings, stem: str, doc_json: dict, all_text: str,
               ft: list[tuple[str, list[list[str]]]]) -> bytes:
    convert = s.convert_numerals
    ne = nonempty_tables(ft)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}_extracted.json", json.dumps(doc_json, ensure_ascii=False, indent=2))
        zf.writestr(f"{stem}_extracted.txt", all_text)
        if ne:
            zf.writestr(f"{stem}_extracted.xlsx", tables_to_xlsx(ft, convert))
        for tid, grid in ne:
            zf.writestr(f"{tid}.csv", grid_to_csv(grid, convert).encode("utf-8-sig"))
    return buf.getvalue()
