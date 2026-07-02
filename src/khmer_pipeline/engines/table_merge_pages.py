from __future__ import annotations
import re
import unicodedata

# Document-level stitching: the real price reports are ONE continuous table split
# across page images (with embedded section-divider rows). The per-page engines
# emit a separate table per page, so this joins consecutive tables that share a
# column structure into one logical table, dropping a repeated header row at each
# page break. A column-count change starts a new logical table, so genuinely
# different sections (e.g. a differently-shaped block) stay separate.

_COL_TOLERANCE = 1


def _cell_text(c: dict) -> str:
    return " ".join(t["text"] for t in (c.get("text_lines") or []) if t.get("text")).strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def _rows(table: dict) -> list[list[dict]]:
    by_row: dict[int, list[dict]] = {}
    for c in table.get("cells", []):
        by_row.setdefault(c.get("row_id", 0), []).append(c)
    return [by_row[r] for r in sorted(by_row)]


def _table_cols(table: dict) -> int:
    cells = table.get("cells", [])
    if cells:
        return max((c.get("col_id", 0) for c in cells), default=-1) + 1
    return len(table.get("cols", []))


def _row_sig(row_cells: list[dict]) -> tuple:
    return tuple((c.get("col_id", 0), _norm(_cell_text(c))) for c in sorted(row_cells, key=lambda c: c.get("col_id", 0)))


def _combine(group: list[tuple[int, dict]]) -> dict:
    # group: [(page_index, table), ...] sharing a column structure → one table.
    out_cells: list[dict] = []
    out_row = 0
    header_sig = None
    source_pages: list[int] = []
    for gi, (page_index, table) in enumerate(group):
        source_pages.append(page_index)
        rows = _rows(table)
        for ri, row_cells in enumerate(rows):
            if gi == 0 and ri == 0:
                header_sig = _row_sig(row_cells)
            elif ri == 0 and header_sig is not None and _row_sig(row_cells) == header_sig:
                continue  # drop the repeated header at a page break
            if all(not _cell_text(c).strip() for c in row_cells):
                continue  # drop fully-empty rows (SLANet over-segmentation noise)
            for c in row_cells:
                nc = dict(c)
                nc["row_id"] = out_row
                out_cells.append(nc)
            out_row += 1
    n_cols = max((c.get("col_id", 0) for c in out_cells), default=-1) + 1
    first = group[0][1]
    return {
        "rows": [{"row_id": i} for i in range(out_row)],
        "cols": [{"col_id": j} for j in range(n_cols)],
        "cells": out_cells,
        "image_bbox": list(first.get("image_bbox", [])),
        "bbox": list(first.get("bbox", [])),
        "source_pages": source_pages,
    }


def merge_document_tables(pages: list) -> list[dict]:
    # pages: objects with .page_index and .tables (SuryaPageResult/CorrectedPageResult).
    seq: list[tuple[int, dict]] = []
    for p in pages:
        for t in p.tables:
            if t.get("cells"):
                seq.append((p.page_index, t))
    if not seq:
        return []
    groups: list[list[tuple[int, dict]]] = [[seq[0]]]
    for page_index, table in seq[1:]:
        ref_cols = _table_cols(groups[-1][0][1])
        if abs(_table_cols(table) - ref_cols) <= _COL_TOLERANCE:
            groups[-1].append((page_index, table))
        else:
            groups.append([(page_index, table)])
    return [_combine(g) for g in groups]
