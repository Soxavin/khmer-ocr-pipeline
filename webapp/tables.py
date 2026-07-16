"""Pure table-shaping helpers, ported from the Streamlit `app.py`.

Two scopes, matching the Track-A fix in app.py:
  • all_export_tables  — every table in the document (drives Downloads).
  • page_export_tables — the tables shown beside the current page image (this page's
    tables when stitch is off; all document tables when stitch is on).
No NiceGUI imports here — kept pure so it is unit-testable.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# Stitching is an EXPORT concern, not an extraction one: the review UI always
# works per-page (so cell↔page-image linking survives), and the analyst's edits
# live as grids — so joining continuation tables happens here, at export, on the
# edited grids. Mirrors engines/table_merge_pages.py at the grid level; the
# pipeline's own `stitch_pages` remains for the CLI.
_COL_TOLERANCE = 1  # same tolerance as merge_document_tables

# (table_id, grid, conf_grid); conf_grid holds per-cell confidence or None.
ExportTable = tuple[str, list[list[str]], list[list[Any]]]


def block_to_table(block: dict) -> ExportTable:
    """Turn one exported table block (row/col cell dicts) into a rectangular grid plus a
    parallel confidence grid (None where the engine set no confidence)."""
    cells = block.get("cells", [])
    max_row = max((c.get("row", 0) for c in cells), default=0) + (1 if cells else 0)
    max_col = max((c.get("col", 0) for c in cells), default=0) + (1 if cells else 0)
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]
    conf = [[None for _ in range(max_col)] for _ in range(max_row)]
    for c in cells:
        r, col = c.get("row", 0), c.get("col", 0)
        if 0 <= r < max_row and 0 <= col < max_col:
            grid[r][col] = c.get("text", "")
            conf[r][col] = c.get("confidence")
    return block["table_id"], grid, conf


def is_stitched(document_json: dict) -> bool:
    return document_json.get("document_tables") is not None


def all_table_blocks(document_json: dict) -> list[dict]:
    """Every table block in the document (stitched doc-tables, else all pages' tables)."""
    doc = document_json.get("document_tables")
    if doc is not None:
        return doc
    return [t for pg in document_json.get("pages", []) for t in pg.get("tables", [])]


def page_table_blocks(document_json: dict, page_idx: int) -> list[dict]:
    """Table blocks to show beside page `page_idx`: all doc-tables when stitched, else
    only that page's tables (position-aligned with the rendered page list)."""
    doc = document_json.get("document_tables")
    if doc is not None:
        return doc
    pages = document_json.get("pages", [])
    return pages[page_idx].get("tables", []) if page_idx < len(pages) else []


def all_export_tables(document_json: dict) -> list[ExportTable]:
    return [block_to_table(b) for b in all_table_blocks(document_json)]


def page_export_tables(document_json: dict, page_idx: int) -> list[ExportTable]:
    return [block_to_table(b) for b in page_table_blocks(document_json, page_idx)]


def _norm_row(row: list[str]) -> tuple:
    """Row signature for header comparison — NFC + collapsed whitespace, matching
    `table_merge_pages._norm` so the two stitchers agree on what a repeat is."""
    return tuple(re.sub(r"\s+", " ", unicodedata.normalize("NFC", c)).strip() for c in row)


def stitch_grids(final_tables: list[tuple[str, list[list[str]]]], stem: str
                 ) -> list[tuple[str, list[list[str]]]]:
    """Join consecutive page tables sharing a column structure into one grid each,
    dropping the header repeated at each page break and fully-empty rows. `stem` names
    the outputs (`{stem}_table1`, …), matching the pipeline's stitched CSV names.
    Input order is page order; a column-count change (beyond ±1) starts a new table."""
    if not final_tables:
        return []
    groups: list[list[list[list[str]]]] = [[final_tables[0][1]]]
    for _tid, grid in final_tables[1:]:
        ref_cols = len(groups[-1][0][0]) if groups[-1][0] else 0
        cols = len(grid[0]) if grid else 0
        if abs(cols - ref_cols) <= _COL_TOLERANCE:
            groups[-1].append(grid)
        else:
            groups.append([grid])

    out: list[tuple[str, list[list[str]]]] = []
    for i, group in enumerate(groups):
        rows: list[list[str]] = []
        header_sig: tuple | None = None
        for gi, grid in enumerate(group):
            for ri, row in enumerate(grid):
                if gi == 0 and ri == 0:
                    header_sig = _norm_row(row)
                elif ri == 0 and header_sig is not None and _norm_row(row) == header_sig:
                    continue  # the header repeated at a page break
                if all(not c.strip() for c in row):
                    continue  # over-segmentation noise
                rows.append(list(row))
        width = max((len(r) for r in rows), default=0)
        rows = [r + [""] * (width - len(r)) for r in rows]
        out.append((f"{stem}_table{i + 1}", rows))
    return out


def patch_table_block(block: dict, grids_by_id: dict[str, list[list[str]]]) -> dict:
    """Rewrite a block's cells from an edited grid so exported JSON matches the CSVs.
    Ported from app.py `_patch_table_block`."""
    grid = grids_by_id.get(block["table_id"])
    if grid is None:
        return block
    patched = dict(block)
    patched["cells"] = [
        {"row": r, "col": c, "text": grid[r][c]}
        for r in range(len(grid))
        for c in range(len(grid[r]))
    ]
    return patched
