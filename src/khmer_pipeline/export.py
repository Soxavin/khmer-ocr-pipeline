from __future__ import annotations
import csv
import io
from datetime import datetime
from pathlib import Path
from .models import PostprocessResult, ExportResult


def export(result: PostprocessResult) -> ExportResult:
    document_json = _build_document_json(result)
    tables_csv: list[tuple[str, str]] = []
    for page in result.pages:
        for t_idx, table in enumerate(page.tables):
            table_id = _make_table_id(result.source_name, page.page_index, t_idx)
            tables_csv.append((table_id, _table_to_csv(table)))
    return ExportResult(
        source_name=result.source_name,
        document_json=document_json,
        tables_csv=tables_csv,
    )


def _make_table_id(source_name: str, page_index: int, table_index: int) -> str:
    return f"{Path(source_name).stem}_page{page_index + 1}_table{table_index + 1}"


def _table_to_csv(table: dict) -> str:
    cells = table.get("cells", [])
    buf = io.StringIO()
    buf.write("﻿")  # UTF-8 BOM — required for Excel to open Khmer text correctly
    writer = csv.writer(buf)
    if not cells:
        return buf.getvalue()
    max_row = max(c["row_id"] for c in cells) + 1
    max_col = max((c.get("col_id") or 0) for c in cells) + 1
    grid = [[""] * max_col for _ in range(max_row)]
    for c in cells:
        r = c["row_id"]
        col = c.get("col_id") or 0
        text = " ".join(
            t["text"] for t in (c.get("text_lines") or []) if t.get("text")
        ).strip()
        if 0 <= r < max_row and 0 <= col < max_col:
            grid[r][col] = text
    writer.writerows(grid)
    return buf.getvalue()


def _build_document_json(result: PostprocessResult) -> dict:
    return {
        "source_name": result.source_name,
        "extracted_at": datetime.utcnow().isoformat() + "Z",
        "page_count": len(result.pages),
        "pages": [
            {
                "page_index": page.page_index,
                "qwen_used": page.qwen_used,
                "corrected_text": page.corrected_text,
                "tables": [
                    {
                        "table_index": t_idx,
                        "table_id": _make_table_id(result.source_name, page.page_index, t_idx),
                        "rows": table["rows"],
                        "cols": table["cols"],
                        "cells": [
                            {
                                "row": c["row_id"],
                                "col": c.get("col_id") or 0,
                                "text": " ".join(
                                    t["text"] for t in (c.get("text_lines") or []) if t.get("text")
                                ).strip(),
                            }
                            for c in table.get("cells", [])
                        ],
                    }
                    for t_idx, table in enumerate(page.tables)
                ],
            }
            for page in result.pages
        ],
    }
