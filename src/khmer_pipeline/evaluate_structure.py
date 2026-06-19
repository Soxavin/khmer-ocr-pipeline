from __future__ import annotations
import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _strip_title_row(grid: list[list[str]]) -> list[list[str]]:
    # Drop row 0 iff it looks like a merged-colspan title:
    # first cell non-empty and all remaining cells empty.
    if not grid:
        return grid
    row0 = grid[0]
    if row0 and row0[0].strip() != "" and all(c.strip() == "" for c in row0[1:]):
        return grid[1:]
    return grid


def _levenshtein(a: str, b: str) -> int:
    # Two-row DP over Unicode codepoints.
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a):
        curr[0] = i + 1
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def gt_table_grid(gt: dict) -> list[list[str]] | None:
    if "tables" in gt:
        return gt["tables"][0]["data"]
    if "data" in gt:
        return gt["data"]
    return None


def gt_paragraph_lines(gt: dict) -> list[str]:
    if "tables" not in gt:
        # isolated table schema — no paragraphs
        return []
    lines: list[str] = []
    for para in gt.get("paragraphs", []):
        for line in para.split("\n"):
            if line:
                lines.append(line)
    footer = gt.get("footer", "")
    for line in footer.split("\n"):
        if line:
            lines.append(line)
    return lines


def pred_table_grid(table: dict) -> list[list[str]]:
    cells = table.get("cells", [])
    if not cells:
        return []
    max_row = max(c.get("row_id", 0) for c in cells) + 1
    max_col = max(c.get("col_id", 0) for c in cells) + 1
    grid = [[""] * max_col for _ in range(max_row)]
    for c in cells:
        r = c.get("row_id", 0)
        col = c.get("col_id", 0)
        text = " ".join(
            t["text"] for t in (c.get("text_lines") or []) if t.get("text")
        ).strip()
        if 0 <= r < max_row and 0 <= col < max_col:
            grid[r][col] = text
    return grid


def _grid_cols(grid: list[list[str]]) -> int:
    return max((len(row) for row in grid), default=0)


def evaluate_table(pred_tables: list[dict], gt_grid: list[list[str]] | None) -> dict:
    if gt_grid is None:
        return {
            "tables_found": 0,
            "gt_rows": 0,
            "gt_cols": 0,
            "pred_rows": 0,
            "pred_cols": 0,
            "cell_accuracy": 0.0,
            "cell_content_recall": 0.0,
            "table_cer": 0.0,
            "cells_total": 0,
            "cells_correct": 0,
        }

    tables_found = len(pred_tables)

    gt_stripped = _strip_title_row(gt_grid)
    gt_rows = len(gt_stripped)
    gt_cols = _grid_cols(gt_stripped)

    if pred_tables:
        pred_stripped = _strip_title_row(pred_table_grid(pred_tables[0]))
    else:
        pred_stripped = []

    pred_rows = len(pred_stripped)
    pred_cols = _grid_cols(pred_stripped)

    cells_total = gt_rows * gt_cols
    cells_correct = 0

    if cells_total > 0:
        for r, gt_row in enumerate(gt_stripped):
            for c in range(gt_cols):
                gt_val = _norm(gt_row[c] if c < len(gt_row) else "")
                if r < len(pred_stripped) and c < len(pred_stripped[r]):
                    pred_val = _norm(pred_stripped[r][c])
                else:
                    pred_val = ""
                if gt_val == pred_val:
                    cells_correct += 1

    cell_accuracy = cells_correct / cells_total if cells_total > 0 else 0.0

    # multiset content recall: non-empty GT cells present in pred multiset
    from collections import Counter
    gt_nonempty = [_norm(gt_stripped[r][c]) for r in range(gt_rows) for c in range(gt_cols)
                   if c < len(gt_stripped[r]) and _norm(gt_stripped[r][c])]
    pred_flat = [_norm(pred_stripped[r][c]) for r in range(pred_rows) for c in range(_grid_cols(pred_stripped))
                 if c < len(pred_stripped[r])]
    pred_counter = Counter(pred_flat)

    matched = 0
    gt_counter = Counter(gt_nonempty)
    for val, count in gt_counter.items():
        matched += min(count, pred_counter.get(val, 0))
    cell_content_recall = matched / len(gt_nonempty) if gt_nonempty else 0.0

    # row-major join for CER
    gt_joined = _norm(" ".join(
        gt_stripped[r][c] if c < len(gt_stripped[r]) else ""
        for r in range(gt_rows) for c in range(gt_cols)
    ))
    pred_joined = _norm(" ".join(
        pred_stripped[r][c] if c < len(pred_stripped[r]) else ""
        for r in range(pred_rows) for c in range(_grid_cols(pred_stripped))
    ))
    table_cer = cer(gt_joined, pred_joined)

    return {
        "tables_found": tables_found,
        "gt_rows": gt_rows,
        "gt_cols": gt_cols,
        "pred_rows": pred_rows,
        "pred_cols": pred_cols,
        "cell_accuracy": cell_accuracy,
        "cell_content_recall": cell_content_recall,
        "table_cer": table_cer,
        "cells_total": cells_total,
        "cells_correct": cells_correct,
    }


def evaluate_text(ocr_text: str, pred_tables: list[dict], gt: dict) -> dict:
    # isolated tables have no paragraphs — return None for all text metrics
    if "tables" not in gt:
        return {"text_cer": None, "paragraph_recall": None, "paragraph_leak": None}

    para_lines = gt_paragraph_lines(gt)
    norm_ocr = _norm(ocr_text)

    if not para_lines:
        gt_text = ""
    else:
        gt_text = _norm(" ".join(para_lines))

    text_cer_val = cer(gt_text, norm_ocr)

    if para_lines:
        found = sum(1 for line in para_lines if _norm(line) in norm_ocr)
        paragraph_recall = found / len(para_lines)
    else:
        paragraph_recall = 0.0

    # collect all pred table cell texts for leak check
    all_pred_cell_texts: list[str] = []
    for table in pred_tables:
        for cell in table.get("cells", []):
            cell_text = " ".join(
                t["text"] for t in (cell.get("text_lines") or []) if t.get("text")
            ).strip()
            all_pred_cell_texts.append(_norm(cell_text))

    paragraph_leak = 0
    for line in para_lines:
        norm_line = _norm(line)
        if any(norm_line in cell_text for cell_text in all_pred_cell_texts):
            paragraph_leak += 1

    return {
        "text_cer": text_cer_val,
        "paragraph_recall": paragraph_recall,
        "paragraph_leak": paragraph_leak,
    }
