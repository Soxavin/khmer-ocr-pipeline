from __future__ import annotations
import difflib
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
    if "tables" in gt and gt["tables"]:
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


def _align_rows(gt_sigs: list[tuple], pred_sigs: list[tuple]) -> list[tuple[int, int]]:
    # Monotonic GT->pred row alignment. equal/replace blocks pair rows by
    # position within the block; delete (GT-only) and insert (extra pred) unmatched.
    pairs: list[tuple[int, int]] = []
    sm = difflib.SequenceMatcher(None, gt_sigs, pred_sigs, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(min(i2 - i1, j2 - j1)):
                pairs.append((i1 + k, j1 + k))
        # delete / insert -> no pairs
    return pairs


def evaluate_table(pred_tables: list[dict], gt_grid: list[list[str]] | None) -> dict:
    if gt_grid is None:
        # No GT table grid (e.g. real docs labelled paragraphs-only): still report
        # how many tables the OCR actually detected, but no cell-level scoring.
        return {
            "tables_found": len(pred_tables),
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

    gt_sigs = [tuple(_norm(c) for c in row) for row in gt_stripped]
    pred_sigs = [tuple(_norm(c) for c in row) for row in pred_stripped]
    cells_correct = 0
    for gi, pj in _align_rows(gt_sigs, pred_sigs):
        gt_row = gt_stripped[gi]
        pred_row = pred_stripped[pj]
        for c in range(gt_cols):
            gt_val = _norm(gt_row[c]) if c < len(gt_row) else ""
            pred_val = _norm(pred_row[c]) if c < len(pred_row) else ""
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


def evaluate_document(ocr_text: str, pred_tables: list[dict], gt: dict) -> dict:
    # Placement-agnostic: pool ALL text on each side, then CER. Robust to whether
    # content was classified as table vs paragraph by either side.
    gt_parts: list[str] = []
    gt_parts.extend(gt.get("paragraphs", []))
    for tbl in gt.get("tables", []) or []:
        for row in tbl.get("data", []):
            gt_parts.extend(row)
    grid = gt.get("data")
    if grid:
        for row in grid:
            gt_parts.extend(row)
    footer = gt.get("footer", "")
    if footer:
        gt_parts.append(footer)

    pred_parts: list[str] = [ocr_text]
    for tbl in pred_tables:
        for cell in tbl.get("cells", []):
            cell_text = " ".join(t["text"] for t in (cell.get("text_lines") or []) if t.get("text")).strip()
            if cell_text:
                pred_parts.append(cell_text)

    gt_pooled = _norm(" ".join(p for p in gt_parts if p))
    pred_pooled = _norm(" ".join(p for p in pred_parts if p))
    return {"document_cer": cer(gt_pooled, pred_pooled)}


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
