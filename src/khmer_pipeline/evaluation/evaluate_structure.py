from __future__ import annotations
import re
import unicodedata

# Khmer→Arabic digit fold. Canonical copy lives in export._KHMER_TO_ARABIC;
# duplicated here (with this note) so the lightweight evaluation module doesn't
# pull in export.py / openpyxl just for 10 entries.
_KHMER_TO_ARABIC = {
    "០": "0", "១": "1", "២": "2", "៣": "3", "៤": "4",
    "៥": "5", "៦": "6", "៧": "7", "៨": "8", "៩": "9",
}

# A folded cell is NUMERIC if it is a bare number: optional +/- sign, digits
# with optional 3-grouped thousands-commas, optional decimal part, optional
# trailing %. (e.g. "7,800", "-3.85%", "123", "0.00%".)
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?%?$")

# The Cambodian convention: comma is the DECIMAL separator and the thousands
# group is a period or a space (spaces are already stripped by _fold_numeric).
# e.g. "០,៧១១៧" = 0.7117, "១.២៣៤,៥៦" = 1234.56.
_NUMERIC_COMMA_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:\.\d{3})+)(?:,\d+)?%?$")

# Longest trailing token still treated as a unit affix ("ដុល្លារ", "រៀល", "លីត្រ",
# "គីឡូក្រាម", "USD", "%"). The cap is what separates a unit from a sentence that
# merely happens to end after a number.
_UNIT_AFFIX_MAX_CHARS = 12


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _fold_numeric(s: str) -> str:
    # NFC-normalize, fold Khmer digits → Arabic, then drop every space so a
    # split number ("7 800") compares equal to its joined form ("7800").
    folded = "".join(_KHMER_TO_ARABIC.get(ch, ch) for ch in _norm(s))
    return folded.replace(" ", "")


def _is_unit_token(tok: str) -> bool:
    # A unit affix is short and carries no digits in either script.
    return (
        0 < len(tok) <= _UNIT_AFFIX_MAX_CHARS
        and not any(ch.isdigit() or ch in _KHMER_TO_ARABIC for ch in tok)
    )


def _strip_unit_affixes(s: str) -> tuple[str, bool]:
    """Strip a leading currency symbol / accounting parens / trailing unit token.

    Returns the remaining number core and whether an affix was found — the latter
    is a *locale signal* used by `_is_numeric` to decide how to read a comma."""
    core = _norm(s)
    found = False
    # Leading currency symbols are matched by Unicode category Sc ("$", "៛")
    # rather than an enumerated list, so a new currency needs no code change.
    if core and unicodedata.category(core[0]) == "Sc":
        core = core[1:].lstrip()
        found = True
    # Accounting convention for negatives, common in budget/TOFE tables: "(1,234)".
    if core.startswith("(") and core.endswith(")"):
        core = core[1:-1].strip()
    # Exactly ONE trailing unit token. Stripping more would let a two-word label
    # ending in a number-shaped token pass as numeric.
    tokens = core.split()
    if len(tokens) > 1 and _is_unit_token(tokens[-1]):
        core = " ".join(tokens[:-1])
        found = True
    return core, found


def _is_numeric(s: str) -> bool:
    """True iff the cell is a single number, optionally carrying a unit affix.

    Two separator conventions are in play. Period-decimal ("7,800.25") is always
    accepted. Comma-decimal ("០,៧១១៧") is accepted only when the cell carries a
    locale signal — Khmer digits, or a currency/unit affix — because the two
    readings are genuinely ambiguous: bare "7,8000" is malformed thousands
    grouping (the known Kiri digit-duplication artifact) but a valid comma-decimal
    number. Requiring the signal keeps that artifact detectable.

    Residual gap: a bare, unit-less "0,7117" in an all-ASCII document reads as
    non-numeric. Resolving it needs document-level context — infer the convention
    once per grid and pass it in — rather than a better per-cell guess."""
    core, has_affix = _strip_unit_affixes(s)
    folded = _fold_numeric(core)
    if _NUMERIC_RE.match(folded):
        return True
    if has_affix or _has_khmer_digit(s):
        return bool(_NUMERIC_COMMA_DECIMAL_RE.match(folded))
    return False


def _has_khmer_digit(s: str) -> bool:
    return any(ch in _KHMER_TO_ARABIC for ch in s)


# A cell counts as Khmer text when at least this fraction of its non-space
# characters are in the Khmer block — same rule surya_kiri_vlm uses to pick
# re-read candidates, so eval and engine agree on what "a Khmer cell" is.
_KHMER_HEAVY_MIN_RATIO = 0.5


def _is_khmer_text(s: str) -> bool:
    """True for Khmer-script *label* cells, excluding Khmer-digit numerals.

    Numerals written in Khmer digits ("១២៣") are semantically numeric and are
    already scored by the numeric metric, so they are excluded here to keep the
    numeric and Khmer cell classes disjoint (no cell is counted twice)."""
    if _is_numeric(s):
        return False
    chars = [ch for ch in _norm(s) if not ch.isspace()]
    if not chars:
        return False
    khmer = sum(1 for ch in chars if "\u1780" <= ch <= "\u17ff")
    return khmer / len(chars) >= _KHMER_HEAVY_MIN_RATIO


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


def _cell(grid: list[list[str]], r: int, c: int) -> str:
    # Normalized cell text, treating out-of-range positions as empty — ragged
    # rows are normal in predicted grids.
    if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
        return _norm(grid[r][c])
    return ""


# Two rows pair only if they are at least this similar (1 - normalized edit
# distance over the joined row text). Real OCR rows are garbled but recognisable
# (§2.42 measured ~0.8 on budget p3); genuinely different rows score far lower.
_ROW_ALIGN_MIN_SIMILARITY = 0.5


def _row_similarity(a: tuple, b: tuple) -> float:
    # 1 - normalized edit distance over the joined row text. Row *signatures*
    # (not raw cells) so this matches what the caller scores on.
    sa, sb = " ".join(a), " ".join(b)
    longest = max(len(sa), len(sb))
    if longest == 0:
        return 1.0
    return 1.0 - _levenshtein(sa, sb) / longest


def _align_rows(gt_sigs: list[tuple], pred_sigs: list[tuple]) -> list[tuple[int, int]]:
    """Monotonic GT->pred row alignment maximising total row similarity.

    Needleman-Wunsch over `_row_similarity`; rows below `_ROW_ALIGN_MIN_SIMILARITY`
    are left unmatched. Deliberately NOT exact-match (difflib) based: on real
    documents OCR garbles every row, so no row compares equal, and exact-match
    opcodes degrade to one big "replace" that pairs rows *positionally* — a single
    extra detected row (e.g. a title) then shifts every row and silently collapses
    the position-sensitive metrics (§2.42).
    """
    n, m = len(gt_sigs), len(pred_sigs)
    if n == 0 or m == 0:
        return []

    # score[i][j] = best total similarity aligning gt[i:] against pred[j:].
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = max(score[i + 1][j], score[i][j + 1])  # skip a gt / pred row
            sim = _row_similarity(gt_sigs[i], pred_sigs[j])
            if sim >= _ROW_ALIGN_MIN_SIMILARITY:
                best = max(best, sim + score[i + 1][j + 1])
            score[i][j] = best

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        sim = _row_similarity(gt_sigs[i], pred_sigs[j])
        if sim >= _ROW_ALIGN_MIN_SIMILARITY and score[i][j] == sim + score[i + 1][j + 1]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif score[i][j] == score[i + 1][j]:
            i += 1
        else:
            j += 1
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
            "numeric_cells_total": 0,
            "numeric_cells_correct": 0,
            "numeric_cell_accuracy": 0.0,
            "numeric_cells_khmer_digit_slips": 0,
            "khmer_cells_total": 0,
            "khmer_cells_correct": 0,
            "khmer_cell_accuracy": 0.0,
            "empty_gt_cells_total": 0,
            "empty_gt_cells_clean": 0,
            "empty_cell_precision": None,
            "grid_shape_match": False,
            "col_count_match": False,
            "row_alignment_rate": 0.0,
            "col_alignment_rate": 0.0,
        }

    tables_found = len(pred_tables)

    gt_stripped = _strip_title_row(gt_grid)
    gt_rows = len(gt_stripped)
    gt_cols = _grid_cols(gt_stripped)

    # combine ALL detected tables (real docs fragment one table into many regions)
    combined = [row for t in pred_tables for row in pred_table_grid(t)]
    pred_stripped = _strip_title_row(combined)

    pred_rows = len(pred_stripped)
    pred_cols = _grid_cols(pred_stripped)

    cells_total = gt_rows * gt_cols

    gt_sigs = [tuple(_norm(c) for c in row) for row in gt_stripped]
    pred_sigs = [tuple(_norm(c) for c in row) for row in pred_stripped]

    # --- Structure-only metrics: script-INDEPENDENT ---
    # An engine with no Khmer ability can still produce the best grid in the field,
    # and every other metric here conflates structure with recognition. These two
    # answer "did it find the right table?" without reading a single character.
    # grid_shape_match is the strict form (exact rows AND cols); row_alignment_rate
    # is the graded form — what fraction of GT rows found a partner at all.
    # col_count_match splits out the COLUMN axis deliberately: measured across the
    # real runs, engines land within ±1 row almost everywhere (a systematic header
    # artifact) which pins grid_shape_match near-always-False and drains it of
    # information, while column count genuinely separates engines — one real run
    # produced 12 columns for a 9-column table while its rivals produced 9.
    grid_shape_match = (pred_rows == gt_rows) and (pred_cols == gt_cols)
    col_count_match = pred_cols == gt_cols

    # Aligned ONCE and reused by every consumer below. _align_rows is
    # Needleman-Wunsch with a Levenshtein similarity per candidate pair, so on a
    # 75-row table each call is expensive; it used to run twice here and adding a
    # third caller made a full re-score time out.
    row_pairs = _align_rows(gt_sigs, pred_sigs)
    row_alignment_rate = len(row_pairs) / gt_rows if gt_rows > 0 else 0.0

    # Columns need the same treatment as rows, for the same reason. Measured: an
    # engine recovered 184/184 GT numeric values on budget p4 yet scored 0.000,
    # because it emitted one extra column and shifted every cell. A column's
    # identity is the values down it, so the column signatures are built from the
    # ALREADY-ALIGNED rows and fed to the same monotonic aligner — which keeps
    # genuinely swapped columns wrong (monotonicity forbids reordering).
    # Signatures are digit-FOLDED: alignment answers "which column is this?", not
    # "is it correct?". An engine that renders ១២៣ where the GT has 123 is still
    # the same column, and must align so the numeric metric can then judge it on
    # value. Scoring below uses the raw text, so folding here loses nothing.
    col_pairs = _align_rows(
        [tuple(_fold_numeric(_cell(gt_stripped, gi, c)) for gi, _ in row_pairs)
         for c in range(gt_cols)],
        [tuple(_fold_numeric(_cell(pred_stripped, pj, c)) for _, pj in row_pairs)
         for c in range(pred_cols)],
    ) if row_pairs else []
    col_alignment_rate = len(col_pairs) / gt_cols if gt_cols > 0 else 0.0

    # Every consumer scores at aligned (row, column) intersections.
    cell_pairs = [((gi, gc), (pj, pc)) for gi, pj in row_pairs for gc, pc in col_pairs]

    cells_correct = sum(
        1 for (gi, gc), (pj, pc) in cell_pairs
        if _cell(gt_stripped, gi, gc) == _cell(pred_stripped, pj, pc)
    )
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

    # --- Numeric-cell accuracy (financial tables are numeral-dominated) ---
    # Denominator = every numeric GT cell in the grid (numeric = well-formed
    # number after digit-folding), so numeric cells in dropped rows count as
    # misses — mirrors how cell_accuracy uses the full gt_rows*gt_cols total.
    numeric_cells_total = sum(
        1
        for r in range(gt_rows)
        for c in range(gt_cols)
        if c < len(gt_stripped[r]) and _is_numeric(gt_stripped[r][c])
    )
    # Correctness + Khmer-digit slips are scored on aligned row pairs only.
    # Both sides are folded before comparison, so a Khmer-digit rendering of the
    # right value scores correct; khmer_digit_slips separately counts paired pred
    # cells that carry any Khmer digit (value-right-but-mixed-script vs value-wrong).
    numeric_cells_correct = 0
    numeric_cells_khmer_digit_slips = 0
    # Khmer-label cells: same denominator rule as numeric (every Khmer GT cell,
    # so cells in dropped rows count as misses). Compared with plain _norm equality
    # — no digit folding, since these are text.
    khmer_cells_total = sum(
        1
        for r in range(gt_rows)
        for c in range(gt_cols)
        if c < len(gt_stripped[r]) and _is_khmer_text(gt_stripped[r][c])
    )
    khmer_cells_correct = 0
    # Empty-cell precision (§2.35): phantom text in empty GT cells (e.g. a cell
    # border read as "|") pollutes exports but is invisible to Recall — count it.
    # Scored over aligned row pairs only (pollution is only measurable where a
    # GT row has a predicted counterpart).
    empty_gt_cells_total = 0
    empty_gt_cells_clean = 0
    for (gi, gc), (pj, pc) in cell_pairs:
        gt_raw = _cell(gt_stripped, gi, gc)
        pred_raw = _cell(pred_stripped, pj, pc)
        if not gt_raw:
            empty_gt_cells_total += 1
            if not pred_raw:
                empty_gt_cells_clean += 1
        if _is_khmer_text(gt_raw) and pred_raw == gt_raw:
            khmer_cells_correct += 1
        if not _is_numeric(gt_raw):
            continue
        if _fold_numeric(pred_raw) == _fold_numeric(gt_raw):
            numeric_cells_correct += 1
        if _has_khmer_digit(pred_raw):
            numeric_cells_khmer_digit_slips += 1
    numeric_cell_accuracy = (
        numeric_cells_correct / numeric_cells_total if numeric_cells_total > 0 else 0.0
    )
    khmer_cell_accuracy = (
        khmer_cells_correct / khmer_cells_total if khmer_cells_total > 0 else 0.0
    )
    empty_cell_precision = (
        empty_gt_cells_clean / empty_gt_cells_total if empty_gt_cells_total > 0 else None
    )

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
        "khmer_cells_total": khmer_cells_total,
        "khmer_cells_correct": khmer_cells_correct,
        "khmer_cell_accuracy": khmer_cell_accuracy,
        "numeric_cells_total": numeric_cells_total,
        "numeric_cells_correct": numeric_cells_correct,
        "numeric_cell_accuracy": numeric_cell_accuracy,
        "numeric_cells_khmer_digit_slips": numeric_cells_khmer_digit_slips,
        "empty_gt_cells_total": empty_gt_cells_total,
        "empty_gt_cells_clean": empty_gt_cells_clean,
        "empty_cell_precision": empty_cell_precision,
        "grid_shape_match": grid_shape_match,
        "col_count_match": col_count_match,
        "row_alignment_rate": row_alignment_rate,
        "col_alignment_rate": col_alignment_rate,
    }


def pool_gt_text(gt: dict) -> str:
    # readable pooled GT: paragraphs + table cells + isolated data + footer, one piece per line
    parts: list[str] = []
    parts.extend(gt.get("paragraphs", []))
    for tbl in gt.get("tables", []) or []:
        for row in tbl.get("data", []):
            parts.extend(row)
    grid = gt.get("data")
    if grid:
        for row in grid:
            parts.extend(row)
    footer = gt.get("footer", "")
    if footer:
        parts.append(footer)
    return "\n".join(p for p in parts if p)


def pool_pred_text(ocr_text: str, pred_tables: list[dict]) -> str:
    parts: list[str] = [ocr_text] if ocr_text else []
    for tbl in pred_tables:
        for cell in tbl.get("cells", []):
            cell_text = " ".join(t["text"] for t in (cell.get("text_lines") or []) if t.get("text")).strip()
            if cell_text:
                parts.append(cell_text)
    return "\n".join(parts)


def evaluate_document(ocr_text: str, pred_tables: list[dict], gt: dict) -> dict:
    # Placement-agnostic: pool ALL text on each side, then CER. Robust to whether
    # content was classified as table vs paragraph by either side.
    return {"document_cer": cer(_norm(pool_gt_text(gt)), _norm(pool_pred_text(ocr_text, pred_tables)))}


def pool_gt_recognition_text(gt: dict) -> str:
    # Single-source recognition GT: avoids pool_gt_text's double-count (our real-doc
    # GT restates table rows in `paragraphs`). Table-bearing GT -> the cells once;
    # text-only GT -> paragraphs + footer.
    grid = gt_table_grid(gt)
    if grid:
        return "\n".join(c for row in grid for c in row if c)
    parts = list(gt.get("paragraphs", []))
    footer = gt.get("footer", "")
    if footer:
        parts.append(footer)
    return "\n".join(p for p in parts if p)


def evaluate_recognition(ocr_text: str, pred_tables: list[dict], gt: dict) -> dict:
    # Recognition-only CER vs a single-source GT (no double-count). Pred is pooled
    # like evaluate_document; a flat external model passes pred_tables=[].
    return {"recognition_cer": cer(_norm(pool_gt_recognition_text(gt)), _norm(pool_pred_text(ocr_text, pred_tables)))}


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
