"""Recall-failure taxonomy for a single document run.

Reproduces the exact pipeline scripts/eval_document.py runs (OCR_ENGINE=surya,
--preprocess, document stitching via merge_document_tables), captures the
predicted grid + verified document GT, row-aligns them with the SAME
difflib-based alignment evaluate_structure.evaluate_table uses, then classifies
WHY each unrecovered GT cell (per the Cell_Content_Recall multiset logic) was
missed:

  ROW-DROPPED  GT row has no aligned predicted row at all (difflib 'delete').
  MERGED       2+ consecutive GT rows in a dropped/aligned block map onto a
               single predicted row (predicted row count in the opcode block
               is smaller than the GT row count spanned -> collapsed rows).
  SPLIT        1 GT row's content is spread across 2+ consecutive predicted
               rows (predicted row count in the block exceeds GT row count).
  CELL-BLANK   row aligned 1:1, predicted cell text is empty.
  WRONG-TEXT   row aligned 1:1, predicted cell text is non-empty but != GT.

Only classifies content misses (recall logic: non-empty GT cell whose value
isn't present with enough multiplicity in the predicted flat multiset). Then
buckets by column and by document section (from the section-divider marker
rows in this document's GT, e.g. "ក/-...", "ខ/-...", "គ/-...").

Usage:
    OCR_ENGINE=surya uv run python scripts/recall_taxonomy.py --preprocess [stem]

Writes a JSON dump of the raw grids + classified misses to
scripts/_recall_taxonomy_output.json for inspection, and prints the taxonomy
table + conclusion to stdout.
"""
from __future__ import annotations
import difflib
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from khmer_pipeline.ingest import ingest
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.preprocess import preprocess, PreprocessConfig
from khmer_pipeline.engines.engine_registry import ACTIVE_OCR_ENGINE
from khmer_pipeline.engines.table_merge_pages import merge_document_tables
from khmer_pipeline.evaluation.evaluate_structure import (
    pred_table_grid,
    _strip_title_row,
    _norm,
    _grid_cols,
)

_REAL_DIR = Path("eval/datasets/real")

_COL_NAMES = [
    "ល.រ (no.)",
    "មុខទំនិញ (item name)",
    "ឯកតា (unit)",
    "08-06-26 បោះដុំ (wholesale)",
    "08-06-26 លក់រាយ (retail)",
    "09-06-26 បោះដុំ (wholesale)",
    "09-06-26 លក់រាយ (retail)",
    "ប្រែប្រួល បោះដុំ% (wholesale %chg)",
    "ប្រែប្រួល លក់រាយ% (retail %chg)",
]


def _load_pages(stem: str, do_preprocess: bool) -> PreprocessResult:
    pngs = sorted(glob.glob(str(_REAL_DIR / f"{stem}_p*.png")))
    images = []
    for p in pngs:
        images.extend(ingest(Path(p).read_bytes(), Path(p).name, dpi=200).page_images)
    if do_preprocess:
        ing = IngestResult(source_name=stem, page_images=images, dpi=200, page_count=len(images))
        return preprocess(ing, PreprocessConfig())
    return PreprocessResult(source_name=stem, page_images=images, dpi=200, page_count=len(images))


def _section_for_row(gi: int, section_rows: list[tuple[int, str]]) -> str:
    # section_rows: sorted [(gt_row_index, label), ...] marker rows (divider rows,
    # col0 non-empty + rest empty). A GT row belongs to the most recent marker
    # at or before it.
    label = "(preamble/header)"
    for idx, lab in section_rows:
        if idx <= gi:
            label = lab
        else:
            break
    return label


def main() -> int:
    argv = sys.argv[1:]
    do_preprocess = "--preprocess" in argv
    positional = [a for a in argv if not a.startswith("--")]
    if positional:
        stem = positional[0]
    else:
        candidates = sorted(_REAL_DIR.glob("*09.06.26_document_gt.json"))
        if not candidates:
            print("No 09.06.26 document_gt.json found under eval/datasets/real/")
            return 1
        stem = candidates[0].name.removesuffix("_document_gt.json")

    engine = getattr(ACTIVE_OCR_ENGINE, "__name__", "ocr")
    print(f"stem={stem}")
    print(f"engine={engine}  preprocess={'on' if do_preprocess else 'off'}")

    pre = _load_pages(stem, do_preprocess)
    if not pre.page_images:
        print(f"No page PNGs for stem: {stem}")
        return 1

    result = ACTIVE_OCR_ENGINE(pre)
    merged = merge_document_tables(result.pages)
    combined = [row for t in merged for row in pred_table_grid(t)]
    pred_grid = _strip_title_row(combined)

    gt_path = _REAL_DIR / f"{stem}_document_gt.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    if gt.get("needs_review_rows"):
        print(f"WARNING: GT has unverified rows: {gt['needs_review_rows']}")
    gt_grid_raw = gt["tables"][0]["data"]
    gt_grid = _strip_title_row(gt_grid_raw)

    gt_rows = len(gt_grid)
    gt_cols = _grid_cols(gt_grid)
    pred_rows = len(pred_grid)
    pred_cols = _grid_cols(pred_grid)
    print(f"GT {gt_rows}x{gt_cols}   pred {pred_rows}x{pred_cols}")

    gt_sigs = [tuple(_norm(c) for c in row) for row in gt_grid]
    pred_sigs = [tuple(_norm(c) for c in row) for row in pred_grid]

    sm = difflib.SequenceMatcher(None, gt_sigs, pred_sigs, autojunk=False)
    opcodes = sm.get_opcodes()

    # 1:1 pairs, exactly as evaluate_structure._align_rows does (equal/replace
    # blocks paired positionally, delete/insert unmatched).
    pairs: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("equal", "replace"):
            for k in range(min(i2 - i1, j2 - j1)):
                pairs[i1 + k] = j1 + k

    # Row-level opcode classification per GT row index, independent of cell content:
    #   equal/replace block with gt_span == pred_span -> ONE_TO_ONE (per-cell class below)
    #   delete block (pred_span == 0)                  -> ROW-DROPPED for all GT rows in it
    #   replace block with pred_span < gt_span          -> MERGED (collapsed) for the
    #                                                       unpaired tail GT rows
    #   replace block with pred_span > gt_span          -> SPLIT credit noted (paired rows
    #                                                       still scored normally; extra
    #                                                       pred rows are noise, not a GT miss)
    gt_row_block_kind: dict[int, str] = {}
    for tag, i1, i2, j1, j2 in opcodes:
        gt_span = i2 - i1
        pred_span = j2 - j1
        if tag == "equal":
            for gi in range(i1, i2):
                gt_row_block_kind[gi] = "ONE_TO_ONE"
        elif tag == "replace":
            if pred_span == 0:
                for gi in range(i1, i2):
                    gt_row_block_kind[gi] = "ROW-DROPPED"
            elif pred_span < gt_span:
                # first pred_span GT rows get positional pairs (1:1-ish), the
                # remaining tail GT rows in this block had no distinct pred row
                # of their own -> collapsed/MERGED into predicted rows of this block.
                for k in range(pred_span):
                    gt_row_block_kind[i1 + k] = "ONE_TO_ONE"
                for gi in range(i1 + pred_span, i2):
                    gt_row_block_kind[gi] = "MERGED"
            elif pred_span > gt_span:
                for k in range(gt_span):
                    gt_row_block_kind[i1 + k] = "ONE_TO_ONE"
                # SPLIT is a pred-side phenomenon (extra pred rows); the paired
                # GT rows are still ONE_TO_ONE for cell scoring, but flag the
                # block so misses inside it can be attributed to SPLIT context.
                for gi in range(i1, i1 + gt_span):
                    gt_row_block_kind[gi] = "SPLIT-BLOCK"
        elif tag == "delete":
            for gi in range(i1, i2):
                gt_row_block_kind[gi] = "ROW-DROPPED"
        # insert: pred-only rows, no GT row involved

    # Section markers: GT rows where col0 non-empty and all other cols empty.
    section_rows: list[tuple[int, str]] = []
    for gi, row in enumerate(gt_grid):
        if row and row[0].strip() and all((row[c].strip() == "" if c < len(row) else True) for c in range(1, gt_cols)):
            section_rows.append((gi, row[0].strip()))

    # Recall multiset (mirrors evaluate_structure exactly) to know global recall.
    gt_nonempty_vals = [
        _norm(gt_grid[r][c]) for r in range(gt_rows) for c in range(gt_cols)
        if c < len(gt_grid[r]) and _norm(gt_grid[r][c])
    ]
    pred_flat_vals = [
        _norm(pred_grid[r][c]) for r in range(pred_rows) for c in range(_grid_cols(pred_grid))
        if c < len(pred_grid[r])
    ]
    pred_counter = Counter(pred_flat_vals)
    gt_counter = Counter(gt_nonempty_vals)
    matched_total = sum(min(cnt, pred_counter.get(val, 0)) for val, cnt in gt_counter.items())
    cell_content_recall = matched_total / len(gt_nonempty_vals) if gt_nonempty_vals else 0.0

    # To classify EACH individual non-empty GT cell as matched/missed under the
    # multiset rule, we need to consume the multiset greedily row-major (same
    # traversal order used to build gt_nonempty_vals) so counts line up.
    remaining_pred_counter = Counter(pred_flat_vals)
    misses: list[dict] = []
    n_matched = 0
    n_total_nonempty = 0
    for gi in range(gt_rows):
        row = gt_grid[gi]
        section = _section_for_row(gi, section_rows)
        block_kind = gt_row_block_kind.get(gi, "ROW-DROPPED")
        for c in range(gt_cols):
            gt_val = _norm(row[c]) if c < len(row) else ""
            if not gt_val:
                continue
            n_total_nonempty += 1
            if remaining_pred_counter.get(gt_val, 0) > 0:
                remaining_pred_counter[gt_val] -= 1
                n_matched += 1
                continue
            # MISS: classify why.
            pj = pairs.get(gi)
            if block_kind == "ROW-DROPPED":
                mode = "ROW-DROPPED"
                pred_val = None
            elif block_kind == "MERGED":
                mode = "MERGED"
                pred_val = None
            else:
                # ONE_TO_ONE or SPLIT-BLOCK: row is aligned to a specific pred row;
                # classify by that pred row's cell content.
                if pj is None:
                    mode = "ROW-DROPPED"
                    pred_val = None
                else:
                    pred_row = pred_grid[pj] if pj < len(pred_grid) else []
                    pred_val = _norm(pred_row[c]) if c < len(pred_row) else ""
                    if block_kind == "SPLIT-BLOCK":
                        mode = "SPLIT"
                    elif pred_val == "":
                        mode = "CELL-BLANK"
                    else:
                        mode = "WRONG-TEXT"
            misses.append({
                "gt_row": gi,
                "col": c,
                "col_name": _COL_NAMES[c] if c < len(_COL_NAMES) else f"col{c}",
                "gt_val": gt_val,
                "pred_val": pred_val,
                "mode": mode,
                "section": section,
            })

    assert n_total_nonempty == len(gt_nonempty_vals)
    # sanity: this greedy per-row-major consumption gives the same total match
    # count as the Counter min-sum used by evaluate_structure (both are
    # multiset intersections over the same multiset).
    assert n_matched == matched_total, (n_matched, matched_total)

    # --- Taxonomy summary ---
    n_miss = len(misses)
    mode_counts = Counter(m["mode"] for m in misses)
    print("\n=== Cell_Content_Recall check ===")
    print(f"  non-empty GT cells: {len(gt_nonempty_vals)}")
    print(f"  matched: {n_matched}   missed: {n_miss}")
    print(f"  cell_content_recall = {cell_content_recall:.4f}")

    print("\n=== Failure-mode taxonomy (of the missed cells) ===")
    print(f"  {'mode':<14}{'count':>8}{'% of misses':>14}{'% of GT cells':>16}")
    for mode in ["ROW-DROPPED", "MERGED", "SPLIT", "CELL-BLANK", "WRONG-TEXT"]:
        cnt = mode_counts.get(mode, 0)
        pct_miss = 100 * cnt / n_miss if n_miss else 0.0
        pct_gt = 100 * cnt / len(gt_nonempty_vals) if gt_nonempty_vals else 0.0
        print(f"  {mode:<14}{cnt:>8}{pct_miss:>13.1f}%{pct_gt:>15.1f}%")

    recognition_modes = {"CELL-BLANK", "WRONG-TEXT"}
    segmentation_modes = {"ROW-DROPPED", "MERGED", "SPLIT"}
    n_recognition = sum(mode_counts.get(m, 0) for m in recognition_modes)
    n_segmentation = sum(mode_counts.get(m, 0) for m in segmentation_modes)
    print(f"\n  RECOGNITION-attributable (CELL-BLANK+WRONG-TEXT): {n_recognition} ({100*n_recognition/n_miss:.1f}% of misses)"
          if n_miss else "  no misses")
    print(f"  SEGMENTATION-attributable (ROW-DROPPED+MERGED+SPLIT): {n_segmentation} ({100*n_segmentation/n_miss:.1f}% of misses)"
          if n_miss else "")

    print("\n=== Misses by column ===")
    col_counts = Counter(m["col_name"] for m in misses)
    for c in range(gt_cols):
        name = _COL_NAMES[c] if c < len(_COL_NAMES) else f"col{c}"
        cnt = col_counts.get(name, 0)
        pct = 100 * cnt / n_miss if n_miss else 0.0
        print(f"  {name:<40}{cnt:>6}  ({pct:.1f}%)")

    print("\n=== Misses by column x mode ===")
    col_mode = defaultdict(Counter)
    for m in misses:
        col_mode[m["col_name"]][m["mode"]] += 1
    header = f"  {'column':<40}" + "".join(f"{m:>13}" for m in ["ROW-DROPPED", "MERGED", "SPLIT", "CELL-BLANK", "WRONG-TEXT"])
    print(header)
    for c in range(gt_cols):
        name = _COL_NAMES[c] if c < len(_COL_NAMES) else f"col{c}"
        row_counts = col_mode.get(name, Counter())
        line = f"  {name:<40}" + "".join(f"{row_counts.get(m,0):>13}" for m in ["ROW-DROPPED", "MERGED", "SPLIT", "CELL-BLANK", "WRONG-TEXT"])
        print(line)

    print("\n=== Misses by section ===")
    sec_counts = Counter(m["section"] for m in misses)
    sec_totals = Counter()
    for gi in range(gt_rows):
        sec = _section_for_row(gi, section_rows)
        row = gt_grid[gi]
        sec_totals[sec] += sum(1 for c in range(gt_cols) if c < len(row) and _norm(row[c]))
    for sec, total in sec_totals.items():
        cnt = sec_counts.get(sec, 0)
        pct_of_section = 100 * cnt / total if total else 0.0
        print(f"  {sec:<30} missed {cnt:>4}/{total:<4} nonempty GT cells  ({pct_of_section:.1f}%)")

    print("\n=== Misses by section x mode ===")
    sec_mode = defaultdict(Counter)
    for m in misses:
        sec_mode[m["section"]][m["mode"]] += 1
    print(f"  {'section':<30}" + "".join(f"{m:>13}" for m in ["ROW-DROPPED", "MERGED", "SPLIT", "CELL-BLANK", "WRONG-TEXT"]))
    for sec in sec_totals:
        row_counts = sec_mode.get(sec, Counter())
        line = f"  {sec:<30}" + "".join(f"{row_counts.get(m,0):>13}" for m in ["ROW-DROPPED", "MERGED", "SPLIT", "CELL-BLANK", "WRONG-TEXT"])
        print(line)

    # sample WRONG-TEXT examples (most diagnostic for recognizer fine-tuning case)
    wrong = [m for m in misses if m["mode"] == "WRONG-TEXT"]
    print(f"\n=== Sample WRONG-TEXT cells (up to 20 of {len(wrong)}) ===")
    for m in wrong[:20]:
        print(f"  row{m['gt_row']:>3} {m['col_name']:<25} GT={m['gt_val']!r:<30} PRED={m['pred_val']!r}")

    blank = [m for m in misses if m["mode"] == "CELL-BLANK"]
    print(f"\n=== Sample CELL-BLANK cells (up to 20 of {len(blank)}) ===")
    for m in blank[:20]:
        print(f"  row{m['gt_row']:>3} {m['col_name']:<25} GT={m['gt_val']!r}")

    out = {
        "stem": stem,
        "engine": engine,
        "preprocess": do_preprocess,
        "gt_rows": gt_rows, "gt_cols": gt_cols,
        "pred_rows": pred_rows, "pred_cols": pred_cols,
        "cell_content_recall": cell_content_recall,
        "n_nonempty_gt_cells": len(gt_nonempty_vals),
        "n_matched": n_matched,
        "n_missed": n_miss,
        "mode_counts": dict(mode_counts),
        "misses": misses,
        "gt_grid": gt_grid,
        "pred_grid": pred_grid,
    }
    out_path = Path("scripts/_recall_taxonomy_output.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n(raw grids + classified misses written to {out_path})")

    print("\n=== CONCLUSION ===")
    if n_miss == 0:
        print("  No misses — recall is effectively 1.0 in this run.")
    elif n_recognition > n_segmentation:
        print(f"  Misses are RECOGNITION-dominated ({n_recognition}/{n_miss} = "
              f"{100*n_recognition/n_miss:.0f}%): rows are structurally aligned but "
              f"cells come back blank or with wrong text on otherwise-legible rows.")
        print("  -> Recognizer fine-tuning is plausible and worth pursuing.")
    else:
        print(f"  Misses are SEGMENTATION-dominated ({n_segmentation}/{n_miss} = "
              f"{100*n_segmentation/n_miss:.0f}%): whole GT rows are dropped, merged, "
              f"or split by layout+stitch before recognition ever sees them as intended.")
        print("  -> Recognizer fine-tuning will NOT fix this; it's a layout/segmentation problem.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
