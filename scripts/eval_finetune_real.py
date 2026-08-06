"""Scores a fine-tuned adapter's predictions against real, hand-verified ARDB ground truth
(eval/datasets/real) -- not the synthetic template-substituted validation split the Colab
notebooks score against. Scoped to ARDB documents only (mentor-limited scope): the corpus's own
"non-ARDB test file" (CambodiaBudgetExecutioninApr-2024) is excluded automatically.

Inference happens in Colab (reusing the notebook's own already-proven FastVisionModel loading,
not a separately maintained local path -- see the "Score against real ARDB docs" cell added to
colab_gemma4_e2b_finetune.ipynb / colab_qwen35_finetune.ipynb), which writes a predictions JSON
keyed by page filename. This script only scores that JSON -- no torch/transformers/peft needed,
runs in the project's normal env:

    uv run python3 scripts/eval_finetune_real.py --predictions real_predictions.json

Reuses this project's own table/text scoring (evaluate_table, evaluate_text, evaluate_document
from khmer_pipeline.evaluation.evaluate_structure) rather than reimplementing CER/row-alignment
-- only the adapter layer (parsing this task's JSON-region output into the pred_tables/ocr_text
shapes those functions expect) is new; see parse_prediction/html_to_grid below.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from khmer_pipeline.evaluation.evaluate_structure import (
    evaluate_document,
    evaluate_table,
    evaluate_text,
    gt_table_grid,
)

_REAL_DIR = Path("eval/datasets/real")
# The corpus's own "non-ARDB test file" -- mentor has scoped fine-tuning/eval work to ARDB
# documents only; this one is a government budget report, structurally unrelated.
_NON_ARDB_STEMS = ("CambodiaBudgetExecutioninApr-2024",)


def ardb_page_names() -> list[str]:
    """Every real-doc page PNG filename in eval/datasets/real, minus the non-ARDB doc's.
    Derived from the actual files present, never hand-typed -- these are Khmer filenames."""
    names = []
    for gt_path in sorted(_REAL_DIR.glob("*_p*_ground_truth.json")):
        stem = re.sub(r"_p\d+_ground_truth\.json$", "", gt_path.name)
        if stem in _NON_ARDB_STEMS:
            continue
        names.append(gt_path.name.replace("_ground_truth.json", ".png"))
    return names


class _TableGridParser(HTMLParser):
    """Parses harvest_table_gt.py's own grid_to_html(...) output back into a plain grid -- the
    inverse of that function. A colspan cell (a whole-row section/title, see
    _html_section_row) puts its text in column 0 only and leaves the remaining spanned columns
    blank, matching how such rows are represented in the GT's own `data` grid (not literally
    spanned there either -- grid_to_html is what introduces the colspan for display)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_chars: list[str] = []
        self._in_cell = False
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_chars = []
            self._colspan = 1
            for k, v in attrs:
                if k == "colspan":
                    try:
                        self._colspan = int(v)
                    except ValueError:
                        pass

    def handle_data(self, data):
        if self._in_cell:
            self._cell_chars.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None:
            text = "".join(self._cell_chars).strip()
            self._row.append(text)
            self._row.extend([""] * (self._colspan - 1))
            self._in_cell = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def html_to_grid(html: str) -> list[list[str]]:
    parser = _TableGridParser()
    parser.feed(html)
    return parser.rows


def _grid_to_pred_table(grid: list[list[str]]) -> dict:
    """Wraps a plain grid as the {"cells": [...]} shape evaluate_structure's pred_table_grid /
    evaluate_table expect (the Surya-engine convention: one dict per detected table, cells keyed
    by row_id/col_id/text_lines)."""
    cells = []
    for r, row in enumerate(grid):
        for c, text in enumerate(row):
            if text:
                cells.append({"row_id": r, "col_id": c, "text_lines": [{"text": text}]})
    return {"cells": cells}


def _try_repair_json(text: str) -> str | None:
    """Recovers two failure modes confirmed in real Colab generations (see
    eval/qwen_finetune_runs.md): missing only the outer list's closing ']', and missing BOTH
    the opening '[' and closing ']' (a bare comma-separated sequence of '{...}' objects with no
    wrapping list at all -- the first fix only handled the former and recovered 0/7 failures on
    a run that turned out to have the latter). Finds the last complete object (last '}'),
    discards anything after it (e.g. a stray end-of-turn token), then adds whichever of '[' / ']'
    is missing. Text cut off mid-field (no trailing '}' at all) is left alone rather than
    guessing at missing content. Mirrors the notebooks' own eval cells' fallback, kept in sync
    so parse-failure counts stay comparable between synthetic-val and real-doc scoring."""
    stripped = text.strip()
    last_brace = stripped.rfind("}")
    if last_brace == -1:
        return None
    core = stripped[:last_brace + 1]
    if not core.startswith("["):
        core = "[" + core
    if not core.endswith("]"):
        core = core + "]"
    if core == stripped:
        return None
    return core


def parse_prediction(pred_text: str) -> tuple[str, list[dict]]:
    """Splits this task's predicted JSON region list into (ocr_text, pred_tables) -- the two
    inputs evaluate_text/evaluate_table/evaluate_document expect. Table region text (HTML) is
    parsed back into a grid; every other non-empty, non-Picture region's text is pooled as
    prose. Malformed JSON returns ("", []) -- callers track parse failures separately, same as
    the notebooks' own eval cells do."""
    regions = None
    for candidate in (pred_text, _try_repair_json(pred_text)):
        if candidate is None:
            continue
        try:
            regions = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        break
    if not isinstance(regions, list):
        return "", []
    prose_parts: list[str] = []
    pred_tables: list[dict] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        text = region.get("text", "")
        if not text:
            continue
        if region.get("label") == "Table":
            pred_tables.append(_grid_to_pred_table(html_to_grid(text)))
        elif region.get("label") != "Picture":
            prose_parts.append(text)
    return "\n".join(prose_parts), pred_tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=Path,
                         help="JSON file from the Colab 'Score against real ARDB docs' cell, keyed by page PNG filename")
    args = parser.parse_args()

    predictions: dict[str, str] = json.loads(args.predictions.read_text(encoding="utf-8"))
    expected_names = ardb_page_names()

    all_table_metrics: list[dict] = []
    all_text_cer: list[float] = []
    all_doc_cer: list[float] = []
    parse_failures = 0
    n_pages = 0

    for name in expected_names:
        if name not in predictions:
            print(f"[skip] {name}: no prediction in {args.predictions.name}")
            continue
        n_pages += 1
        gt_path = _REAL_DIR / name.replace(".png", "_ground_truth.json")
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ocr_text, pred_tables = parse_prediction(predictions[name])
        if not pred_tables and not ocr_text:
            parse_failures += 1
            print(f"[{n_pages}] {name}: parse failure (malformed/empty JSON)")
            continue

        table_metrics = evaluate_table(pred_tables, gt_table_grid(gt))
        text_metrics = evaluate_text(ocr_text, pred_tables, gt)
        doc_metrics = evaluate_document(ocr_text, pred_tables, gt)

        all_table_metrics.append(table_metrics)
        if text_metrics["text_cer"] is not None:
            all_text_cer.append(text_metrics["text_cer"])
        all_doc_cer.append(doc_metrics["document_cer"])

        print(f"[{n_pages}] {name}: table_cer={table_metrics['table_cer']:.3f} "
              f"cell_accuracy={table_metrics['cell_accuracy']:.3f} "
              f"text_cer={text_metrics['text_cer']} document_cer={doc_metrics['document_cer']:.3f} "
              f"grid={table_metrics['pred_rows']}x{table_metrics['pred_cols']} "
              f"(gt {table_metrics['gt_rows']}x{table_metrics['gt_cols']}, "
              f"shape_match={table_metrics['grid_shape_match']} "
              f"col_match={table_metrics['col_count_match']})")

    print(f"\n{'=' * 60}")
    print(f"Pages scored: {len(all_doc_cer)} / {len(expected_names)}  (parse failures: {parse_failures})")
    if all_table_metrics:
        print(f"Mean table_cer: {sum(m['table_cer'] for m in all_table_metrics) / len(all_table_metrics):.4f}")
        print(f"Mean cell_accuracy: {sum(m['cell_accuracy'] for m in all_table_metrics) / len(all_table_metrics):.4f}")
        print(f"Mean numeric_cell_accuracy: {sum(m['numeric_cell_accuracy'] for m in all_table_metrics) / len(all_table_metrics):.4f}")
        # Structure-only (script-independent) shape metrics -- "did it find the right table
        # dimensions" separated from "did it read the characters right." See evaluate_table's
        # own docstring comments for why row/col count are tracked separately (±1-row header
        # drift is common and uninformative; column-count drift is rarer and more diagnostic).
        n = len(all_table_metrics)
        print(f"Grid shape match rate (exact rows AND cols): "
              f"{sum(m['grid_shape_match'] for m in all_table_metrics) / n:.4f}")
        print(f"Column count match rate: "
              f"{sum(m['col_count_match'] for m in all_table_metrics) / n:.4f}")
        print(f"Mean row alignment rate (fraction of GT rows found at all): "
              f"{sum(m['row_alignment_rate'] for m in all_table_metrics) / n:.4f}")
        print(f"Mean column alignment rate (fraction of GT columns found at all): "
              f"{sum(m['col_alignment_rate'] for m in all_table_metrics) / n:.4f}")
    if all_text_cer:
        print(f"Mean text_cer (prose): {sum(all_text_cer) / len(all_text_cer):.4f} (n={len(all_text_cer)})")
    if all_doc_cer:
        print(f"Mean document_cer (whole page pooled): {sum(all_doc_cer) / len(all_doc_cer):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
