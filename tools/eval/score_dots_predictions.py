"""Score dots.ocr Colab predictions against the local ground truth.

Companion to scripts/colab_dots_ocr.ipynb. The notebook emits
`predictions.json = {"<image.png>": "<raw dots.ocr output>"}`; this scores each
page identically to the local engines, reusing the production HTML-table parser
and the same metric, so a challenger number is directly comparable to Surya's.

    uv run python scripts/score_dots_predictions.py --predictions predictions.json

No new table logic: dots.ocr emits tables as HTML, and _parse_html_table_with_spans
-> _build_table_from_grid is exactly what Surya's own VLM output already flows
through. Honours scoring_scope (text-layer GT masks non-numeric columns) and the
GT circularity guard, matching scripts/compare_engines_ab.py.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from khmer_pipeline.engines.surya import _parse_html_table_with_spans, _build_table_from_grid
from khmer_pipeline.evaluation.evaluate_structure import evaluate_table, gt_table_grid
from khmer_pipeline.evaluation.gt_provenance import is_circular, circularity_note

# Where per-page GT lives. A prediction filename's stem maps to <stem>_ground_truth.json
# in one of these; document-level ARDB GT (*_document_gt.json) is not used here since
# the notebook scores per uploaded page image.
_GT_DIRS = [
    Path("eval/datasets/real"),
    Path("eval/datasets/budget_textlayer"),
    Path("eval/datasets/moc_gas"),
]

# Same mask as compare_engines_ab: a restricted-scope GT cannot support these.
_SCOPE_MASKED_METRICS: dict[str, set[str]] = {
    "numeric_and_structure": {"cell_accuracy", "khmer_cell_accuracy", "table_cer"},
}

_ENGINE_KEY = "dots_ocr"  # for the circularity check; dots is not an LLM-GT family, so never circular


def _find_gt(stem: str) -> Path | None:
    for d in _GT_DIRS:
        p = d / f"{stem}_ground_truth.json"
        if p.exists():
            return p
    return None


def _parse_layout(raw: str) -> list[dict] | None:
    """Decode dots.ocr's layout output; tolerate a trailing-text tail or an
    object wrapper, but never bracket-slice (bbox values contain brackets too)."""
    s = raw.strip()
    # Strip a ```json fence if the model added one.
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value, _ = json.JSONDecoder().raw_decode(s)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        # Some prompt variants wrap the list under a key.
        for v in value.values():
            if isinstance(v, list):
                return v
        return None
    return value if isinstance(value, list) else value


def _tables_from_layout(elements: list[dict]) -> list[dict]:
    """Every Table element's HTML -> a pipeline Table via the shared parser."""
    tables: list[dict] = []
    for el in elements:
        if el.get("category") != "Table":
            continue
        html = el.get("text") or ""
        grid_map, _spans = _parse_html_table_with_spans(html)
        bbox = el.get("bbox") or [0, 0, 0, 0]
        tables.append(_build_table_from_grid(grid_map, html, [float(v) for v in bbox]))
    return tables


def _span_recovery(elements: list[dict]) -> int:
    """Total colspan/rowspan attributes dots.ocr emitted across all tables — the
    §2.40 capability Surya 2 no longer produces."""
    return sum((el.get("text") or "").count("colspan") + (el.get("text") or "").count("rowspan")
               for el in elements if el.get("category") == "Table")


def _fmt(value, key: str, masked: set[str]) -> str:
    return "—" if key in masked else f"{value:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    args = ap.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    print(f"{'page':52} {'cell':>7} {'numeric':>8} {'khmer':>7} {'rowaln':>7} "
          f"{'colaln':>7} {'spans':>6}  notes")
    print("-" * 110)

    for fname in sorted(preds):
        stem = Path(fname).stem
        gt_path = _find_gt(stem)
        if gt_path is None:
            print(f"{stem[:52]:52} {'':>7} {'':>8}  no GT found — skipped")
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))

        note = circularity_note(_ENGINE_KEY, gt)
        if note:
            print(f"{stem[:52]:52}  SKIP (circular): {note}")
            continue

        elements = _parse_layout(preds[fname])
        if elements is None:
            print(f"{stem[:52]:52}  parse failed (truncated / not JSON)")
            continue

        tables = _tables_from_layout(elements)
        m = evaluate_table(tables, gt_table_grid(gt))
        masked = _SCOPE_MASKED_METRICS.get(gt.get("scoring_scope") or "", set())
        spans = _span_recovery(elements)
        extra = []
        if gt.get("scoring_scope"):
            extra.append(f"scope={gt['scoring_scope']}")
        if m["pred_rows"] != m["gt_rows"] or m["pred_cols"] != m["gt_cols"]:
            extra.append(f"shape {m['pred_rows']}x{m['pred_cols']} vs GT {m['gt_rows']}x{m['gt_cols']}")
        print(f"{stem[:52]:52} {_fmt(m['cell_accuracy'], 'cell_accuracy', masked):>7} "
              f"{_fmt(m['numeric_cell_accuracy'], 'numeric_cell_accuracy', set()):>8} "
              f"{_fmt(m['khmer_cell_accuracy'], 'khmer_cell_accuracy', masked):>7} "
              f"{m['row_alignment_rate']:>7.3f} {m['col_alignment_rate']:>7.3f} {spans:>6}  "
              + " ".join(extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
