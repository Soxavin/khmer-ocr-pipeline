"""Track A Gate 2: does the fine-tuned layout detector beat Surya's layout end-to-end?

Runs the production path (ingest → preprocess → engine) over every verified GT page,
for each engine × {layout on, off} × N runs, and scores with evaluate_table.

The bet (§2.37): Surya's layout is non-deterministic — the budget table's column count
swings 14–21 between identical runs. So this reports STABILITY (do repeated runs agree
on the table's shape?) alongside accuracy, because a detector that is merely as accurate
but deterministic is already a win — and one with beautiful boxes that amputates a label
column is a loss (§2.24). Recall + Numeric_Cell_Accuracy are the honest pair; raw
Cell_Accuracy inflates on sparse tables.

Scored on the §2.42-fixed row aligner — pre-§2.42 numbers are not comparable.

Usage (repo root), weights from the Colab notebook:

    KHMER_LAYOUT_WEIGHTS=/path/khmer_layout_best.onnx \
        uv run python experiments/layout_yolo/gate.py [--runs 3] [--tag v1]

Writes experiments/layout_yolo/gate_<tag>.json. Inference only (onnxruntime) — safe
to run locally; TRAINING must stay on Colab (it freezes the 24GB Mac).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from khmer_pipeline.ingest import ingest
from khmer_pipeline.preprocess import preprocess
from khmer_pipeline.engines.engine_registry import get_ocr_engine
from khmer_pipeline.evaluation.evaluate_structure import evaluate_table

_REAL = _REPO / "eval/datasets/real"
_ENGINES = ("surya", "surya_kiri")
_METRICS = ("cell_content_recall", "numeric_cell_accuracy", "cell_accuracy", "table_cer")
_WEIGHTS_ENV = "KHMER_LAYOUT_WEIGHTS"


def _gt_pages() -> list[tuple[str, Path, list]]:
    """Every GT page that has both a table grid and a page PNG: (id, png, grid)."""
    out = []
    for gt_path in sorted(_REAL.glob("*_ground_truth.json")):
        gt = json.loads(gt_path.read_text())
        grids = [t.get("data") for t in gt.get("tables", []) if t.get("data")]
        png = gt_path.with_name(gt_path.name.replace("_ground_truth.json", ".png"))
        if grids and png.exists():
            out.append((png.stem[-20:], png, grids[0]))
    return out


def _score(engine, png: Path, grid: list) -> dict:
    pre = preprocess(ingest(png.read_bytes(), png.name))
    result = engine(pre)
    m = evaluate_table(result.pages[0].tables, grid)
    scored = {k: m.get(k) for k in _METRICS}
    scored["dims"] = f"{m['pred_rows']}x{m['pred_cols']}"
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(description="Track A end-to-end layout A/B.")
    ap.add_argument("--runs", type=int, default=3, help="repeats per config (stability)")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    # Weights are optional: the stock Apache-2.0 PP-DocLayout model needs none, so
    # the off-the-shelf detector can be baselined before any training happens.
    weights = os.environ.get(_WEIGHTS_ENV)
    if weights and not os.path.isfile(weights):
        sys.exit(f"{_WEIGHTS_ENV}={weights!r}: no such file")
    model = os.environ.get("KHMER_LAYOUT_MODEL", "pp_doc_layoutv2 (default)")
    print(f"detector: model={model} weights={weights or '(stock)'}")

    pages = _gt_pages()
    print(f"{len(pages)} GT page(s) x {len(_ENGINES)} engine(s) x layout on/off x {args.runs} run(s)\n")

    results: dict = {}
    os.environ.pop("KHMER_LAYOUT_DETECTOR", None)
    for engine_name in _ENGINES:
        engine = get_ocr_engine(engine_name)
        for layout_on in (False, True):
            # detect_table_boxes reads the env at call time, so toggling it here
            # flips the layout source without reimporting anything.
            if layout_on:
                os.environ["KHMER_LAYOUT_DETECTOR"] = "rapid"
                if weights:
                    os.environ[_WEIGHTS_ENV] = weights
            else:
                os.environ.pop("KHMER_LAYOUT_DETECTOR", None)
            cfg = f"{engine_name}/layout_{'on' if layout_on else 'off'}"

            for page_id, png, grid in pages:
                runs = [_score(engine, png, grid) for _ in range(args.runs)]
                dims = {r["dims"] for r in runs}
                entry = {
                    # Mean over runs: Surya's layout is non-deterministic, so a single
                    # run is not a measurement of it.
                    **{k: round(sum(r[k] for r in runs) / len(runs), 4) for k in _METRICS},
                    "dims_seen": sorted(dims),
                    "stable": len(dims) == 1,
                }
                results[f"{cfg}/{page_id}"] = entry
                flag = "" if entry["stable"] else "  <-- UNSTABLE"
                print(f"{cfg:26s} {page_id:22s} "
                      f"recall={entry['cell_content_recall']:.3f} "
                      f"numacc={entry['numeric_cell_accuracy']:.3f} "
                      f"dims={','.join(entry['dims_seen'])}{flag}")
            print()

    os.environ.pop("KHMER_LAYOUT_DETECTOR", None)
    out = _REPO / f"experiments/layout_yolo/gate_{args.tag}.json"
    out.write_text(json.dumps({"weights": weights, "runs": args.runs,
                               "configs": results}, indent=2))
    print(f"→ {out}")
    print("\nGO = layout_on is stable everywhere with no recall/numacc loss vs layout_off.")


if __name__ == "__main__":
    main()
