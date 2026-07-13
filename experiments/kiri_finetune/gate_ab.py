"""Track B end-to-end gate: score surya_kiri on every verified GT page (one config/run).

Runs the production path (ingest → preprocess → surya_kiri) on each
eval/datasets/real/*_p*_ground_truth.json page that has a table grid + PNG, and
scores with evaluate_table. The Kiri weights are whatever KHMER_KIRI_WEIGHTS
says (unset = stock pinned checkpoint), so run once per config:

    uv run python experiments/kiri_finetune/gate_ab.py --tag baseline
    KHMER_KIRI_WEIGHTS=experiments/kiri_finetune/run1 \
        uv run python experiments/kiri_finetune/gate_ab.py --tag finetuned

Writes experiments/kiri_finetune/gate_<tag>.json; compare with gate_compare.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("OCR_ENGINE", "surya_kiri")

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from khmer_pipeline.ingest import ingest
from khmer_pipeline.preprocess import preprocess
from khmer_pipeline.engines.engine_registry import ACTIVE_OCR_ENGINE
from khmer_pipeline.evaluation.evaluate_structure import evaluate_table

_REAL = _REPO / "eval/datasets/real"
_METRICS = ("cell_content_recall", "numeric_cell_accuracy", "table_cer",
            "empty_cell_precision", "cell_accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score surya_kiri vs all GT pages.")
    parser.add_argument("--tag", required=True, help="output label (baseline/finetuned)")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for gt_path in sorted(_REAL.glob("*_ground_truth.json")):
        gt = json.loads(gt_path.read_text())
        grids = [t.get("data") for t in gt.get("tables", []) if t.get("data")]
        png = gt_path.with_name(gt_path.name.replace("_ground_truth.json", ".png"))
        if not grids or not png.exists():
            continue
        page_id = png.stem[-20:]
        ing = ingest(png.read_bytes(), png.name)
        pre = preprocess(ing)
        engine_result = ACTIVE_OCR_ENGINE(pre)
        m = evaluate_table(engine_result.pages[0].tables, grids[0])
        results[page_id] = {k: m.get(k) for k in _METRICS}
        results[page_id]["pred_dims"] = f"{m['pred_rows']}x{m['pred_cols']}"
        print(f"{page_id}: " + " ".join(
            f"{k}={m.get(k):.3f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
            for k in _METRICS))

    out = _REPO / f"experiments/kiri_finetune/gate_{args.tag}.json"
    out.write_text(json.dumps({"weights": os.environ.get("KHMER_KIRI_WEIGHTS", "stock"),
                               "pages": results}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
