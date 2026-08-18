"""Cap A/B gate: score surya_kiri on every verified GT page at a given resolution cap.

Follows the §2.39 gate_ab.py pattern (ingest → preprocess → surya_kiri → evaluate_table).
The only variable is _CAP_RESOLUTION_MAX_DIM. ARDB pages (2000x2000 native, under
both caps) are an unaffected CONTROL — any movement there means the change leaked.

Usage (repo root):
    uv run python experiments/crop_scale/gate_cap.py --cap 2048 --tag cap2048
    uv run python experiments/crop_scale/gate_cap.py --cap 2900 --tag cap2900
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
from khmer_pipeline import preprocess as _pre_mod
from khmer_pipeline.preprocess import preprocess
from khmer_pipeline.engines.engine_registry import ACTIVE_OCR_ENGINE
from khmer_pipeline.evaluation.evaluate_structure import evaluate_table

_REAL = _REPO / "eval/datasets/real"
_METRICS = ("cell_content_recall", "numeric_cell_accuracy", "table_cer",
            "empty_cell_precision", "cell_accuracy")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score surya_kiri at a given resolution cap.")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    # _cap_resolution binds the threshold as a default arg at def time, so rebinding
    # the module global is a no-op — wrap the function instead.
    _orig_cap = _pre_mod._cap_resolution
    _pre_mod._cap_resolution = lambda bgr, max_dim=args.cap: _orig_cap(bgr, max_dim)

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

    out = _REPO / f"experiments/crop_scale/gate_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cap": args.cap, "pages": results}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
