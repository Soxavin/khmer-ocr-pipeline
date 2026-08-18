"""Recognition A/B: per-page recognition CER (single-source GT) across engines.

Isolates RECOGNITION from layout: pools all text on each side and scores CER
against a single-source GT (no paragraph/table double-count — see
evaluate_recognition). Goal: baseline + where Surya fails.

Local engines (swap via OCR_ENGINE):
    OCR_ENGINE=surya     uv run python scripts/eval_recognizers.py
    OCR_ENGINE=tesseract uv run python scripts/eval_recognizers.py

External model (predictions produced on Colab, scored locally on the SAME basis):
    uv run python scripts/eval_recognizers.py --predictions preds.json --name qwen2.5-vl-7b

predictions.json shape: {"<image_file.png>": "<recognized text>", ...}

Scores every eval/datasets/real/*_ground_truth.json that has a matching .png and
writes eval/runs/<ts>_recog_<engine>/recognition.csv.
"""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from khmer_pipeline.ingest import ingest
from khmer_pipeline.models import PreprocessResult
from khmer_pipeline.engines.engine_registry import ACTIVE_OCR_ENGINE
from khmer_pipeline.evaluation.evaluate_structure import (
    evaluate_recognition,
    pool_gt_recognition_text,
    pool_pred_text,
)
from khmer_pipeline.utils.memory import clear_device_cache

_REAL = Path("eval/datasets/real")
_RUNS = Path("eval/runs")
_FIELDS = ["Engine", "Image_File", "GT_Chars", "Pred_Chars", "Recognition_CER"]


def _gt_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for gt_path in sorted(_REAL.glob("*_ground_truth.json")):
        png = gt_path.with_name(gt_path.name.replace("_ground_truth.json", ".png"))
        if png.exists():
            pairs.append((png, gt_path))
    return pairs


def _row(engine: str, name: str, gt: dict, pred_text: str, cerv: float) -> dict:
    return {
        "Engine": engine,
        "Image_File": name,
        "GT_Chars": len(pool_gt_recognition_text(gt)),
        "Pred_Chars": len(pred_text),
        "Recognition_CER": cerv,
    }


def _score_local() -> tuple[str, list[dict]]:
    engine = getattr(ACTIVE_OCR_ENGINE, "__name__", "ocr")
    rows = []
    for png, gt_path in _gt_pairs():
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ing = ingest(png.read_bytes(), png.name, dpi=200)
        pre = PreprocessResult(source_name=ing.source_name, page_images=ing.page_images,
                               dpi=ing.dpi, page_count=ing.page_count)
        res = ACTIVE_OCR_ENGINE(pre)
        ocr_text = "\n".join(p.ocr_text for p in res.pages)
        pred_tables = [t for p in res.pages for t in p.tables]
        cerv = evaluate_recognition(ocr_text, pred_tables, gt)["recognition_cer"]
        rows.append(_row(engine, png.name, gt, pool_pred_text(ocr_text, pred_tables), cerv))
        print(f"[{engine}] {png.name}: CER={cerv:.3f}")
        clear_device_cache()
    return engine, rows


def _score_external(pred_path: Path, name: str | None) -> tuple[str, list[dict]]:
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    engine = name or pred_path.stem
    rows = []
    for png, gt_path in _gt_pairs():
        if png.name not in preds:
            print(f"[{engine}] {png.name}: (no prediction — skipped)")
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        text = preds[png.name]
        cerv = evaluate_recognition(text, [], gt)["recognition_cer"]
        rows.append(_row(engine, png.name, gt, text, cerv))
        print(f"[{engine}] {png.name}: CER={cerv:.3f}")
    return engine, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognition A/B (per-page recognition CER).")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="External model predictions JSON (skips local engine).")
    parser.add_argument("--name", default=None, help="Engine label for external predictions.")
    args = parser.parse_args()

    if not _gt_pairs():
        print(f"No *_ground_truth.json + .png pairs in {_REAL}")
        return 1

    if args.predictions:
        engine, rows = _score_external(args.predictions, args.name)
    else:
        engine, rows = _score_local()

    if not rows:
        print("No pages scored.")
        return 1

    mean = sum(r["Recognition_CER"] for r in rows) / len(rows)
    print(f"\n{engine}: mean Recognition_CER over {len(rows)} pages = {mean:.3f}")

    run_dir = _RUNS / f"{datetime.now():%Y%m%d_%H%M%S}_recog_{engine}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "recognition.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "Recognition_CER": f"{r['Recognition_CER']:.3f}"})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
