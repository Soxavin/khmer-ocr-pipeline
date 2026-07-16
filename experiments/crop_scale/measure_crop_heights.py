"""Measure production cell-crop heights vs Kiri's fixed IMG_H, per GT page.

Tests the "resolution is throttled" hypothesis on the REAL production path
(ingest → preprocess → surya_kiri → per-cell crops), not the harvest path.
Kiri scales crops by HEIGHT to CFG.IMG_H (kiri_vendor/model.py); crops taller
than IMG_H are downscaled (information discarded), shorter ones are upsampled
(no new information). The hypothesis needs crops < IMG_H to have any upside.

Instruments by monkeypatching surya_kiri_engine.recognize_cells_conf to record
crop.shape[0] before delegating — no source change.

Usage (repo root):
    uv run python experiments/crop_scale/measure_crop_heights.py [--cap 2048]
"""

from __future__ import annotations

import os

os.environ.setdefault("OCR_ENGINE", "surya_kiri")

import argparse
import json
import statistics as st
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from khmer_pipeline.ingest import ingest
from khmer_pipeline import preprocess as _pre_mod
from khmer_pipeline.preprocess import preprocess
from khmer_pipeline.engines import surya_kiri_engine
from khmer_pipeline.engines.engine_registry import ACTIVE_OCR_ENGINE
from khmer_pipeline.engines.kiri_vendor.model import CFG

_REAL = _REPO / "eval/datasets/real"
_IMG_H = CFG().IMG_H


def _summarize(heights: list[int]) -> dict:
    """Percentiles + the share of crops Kiri must upsample (h < IMG_H)."""
    hs = sorted(heights)
    n = len(hs)
    under = sum(1 for h in hs if h < _IMG_H)
    return {
        "n_cells": n,
        "min": hs[0],
        "p10": hs[n // 10],
        "median": round(st.median(hs), 1),
        "p90": hs[9 * n // 10],
        "max": hs[-1],
        "pct_below_img_h": round(100 * under / n, 1),
        "median_scale_into_kiri": round(_IMG_H / st.median(hs), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure production cell-crop heights.")
    ap.add_argument("--cap", type=int, default=None,
                    help="override _CAP_RESOLUTION_MAX_DIM for this run")
    ap.add_argument("--tag", default=None, help="output label (default: cap value)")
    args = ap.parse_args()

    cap = args.cap or _pre_mod._CAP_RESOLUTION_MAX_DIM
    if args.cap:
        # _cap_resolution binds _CAP_RESOLUTION_MAX_DIM as a default arg at def
        # time, so rebinding the module global is a no-op — wrap the function.
        _orig_cap = _pre_mod._cap_resolution
        _pre_mod._cap_resolution = lambda bgr, max_dim=args.cap: _orig_cap(bgr, max_dim)
    tag = args.tag or f"cap{cap}"

    captured: list[int] = []
    _orig = surya_kiri_engine.recognize_cells_conf

    def _spy(crops, warning_sink=None):
        captured.extend(int(c.shape[0]) for c in crops)
        return _orig(crops, warning_sink=warning_sink)

    surya_kiri_engine.recognize_cells_conf = _spy

    results: dict[str, dict] = {}
    for png in sorted(_REAL.glob("*.png")):
        captured.clear()
        ing = ingest(png.read_bytes(), png.name)
        pre = preprocess(ing)
        page_px = f"{pre.page_images[0].shape[1]}x{pre.page_images[0].shape[0]}"
        ACTIVE_OCR_ENGINE(pre)
        if not captured:
            print(f"{png.stem[:34]:36s} no cells recognized")
            continue
        summary = _summarize(list(captured))
        summary["page_px_after_preprocess"] = page_px
        results[png.stem] = summary
        print(f"{png.stem[:34]:36s} page={page_px:>11s} cells={summary['n_cells']:4d} "
              f"h: p10={summary['p10']:3d} med={summary['median']:5.1f} p90={summary['p90']:3d} "
              f"| <IMG_H({_IMG_H}): {summary['pct_below_img_h']:5.1f}% "
              f"| scale={summary['median_scale_into_kiri']:.2f}x")

    out = _REPO / f"experiments/crop_scale/heights_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cap": cap, "img_h": _IMG_H, "pages": results}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
