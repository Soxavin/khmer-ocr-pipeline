"""Quantify what stamp removal actually erases on a real page.

The colour mask in `preprocess._stamp_ink_mask` thresholds red+blue over the whole
page, so coloured *body text* is masked alongside stamps. This probe measures that
directly — deterministic and instant, unlike an OCR ablation which is confounded by
Surya's run-to-run variance.

Reports masked-pixel counts before and after the shape gate, and writes a
side-by-side PNG (original · mask · result) for visual confirmation.

    uv run python scripts/probe_stamp_mask.py <page.png> [--out probe.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from khmer_pipeline.preprocess import _remove_stamps, _stamp_ink_mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("stamp_probe.png"))
    args = ap.parse_args()

    rgb = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if rgb is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    bgr = rgb  # imread already gives BGR, which is what the preprocess helpers expect

    h, w = bgr.shape[:2]
    total = h * w
    mask = _stamp_ink_mask(bgr)
    masked = int(cv2.countNonZero(mask))

    result = _remove_stamps(bgr)
    changed = int(np.count_nonzero(np.any(bgr != result, axis=2)))

    print(f"page            : {args.image.name}  ({w}x{h} = {total:,} px)")
    print(f"colour-masked   : {masked:,} px  ({masked / total * 100:.2f}% of page)")
    print(f"pixels CHANGED  : {changed:,} px  ({changed / total * 100:.2f}% of page)")
    if masked:
        print(f"dilation blowup : {changed / masked:.2f}x the colour mask")

    panel = np.hstack([bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
    cv2.imwrite(str(args.out), panel)
    print(f"wrote           : {args.out}  (original | mask | result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
