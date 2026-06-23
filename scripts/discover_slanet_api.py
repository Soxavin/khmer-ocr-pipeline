"""Probe the SLANet (rapid_table) table-structure API used by the hybrid engine.

Run on a cropped table image to see the exact output schema we build against:
    uv run python scripts/discover_slanet_api.py <image.png> [x0 y0 x1 y1]

If a crop box is given, the image is cropped to it first (SLANet wants a table
region, not a full page). Prints cell count, the cell_bboxes / logic_points
shapes + samples, and the derived rows x cols grid. Structure only (use_ocr=False)
— text comes from Surya per-cell in the hybrid engine.
"""
from __future__ import annotations
import sys
import numpy as np
from PIL import Image
from rapid_table import RapidTable, RapidTableInput, ModelType


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    img = Image.open(sys.argv[1]).convert("RGB")
    if len(sys.argv) >= 6:
        box = tuple(int(v) for v in sys.argv[2:6])
        img = img.crop(box)
    arr = np.array(img)
    print("input crop shape:", arr.shape)

    eng = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False))
    out = eng(arr)

    cb = np.array(out.cell_bboxes)[0]   # (N, 8) quad coords, crop-relative
    lp = np.array(out.logic_points)[0]  # (N, 4) [row_start, row_end, col_start, col_end]
    print("output fields:", [a for a in dir(out) if not a.startswith("_")])
    print("n_cells:", lp.shape[0])
    print("cell_bboxes shape:", cb.shape, "sample:", cb[0].tolist())
    print("logic_points shape:", lp.shape, "sample(first 5):", lp[:5].tolist())
    print("grid:", int(lp[:, 1].max()) + 1, "rows x", int(lp[:, 3].max()) + 1, "cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
