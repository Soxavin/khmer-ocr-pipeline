from __future__ import annotations
import numpy as np

# Isolated SLANet (rapid_table) table-structure wrapper. Returns a unified cell
# grid with coordinates for a cropped table image — the structure the hybrid
# engine needs (Surya reads the text per cell). Kept separate so the structure
# model can be swapped without touching the engine. use_ocr=False: structure only.

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapid_table import RapidTable, RapidTableInput, ModelType
        _engine = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False))
    return _engine


def _quad_to_bbox(q) -> list[float]:
    # q: 8 numbers (4 corner points) → axis-aligned [x0, y0, x1, y1]
    xs = q[0::2]
    ys = q[1::2]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def predict_cells(table_img_rgb: np.ndarray) -> list[dict]:
    # Returns one dict per detected cell, coords RELATIVE TO THE INPUT CROP:
    #   {row_id, col_id, row_span, col_span, bbox: [x0, y0, x1, y1]}
    out = _get_engine()(table_img_rgb)
    if not out.cell_bboxes or not out.logic_points:
        return []
    cell_bboxes = np.array(out.cell_bboxes[0])    # (N, 8)
    logic_points = np.array(out.logic_points[0])  # (N, 4): row_start,row_end,col_start,col_end
    cells: list[dict] = []
    for quad, (r0, r1, c0, c1) in zip(cell_bboxes, logic_points):
        cells.append({
            "row_id": int(r0),
            "col_id": int(c0),
            "row_span": int(r1) - int(r0) + 1,
            "col_span": int(c1) - int(c0) + 1,
            "bbox": _quad_to_bbox(quad),
        })
    return cells
