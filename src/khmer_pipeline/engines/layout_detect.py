from __future__ import annotations
import numpy as np

# Isolated DocLayout-YOLO (rapid_layout) page-layout wrapper. Detects the dense
# table as a single clean box, an alternative to Surya's fragmented layout boxes
# + geometric stitcher (table_stitch.py). Kept separate so the detector can be
# swapped without touching the hybrid engine.

_MODEL_TYPE = "doclayout_docstructbench"
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapid_layout import RapidLayout
        _engine = RapidLayout(model_type=_MODEL_TYPE)
    return _engine


def detect_table_boxes(page_img_rgb: np.ndarray) -> list[list[float]]:
    # Returns axis-aligned [x0, y0, x1, y1] (floats) per detected table region,
    # coords in the page image's pixel space. [] when nothing matches.
    res = _get_engine()(page_img_rgb)
    names = res.class_names if res.class_names is not None else []
    boxes = res.boxes if res.boxes is not None else []
    return [[float(v) for v in b] for b, c in zip(boxes, names) if c.lower() == "table"]
