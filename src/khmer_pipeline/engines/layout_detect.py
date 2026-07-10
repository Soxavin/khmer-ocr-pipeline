from __future__ import annotations
import os
import numpy as np

# Isolated page-layout wrapper with two backends, an alternative to Surya's
# fragmented layout boxes + geometric stitcher (table_stitch.py):
#  - default: off-the-shelf DocLayout-YOLO via rapid_layout (§2.23/24)
#  - KHMER_LAYOUT_WEIGHTS=<path/to/best.pt>: an Ultralytics YOLO checkpoint
#    fine-tuned on our documents (Track A) — needs the `ultralytics` package.
# Kept separate so the detector can be swapped without touching the hybrid engine.

_MODEL_TYPE = "doclayout_docstructbench"
_WEIGHTS_ENV = "KHMER_LAYOUT_WEIGHTS"
_YOLO_TABLE_CLASS = "Table"  # class name in our fine-tune dataset (ardb_layout_coco_v1)
_YOLO_MIN_CONF = 0.25
_engine = None
_yolo_engine = None  # (weights_path, model)


def _get_engine():
    global _engine
    if _engine is None:
        from rapid_layout import RapidLayout
        _engine = RapidLayout(model_type=_MODEL_TYPE)
    return _engine


def _get_yolo(weights: str):
    global _yolo_engine
    if _yolo_engine is None or _yolo_engine[0] != weights:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"{_WEIGHTS_ENV}={weights!r}: no such weights file")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                f"{_WEIGHTS_ENV} is set but the `ultralytics` package is not installed; "
                "run `uv add 'ultralytics>=8.3,<9'` to use fine-tuned layout weights."
            ) from e
        _yolo_engine = (weights, YOLO(weights))
    return _yolo_engine[1]


def _yolo_table_boxes(model, page_img_rgb: np.ndarray) -> list[list[float]]:
    res = model.predict(page_img_rgb, conf=_YOLO_MIN_CONF, verbose=False)[0]
    names = res.names  # class-id → name
    return [[float(v) for v in box.xyxy[0].tolist()]
            for box in res.boxes if names[int(box.cls)] == _YOLO_TABLE_CLASS]


def detect_table_boxes(page_img_rgb: np.ndarray) -> list[list[float]]:
    """Return axis-aligned [x0, y0, x1, y1] per detected table region (page-pixel
    coords, [] when none). Uses the fine-tuned YOLO checkpoint when
    KHMER_LAYOUT_WEIGHTS is set, else the stock DocLayout-YOLO backend."""
    weights = os.environ.get(_WEIGHTS_ENV)
    if weights:
        return _yolo_table_boxes(_get_yolo(weights), page_img_rgb)
    res = _get_engine()(page_img_rgb)
    names = res.class_names if res.class_names is not None else []
    boxes = res.boxes if res.boxes is not None else []
    return [[float(v) for v in b] for b, c in zip(boxes, names) if c.lower() == "table"]
