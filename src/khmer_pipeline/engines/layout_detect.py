from __future__ import annotations
import os
import numpy as np

# Isolated page-layout wrapper with three backends, an alternative to Surya's
# fragmented layout boxes + geometric stitcher (table_stitch.py):
#  - default: off-the-shelf DocLayout-YOLO via rapid_layout (§2.23/24)
#  - KHMER_LAYOUT_WEIGHTS=<path/to/best.onnx>: our DocLayout-YOLO fine-tuned on
#    our documents (Track A, §2.43), served through rapid_layout — no new dependency.
#  - KHMER_LAYOUT_WEIGHTS=<path/to/best.pt>: a stock Ultralytics checkpoint (e.g. a
#    yolo11s fine-tune) — needs the `ultralytics` package.
# Kept separate so the detector can be swapped without touching the hybrid engine.
#
# Why the fine-tune arrives as ONNX and not .pt: it trains with the `doclayout_yolo`
# research fork, which pickles its own module paths into the checkpoint — stock
# ultralytics cannot unpickle it (ModuleNotFoundError: doclayout_yolo). Installing the
# fork here is not an option either: it requires `opencv-python`, which collides with
# this project's `opencv-python-headless` (both provide `cv2`). ONNX carries no Python
# class dependency, so the weights cross the venv boundary cleanly.

_MODEL_TYPE = "doclayout_docstructbench"
_WEIGHTS_ENV = "KHMER_LAYOUT_WEIGHTS"
_YOLO_TABLE_CLASS = "Table"  # class name in our fine-tune dataset (ardb_layout_coco_v1)
_YOLO_MIN_CONF = 0.25
_engine = None
_yolo_engine = None  # (weights_path, model)
_onnx_engine = None  # (weights_path, RapidLayout)


def _get_engine():
    global _engine
    if _engine is None:
        from rapid_layout import RapidLayout
        _engine = RapidLayout(model_type=_MODEL_TYPE)
    return _engine


def _get_yolo(weights: str):
    global _yolo_engine
    if _yolo_engine is None or _yolo_engine[0] != weights:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                f"{_WEIGHTS_ENV} is set but the `ultralytics` package is not installed; "
                "run `uv add 'ultralytics>=8.3,<9'` to use fine-tuned layout weights."
            ) from e
        _yolo_engine = (weights, YOLO(weights))
    return _yolo_engine[1]


def _get_onnx_engine(weights: str):
    global _onnx_engine
    if _onnx_engine is None or _onnx_engine[0] != weights:
        from rapid_layout import RapidLayout
        from rapid_layout.utils.typings import ModelType, RapidLayoutInput

        # model_type selects DocLayout's pre/post-processing; model_dir_or_path points
        # it at our weights instead of the bundled ones. rapid_layout reads the class
        # list from the ONNX's `character` metadata key — experiments/layout_yolo/
        # export_onnx.py injects it, since an exported model has no such key.
        _onnx_engine = (weights, RapidLayout(RapidLayoutInput(
            model_type=ModelType(_MODEL_TYPE),
            model_dir_or_path=weights,
            conf_thresh=_YOLO_MIN_CONF,
        )))
    return _onnx_engine[1]


def _table_boxes_from_result(res) -> list[list[float]]:
    """Filter a rapid_layout result down to Table-labelled boxes ([] when none)."""
    names = res.class_names if res.class_names is not None else []
    boxes = res.boxes if res.boxes is not None else []
    return [[float(v) for v in b] for b, c in zip(boxes, names)
            if c.lower() == _YOLO_TABLE_CLASS.lower()]


def _yolo_table_boxes(model, page_img_rgb: np.ndarray) -> list[list[float]]:
    res = model.predict(page_img_rgb, conf=_YOLO_MIN_CONF, verbose=False)[0]
    names = res.names  # class-id → name
    return [[float(v) for v in box.xyxy[0].tolist()]
            for box in res.boxes if names[int(box.cls)] == _YOLO_TABLE_CLASS]


def detect_table_boxes(page_img_rgb: np.ndarray) -> list[list[float]]:
    """Return axis-aligned [x0, y0, x1, y1] per detected table region (page-pixel
    coords, [] when none). Uses our fine-tuned weights when KHMER_LAYOUT_WEIGHTS is
    set (`.onnx` → rapid_layout, else Ultralytics), otherwise the stock backend."""
    weights = os.environ.get(_WEIGHTS_ENV)
    if weights:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"{_WEIGHTS_ENV}={weights!r}: no such weights file")
        if weights.endswith(".onnx"):
            return _table_boxes_from_result(_get_onnx_engine(weights)(page_img_rgb))
        return _yolo_table_boxes(_get_yolo(weights), page_img_rgb)
    return _table_boxes_from_result(_get_engine()(page_img_rgb))
