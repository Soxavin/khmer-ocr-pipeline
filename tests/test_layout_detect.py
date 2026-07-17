from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import khmer_pipeline.engines.layout_detect as ld


def _img() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _engine(class_names, boxes):
    return lambda img: SimpleNamespace(class_names=class_names, boxes=boxes)


def test_detect_table_boxes_filters_to_table_class_only():
    names = ["text", "table", "figure"]
    boxes = [[0, 0, 10, 10], [5, 5, 50, 50], [60, 60, 90, 90]]
    with patch.object(ld, "_get_engine", return_value=_engine(names, boxes)):
        out = ld.detect_table_boxes(_img())
    assert out == [[5.0, 5.0, 50.0, 50.0]]


def test_detect_table_boxes_is_case_insensitive():
    names = ["Table"]
    boxes = [[1, 2, 3, 4]]
    with patch.object(ld, "_get_engine", return_value=_engine(names, boxes)):
        out = ld.detect_table_boxes(_img())
    assert out == [[1.0, 2.0, 3.0, 4.0]]


def test_detect_table_boxes_returns_all_table_regions():
    names = ["table", "text", "table"]
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
    with patch.object(ld, "_get_engine", return_value=_engine(names, boxes)):
        out = ld.detect_table_boxes(_img())
    assert out == [[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 50.0, 50.0]]


def test_detect_table_boxes_none_class_names_returns_empty():
    with patch.object(ld, "_get_engine", return_value=_engine(None, [[0, 0, 1, 1]])):
        out = ld.detect_table_boxes(_img())
    assert out == []


def test_detect_table_boxes_none_boxes_returns_empty():
    with patch.object(ld, "_get_engine", return_value=_engine(["table"], None)):
        out = ld.detect_table_boxes(_img())
    assert out == []


def test_detect_table_boxes_no_matches_returns_empty():
    with patch.object(ld, "_get_engine", return_value=_engine(["text", "figure"], [[0, 0, 1, 1], [2, 2, 3, 3]])):
        out = ld.detect_table_boxes(_img())
    assert out == []


def test_detect_table_boxes_returns_floats():
    names = ["table"]
    boxes = [[1, 2, 3, 4]]
    with patch.object(ld, "_get_engine", return_value=_engine(names, boxes)):
        out = ld.detect_table_boxes(_img())
    assert all(isinstance(v, float) for box in out for v in box)


# --- fine-tuned YOLO backend (KHMER_LAYOUT_WEIGHTS, Track A) ---

def test_missing_weights_file_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setenv("KHMER_LAYOUT_WEIGHTS", str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="no such weights file"):
        ld.detect_table_boxes(_img())


def test_weights_env_routes_to_yolo_backend(monkeypatch, tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"")  # the existence guard in detect_table_boxes runs before routing
    monkeypatch.setenv("KHMER_LAYOUT_WEIGHTS", str(pt))
    sentinel_model = object()
    with patch.object(ld, "_get_yolo", return_value=sentinel_model) as gy, \
         patch.object(ld, "_yolo_table_boxes", return_value=[[1.0, 2.0, 3.0, 4.0]]) as yb:
        out = ld.detect_table_boxes(_img())
    assert out == [[1.0, 2.0, 3.0, 4.0]]
    gy.assert_called_once_with(str(pt))
    yb.assert_called_once()


def test_no_env_keeps_stock_backend(monkeypatch):
    monkeypatch.delenv("KHMER_LAYOUT_WEIGHTS", raising=False)
    with patch.object(ld, "_get_engine", return_value=_engine(["table"], [[0, 0, 1, 1]])):
        assert ld.detect_table_boxes(_img()) == [[0.0, 0.0, 1.0, 1.0]]


# --- fine-tuned DocLayout-YOLO ONNX backend (Track A, §2.43) ---
# The fine-tune trains in an isolated venv (the doclayout_yolo fork can't live in the
# project venv — it needs opencv-python, which collides with opencv-python-headless),
# so the weights cross the boundary as ONNX and are served by rapid_layout.

def test_onnx_weights_route_to_onnx_backend_not_ultralytics(monkeypatch, tmp_path):
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"")
    monkeypatch.setenv("KHMER_LAYOUT_WEIGHTS", str(onnx))
    with patch.object(ld, "_get_onnx_engine",
                      return_value=_engine(["text", "Table"], [[0, 0, 1, 1], [5, 5, 9, 9]])) as go, \
         patch.object(ld, "_get_yolo") as gy:
        out = ld.detect_table_boxes(_img())
    assert out == [[5.0, 5.0, 9.0, 9.0]]
    go.assert_called_once_with(str(onnx))
    gy.assert_not_called()


def test_pt_weights_still_route_to_ultralytics(monkeypatch, tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"")
    monkeypatch.setenv("KHMER_LAYOUT_WEIGHTS", str(pt))
    with patch.object(ld, "_get_yolo", return_value=object()), \
         patch.object(ld, "_yolo_table_boxes", return_value=[[1.0, 2.0, 3.0, 4.0]]), \
         patch.object(ld, "_get_onnx_engine") as go:
        out = ld.detect_table_boxes(_img())
    assert out == [[1.0, 2.0, 3.0, 4.0]]
    go.assert_not_called()


def test_missing_onnx_weights_file_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setenv("KHMER_LAYOUT_WEIGHTS", str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError, match="no such weights file"):
        ld.detect_table_boxes(_img())


def test_onnx_backend_empty_result_returns_empty(monkeypatch, tmp_path):
    # An empty detection must return [] so callers can fall back to Surya's boxes
    # rather than silently dropping the table.
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"")
    monkeypatch.setenv("KHMER_LAYOUT_WEIGHTS", str(onnx))
    with patch.object(ld, "_get_onnx_engine", return_value=_engine(None, None)):
        assert ld.detect_table_boxes(_img()) == []


# --- model-type selection (licence: PP is Apache-2.0, DocLayout is AGPL-3.0) ---

def test_default_model_type_is_apache_licensed_pp(monkeypatch):
    monkeypatch.delenv("KHMER_LAYOUT_MODEL", raising=False)
    # The default must stay Apache-2.0: DocLayout-YOLO is an Ultralytics derivative
    # (AGPL-3.0) and this deliverable is served over a web UI to a government dept.
    assert ld._model_type() == "pp_doc_layoutv2"


def test_model_type_is_env_overridable(monkeypatch):
    monkeypatch.setenv("KHMER_LAYOUT_MODEL", "doclayout_docstructbench")
    assert ld._model_type() == "doclayout_docstructbench"
