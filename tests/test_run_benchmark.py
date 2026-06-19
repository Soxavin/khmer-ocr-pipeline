from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from khmer_pipeline.run_benchmark import (
    _default_run_dir,
    _done_keys,
    _engine_name,
    _git_commit,
    _raw_preprocess_result,
    _tool_versions,
    _write_manifest,
    _DEFAULT_DATASETS,
)
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.analyze_benchmark import summarize


def test_engine_name(monkeypatch):
    def fake_ocr(pre):
        pass
    fake_ocr.__name__ = "run_fake"
    monkeypatch.setattr("khmer_pipeline.run_benchmark.ACTIVE_OCR_ENGINE", fake_ocr)
    assert _engine_name() == "run_fake"


def test_engine_name_fallback(monkeypatch):
    # object with no __name__ falls back to "ocr"
    class NoName:
        pass
    monkeypatch.setattr("khmer_pipeline.run_benchmark.ACTIVE_OCR_ENGINE", NoName())
    assert _engine_name() == "ocr"


def test_default_run_dir():
    result = _default_run_dir("run_surya", datetime(2026, 6, 19, 13, 5, 9))
    assert result == Path("eval/runs/20260619_130509_run_surya")


def test_done_keys_missing_file(tmp_path):
    assert _done_keys(tmp_path / "nonexistent.csv") == set()


def test_done_keys_parses_pairs(tmp_path):
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Engine", "Corrected", "Dataset", "Image_File", "Font"])
        writer.writeheader()
        writer.writerow({"Engine": "run_surya", "Corrected": "False", "Dataset": "synthetic_data", "Image_File": "table_0_Battambang.png", "Font": "Battambang"})
        writer.writerow({"Engine": "run_surya", "Corrected": "False", "Dataset": "synthetic_documents", "Image_File": "doc_1_Hanuman.png", "Font": "Hanuman"})

    keys = _done_keys(csv_path)
    assert ("synthetic_data", "table_0_Battambang.png") in keys
    assert ("synthetic_documents", "doc_1_Hanuman.png") in keys
    assert len(keys) == 2


def test_raw_preprocess_result_preserves_images():
    imgs = [np.zeros((10, 10, 3), dtype=np.uint8)]
    ingest_result = IngestResult(
        source_name="test.png",
        page_images=imgs,
        dpi=200,
        page_count=1,
    )
    pre = _raw_preprocess_result(ingest_result)
    assert isinstance(pre, PreprocessResult)
    assert pre.source_name == "test.png"
    assert pre.page_images is imgs  # same object — no copy
    assert pre.dpi == 200
    assert pre.page_count == 1


def test_git_commit_returns_types():
    sha, dirty = _git_commit()
    assert isinstance(sha, str)
    assert isinstance(dirty, bool)


def test_git_commit_never_raises():
    # should never raise regardless of environment
    result = _git_commit()
    assert len(result) == 2


def test_tool_versions_has_keys():
    v = _tool_versions()
    assert isinstance(v, dict)
    assert "surya_ocr" in v
    assert "python" in v


def test_tool_versions_never_raises():
    v = _tool_versions()
    assert v is not None


def test_write_manifest(tmp_path):
    run_dir = tmp_path / "20260619_130509_run_test"
    run_dir.mkdir()
    dataset_counts = [
        ("synthetic_tables", Path("eval/datasets/synthetic_tables"), 5),
        ("synthetic_documents", Path("eval/datasets/synthetic_documents"), 5),
    ]
    aggregates = {
        "avg_cell_accuracy": 0.643,
        "avg_cell_content_recall": 0.78,
        "avg_table_cer": 0.147,
        "avg_text_cer": 0.219,
    }
    _write_manifest(run_dir, "run_test", False, dataset_counts, aggregates)

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "run_id" in manifest
    assert manifest["engine"] == "run_test"
    assert "correction" in manifest
    assert "git_commit" in manifest
    assert "versions" in manifest
    assert "datasets" in manifest
    assert len(manifest["datasets"]) == 2
    assert "image_count" in manifest
    assert manifest["image_count"] == 10
    assert "aggregates" in manifest


def test_argparse_data_dir_default():
    import argparse
    # reconstruct the parser the same way __main__ does
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", nargs="+", default=_DEFAULT_DATASETS, type=Path)
    parser.add_argument("--run-dir", default=None, type=Path)
    parser.add_argument("--with-correction", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args([])
    assert args.data_dir == _DEFAULT_DATASETS
    assert args.data_dir[0] == Path("eval/datasets/synthetic_tables")
    assert args.data_dir[1] == Path("eval/datasets/synthetic_documents")


def test_end_to_end_run_benchmark(tmp_path, monkeypatch):
    # set up a synthetic_data dir with one GT JSON + dummy PNG
    data_dir = tmp_path / "synthetic_data"
    data_dir.mkdir()

    # minimal ground truth (isolated table schema)
    gt = {
        "font_family": "Battambang",
        "template": "test_template",
        "data": [["header"], ["row1"]],
    }
    gt_path = data_dir / "table_0_Battambang_ground_truth.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")

    # dummy 1x1 white PNG
    img_path = data_dir / "table_0_Battambang.png"
    import struct, zlib
    def _minimal_png():
        sig = b'\x89PNG\r\n\x1a\n'
        def chunk(t, d):
            return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
        ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
        raw = b'\x00\xff\xff\xff'
        idat = chunk(b'IDAT', zlib.compress(raw))
        iend = chunk(b'IEND', b'')
        return sig + ihdr + idat + iend
    img_path.write_bytes(_minimal_png())

    # canned return values
    canned_ocr = type("FakeOCR", (), {})()
    fake_page = type("FakePage", (), {
        "tables": [],
        "ocr_text": "hello",
    })()
    canned_ocr.pages = [fake_page]

    canned_table_metrics = {
        "tables_found": 0,
        "gt_rows": 2, "gt_cols": 1,
        "pred_rows": 0, "pred_cols": 0,
        "cell_accuracy": 0.0,
        "cell_content_recall": 0.0,
        "table_cer": 1.0,
    }
    canned_text_metrics = {
        "text_cer": 0.5,
        "paragraph_recall": 0.0,
        "paragraph_leak": 0,
    }

    def fake_ingest(img_bytes, name, dpi):
        imgs = [np.zeros((10, 10, 3), dtype=np.uint8)]
        return IngestResult(source_name=name, page_images=imgs, dpi=dpi, page_count=1)

    def fake_ocr_engine(pre):
        return canned_ocr

    fake_ocr_engine.__name__ = "run_fake"

    monkeypatch.setattr("khmer_pipeline.run_benchmark.ingest", fake_ingest)
    monkeypatch.setattr("khmer_pipeline.run_benchmark.ACTIVE_OCR_ENGINE", fake_ocr_engine)
    monkeypatch.setattr("khmer_pipeline.run_benchmark.evaluate_table", lambda pred, gt_grid: canned_table_metrics)
    monkeypatch.setattr("khmer_pipeline.run_benchmark.evaluate_text", lambda text, pred, gt: canned_text_metrics)
    monkeypatch.setattr("khmer_pipeline.run_benchmark.clear_device_cache", lambda: None)

    run_dir = tmp_path / "run"

    from khmer_pipeline.run_benchmark import run_benchmark
    run_benchmark([data_dir], run_dir=run_dir)

    # run dir must exist and contain the three artifacts
    assert run_dir.exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.txt").exists()

    with (run_dir / "results.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field_names = reader.fieldnames
        data_rows = list(reader)

    # Engine must be the first column
    assert field_names[0] == "Engine"
    # one data row for the one image
    assert len(data_rows) == 1
    assert data_rows[0]["Engine"] == "run_fake"
    assert data_rows[0]["Dataset"] == "synthetic_data"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_count"] == 1


def test_summarize_empty():
    assert summarize([]) == "No results."


def test_summarize_with_rows():
    rows = [
        {
            "Engine": "run_fake", "Corrected": "False",
            "Dataset": "synthetic_tables", "Image_File": "table_0.png",
            "Font": "Battambang", "Template": "tmpl_a",
            "Tables_Expected": "1", "Tables_Found": "1",
            "GT_Rows": "2", "GT_Cols": "2", "Pred_Rows": "2", "Pred_Cols": "2",
            "Cell_Accuracy": "0.800", "Cell_Content_Recall": "0.900",
            "Table_CER": "0.100", "Text_CER": "0.150",
            "Paragraph_Recall": "1.000", "Paragraph_Leak": "0",
            "Error": "",
        },
        {
            "Engine": "run_fake", "Corrected": "False",
            "Dataset": "synthetic_tables", "Image_File": "table_1.png",
            "Font": "Hanuman", "Template": "tmpl_b",
            "Tables_Expected": "1", "Tables_Found": "1",
            "GT_Rows": "3", "GT_Cols": "2", "Pred_Rows": "3", "Pred_Cols": "2",
            "Cell_Accuracy": "0.600", "Cell_Content_Recall": "0.700",
            "Table_CER": "0.200", "Text_CER": "0.250",
            "Paragraph_Recall": "0.800", "Paragraph_Leak": "0",
            "Error": "",
        },
    ]
    result = summarize(rows)
    assert isinstance(result, str)
    assert "Per-Engine" in result
    assert "Per-Font" in result
    assert "Per-Template" in result
    assert "Per-Dataset" in result
    assert "Best & Worst by Cell_Accuracy" in result
    assert "Lowest Table_CER" in result
