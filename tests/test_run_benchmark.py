from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from khmer_pipeline.run_benchmark import (
    _default_output,
    _done_keys,
    _engine_name,
    _raw_preprocess_result,
)
from khmer_pipeline.models import IngestResult, PreprocessResult


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


def test_default_output():
    result = _default_output("run_surya", datetime(2026, 6, 19, 13, 5, 9))
    assert result == Path("benchmark_results_run_surya_20260619_130509.csv")


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


def test_argparse_data_dir_default():
    import argparse
    # reconstruct the parser the same way __main__ does
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", nargs="+", default=[Path("./synthetic_data"), Path("./synthetic_documents")], type=Path)
    parser.add_argument("--output-csv", default=None, type=Path)
    parser.add_argument("--with-correction", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args([])
    assert isinstance(args.data_dir, list)
    assert all(isinstance(d, Path) for d in args.data_dir)


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

    output_csv = tmp_path / "out.csv"

    from khmer_pipeline.run_benchmark import run_benchmark
    run_benchmark([data_dir], output_csv=output_csv)

    assert output_csv.exists()
    with output_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field_names = reader.fieldnames
        data_rows = list(reader)

    # Engine must be the first column
    assert field_names[0] == "Engine"
    # one data row for the one image
    assert len(data_rows) == 1
    assert data_rows[0]["Engine"] == "run_fake"
    assert data_rows[0]["Dataset"] == "synthetic_data"
