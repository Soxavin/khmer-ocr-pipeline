from __future__ import annotations


def test_run_imports_without_error():
    from khmer_pipeline.pipeline import run
    assert callable(run)


def test_run_signature_has_all_parameters():
    import inspect
    from khmer_pipeline.pipeline import run
    params = inspect.signature(run).parameters
    for p in ["source_path", "output_dir", "dpi", "remove_stamps",
              "sharpen", "normalise", "skip_qwen", "convert_numerals"]:
        assert p in params, f"Missing parameter: {p}"


def test_run_passes_provenance_to_export(tmp_path, monkeypatch):
    """C6: run() must build a provenance block and pass it to export()."""
    import numpy as np
    import khmer_pipeline.pipeline as pl
    from khmer_pipeline.models import (
        IngestResult, PreprocessResult, SuryaResult, SuryaPageResult,
        PostprocessResult, CorrectedPageResult, ExportResult,
    )

    captured: dict = {}

    def fake_ingest(data, name, dpi=200):
        return IngestResult(source_name=name,
                            page_images=[np.zeros((4, 4, 3), dtype=np.uint8)],
                            dpi=dpi, page_count=1)

    def fake_preprocess(ing, cfg):
        return PreprocessResult(source_name=ing.source_name, page_images=ing.page_images,
                                dpi=ing.dpi, page_count=ing.page_count)

    def fake_ocr(pre):
        return SuryaResult(source_name=pre.source_name,
                           pages=[SuryaPageResult(0, [], [], "")])
    fake_ocr.__name__ = "run_fake_engine"

    def fake_correction(surya, skip_qwen=True, anomaly_threshold=0.15):
        return PostprocessResult(source_name=surya.source_name,
                                 pages=[CorrectedPageResult(0, [], [], "", "", "", False)])

    def fake_export(result, convert_numerals=False, repair_tables=False,
                    stitch_pages=False, provenance=None):
        captured["provenance"] = provenance
        return ExportResult(source_name=result.source_name,
                            document_json={"pages": []}, tables_csv=[])

    monkeypatch.setattr(pl, "ingest", fake_ingest)
    monkeypatch.setattr(pl, "preprocess", fake_preprocess)
    monkeypatch.setattr(pl, "ACTIVE_OCR_ENGINE", fake_ocr)
    monkeypatch.setattr(pl, "ACTIVE_CORRECTION_ENGINE", fake_correction)
    monkeypatch.setattr(pl, "export", fake_export)
    monkeypatch.setattr(pl, "clear_device_cache", lambda: None)

    src = tmp_path / "in.pdf"
    src.write_bytes(b"dummy")
    pl.run(src, tmp_path / "out", dpi=222, stitch_pages=False,
           convert_numerals=True, repair_tables=True, deskew=False)

    prov = captured["provenance"]
    assert prov is not None
    assert prov["engine"] == "run_fake_engine"
    assert prov["dpi"] == 222
    assert prov["stitch_pages"] is False
    assert prov["convert_numerals"] is True
    assert prov["repair_tables"] is True
    assert prov["preprocess"]["deskew"] is False
    assert set(prov["preprocess"]) == {
        "remove_stamps", "sharpen", "normalise", "deskew", "normalise_table_backgrounds",
    }
    assert "surya_ocr_version" in prov
