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
