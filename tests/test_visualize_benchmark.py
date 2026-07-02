from __future__ import annotations
from pathlib import Path
import csv
import pytest
from khmer_pipeline.evaluation.visualize_benchmark import (
    _coerce_float,
    _coerce_int,
    _mean,
    _group_by,
    _partition,
    _aggregate_metric,
    _aggregate_partitioned,
    _parse_corrected,
    _has_two_distinct,
    _plan_charts,
    _load_rows,
    visualize,
)


# --- _coerce_float ---

def test_coerce_float_valid():
    assert _coerce_float("0.5") == 0.5
    assert _coerce_float("42") == 42.0
    assert _coerce_float(3.14) == 3.14


def test_coerce_float_empty():
    assert _coerce_float("") is None
    assert _coerce_float(None) is None


def test_coerce_float_garbage():
    assert _coerce_float("not a number") is None
    assert _coerce_float([]) is None
    assert _coerce_float({}) is None


# --- _coerce_int ---

def test_coerce_int_valid():
    assert _coerce_int("7") == 7
    assert _coerce_int("1") == 1
    # tolerate '1.0' from float-formatted CSV
    assert _coerce_int("1.0") == 1


def test_coerce_int_empty():
    assert _coerce_int("") is None
    assert _coerce_int(None) is None


# --- _mean ---

def test_mean_empty_returns_none():
    assert _mean([]) is None


def test_mean_basic():
    assert _mean([1.0, 2.0, 3.0]) == 2.0
    assert _mean([0.5]) == 0.5


def test_mean_skips_none():
    assert _mean([1.0, None, 3.0]) == 2.0
    assert _mean([None, None]) is None


# --- _group_by ---

def test_group_by_preserves_first_seen_order():
    rows = [
        {"Dataset": "b", "v": 1},
        {"Dataset": "a", "v": 2},
        {"Dataset": "b", "v": 3},
        {"Dataset": "c", "v": 4},
    ]
    out = _group_by(rows, "Dataset")
    assert list(out.keys()) == ["b", "a", "c"]
    assert len(out["b"]) == 2
    assert len(out["a"]) == 1


# --- _partition ---

def test_partition_basic():
    rows = [
        {"Engine": "run_surya"},
        {"Engine": "run_tesseract"},
        {"Engine": "run_surya"},
    ]
    out = _partition(rows, "Engine")
    assert set(out.keys()) == {"run_surya", "run_tesseract"}
    assert len(out["run_surya"]) == 2


def test_partition_preserves_first_seen_order():
    rows = [
        {"Corrected": "True", "v": 1},
        {"Corrected": "False", "v": 2},
        {"Corrected": "True", "v": 3},
    ]
    out = _partition(rows, "Corrected")
    assert list(out.keys()) == ["True", "False"]


def test_partition_drops_empty_values():
    rows = [
        {"Engine": "run_surya"},
        {"Engine": ""},
        {"Engine": None},
        {"Engine": "run_tesseract"},
    ]
    out = _partition(rows, "Engine")
    assert list(out.keys()) == ["run_surya", "run_tesseract"]


# --- _aggregate_metric ---

def test_aggregate_metric_per_group_mean():
    rows = [
        {"Dataset": "synthetic_tables", "Cell_Accuracy": "0.5"},
        {"Dataset": "synthetic_tables", "Cell_Accuracy": "0.7"},
        {"Dataset": "real", "Cell_Accuracy": "0.3"},
    ]
    out = _aggregate_metric(rows, "Dataset", ["Cell_Accuracy"])
    assert out["synthetic_tables"]["Cell_Accuracy"] == pytest.approx(0.6)
    assert out["real"]["Cell_Accuracy"] == pytest.approx(0.3)


def test_aggregate_metric_skips_none_within_group():
    # Document_CER is empty for some synthetic_tables rows in the real data;
    # the mean should only consider populated values.
    rows = [
        {"Dataset": "synthetic_tables", "Document_CER": "0.1"},
        {"Dataset": "synthetic_tables", "Document_CER": ""},
        {"Dataset": "synthetic_tables", "Document_CER": "0.3"},
    ]
    out = _aggregate_metric(rows, "Dataset", ["Document_CER"])
    assert out["synthetic_tables"]["Document_CER"] == pytest.approx(0.2)


# --- _aggregate_partitioned ---

def test_aggregate_partitioned_aligns_groups_across_buckets():
    rows = [
        {"Engine": "run_surya", "Dataset": "synthetic_tables", "Document_CER": "0.1"},
        {"Engine": "run_surya", "Dataset": "real", "Document_CER": "0.5"},
        {"Engine": "run_tesseract", "Dataset": "real", "Document_CER": "0.8"},
        # run_tesseract has no synthetic_tables row → matrix should have None
    ]
    keys, matrix = _aggregate_partitioned(rows, "Dataset", "Engine", "Document_CER")
    assert keys == ["run_surya", "run_tesseract"]
    assert matrix["synthetic_tables"]["run_surya"] == pytest.approx(0.1)
    assert matrix["synthetic_tables"]["run_tesseract"] is None  # missing → not zero
    assert matrix["real"]["run_surya"] == pytest.approx(0.5)
    assert matrix["real"]["run_tesseract"] == pytest.approx(0.8)


# --- _parse_corrected ---

def test_parse_corrected_true():
    assert _parse_corrected("True") is True


def test_parse_corrected_false():
    assert _parse_corrected("False") is False


def test_parse_corrected_garbage_returns_none():
    assert _parse_corrected("true") is None  # case-sensitive
    assert _parse_corrected("1") is None
    assert _parse_corrected("") is None
    assert _parse_corrected(None) is None


# --- _has_two_distinct ---

def test_has_two_distinct_false_when_constant():
    rows = [{"Engine": "run_surya"}, {"Engine": "run_surya"}]
    assert _has_two_distinct(rows, "Engine") is False


def test_has_two_distinct_true_with_two_values():
    rows = [{"Engine": "run_surya"}, {"Engine": "run_tesseract"}]
    assert _has_two_distinct(rows, "Engine") is True


def test_has_two_distinct_true_with_parser():
    rows = [
        {"Corrected": "True"},
        {"Corrected": "False"},
        {"Corrected": "True"},
    ]
    assert _has_two_distinct(rows, "Corrected", parser=_parse_corrected) is True


# --- _plan_charts ---

def _row(dataset="synthetic_tables", engine="run_surya", corrected="True", **kwargs):
    base = {
        "Engine": engine,
        "Corrected": corrected,
        "Dataset": dataset,
        "Image_File": "x.png",
        "Font": "Battambang",
        "Template": "t",
        "Tables_Expected": "1",
        "Tables_Found": "1",
        "GT_Rows": "7",
        "GT_Cols": "4",
        "Pred_Rows": "7",
        "Pred_Cols": "4",
        "Cell_Accuracy": "1.0",
        "Cell_Content_Recall": "1.0",
        "Table_CER": "0.0",
        "Text_CER": "",
        "Document_CER": "0.1",
        "Paragraph_Recall": "",
        "Paragraph_Leak": "",
        "Error": "",
    }
    base.update(kwargs)
    return base


def test_plan_charts_includes_always_on():
    rows = [_row()]
    plans = _plan_charts(rows)
    names = {p["name"] for p in plans if p["should_render"]}
    assert {"cer_by_dataset", "accuracy_by_font", "table_fragmentation"} <= names


def test_plan_charts_omits_engine_comparison_for_single_engine():
    rows = [_row(engine="run_surya")]
    plans = _plan_charts(rows)
    ec = next(p for p in plans if p["name"] == "engine_comparison")
    assert ec["should_render"] is False
    assert "Engine" in ec["reason_if_skipped"]


def test_plan_charts_includes_engine_comparison_for_two_engines():
    rows = [_row(engine="run_surya"), _row(engine="run_tesseract")]
    plans = _plan_charts(rows)
    ec = next(p for p in plans if p["name"] == "engine_comparison")
    assert ec["should_render"] is True
    assert ec["partition_column"] == "Engine"


def test_plan_charts_omits_correction_ab_for_constant_corrected():
    rows = [_row(corrected="True")]
    plans = _plan_charts(rows)
    cab = next(p for p in plans if p["name"] == "correction_ab")
    assert cab["should_render"] is False
    assert "Corrected" in cab["reason_if_skipped"]


def test_plan_charts_includes_correction_ab_with_parsed_bools():
    rows = [_row(corrected="True"), _row(corrected="False")]
    plans = _plan_charts(rows)
    cab = next(p for p in plans if p["name"] == "correction_ab")
    assert cab["should_render"] is True
    assert cab["partition_column"] == "Corrected"


# --- _load_rows ---

def test_load_rows_reads_csv_from_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "Engine,Dataset,Document_CER\nrun_surya,real,0.5\n", encoding="utf-8"
    )
    rows = _load_rows([run_dir])
    assert len(rows) == 1
    assert rows[0]["Engine"] == "run_surya"


def test_load_rows_accepts_csv_path_directly(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "Engine,Dataset,Document_CER\nrun_surya,real,0.5\n", encoding="utf-8"
    )
    rows = _load_rows([csv_path])
    assert len(rows) == 1


def test_load_rows_skips_missing_with_notice(tmp_path, capsys):
    rows = _load_rows([tmp_path / "does_not_exist"])
    assert rows == []
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower()


# --- visualize (integration, renderers monkeypatched) ---

def _write_run_csv(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_visualize_empty_rows_writes_no_files(tmp_path, monkeypatch):
    rendered = []
    monkeypatch.setattr(
        "khmer_pipeline.evaluation.visualize_benchmark._render_grouped_bars",
        lambda *a, **kw: rendered.append(kw.get("out_path")),
    )
    written = visualize([], tmp_path / "out")
    assert written == []


def test_visualize_calls_renderer_for_each_renderable_chart(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_run_csv(
        run_dir,
        [
            _row(dataset="synthetic_tables", font="Battambang"),
            _row(dataset="synthetic_tables", font="Fasthand"),
            _row(dataset="real", font="Battambang"),
        ],
    )
    calls = []
    monkeypatch.setattr(
        "khmer_pipeline.evaluation.visualize_benchmark._render_grouped_bars",
        lambda agg, group_col, series_labels, ylabel, title, out_path: calls.append(out_path.name),
    )
    out = tmp_path / "figures"
    written = visualize([run_dir], out)
    # 3 always-on charts: cer_by_dataset, accuracy_by_font, table_fragmentation
    assert len(written) == 3
    assert {p.name for p in written} == {
        "cer_by_dataset.png",
        "accuracy_by_font.png",
        "table_fragmentation.png",
    }
    assert calls == ["cer_by_dataset.png", "accuracy_by_font.png", "table_fragmentation.png"]


def test_visualize_creates_out_dir(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_run_csv(run_dir, [_row()])
    monkeypatch.setattr(
        "khmer_pipeline.evaluation.visualize_benchmark._render_grouped_bars",
        lambda *a, **kw: None,
    )
    out = tmp_path / "does_not_exist" / "yet"
    written = visualize([run_dir], out)
    assert out.is_dir()
    assert len(written) >= 1


def test_visualize_returns_list_of_paths(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_run_csv(run_dir, [_row()])
    monkeypatch.setattr(
        "khmer_pipeline.evaluation.visualize_benchmark._render_grouped_bars",
        lambda *a, **kw: None,
    )
    written = visualize([run_dir], tmp_path / "out")
    assert isinstance(written, list)
    for p in written:
        assert isinstance(p, Path)
        assert p.name.endswith(".png")


def test_visualize_skips_charts_with_no_data(tmp_path, monkeypatch, capsys):
    # Font column is always empty → accuracy_by_font should be skipped.
    run_dir = tmp_path / "run"
    rows = [_row()]
    rows[0]["Font"] = ""
    _write_run_csv(run_dir, rows)
    skipped_charts = []
    monkeypatch.setattr(
        "khmer_pipeline.evaluation.visualize_benchmark._render_grouped_bars",
        lambda *a, **kw: None,
    )
    written = visualize([run_dir], tmp_path / "out")
    captured = capsys.readouterr()
    assert "accuracy_by_font" in captured.out
    assert "accuracy_by_font.png" not in {p.name for p in written}


def test_visualize_skips_engine_comparison_for_single_engine(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    _write_run_csv(run_dir, [_row(engine="run_surya")])
    monkeypatch.setattr(
        "khmer_pipeline.evaluation.visualize_benchmark._render_grouped_bars",
        lambda *a, **kw: None,
    )
    written = visualize([run_dir], tmp_path / "out")
    captured = capsys.readouterr()
    assert "engine_comparison" in captured.out
    assert "engine_comparison.png" not in {p.name for p in written}


def test_visualize_renders_correction_ab_for_two_corrected_values(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_run_csv(
        run_dir,
        [
            _row(corrected="True", dataset="real"),
            _row(corrected="False", dataset="real"),
        ],
    )
    calls = []
    monkeypatch.setattr(
        "khmer_pipeline.evaluation.visualize_benchmark._render_grouped_bars",
        lambda agg, group_col, series_labels, ylabel, title, out_path: calls.append((tuple(series_labels), out_path.name)),
    )
    written = visualize([run_dir], tmp_path / "out")
    cab_call = [c for c in calls if c[1] == "correction_ab.png"]
    assert len(cab_call) == 1
    # The series labels should be the two Corrected values, in first-seen order
    assert cab_call[0][0] == ("True", "False")
