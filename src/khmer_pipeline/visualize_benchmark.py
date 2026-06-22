from __future__ import annotations
import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

# Report only uses Latin-safe columns (Dataset, Font, Engine). Template is
# Khmer and would render as tofu boxes in matplotlib's default font, so it's
# deliberately never used as a chart label.

_FIG_DPI = 150
_FIG_FIGSIZE = (10, 6)
_BAR_WIDTH = 0.25
_CER_METRICS = ("Document_CER", "Text_CER", "Table_CER")
_ACCURACY_METRICS = ("Cell_Accuracy", "Cell_Content_Recall")
_FRAGMENTATION_METRICS = ("Tables_Expected", "Tables_Found")


# --- pure aggregation helpers (no pandas, no matplotlib) ---

def _coerce_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _coerce_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _mean(values) -> float | None:
    # Filter out None so callers can pass already-coerced lists without
    # pre-filtering (matches what callers see after _coerce_float).
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _group_by(rows: list[dict], column: str) -> "OrderedDict[str, list[dict]]":
    # First-seen order of distinct values matters: it determines x-axis order.
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        key = r.get(column)
        if key is None or key == "":
            continue
        groups.setdefault(key, []).append(r)
    return groups


def _partition(rows: list[dict], column: str) -> "OrderedDict[str, list[dict]]":
    # Partition = group by with a specific intent: split rows into disjoint
    # buckets keyed by a comparison column (Engine, Corrected). Same impl.
    return _group_by(rows, column)


def _aggregate_metric(
    rows: list[dict],
    column: str,
    metric_columns: list[str],
) -> dict[str, dict[str, float | None]]:
    groups = _group_by(rows, column)
    result: dict[str, dict[str, float | None]] = {}
    for group, items in groups.items():
        result[group] = {}
        for m in metric_columns:
            vals = [v for v in (_coerce_float(item.get(m)) for item in items) if v is not None]
            result[group][m] = _mean(vals)
    return result


def _aggregate_partitioned(
    rows: list[dict],
    group_column: str,
    partition_column: str,
    metric: str,
) -> tuple[list[str], dict[str, dict[str, float | None]]]:
    # Partition rows by partition_column; aggregate each bucket by group_column
    # for a single metric. Returns (partition_keys_in_order, matrix) where
    # matrix[group][partition_key] = mean_or_None. Missing combinations are
    # explicitly populated with None (NOT zero) so the renderer can skip them.
    buckets = _partition(rows, partition_column)
    keys = list(buckets.keys())
    # Pre-collect all group keys across buckets so we can backfill None for
    # combinations a particular bucket lacks.
    all_groups: "OrderedDict[str, None]" = OrderedDict()
    for bucket_rows in buckets.values():
        for r in bucket_rows:
            g = r.get(group_column)
            if g is not None and g != "" and g not in all_groups:
                all_groups[g] = None
    matrix: dict[str, dict[str, float | None]] = {}
    for group in all_groups:
        matrix[group] = {key: None for key in keys}
    for key in keys:
        bucket_agg = _aggregate_metric(buckets[key], group_column, [metric])
        for group, m in bucket_agg.items():
            matrix[group][key] = m[metric]
    return keys, matrix


def _parse_corrected(value) -> bool | None:
    # csv.DictWriter writes Python bools as 'True'/'False' (capitalised).
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _has_two_distinct(rows: list[dict], column: str, parser=str) -> bool:
    seen: set = set()
    for r in rows:
        v = r.get(column)
        if v is None or v == "":
            continue
        parsed = parser(v)
        if parsed is None:
            continue
        seen.add(parsed)
        if len(seen) >= 2:
            return True
    return False


# --- row loader ---

def _load_rows(paths) -> list[dict]:
    # Mirror analyze_benchmark._load_rows: each path is either a run-dir
    # (we read results.csv inside) or a direct results.csv path.
    rows: list[dict] = []
    for p in paths:
        p = Path(p)
        csv_path = p / "results.csv" if p.is_dir() else p
        if not csv_path.exists():
            print(f"Note: {csv_path} not found — skipping.")
            continue
        rows.extend(csv.DictReader(csv_path.open(encoding="utf-8")))
    return rows


# --- chart planning (declarative; no matplotlib here) ---

def _plan_charts(rows: list[dict]) -> list[dict]:
    # Always-on charts first; conditional charts with should_render flags.
    return [
        {
            "name": "cer_by_dataset",
            "title": "CER by Dataset (lower is better)",
            "group_column": "Dataset",
            "series_kind": "metrics",  # one series per metric_column
            "metric_columns": list(_CER_METRICS),
            "partition_column": None,
            "ylabel": "CER",
            "should_render": True,
            "reason_if_skipped": None,
        },
        {
            "name": "accuracy_by_font",
            "title": "Cell Accuracy & Content Recall by Font (higher is better)",
            "group_column": "Font",
            "series_kind": "metrics",
            "metric_columns": list(_ACCURACY_METRICS),
            "partition_column": None,
            "ylabel": "Score",
            "should_render": True,
            "reason_if_skipped": None,
        },
        {
            "name": "table_fragmentation",
            "title": "Table Fragmentation: Tables Found vs Tables Expected by Dataset",
            "group_column": "Dataset",
            "series_kind": "metrics",
            "metric_columns": list(_FRAGMENTATION_METRICS),
            "partition_column": None,
            "ylabel": "Number of Tables",
            "should_render": True,
            "reason_if_skipped": None,
        },
        {
            "name": "engine_comparison",
            "title": "Engine Comparison: Document CER by Dataset (lower is better)",
            "group_column": "Dataset",
            "series_kind": "partition",
            "metric_columns": ["Document_CER"],
            "partition_column": "Engine",
            "ylabel": "Document CER",
            "should_render": _has_two_distinct(rows, "Engine"),
            "reason_if_skipped": "need >=2 distinct Engine values across runs",
        },
        {
            "name": "correction_ab",
            "title": "Correction A/B: Document CER by Dataset (lower is better)",
            "group_column": "Dataset",
            "series_kind": "partition",
            "metric_columns": ["Document_CER"],
            "partition_column": "Corrected",
            "ylabel": "Document CER",
            "should_render": _has_two_distinct(rows, "Corrected", parser=_parse_corrected),
            "reason_if_skipped": "need both Corrected=True and Corrected=False across runs",
        },
    ]


# --- rendering (matplotlib imports scoped to these functions so pure
#     helpers stay importable in headless / unit-test contexts) ---

def _render_grouped_bars(
    matrix: dict[str, dict[str, float | None]],
    group_column: str,
    series_labels: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Union of group keys, preserving first-seen order.
    groups: list[str] = []
    seen: set[str] = set()
    for g in matrix.keys():
        if g not in seen:
            groups.append(g)
            seen.add(g)

    x = np.arange(len(groups))
    n_series = len(series_labels)
    if n_series == 0:
        return
    width = _BAR_WIDTH if n_series <= 3 else _BAR_WIDTH * (3 / n_series)
    fig, ax = plt.subplots(figsize=_FIG_FIGSIZE)
    for i, series in enumerate(series_labels):
        # Plot all groups; missing (None) values are rendered as zero-height
        # bars with a label so the user sees "no data" rather than a gap.
        plot_x: list[int] = []
        plot_y: list[float] = []
        plot_labels: list[str] = []
        for j, group in enumerate(groups):
            v = matrix[group].get(series)
            offset = (i - (n_series - 1) / 2) * width
            plot_x.append(x[j] + offset)
            if v is None:
                plot_y.append(0.0)
                plot_labels.append("")
            else:
                plot_y.append(v)
                plot_labels.append(f"{v:.3f}")
        bars = ax.bar(plot_x, plot_y, width, label=series)
        for bar, label in zip(bars, plot_labels):
            if label:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel(group_column)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, out_path)


def _save_figure(fig, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# --- orchestrator ---

def visualize(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(run_dirs)
    if not rows:
        print("No results to visualize.")
        return []
    plans = _plan_charts(rows)
    written: list[Path] = []
    for spec in plans:
        if not spec["should_render"]:
            print(f"Skipping {spec['name']}: {spec['reason_if_skipped']}.")
            continue
        out_path = out_dir / f"{spec['name']}.png"
        if spec["series_kind"] == "metrics":
            agg = _aggregate_metric(rows, spec["group_column"], spec["metric_columns"])
            if not agg:
                print(f"Skipping {spec['name']}: no data for group column {spec['group_column']!r}.")
                continue
            series_labels = list(spec["metric_columns"])
        else:  # "partition"
            metric = spec["metric_columns"][0]
            series_labels, agg = _aggregate_partitioned(
                rows, spec["group_column"], spec["partition_column"], metric,
            )
            if len(series_labels) < 2:
                print(f"Skipping {spec['name']}: partition column {spec['partition_column']!r} yielded <2 buckets.")
                continue
        _render_grouped_bars(
            agg, spec["group_column"], series_labels,
            spec["title"], spec["ylabel"], out_path,
        )
        written.append(out_path)
    return written


# --- CLI ---

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate benchmark figures (matplotlib) from one or more run dirs.",
    )
    parser.add_argument(
        "run_dir",
        nargs="+",
        type=Path,
        help="One or more run dirs (containing results.csv) or direct results.csv paths.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/figures"),
        help="Output directory for PNGs (default: eval/figures).",
    )
    args = parser.parse_args(argv)
    written = visualize(args.run_dir, args.out)
    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
