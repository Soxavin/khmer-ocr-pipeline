"""Plots the headline real-document comparison from eval/real_doc_eval.csv: the existing OCR
pipeline vs. each fine-tuned model's best-tested adapter, on the same 15 hand-verified real ARDB
pages. This is the centerpiece "does it actually work" chart -- unlike plot_run_metrics.py's
per-run trend charts (which track the epoch sweep on the in-distribution synthetic validation
split), this is a single, one-shot comparison across approaches, so it gets its own script rather
than being forced into that trend-over-runs shape.

Two panels, split by metric direction so nothing is misread:
  - left: "higher is better" metrics (cell/numeric-cell accuracy, grid-shape/col-count match rate)
  - right: "lower is better" metrics (table/document CER)

A row with 100% parse failures (Qwen, as of 2026-08-06 -- see eval/qwen_finetune_runs.md) has
blank CER/accuracy fields in the CSV, since those numbers are undefined when nothing parsed, not
zero. Rendered as an explicit "no output" annotation instead of a bar, on BOTH panels -- silently
treating it as a 0-height bar would misleadingly imply a defined-but-bad score on the CER panel
(where a real 0 would mean a PERFECT score, the opposite of the truth).

CLI:
    python scripts/plot_real_doc_comparison.py eval/real_doc_eval.csv --out docs/figures/finetune_eval/real_doc_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _report_style import apply_report_style, save_all_formats

# (csv column, display label)
_HIGHER_IS_BETTER = [
    ("mean_cell_accuracy", "cell\naccuracy"),
    ("mean_numeric_cell_accuracy", "numeric-cell\naccuracy"),
    ("mean_grid_shape_match_rate", "grid shape\nmatch rate"),
    ("mean_col_count_match_rate", "col count\nmatch rate"),
]
_LOWER_IS_BETTER = [
    ("mean_table_cer", "table\nCER"),
    ("mean_document_cer", "document\nCER"),
]

# Short, report-friendly labels for the three approaches, keyed by the CSV's `model` column.
_APPROACH_LABEL = {
    "ocr_pipeline_surya": "OCR pipeline\n(Surya, no fine-tune)",
    "gemma": "Gemma 4 E2B\n(fine-tuned, v5-e3)",
    "qwen": "Qwen3.5-0.8B\n(fine-tuned, v5-e3-ga2)",
}
_APPROACH_ORDER = ["ocr_pipeline_surya", "gemma", "qwen"]


def _panel(ax, df: pd.DataFrame, metrics: list[tuple[str, str]], title: str, ylim_top: float) -> None:
    approaches = [a for a in _APPROACH_ORDER if a in df["model"].values]
    n_metrics = len(metrics)
    n_approaches = len(approaches)
    bar_width = 0.8 / n_approaches
    x = np.arange(n_metrics)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, approach in enumerate(approaches):
        row = df[df["model"] == approach].iloc[0]
        offsets = x - 0.4 + bar_width * (i + 0.5)
        heights, failed_positions = [], []
        for j, (col, _) in enumerate(metrics):
            val = row[col]
            if pd.isna(val):
                heights.append(0.0)
                failed_positions.append(j)
            else:
                heights.append(float(val))
        bars = ax.bar(offsets, heights, width=bar_width, label=_APPROACH_LABEL.get(approach, approach),
                       color=color_cycle[i % len(color_cycle)])
        for j in failed_positions:
            ax.annotate("no\noutput", (offsets[j], 0.015), ha="center", va="bottom",
                        fontsize=7, color=color_cycle[i % len(color_cycle)], fontweight="bold")
        for bar, h, j in zip(bars, heights, range(n_metrics)):
            if j not in failed_positions:
                ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=8)
    ax.set_ylim(0, ylim_top)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


def plot_real_doc_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_hi, ax_lo) = plt.subplots(1, 2, figsize=(12, 5.5))
    # Left panel is a genuinely bounded [0, 1] metric family (accuracy/match rate), so a fixed
    # 1.05 ceiling is meaningful -- 1.0 means "perfect". Right panel (CER) is unbounded and its
    # visible values top out well under 1.0 here, so scaling it to the left panel's ceiling
    # would just waste over a third of the panel as dead space above the tallest bar -- scale
    # it to its own data instead, with a floor so a small-value chart doesn't look absurdly tall.
    lower_max = df[[c for c, _ in _LOWER_IS_BETTER]].max(numeric_only=True).max()
    lower_ylim = max(0.3, lower_max * 1.2) if pd.notna(lower_max) else 1.05
    _panel(ax_hi, df, _HIGHER_IS_BETTER, "Higher is better", ylim_top=1.05)
    _panel(ax_lo, df, _LOWER_IS_BETTER, "Lower is better", ylim_top=lower_ylim)
    handles, labels = ax_hi.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=9)
    fig.suptitle("Real-document evaluation: same 15 hand-verified ARDB pages, all three approaches", y=1.16, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_all_formats(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="eval/real_doc_eval.csv")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/finetune_eval/real_doc_comparison.png"))
    args = parser.parse_args()

    apply_report_style()
    df = pd.read_csv(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_real_doc_comparison(df, args.out)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
