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

The file this script writes is the LEAN render (see plot_run_metrics.py's header for the
two-tier arrangement): everything above stays, but the figure-level footnote restating Qwen's
15/15 failures in prose does not -- it lives on the full-detail block in
scripts/plot_finetune_dashboard.py, drawn by this module's own compose function with
`detail=True`.

CLI:
    python scripts/plot_real_doc_comparison.py eval/real_doc_eval.csv --out docs/figures/finetune_eval/real_doc_comparison.png
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _report_style import (
    ANNOTATION_SIZE,
    NOTE_SIZE,
    apply_report_style,
    model_color,
    model_hatch,
    save_all_formats,
    titled,
)

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
        # hatch + white edge: the redundant, non-color encoding of approach identity (see
        # _report_style.MODEL_HATCH). It also propagates into the legend swatches automatically,
        # which is where a grayscale reader most needs it.
        bars = ax.bar(offsets, heights, width=bar_width, label=_APPROACH_LABEL.get(approach, approach),
                       color=model_color(approach), hatch=model_hatch(approach),
                       edgecolor="white", linewidth=0.8)
        for j in failed_positions:
            # Rotated so the full phrase fits inside a single bar's slot: at slide-legible type a
            # horizontal "no output" is already wider than the slot, and the ambiguity this
            # annotation exists to kill (empty slot = "no bar drawn" vs. "score of zero") is worth
            # more than the reading comfort of a horizontal label. Sits in the empty slot itself,
            # so it's unmistakably about that approach and not the neighbouring bar.
            ax.annotate("no output", (offsets[j], ylim_top * 0.03), ha="center", va="bottom",
                        rotation=90, fontsize=ANNOTATION_SIZE + 1,
                        color=model_color(approach), fontweight="bold")
        for bar, h, j in zip(bars, heights, range(n_metrics)):
            if j not in failed_positions:
                ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=ANNOTATION_SIZE + 1.5,
                            xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=11)
    # The rightmost group's "no output" label is the last thing in the panel and sat flush against
    # the spine at the default bar margins -- enough to look clipped in the render.
    ax.margins(x=0.06)
    ax.set_ylim(0, ylim_top)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)


def _no_output_note(df: pd.DataFrame) -> str:
    """One figure-level sentence naming each approach that produced nothing scorable, and how
    many pages that was. The per-bar 'no output' labels say an empty slot isn't a zero; this says
    what actually happened, in numbers, without the viewer needing the README on screen."""
    parts = []
    for _, row in df.iterrows():
        failures, pages = row.get("parse_failures"), row.get("n_pages")
        if pd.notna(failures) and pd.notna(pages) and int(failures) == int(pages) and int(pages) > 0:
            name = _APPROACH_LABEL.get(row["model"], row["model"]).split("\n")[0]
            parts.append(f"{name}: all {int(pages)} pages failed to parse, so it has no bar on "
                          f"either panel (an empty slot is not a score of zero)")
    return "  ".join(parts)


def compose_real_doc_comparison(container, df: pd.DataFrame, *, detail: bool = False,
                                title_y: float = 1.19, legend_y: float = 1.05,
                                note_y: float = -0.03, heading_prefix: str = "") -> None:
    """Draws the two-panel real-document comparison into `container` (a Figure for the standalone
    file, a SubFigure for the dashboard poster).

    Lean vs. `detail=True` differ only in the figure-level footnote spelling out Qwen's 15/15
    parse failures. Everything else here is load-bearing at any density and stays on both: the
    per-bar value labels are the numbers a viewer is meant to compare, and the "no output"
    markers are what stop an empty slot being read as a score of zero -- which on the CER panel
    would mean a PERFECT transcription, the exact opposite of what happened."""
    ax_hi, ax_lo = container.subplots(1, 2)
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
    container.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=3)
    note = _no_output_note(df) if detail else ""
    if note:
        container.text(0.5, note_y, textwrap.fill(note, 120), ha="center", va="top",
                       fontsize=NOTE_SIZE, style="italic", color="0.35", linespacing=1.5)
    n_pages = df["n_pages"].dropna()
    pages_phrase = f"same {int(n_pages.iloc[0])} hand-verified ARDB pages" if len(n_pages) else "same held-out ARDB pages"
    titled(container,
           heading_prefix + f"Real-document evaluation: {pages_phrase}, all three approaches",
           "The existing OCR pipeline wins on every metric measured — with no fine-tuning at all.",
           y=title_y)


def plot_real_doc_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Writes the standalone real-document figure -- always the LEAN render. The file at this path
    is what slides and docs/REPORT.md embed; the annotated version of the same chart is a block on
    finetune_dashboard.png, never a second file here."""
    fig = plt.figure(figsize=(13, 6))
    compose_real_doc_comparison(fig, df)
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
