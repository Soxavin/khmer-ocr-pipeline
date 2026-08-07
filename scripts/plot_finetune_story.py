"""Builds `finetune_story_overview.png` -- a single opening graphic for the fine-tuning arc,
meant to be the FIRST slide of the §4.10 sequence, before the four detail figures.

Why this exists rather than just reusing real_doc_comparison.png: the four detail charts each
answer one question well, but nothing showed the two questions *in sequence*, which is the whole
argument -- "we swept epochs, both models peaked at the same setting, and even that peak lost to
the pipeline we already ship." On a slide with no README and no surrounding prose, that sequence
is what a viewer needs in the first twenty seconds; the detail charts then earn their airtime.

Deliberately lossy relative to those detail charts, and it should stay that way:
  - left panel plots the v5 sweep only (the controlled one), no v2 history, no step-count
    provenance labels -- plot_parse_failure_rate_by_run.png is where that detail lives;
  - right panel plots ONE metric (table CER), not the six in real_doc_comparison.png -- one
    metric means one "is lower better?" question instead of two panels' worth.
Every number here also appears, with its caveats, on a detail chart; nothing is computed here
that isn't computed there. If the two ever disagree, the detail chart is right.

CLI:
    python scripts/plot_finetune_story.py eval/gemma_finetune_runs.csv eval/qwen_finetune_runs.csv \
        --real-doc-csv eval/real_doc_eval.csv \
        --out docs/figures/finetune_eval/finetune_story_overview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _report_style import (
    ANNOTATION_SIZE,
    HIGHLIGHT_EPOCH,
    NOTE_SIZE,
    apply_report_style,
    dataset_short_label,
    highlight_epoch,
    model_color,
    model_display_name,
    model_hatch,
    model_marker,
    save_all_formats,
    titled,
)

_PRIMARY_DATASET = "v5"
_VERDICT_METRIC = ("mean_table_cer", "Table CER on 15 real pages (lower is better)")
_APPROACH_ORDER = ["ocr_pipeline_surya", "gemma", "qwen"]
_APPROACH_SHORT = {
    "ocr_pipeline_surya": "OCR pipeline (Surya)\n(already shipped,\nno fine-tuning)",
    "gemma": "Gemma 4 E2B\n(best adapter:\n3 epochs)",
    "qwen": "Qwen3.5-0.8B\n(best adapter:\n3 epochs)",
}


def _sweep_panel(ax, runs: pd.DataFrame) -> None:
    """Left panel: JSON parse-failure rate vs. epoch count, v5 runs only, one line per model.

    Series are labelled directly at their last point instead of via a legend box -- with only two
    lines a legend costs a viewer an extra saccade and a color-match for information the line
    itself can carry, and on a slide that lookup is the difference between reading the chart and
    reading the legend."""
    v5 = runs[runs["dataset_version"].apply(dataset_short_label) == _PRIMARY_DATASET].copy()
    v5 = v5.drop_duplicates(subset=["run_id"])
    v5["failure_rate"] = v5["json_parse_failures"] / v5["n_validation_rows"]

    highlight_epoch(ax, HIGHLIGHT_EPOCH, caption="best setting\nfor both models")

    models = sorted(v5["model"].unique())
    # Same per-model x-offset trick as the detail chart: at 2 epochs both models sit at exactly
    # 1.0, and without it one marker hides the other entirely.
    dx = {m: (i - (len(models) - 1) / 2) * 0.12 for i, m in enumerate(models)}
    for model in models:
        # One point per epoch. Where a model has two runs at the same epoch count (Qwen's 3-epoch
        # pair, 39 vs. 78 steps), take the higher step count -- that's the run with the effective
        # batch size the sweep held constant across both models so epoch counts stay comparable,
        # and it is the adapter carried forward to the real-document panel on the right. Averaging
        # the pair instead would put a number on this chart (0.67) that appears in no run log and
        # contradicts §4.10's own table, which is worse than lossy.
        group = (v5[v5["model"] == model].sort_values("steps")
                 .drop_duplicates(subset=["epochs"], keep="last").sort_values("epochs"))
        ax.plot(group["epochs"] + dx[model], group["failure_rate"], marker=model_marker(model),
                color=model_color(model), linewidth=3.0, markersize=13,
                markeredgecolor="0.15", markeredgewidth=0.8, zorder=3)
        last = group.iloc[-1]
        ax.annotate(model_display_name(model), (last["epochs"] + dx[model], last["failure_rate"]),
                    xytext=(14, 0), textcoords="offset points", va="center",
                    color=model_color(model), fontweight="bold", fontsize=ANNOTATION_SIZE + 1.5)

    epochs = sorted(v5["epochs"].dropna().unique())
    ax.set_xticks(epochs)
    # Right margin carries the direct labels, so it is much wider than the left.
    ax.set_xlim(min(epochs) - 0.5, max(epochs) + 2.0)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("training epochs")
    ax.set_ylabel("share of pages with unusable output")
    ax.set_title("1.  The sweep: does the output even parse?")


def _verdict_panel(ax, real: pd.DataFrame) -> None:
    """Right panel: one metric, three approaches. A single metric keeps the panel to one
    "which direction is good?" question -- real_doc_comparison.png splits four higher-is-better
    and two lower-is-better metrics across two panels precisely because mixing them is a
    misreading risk, and that risk isn't worth taking on an overview slide."""
    col, ylabel = _VERDICT_METRIC
    approaches = [a for a in _APPROACH_ORDER if a in real["model"].values]
    values, failed = [], []
    for i, approach in enumerate(approaches):
        row = real[real["model"] == approach].iloc[0]
        if pd.isna(row[col]):
            values.append(0.0)
            failed.append(i)
        else:
            values.append(float(row[col]))

    top = max(values) * 1.35 if max(values) else 1.0
    ax.bar(np.arange(len(approaches)), values, width=0.6,
           color=[model_color(a) for a in approaches],
           hatch=[model_hatch(a) for a in approaches], edgecolor="white", linewidth=0.8)
    for i, v in enumerate(values):
        if i in failed:
            # An empty slot has to say which of "no bar" and "scored zero" it is -- on a CER axis
            # a real 0.0 would mean a PERFECT transcription, the exact opposite of what happened.
            ax.annotate("no usable\noutput at all\n(15/15 pages\nfailed to parse)", (i, top * 0.04),
                        ha="center", va="bottom", fontsize=ANNOTATION_SIZE,
                        color=model_color(approaches[i]), fontweight="bold", linespacing=1.4)
        else:
            ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points", fontsize=ANNOTATION_SIZE + 3,
                        fontweight="bold")

    ax.set_xticks(np.arange(len(approaches)))
    ax.set_xticklabels([_APPROACH_SHORT.get(a, a) for a in approaches], fontsize=11)
    ax.set_ylim(0, top)
    ax.set_ylabel(ylabel)
    ax.set_title("2.  The verdict: best adapter vs. the existing pipeline")
    ax.grid(True, axis="y", alpha=0.3)


def plot_story_overview(runs: pd.DataFrame, real: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_sweep, ax_verdict) = plt.subplots(1, 2, figsize=(14.5, 6.4),
                                               gridspec_kw={"width_ratios": [1.15, 1]})
    _sweep_panel(ax_sweep, runs)
    _verdict_panel(ax_verdict, real)

    titled(fig, "Fine-tuning experiment at a glance",
           "Two models, three epoch settings each. Both peaked at 3 epochs — and the best of "
           "them still lost to the pipeline already in production.",
           y=1.07)
    fig.text(0.5, -0.02,
             "Left: in-distribution synthetic validation split (9 pages/run, one run per point). "
             "Right: 15 held-out, hand-verified real pages.\n"
             "Single seed throughout — no error bars. Per-run detail, sample sizes and caveats: "
             "the four charts that follow, and this folder's README.",
             ha="center", va="top", fontsize=NOTE_SIZE, style="italic", color="0.35",
             linespacing=1.5)
    fig.tight_layout()
    save_all_formats(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", type=Path, nargs="+",
                        help="Run-log CSVs, e.g. eval/gemma_finetune_runs.csv eval/qwen_finetune_runs.csv")
    parser.add_argument("--real-doc-csv", type=Path, default=Path("eval/real_doc_eval.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("docs/figures/finetune_eval/finetune_story_overview.png"))
    args = parser.parse_args()

    apply_report_style()
    # Reuses plot_run_metrics.load_combined so the model tagging and run_id construction can't
    # drift between this overview and the detail charts it summarises.
    from plot_run_metrics import load_combined
    runs = load_combined(args.csvs)
    real = pd.read_csv(args.real_doc_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_story_overview(runs, real, args.out)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
