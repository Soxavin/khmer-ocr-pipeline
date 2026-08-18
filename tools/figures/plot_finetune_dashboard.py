"""Builds `finetune_dashboard.png` -- ONE full-detail reference sheet holding all four detail
charts of the fine-tuning arc (real-document comparison, parse-failure rate, per-label CER,
training loss) in a 2x2 grid.

Why this exists: the four standalone charts are deliberately LEAN. They're what goes on a slide
and into the report body, where per-point sample sizes, step counts, dates and footnotes were
competing for attention with the shape of the data -- the thing those charts exist to show. That
detail is still worth having somewhere, just not on the slide, so it lives here: one image to
open when a specific follow-up question gets asked ("how many pages was that average over?",
"which of the two 3-epoch runs is that point?"), rather than four separate annotated files that
would have to be kept in sync with the lean ones.

This is NOT a presentation asset. It's read on screen at reference scale, so it can be far denser
than the slide charts and doesn't need projector-scale type.

Nothing is drawn here that the other scripts don't draw: every block calls the same
compose_*() function the standalone chart calls, with detail=True. There is no second copy of any
plotting code -- a lean chart and its dashboard block cannot disagree, because they are the same
code path at two text densities.

CLI:
    python scripts/plot_finetune_dashboard.py eval/gemma_finetune_runs.csv eval/qwen_finetune_runs.csv \
        --real-doc-csv eval/real_doc_eval.csv --loss-csv eval/loss_history.csv \
        --out docs/figures/finetune_eval/finetune_dashboard.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _report_style import NOTE_SIZE, apply_report_style, save_all_formats
from plot_real_doc_comparison import compose_real_doc_comparison
from plot_run_metrics import (
    _run_label_lookup,
    compose_cer_by_run,
    compose_loss_by_run,
    compose_parse_failure_rate_by_run,
    load_combined,
)

# One block per original chart. 2x2 rather than a 4-high column: at four blocks a single column
# is ~30 inches tall and can only be read by scrolling, which defeats "one sheet you can scan",
# while 2x2 keeps the whole reference visible at once on a normal screen and prints on one
# landscape page.
_FIGSIZE = (31.0, 17.0)

# Read on screen, not projected: full report DPI here would mean a ~50-megapixel PNG for an image
# nobody scales up. The .pdf companion is vector regardless, so nothing is actually lost.
_DASHBOARD_DPI = 150

_HEADER_TITLE = "Fine-tuning evaluation — full-detail reference sheet"
_HEADER_NOTE = (
    "Every chart here also exists as its own slide-ready file in this folder, drawn from the same "
    "code with the per-point labels, provenance and footnotes below stripped out.\n"
    "Those are the versions to present or embed; this sheet is the backup for a specific "
    "follow-up question. Numbers come straight from eval/*.csv — nothing is recomputed here."
)


def _block_heading(index: int) -> str:
    """'1 · ' etc. -- prefixed to each block's own chart title so a reader can say which block
    they're looking at without decoding four titles, and so the numbering matches the order the
    charts are discussed in docs/figures/finetune_eval/README.md."""
    return f"{index} · "


def plot_dashboard(runs: pd.DataFrame, real: pd.DataFrame, loss_df: pd.DataFrame,
                   run_meta: pd.DataFrame, label_lookup: dict[str, str], out_path: Path) -> None:
    fig = plt.figure(figsize=_FIGSIZE)
    header, body = fig.subfigures(2, 1, height_ratios=[0.10, 1.0])
    header.text(0.5, 0.80, _HEADER_TITLE, ha="center", va="top", fontsize=24, fontweight="bold")
    header.text(0.5, 0.42, _HEADER_NOTE, ha="center", va="top", fontsize=NOTE_SIZE + 1,
                color="0.32", linespacing=1.5)

    blocks = body.subfigures(2, 2, wspace=0.02, hspace=0.03)
    (top_left, top_right), (bottom_left, bottom_right) = blocks
    # A faint panel behind each block, so four dense charts on one canvas read as four separate
    # things rather than one continuous field of marks. Cheaper than rules or boxes: no extra
    # lines to compete with the chart content.
    for row in blocks:
        for block in row:
            block.set_facecolor("#F3F4F6")

    finishers = []

    # 1. Real-document comparison -- widest internal content (two panels + a 3-column legend),
    #    so it takes the top-left slot a reader lands on first, matching the report's own order.
    compose_real_doc_comparison(top_left, real, detail=True, title_y=0.99, legend_y=0.88,
                                note_y=0.075, heading_prefix=_block_heading(1))
    top_left.subplots_adjust(top=0.70, bottom=0.20, left=0.075, right=0.975, wspace=0.18)

    # 2. Parse-failure rate -- the legend sits outside the axes on the right, hence the tight
    #    right margin.
    finishers += compose_parse_failure_rate_by_run(
        top_right, runs, detail=True, title_y=0.99, note_y=0.075,
        heading_prefix=_block_heading(2))
    top_right.subplots_adjust(top=0.84, bottom=0.20, left=0.08, right=0.74)

    # 3. Per-label CER -- densest block: two panels, a shared legend above them, per-point n=
    #    labels with leader lines, and the longest footnote of the four.
    finishers += compose_cer_by_run(bottom_left, runs, detail=True, title_y=0.99, legend_y=0.845,
                                    note_y=0.135, heading_prefix=_block_heading(3))
    bottom_left.subplots_adjust(top=0.74, bottom=0.26, left=0.075, right=0.975, wspace=0.16)

    # 4. Training loss -- no annotations at all, so it gets the least vertical furniture.
    compose_loss_by_run(bottom_right, loss_df, run_meta, label_lookup, detail=True,
                        title_y=0.99, heading_prefix=_block_heading(4))
    bottom_right.subplots_adjust(top=0.83, bottom=0.16, left=0.07, right=0.98, wspace=0.10)

    # Only now: _place_annotations declutters in display space, which needs each axes' final
    # geometry -- i.e. after every subplots_adjust above.
    for finish in finishers:
        finish()
    save_all_formats(fig, out_path, dpi=_DASHBOARD_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", type=Path, nargs="+",
                        help="Run-log CSVs, e.g. eval/gemma_finetune_runs.csv eval/qwen_finetune_runs.csv")
    parser.add_argument("--real-doc-csv", type=Path, default=Path("eval/real_doc_eval.csv"))
    parser.add_argument("--loss-csv", type=Path, default=Path("eval/loss_history.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("docs/figures/finetune_eval/finetune_dashboard.png"))
    args = parser.parse_args()

    apply_report_style()
    runs = load_combined(args.csvs)
    real = pd.read_csv(args.real_doc_csv)
    if args.loss_csv.is_file():
        loss_df = pd.read_csv(args.loss_csv)
        loss_df["run_id"] = loss_df["model"] + "/" + loss_df["run"].astype(str)
    else:
        loss_df = pd.DataFrame(columns=["run_id", "step", "loss"])
    run_meta = runs.drop_duplicates(subset=["run_id"])[["run_id", "dataset_version", "epochs", "steps"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_dashboard(runs, real, loss_df, run_meta, _run_label_lookup(runs), args.out)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
