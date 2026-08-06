"""Plot fine-tune run-log trends (per-label CER, JSON-parse-failure rate) from one or more
structured CSV companions to the narrative run logs (eval/gemma_finetune_runs.md,
eval/qwen_finetune_runs.md).

Each model's runs live in their own CSV/MD pair (not merged into one file, since they track
different models with partly different columns) -- this script is what combines them into
one comparison chart, the "CER & visualize" proof artifact requested directly by the mentor,
not eyeballed from prose.

CLI:
    python scripts/plot_run_metrics.py eval/gemma_finetune_runs.csv --out-dir docs/figures/finetune_eval
    python scripts/plot_run_metrics.py eval/gemma_finetune_runs.csv eval/qwen_finetune_runs.csv \
        --out-dir docs/figures/finetune_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _report_style import (
    MARKERS,
    apply_report_style,
    dataset_linestyle,
    dataset_short_label,
    model_color,
    model_display_name,
    model_marker,
    run_display_label,
    save_all_formats,
)

_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = MARKERS


def _jitter_duplicate_epochs(group: pd.DataFrame, spread: float = 0.12) -> pd.Series:
    """x-position per row, aligned to group.index: the row's epoch value, except when more than
    one row in this single (model[, label], dataset-version) line shares the same epoch value --
    a same-config repeat (e.g. Gemma's v2-run2/run3, an identical-config rerun after a GT-typo
    fix) or a different-batch-size run at the same epoch count (e.g. Qwen's two 3-epoch runs, 39
    vs. 78 steps). Those rows are spread symmetrically around the shared epoch value, ordered by
    step count then run_id, so neither point silently lands exactly on top of the other."""
    xs = pd.Series(index=group.index, dtype=float)
    for epochs, sub in group.groupby("epochs"):
        sub = sub.sort_values(["steps", "run_id"], kind="stable")
        n = len(sub)
        for i, idx in enumerate(sub.index):
            xs[idx] = float(epochs) + spread * (i - (n - 1) / 2)
    return xs.reindex(group.index)


def _place_annotations(ax, entries: list[tuple[float, float, str, float]],
                        base_dy: float = 8, tier_gap: float = 13) -> None:
    """entries: (x, y, text, epoch_bucket) tuples. Groups by epoch_bucket -- the actual epoch
    count, not the jittered x -- since every observed annotation collision in these charts
    happens between points that share an epoch, then stacks that bucket's labels in
    ascending-y order with a small, consistent tier gap. Every annotation gets a thin leader
    line back to its real point (arrowprops), so even a label pushed up several tiers to clear
    a crowded cluster stays visually tied to its marker -- an earlier version offset each LINE
    by a fixed amount regardless of how isolated its points were, which pushed isolated points'
    labels far from their marker with nothing connecting them, reading as disconnected floating
    text rather than an annotation."""
    from collections import defaultdict
    buckets: dict[float, list[tuple[float, float, str]]] = defaultdict(list)
    for x, y, text, epoch in entries:
        buckets[epoch].append((x, y, text))
    for pts in buckets.values():
        for tier, (x, y, text) in enumerate(sorted(pts, key=lambda t: t[1])):
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(6, base_dy + tier_gap * tier), fontsize=6.5, color="0.25",
                        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5, shrinkA=0, shrinkB=3))


def _epoch_disambiguator(run_id: str, label_lookup: dict[str, str]) -> str:
    """The part of a run's display label beyond its dataset version -- e.g. 'v5\\n3ep, 78 steps'
    becomes '3ep, 78 steps' -- used as a per-point annotation on epoch-axis charts, where the
    dataset version is already conveyed by linestyle/panel and the epoch count by x-position, so
    only the disambiguating remainder (steps, or a date tiebreak) is worth spelling out again."""
    full = label_lookup.get(run_id, run_id)
    parts = full.split("\n", 1)
    return parts[1] if len(parts) > 1 else full


def _infer_model_name(csv_path: Path) -> str:
    """e.g. 'gemma_finetune_runs.csv' -> 'gemma', 'qwen_finetune_runs.csv' -> 'qwen'."""
    return csv_path.stem.removesuffix("_finetune_runs")


def load_combined(csv_paths: list[Path], model_names: list[str] | None = None) -> pd.DataFrame:
    """Reads one or more run-log CSVs, tags each row with its model, and gives every row a
    globally-unique run_id (model/run) so runs from different models never collide on the
    same x-axis position even if their own `run` names happen to match."""
    if model_names is None:
        model_names = [_infer_model_name(p) for p in csv_paths]
    if len(model_names) != len(csv_paths):
        raise ValueError(f"Got {len(csv_paths)} CSVs but {len(model_names)} labels -- must match 1:1.")

    frames = []
    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        df["model"] = name
        df["run_id"] = name + "/" + df["run"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _run_label_lookup(df: pd.DataFrame) -> dict[str, str]:
    """run_id -> a label a report reader can understand with no cross-referencing (dataset
    version + epoch + step count), built from each row's own dataset_version/epochs/steps
    columns. Even (dataset, epochs, steps) can collide for the SAME model -- e.g. Gemma's
    v2-run2/v2-run3 are a genuine identical-config repeat (a GT-typo fix happened between
    them, see eval/gemma_finetune_runs.md) -- so any same-model collision gets the run's date
    appended as a tiebreaker, since two runs with truly identical config can only still be
    told apart by when they happened."""
    lookup: dict[str, str] = {}
    rows: dict[str, pd.Series] = {}
    for run_id, group in df.groupby("run_id"):
        row = group.iloc[0]
        rows[run_id] = row
        lookup[run_id] = run_display_label(row.get("dataset_version"), row.get("epochs"), row.get("steps"))

    by_model_label: dict[tuple[str, str], list[str]] = {}
    for run_id, label in lookup.items():
        key = (rows[run_id].get("model"), label)
        by_model_label.setdefault(key, []).append(run_id)
    for (_, label), run_ids in by_model_label.items():
        if len(run_ids) > 1:
            for run_id in run_ids:
                date = rows[run_id].get("date")
                lookup[run_id] = f"{label} ({date})" if pd.notna(date) else f"{label} ({run_id})"
    return lookup


def plot_cer_by_run(df: pd.DataFrame, out_path: Path) -> None:
    """One panel per model, one line per (content label, dataset version) within it, CER vs.
    EPOCH COUNT (skipping rows with no computable CER, e.g. a run where every validation row
    failed to parse). Split into per-model panels rather than one shared axis for two reasons:
    (1) a single Qwen outlier (Page-Furniture CER 5.18 on v5-run1) would otherwise compress
    every other, more relevant point into an unreadable cluster near zero on a shared y-axis;
    (2) each model's own panel keeps its own epoch range legible.

    x = epoch count, not run order: the earlier run-order x-axis meant equal-epoch points from
    different dataset versions or step counts weren't visually aligned, obscuring exactly the
    comparison this chart exists to support. Dataset version (v2 vs. v5) is now carried by
    linestyle instead (solid v5, dashed v2 -- see _report_style.dataset_linestyle); a same-label
    point that collides on epoch count (a same-config rerun, or a different-batch-size run at
    the same epoch count) is jittered apart via _jitter_duplicate_epochs and annotated with the
    disambiguating detail (steps, or a date) rather than silently overlapping.
    """
    labeled = df.dropna(subset=["label_cer"]).copy()
    labeled["dataset_short"] = labeled["dataset_version"].apply(dataset_short_label)
    label_lookup = _run_label_lookup(df)
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5.5), squeeze=False)
    axes = axes[0]
    # Collected across panels for one shared external legend instead of a per-panel one --
    # label->color/marker mapping is consistent across panels (both sort their own labels
    # alphabetically, and Gemma's one extra label, "Text", sorts last, so shared labels always
    # land on the same palette index in both panels), so a single legend below the figure is
    # correct, not just more compact.
    handles_by_label: dict[str, object] = {}

    for ax, model in zip(axes, models):
        model_data = labeled[labeled["model"] == model]
        if model_data.empty:
            ax.text(0.5, 0.5, "no CER data yet", ha="center", va="center", transform=ax.transAxes)
        else:
            labels_present = sorted(model_data["label"].unique())
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color_by_label = {lbl: color_cycle[i % len(color_cycle)] for i, lbl in enumerate(labels_present)}
            # Marker shape, not just color, per label -- a colorblind-safe palette still relies
            # on hue, which a true grayscale/luminance-only rendering collapses entirely. Shape
            # carries the same distinction redundantly so the chart survives desaturation too.
            marker_by_label = {lbl: MARKERS[i % len(MARKERS)] for i, lbl in enumerate(labels_present)}
            # Collected across all (label, dataset-version) lines in this panel, then placed
            # once via _place_annotations -- sample-size annotations are grouped by which
            # EPOCH they actually collide at, not offset by a fixed per-line amount regardless
            # of how isolated the point is (see _place_annotations docstring for why that
            # earlier approach produced disconnected-looking floating text).
            annotation_entries: list[tuple[float, float, str, float]] = []
            for (label, ds), group in model_data.groupby(["label", "dataset_short"]):
                group = group.sort_values("epochs")
                xs = _jitter_duplicate_epochs(group)
                # markeredgecolor: the Okabe-Ito palette's yellow is colorblind-safe but has
                # very high luminance -- confirmed via an actual grayscale conversion that it
                # nearly vanishes against a white background without a dark edge. A dark edge
                # keeps every marker visible regardless of which color in the cycle it lands on.
                line, = ax.plot(xs.values, group["label_cer"], marker=marker_by_label[label],
                                 color=color_by_label[label], linestyle=dataset_linestyle(ds),
                                 label=label, markeredgecolor="0.15", markeredgewidth=0.6)
                handles_by_label.setdefault(label, line)
                # Sample-size annotation on every point -- a single-sample point (e.g. Gemma's
                # v5/5ep run, where 8/9 rows failed to parse and the one that did was missing
                # most of its regions) looks identical to a well-sampled point otherwise;
                # nothing on the line itself signals "this is one row, not sixteen."
                for run_id, x, y, ep, n in zip(group["run_id"], xs.values, group["label_cer"],
                                                group["epochs"], group["label_n_matched"]):
                    if pd.notna(n):
                        disamb = _epoch_disambiguator(run_id, label_lookup)
                        annotation_entries.append((x, y, f"n={int(n)} ({disamb})", float(ep)))
            _place_annotations(ax, annotation_entries)
            ax.set_xlabel("epochs")
            epochs_present = sorted(model_data["epochs"].dropna().unique())
            ax.set_xticks(epochs_present)
            ax.set_xlim(min(epochs_present) - 0.6, max(epochs_present) + 0.6)
        ax.set_ylabel("CER (lower is better)")
        ax.set_title(model_display_name(model))
        ax.margins(y=0.18)
        # CER can never be negative -- margins() extends symmetrically for annotation headroom
        # above the highest point, but the same symmetric extension below 0 would be dishonest
        # for a metric that's bounded at zero. Keep the auto top, clamp the bottom explicitly.
        ax.set_ylim(bottom=0)

    notes = ["line style shows dataset version (solid = v5, dashed = v2; Qwen has only v5 data)"]
    for model, mgroup in df.groupby("model"):
        meta = mgroup[["run_id", "dataset_version", "epochs", "steps"]].drop_duplicates()
        for (ds, ep), sub in meta.groupby(["dataset_version", "epochs"]):
            steps_here = sorted(sub["steps"].dropna().unique())
            if len(steps_here) > 1:
                steps_str = " vs. ".join(str(int(s)) for s in steps_here)
                notes.append(f"{model_display_name(model)}'s {dataset_short_label(ds)}, "
                              f"{int(ep)}-epoch runs differ in step count ({steps_str} steps = "
                              f"different effective batch size), not duplicate configs")
    fig.text(0.5, -0.05, "Note: " + "; ".join(notes) + ".", ha="center", fontsize=8, style="italic")

    fig.suptitle("Per-label CER vs. epoch count (one panel per model, own y-scale)", y=1.14)
    if handles_by_label:
        ordered_labels = sorted(handles_by_label)
        fig.legend([handles_by_label[l] for l in ordered_labels], ordered_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(ordered_labels), fontsize=9)
    fig.tight_layout()
    save_all_formats(fig, out_path)
    plt.close(fig)


def plot_parse_failure_rate_by_run(df: pd.DataFrame, out_path: Path) -> None:
    """One point per run (deduplicated by run_id -- json_parse_failures/n_validation_rows is a
    run-level number, repeated across that run's per-label rows in the flat CSV). x = EPOCH
    COUNT, the actual sweep variable -- not run order. This is the single most load-bearing
    chart in the project for showing that 3 epochs is a non-monotonic optimum for BOTH models
    independently: with epoch count on the x-axis, both models' V-shapes land at the same x
    position and the pattern is visible at a glance, instead of requiring a reader to read every
    x-tick label to notice it (the earlier run-order x-axis put Gemma's 3-epoch point and Qwen's
    3-epoch point several slots apart for no reason other than list order).

    Color = model (see _report_style.MODEL_COLOR, consistent with real_doc_comparison.png).
    Linestyle = dataset version (solid v5, dashed v2 -- Qwen has only v5 data, so its line is
    always solid). A same-model/same-epoch pair that used a different effective batch size
    (Qwen's two 3-epoch runs, 39 vs. 78 steps) or is a same-config rerun (Gemma's v2-run2/run3)
    would otherwise land on the exact same point -- jittered apart via _jitter_duplicate_epochs
    and annotated with the disambiguating detail instead of silently overlapping."""
    per_run = df.drop_duplicates(subset=["run_id"]).copy()
    label_lookup = _run_label_lookup(df)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    if per_run.empty or per_run["n_validation_rows"].isna().all():
        ax.text(0.5, 0.5, "no run data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        per_run["failure_rate"] = per_run["json_parse_failures"] / per_run["n_validation_rows"]
        per_run["dataset_short"] = per_run["dataset_version"].apply(dataset_short_label)

        legend_handles: dict[str, object] = {}
        # Collected across all (model, dataset-version) lines, then placed once via
        # _place_annotations -- grouped by which epoch they actually collide at (e.g. Gemma's
        # v2 5-epoch run and Qwen's v5 5-epoch run both sit at (5, 1.0)), not offset by a fixed
        # per-line amount regardless of how isolated the point is.
        annotation_entries: list[tuple[float, float, str, float]] = []
        for (model, ds), group in per_run.groupby(["model", "dataset_short"]):
            group = group.sort_values("epochs")
            xs = _jitter_duplicate_epochs(group)
            line, = ax.plot(xs.values, group["failure_rate"], marker=model_marker(model),
                             color=model_color(model), linestyle=dataset_linestyle(ds),
                             markeredgecolor="0.15", markeredgewidth=0.6)
            legend_handles[f"{model_display_name(model)} — {ds}"] = line
            for run_id, x, y, ep in zip(group["run_id"], xs.values, group["failure_rate"], group["epochs"]):
                disamb = _epoch_disambiguator(run_id, label_lookup)
                annotation_entries.append((x, y, disamb, float(ep)))
        _place_annotations(ax, annotation_entries)

        ordered_keys = sorted(legend_handles)
        ax.legend([legend_handles[k] for k in ordered_keys], ordered_keys,
                   fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        epochs_present = sorted(per_run["epochs"].dropna().unique())
        ax.set_xticks(epochs_present)
        ax.set_xlim(min(epochs_present) - 0.75, max(epochs_present) + 0.75)
    ax.set_ylabel("JSON parse-failure rate")
    ax.set_xlabel("epochs")
    ax.set_title("JSON parse-failure rate vs. epoch count")
    # Top clamp well above 1.0 (not 1.05): with up to 3 stacked annotation tiers on points
    # that sit at failure_rate=1.0 (a common value -- several runs are 100% failures), the
    # highest tier's point-offset text needs real data-space headroom below the title, or it
    # renders on top of the title text (annotate() isn't clipped to the axes by default).
    ax.set_ylim(-0.05, 1.35)
    fig.tight_layout()
    save_all_formats(fig, out_path)
    plt.close(fig)


def plot_loss_by_run(loss_df: pd.DataFrame, out_path: Path, run_meta: pd.DataFrame,
                      label_lookup: dict[str, str] | None = None) -> None:
    """One panel per model (Gemma left, Qwen right), one line per run within it, from
    eval/loss_history.csv's long-format {model, run, step, loss} rows -- the actual per-step
    training-loss curve, not just the start/end summary logged in the narrative run logs.
    Faceting by model halves the line count each panel has to hold (6 -> up to 6 total but
    split, 4 for Qwen) and matches plot_cer_by_run's existing per-model-panel layout so the
    figure set reads as one consistent design instead of a one-off layout per chart. Linestyle
    within a panel still encodes dataset version (solid v5, dashed v2), the same convention used
    in the other two trend charts.

    This is a convergence sanity check, not a quality signal: a clean drop here does not imply
    good generalization (Gemma v2-run3's loss dropped just as cleanly as every other run despite
    9/9 eval JSON-parse failures -- see eval/gemma_finetune_runs.md). Use alongside
    plot_cer_by_run/plot_parse_failure_rate_by_run, never in place of them."""
    label_lookup = label_lookup or {}
    if loss_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "no loss history yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("training loss (log scale)")
        ax.set_xlabel("training step")
        ax.set_title("Training loss by step, across fine-tune runs")
        fig.tight_layout()
        save_all_formats(fig, out_path)
        plt.close(fig)
        return

    loss_df = loss_df.merge(run_meta[["run_id", "dataset_version"]], on="run_id", how="left")
    loss_df["dataset_short"] = loss_df["dataset_version"].apply(dataset_short_label)
    models = sorted(loss_df["run_id"].str.split("/").str[0].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5.5), squeeze=False, sharey=True)
    axes = axes[0]

    for ax, model in zip(axes, models):
        model_loss = loss_df[loss_df["run_id"].str.startswith(model + "/")]
        run_ids = sorted(model_loss["run_id"].unique())
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_by_run = {r: color_cycle[i % len(color_cycle)] for i, r in enumerate(run_ids)}
        # Sparse markers with a dark edge, not just a colored line -- confirmed via an actual
        # grayscale conversion that a high-luminance palette color (Okabe-Ito's yellow) nearly
        # disappears as a plain line against white. A marker every ~10 steps, per-run shape,
        # gives the same color+shape redundancy the other two trend charts already have.
        marker_by_run = {r: MARKERS[i % len(MARKERS)] for i, r in enumerate(run_ids)}
        for run_id, group in model_loss.groupby("run_id"):
            group = group.sort_values("step")
            ds = group["dataset_short"].iloc[0]
            display = label_lookup.get(run_id, run_id).replace("\n", ", ")
            ax.plot(group["step"], group["loss"], color=color_by_run[run_id],
                    linestyle=dataset_linestyle(ds), label=display,
                    marker=marker_by_run[run_id], markevery=10, markersize=5,
                    markeredgecolor="0.15", markeredgewidth=0.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_yscale("log")
        ax.set_title(model_display_name(model))
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.3, which="both")

    axes[0].set_ylabel("training loss (log scale)")
    fig.suptitle("Training loss by step (one panel per model, solid = v5 / dashed = v2)", y=1.02)
    fig.tight_layout()
    save_all_formats(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fine-tune run-log trends from one or more CSVs.")
    parser.add_argument("csvs", type=Path, nargs="+",
                        help="One or more structured run-log CSVs, e.g. eval/gemma_finetune_runs.csv "
                             "eval/qwen_finetune_runs.csv")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Model name per CSV, same order as csvs (default: inferred from "
                             "filename, stripping a trailing '_finetune_runs')")
    parser.add_argument("--loss-csv", type=Path, default=Path("eval/loss_history.csv"),
                        help="Long-format {model,run,step,loss} CSV (default eval/loss_history.csv); "
                             "skipped if the file doesn't exist")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures/finetune_eval"),
                        help="Folder to write PNG charts into (default docs/figures/finetune_eval)")
    args = parser.parse_args()

    apply_report_style()
    df = load_combined(args.csvs, args.labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cer_by_run(df, args.out_dir / "cer_by_run.png")
    plot_parse_failure_rate_by_run(df, args.out_dir / "parse_failure_rate_by_run.png")
    written = [args.out_dir / "cer_by_run.png", args.out_dir / "cer_by_run.pdf",
               args.out_dir / "parse_failure_rate_by_run.png", args.out_dir / "parse_failure_rate_by_run.pdf"]

    if args.loss_csv.is_file():
        loss_df = pd.read_csv(args.loss_csv)
        loss_df["run_id"] = loss_df["model"] + "/" + loss_df["run"].astype(str)
        run_meta = df.drop_duplicates(subset=["run_id"])[["run_id", "dataset_version"]]
        plot_loss_by_run(loss_df, args.out_dir / "loss_by_run.png", run_meta=run_meta,
                          label_lookup=_run_label_lookup(df))
        written += [args.out_dir / "loss_by_run.png", args.out_dir / "loss_by_run.pdf"]

    print(f"Wrote {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
