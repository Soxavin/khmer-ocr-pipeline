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
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _report_style import (
    ANNOTATION_SIZE,
    HIGHLIGHT_EPOCH,
    MARKERS,
    NOTE_SIZE,
    apply_report_style,
    content_label_color,
    content_label_marker,
    dataset_linestyle,
    dataset_short_label,
    epoch_color,
    highlight_epoch,
    model_color,
    model_display_name,
    model_marker,
    run_display_label,
    save_all_formats,
    titled,
)

_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = MARKERS

# The dataset version the epoch sweep was actually run as a controlled sweep on. v2 runs predate
# it (smaller dataset, and a ground-truth typo was fixed partway through -- see
# eval/gemma_finetune_runs.md), so they're kept on the charts for completeness but drawn
# recessively: at full weight a v2 line reads as a peer of the v5 sweep and invites exactly the
# wrong conclusion on a slide, since v2's 2-epoch point is the lowest failure rate anywhere on
# the chart while measuring a different dataset under different ground truth.
_PRIMARY_DATASET = "v5"
# Two weights, because the same de-emphasis does different work on the two epoch-axis charts.
# On the parse-failure chart the v2 line actively competes with the finding (its 2-epoch point is
# the lowest failure rate anywhere on the chart, measured on a different dataset under different
# ground truth), so it's pushed well back. On the CER chart it carries most of Gemma's data and
# the colors there encode content region rather than model, so the same weight would leave that
# panel looking empty for no gain.
_SECONDARY_ALPHA = 0.4
_SECONDARY_ALPHA_CER = 0.7

# Figure-width footnotes are laid out as one long line by default, and savefig(bbox="tight")
# then widens the whole canvas to fit it -- which silently stretched the CER figure to a 2.7:1
# letterbox and squashed both panels. Wrapped instead, so the note grows downward.
_NOTE_WRAP = 128

# At or below this many matched pages, a per-label CER average is one or two documents' worth of
# evidence, not a trend -- drawn hollow on the CER chart so the distinction is visible without
# reading the n= label. Deliberately not a statistical threshold; it's a "read this point
# cautiously" flag, and the README says so.
_THIN_SAMPLE_N = 3


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
                        base_dy: float = 9, min_gap: float = 18, dx: float = 12) -> None:
    """entries: (x, y, text, epoch_bucket) tuples. Groups by epoch_bucket -- the actual epoch
    count, not the jittered x -- since every observed annotation collision in these charts
    happens between points that share an epoch. Every annotation gets a thin leader line back to
    its real point (arrowprops), so a label nudged upward to clear a neighbour stays visually
    tied to its marker -- an earlier version offset each LINE by a fixed amount regardless of how
    isolated its points were, which pushed isolated points' labels far from their marker with
    nothing connecting them, reading as disconnected floating text rather than an annotation.

    Within a bucket, labels are decluttered greedily in *display* space rather than assigned
    fixed tier offsets by rank: a fixed-tier scheme adds the same offset to the 2nd-lowest point
    whether it sits next to its neighbour or half a panel away, and at 5 epochs that pushed
    Gemma's label off its own point and onto Qwen's marker at a different failure rate -- a label
    that names the wrong series is worse than a crowded one. Each label instead asks for
    `base_dy` above its own point and is only raised if that would land within `min_gap` of the
    previous label in the same bucket.

    MUST be called after the axes' final limits and layout are set (see callers): the display
    transform used to measure those gaps is only meaningful once the axes has its final size.
    base_dy/min_gap are in points and scale with ANNOTATION_SIZE, which was raised from 6.5pt
    (fine on a report page, unreadable projected) to ~9.5pt."""
    from collections import defaultdict
    buckets: dict[float, list[tuple[float, float, str]]] = defaultdict(list)
    for x, y, text, epoch in entries:
        buckets[epoch].append((x, y, text))
    px_per_pt = ax.figure.dpi / 72.0
    for pts in buckets.values():
        prev_top_px = None
        for x, y, text in sorted(pts, key=lambda t: t[1]):
            point_px = ax.transData.transform((x, y))[1]
            label_px = point_px + base_dy * px_per_pt
            if prev_top_px is not None:
                label_px = max(label_px, prev_top_px + min_gap * px_per_pt)
            # A multi-line label grows upward from its anchor, so the next label has to clear the
            # whole block, not just the anchor line.
            extra_lines = text.count("\n")
            prev_top_px = label_px + extra_lines * ANNOTATION_SIZE * 1.3 * px_per_pt
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(dx, (label_px - point_px) / px_per_pt),
                        fontsize=ANNOTATION_SIZE, color="0.25",
                        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6, shrinkA=0, shrinkB=3))


def _ambiguous_epochs(group_keys: pd.DataFrame, key_cols: list[str]) -> set:
    """The (…, epochs) buckets that hold more than one distinct run, i.e. the only places where a
    step-count annotation actually disambiguates anything. Used to keep per-point labels
    selective: annotating every point with its full '3ep, 78 steps' provenance duplicates the
    x-axis on isolated points and, at projector-legible type, turns the chart into a wall of
    grey text. Where a bucket holds one run, position alone already identifies it."""
    counts = group_keys.drop_duplicates().groupby(key_cols, dropna=False).size()
    return {k for k, n in counts.items() if n > 1}


def _no_cer_epochs(model_runs: pd.DataFrame, scored: pd.DataFrame) -> dict[float, str]:
    """Epochs this model was run at but has no scorable CER for -> the reason, stated only as far
    as the run log supports it. An epoch column that is simply blank reads as "we didn't try that"
    when in fact it's "we tried it and nothing the model emitted could be scored" -- which for
    Qwen's 2- and 5-epoch runs is the actual finding, not an omission."""
    scored_epochs = {float(e) for e in scored["epochs"].dropna().unique()}
    reasons: dict[float, str] = {}
    for epoch, runs in model_runs.groupby("epochs"):
        if float(epoch) in scored_epochs:
            continue
        failures = runs["json_parse_failures"].sum(min_count=1)
        rows = runs["n_validation_rows"].sum(min_count=1)
        total_failure = pd.notna(failures) and pd.notna(rows) and rows > 0 and failures == rows
        reasons[float(epoch)] = ("no CER —\nevery page failed\nto parse" if total_failure
                                  else "no CER —\nnothing scorable")
    return reasons


def _run_provenance(row: pd.Series, peers: pd.DataFrame) -> str:
    """Short 'which run is this point' label for an epoch-axis chart: dataset version + step
    count, e.g. 'v5, 130 steps'. Epoch count is deliberately omitted (the x-position already
    states it) and a date is appended only when a peer row on the same line shares BOTH epoch
    and step count -- Gemma's v2-run2/v2-run3 are a genuine identical-config repeat around a
    ground-truth typo fix, so the date is the only thing that tells them apart."""
    ds = dataset_short_label(row["dataset_version"])
    parts = [ds]
    if pd.notna(row.get("steps")):
        parts.append(f"{int(float(row['steps']))} steps")
    text = ", ".join(parts)
    same_config = peers[(peers["epochs"] == row["epochs"]) & (peers["steps"] == row["steps"])]
    if len(same_config) > 1 and pd.notna(row.get("date")):
        text += f"\n{row['date']}"
    return text


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
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 6.0), squeeze=False)
    axes = axes[0]
    # Collected across panels for one shared external legend instead of a per-panel one. Colors
    # and markers now come from a fixed label->style dict (_report_style.CONTENT_LABEL_COLOR),
    # not from each panel's own cycle position, so a label missing from one panel (Qwen never
    # produced a scorable "Text" region) can't shift the remaining labels onto different colors
    # than the neighbouring panel gives them -- the shared legend is then correct by
    # construction rather than by an alphabetical-ordering coincidence.
    handles_by_label: dict[str, object] = {}
    pending_annotations: list[tuple[object, list]] = []

    for ax, model in zip(axes, models):
        model_data = labeled[labeled["model"] == model]
        model_runs = df[df["model"] == model].drop_duplicates(subset=["run_id"])
        # Every epoch this model was actually RUN at, including ones with no scorable CER --
        # dropping those from the axis silently hid two thirds of Qwen's sweep and made its panel
        # look like a single-epoch experiment. See _no_cer_epochs for what gets drawn there.
        epochs_run = sorted(model_runs["epochs"].dropna().unique())
        if model_data.empty:
            ax.text(0.5, 0.5, "no CER data yet", ha="center", va="center", transform=ax.transAxes)
        else:
            highlight_epoch(ax, HIGHLIGHT_EPOCH)
            ambiguous = _ambiguous_epochs(model_data[["label", "epochs", "run_id"]],
                                          ["label", "epochs"])
            annotation_entries: list[tuple[float, float, str, float]] = []
            for i, ((label, ds), group) in enumerate(model_data.groupby(["label", "dataset_short"])):
                group = group.sort_values("epochs")
                xs = _jitter_duplicate_epochs(group)
                color = content_label_color(label, i)
                primary = ds == _PRIMARY_DATASET
                # markeredgecolor: a colorblind-safe palette still relies on hue, which grayscale
                # collapses; a dark edge keeps every marker readable whatever its fill.
                line, = ax.plot(xs.values, group["label_cer"], marker=content_label_marker(label, i),
                                 color=color, linestyle=dataset_linestyle(ds),
                                 label=label, markeredgecolor="0.15", markeredgewidth=0.6,
                                 linewidth=2.4 if primary else 1.6,
                                 markersize=10 if primary else 7,
                                 alpha=1.0 if primary else _SECONDARY_ALPHA_CER,
                                 zorder=3 if primary else 2)
                handles_by_label.setdefault(label, line)
                # A point averaged over 1-3 pages is drawn hollow. The n= text below says the same
                # thing, but only if the viewer reads it -- on a slide the shape difference is
                # what actually lands, and it's the difference between "this line dropped" and
                # "this line dropped according to one page." Encoded by fill, not color, so it
                # survives grayscale and CVD alongside everything else here.
                thin = group[group["label_n_matched"] <= _THIN_SAMPLE_N]
                if not thin.empty:
                    ax.plot(xs[thin.index].values, thin["label_cer"], linestyle="none",
                            marker=content_label_marker(label, i), markersize=10 if primary else 7,
                            markerfacecolor="white", markeredgecolor=color, markeredgewidth=2.0,
                            alpha=1.0 if primary else _SECONDARY_ALPHA_CER, zorder=4)
                for _, row in group.iterrows():
                    n = row["label_n_matched"]
                    if pd.isna(n):
                        continue
                    text = f"n={int(n)}"
                    # Step count only where two runs of this label share an epoch (Qwen's two
                    # 3-epoch runs at different effective batch sizes) -- elsewhere the x-position
                    # already identifies the run, and repeating it on every point turned the panel
                    # into a wall of grey provenance text at projector-legible type.
                    if (row["label"], row["epochs"]) in ambiguous and pd.notna(row.get("steps")):
                        text += f" · {int(float(row['steps']))} steps"
                    annotation_entries.append((xs[row.name], row["label_cer"], text,
                                               float(row["epochs"])))
            pending_annotations.append((ax, annotation_entries))
            ax.set_xlabel("training epochs")
            ax.set_xticks(epochs_run)
            # Asymmetric right margin: annotations are placed to the RIGHT of their point, so the
            # rightmost epoch's labels ran past the panel and into the gutter between panels.
            ax.set_xlim(min(epochs_run) - 0.7, max(epochs_run) + 1.1)
        ax.set_ylabel("CER (lower is better)")
        ax.set_title(model_display_name(model))
        ax.margins(y=0.20)
        # CER can never be negative -- margins() extends symmetrically for annotation headroom
        # above the highest point, but the same symmetric extension below 0 would be dishonest
        # for a metric that's bounded at zero. Keep the auto top, clamp the bottom explicitly.
        ax.set_ylim(bottom=0)
        for epoch, reason in _no_cer_epochs(model_runs, model_data).items():
            # An empty column is ambiguous between "not run" and "run, nothing scorable"; the
            # difference is the entire point of the 2- and 5-epoch Qwen runs, so it's stated.
            ax.text(epoch, 0.06, reason, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=ANNOTATION_SIZE, color="0.35", style="italic",
                    linespacing=1.4)

    notes = ["line style = dataset version (solid = v5, the controlled sweep; dashed = v2, an "
             "earlier/smaller dataset, faded); hollow marker = averaged over "
             f"{_THIN_SAMPLE_N} pages or fewer"]
    for model, mgroup in df.groupby("model"):
        meta = mgroup[["run_id", "dataset_version", "epochs", "steps"]].drop_duplicates()
        for (ds, ep), sub in meta.groupby(["dataset_version", "epochs"]):
            steps_here = sorted(sub["steps"].dropna().unique())
            if len(steps_here) > 1:
                steps_str = " vs. ".join(str(int(s)) for s in steps_here)
                notes.append(f"{model_display_name(model)}'s {dataset_short_label(ds)}, "
                              f"{int(ep)}-epoch runs differ in step count ({steps_str} steps = "
                              f"different effective batch size), not duplicate configs")
    fig.text(0.5, -0.06, textwrap.fill("Note: " + "; ".join(notes) + ".", _NOTE_WRAP),
             ha="center", va="top", fontsize=NOTE_SIZE, style="italic", color="0.35",
             linespacing=1.5)

    titled(fig, "Per-label CER vs. epoch count (one panel per model, own y-scale)",
           "n = pages behind each average. It shrinks as parse failures rise, so the "
           "worst-performing runs are also the ones resting on the least data.",
           y=1.16)
    if handles_by_label:
        ordered_labels = sorted(handles_by_label)
        fig.legend([handles_by_label[l] for l in ordered_labels], ordered_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(ordered_labels))
    fig.tight_layout()
    # After tight_layout so the display-space declutter measures each axes' final geometry.
    # Wider dx/min_gap than the parse-failure chart: this chart's points cluster much more tightly
    # (all five of Qwen's scorable points sit at 3 epochs), so a label offset that clears its own
    # marker still landed on a *neighbouring* label's marker at the default spacing.
    for ax, entries in pending_annotations:
        _place_annotations(ax, entries, min_gap=20, dx=18)
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
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    annotation_entries: list[tuple[float, float, str, float]] = []
    if per_run.empty or per_run["n_validation_rows"].isna().all():
        ax.text(0.5, 0.5, "no run data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        per_run["failure_rate"] = per_run["json_parse_failures"] / per_run["n_validation_rows"]
        per_run["dataset_short"] = per_run["dataset_version"].apply(dataset_short_label)

        # Drawn before any series so the band sits behind the lines.
        highlight_epoch(ax, HIGHLIGHT_EPOCH, caption="lowest failure rate\nfor both models (v5)")

        # Only buckets holding more than one run get per-point provenance labels -- see
        # _ambiguous_epochs. Here that's exactly the places a viewer would otherwise ask "why are
        # there two Gemma points at 2 epochs?", and nowhere else.
        ambiguous = _ambiguous_epochs(per_run[["model", "epochs", "run_id"]], ["model", "epochs"])

        # _jitter_duplicate_epochs only separates points *within* one line, so two different
        # models landing on the identical (epoch, rate) still drew on top of each other -- at
        # (2 epochs, 1.0) Qwen's square completely hid Gemma's circle, silently turning "both
        # models are at 100% failure here" into "only Qwen is". A small constant per-model
        # x-offset makes coincident points visible as two points. It's cosmetic and well inside
        # the 3-epoch highlight band's width, so it never moves a point to a different epoch.
        models_present = sorted(per_run["model"].unique())
        model_dx = {m: ((i - (len(models_present) - 1) / 2) * 0.07)
                    for i, m in enumerate(models_present)}

        legend_handles: dict[str, object] = {}
        # Collected across all (model, dataset-version) lines, then placed once via
        # _place_annotations after layout is final -- grouped by which epoch they actually
        # collide at (e.g. Gemma's v2 5-epoch run and Qwen's v5 5-epoch run both sit at (5, 1.0)).
        for (model, ds), group in per_run.groupby(["model", "dataset_short"]):
            group = group.sort_values("epochs")
            xs = _jitter_duplicate_epochs(group) + model_dx[model]
            primary = ds == _PRIMARY_DATASET
            line, = ax.plot(xs.values, group["failure_rate"], marker=model_marker(model),
                             color=model_color(model), linestyle=dataset_linestyle(ds),
                             linewidth=2.8 if primary else 1.8,
                             markersize=11 if primary else 7,
                             alpha=1.0 if primary else _SECONDARY_ALPHA,
                             zorder=3 if primary else 2,
                             markeredgecolor="0.15", markeredgewidth=0.6)
            key = (f"{model_display_name(model)} — {ds}" if primary
                   else f"{model_display_name(model)} — {ds} (earlier dataset)")
            legend_handles[key] = (line, primary)
            for _, row in group.iterrows():
                if (row["model"], row["epochs"]) not in ambiguous:
                    continue
                x = xs[row.name]
                annotation_entries.append((x, row["failure_rate"],
                                            _run_provenance(row, group), float(row["epochs"])))

        # Primary (v5) series first in the legend, so the sweep this chart is actually about
        # reads before the recessive historical lines rather than after them alphabetically.
        ordered_keys = sorted(legend_handles, key=lambda k: (not legend_handles[k][1], k))
        ax.legend([legend_handles[k][0] for k in ordered_keys], ordered_keys,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0))
        epochs_present = sorted(per_run["epochs"].dropna().unique())
        ax.set_xticks(epochs_present)
        ax.set_xlim(min(epochs_present) - 0.75, max(epochs_present) + 0.75)
    ax.set_ylabel("JSON parse-failure rate (lower is better)")
    ax.set_xlabel("training epochs")
    # Top clamp well above 1.0 (not 1.05): with up to 3 stacked annotation tiers on points
    # that sit at failure_rate=1.0 (a common value -- several runs are 100% failures), the
    # highest tier's point-offset text needs real data-space headroom below the title, or it
    # renders on top of the title text (annotate() isn't clipped to the axes by default).
    ax.set_ylim(-0.05, 1.28)
    n_val = per_run["n_validation_rows"].dropna()
    n_note = (f" Each point is one run scored on {int(n_val.iloc[0])} validation pages."
              if len(n_val) and n_val.nunique() == 1 else "")
    fig.text(0.5, -0.02, textwrap.fill(
                 "Line style = dataset version (solid = v5, the controlled sweep; dashed = v2, "
                 "an earlier/smaller dataset, shown faded for context only)." + n_note, _NOTE_WRAP),
             ha="center", va="top", fontsize=NOTE_SIZE, style="italic", color="0.35",
             linespacing=1.5)
    titled(fig, "JSON parse-failure rate vs. epoch count",
           "Both models fail least at 3 epochs on v5; 2 and 5 epochs are worse. "
           "One run per point — a direction, not a tested effect.",
           y=1.08)
    fig.tight_layout()
    # After tight_layout so the display-space declutter measures the axes' final geometry.
    _place_annotations(ax, annotation_entries)
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

    meta_cols = [c for c in ["run_id", "dataset_version", "epochs", "steps"] if c in run_meta]
    loss_df = loss_df.merge(run_meta[meta_cols], on="run_id", how="left")
    loss_df["dataset_short"] = loss_df["dataset_version"].apply(dataset_short_label)
    models = sorted(loss_df["run_id"].str.split("/").str[0].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5.8), squeeze=False,
                             sharey=True)
    axes = axes[0]
    all_epochs = sorted(loss_df["epochs"].dropna().unique())

    for ax, model in zip(axes, models):
        model_loss = loss_df[loss_df["run_id"].str.startswith(model + "/")]
        # Ordered by epoch count so the legend reads 2 -> 3 -> 5 rather than by run name, and so
        # the light-to-dark ramp below appears in ramp order.
        run_ids = sorted(model_loss["run_id"].unique(),
                         key=lambda r: (float(model_loss.loc[model_loss["run_id"] == r, "epochs"].iloc[0]), r))
        for run_id in run_ids:
            group = model_loss[model_loss["run_id"] == run_id].sort_values("step")
            ds = group["dataset_short"].iloc[0]
            epochs = group["epochs"].iloc[0]
            steps = group["steps"].iloc[0]
            # Color keyed to EPOCH COUNT, not to position in a sorted run list. The old
            # cycle-by-index scheme only happened to give both panels the same color per epoch
            # because each model's runs sorted into the same order -- add one run to either model
            # and "the green line" silently means a different epoch count in each panel. Keying it
            # to the value makes cross-panel comparison correct by construction. Epoch count is
            # ordinal, so the ramp is light->dark rather than three unrelated hues (see
            # _report_style.epoch_color); marker shape carries the same distinction for grayscale.
            color = epoch_color(epochs, all_epochs)
            marker = MARKERS[all_epochs.index(epochs) % len(MARKERS)] if epochs in all_epochs else "o"
            display = (f"{int(float(epochs))} epochs ({int(float(steps))} steps)"
                       if pd.notna(epochs) and pd.notna(steps)
                       else label_lookup.get(run_id, run_id).replace("\n", ", "))
            if dataset_short_label(ds) != _PRIMARY_DATASET:
                display += f" — {dataset_short_label(ds)}"
            ax.plot(group["step"], group["loss"], color=color,
                    linestyle=dataset_linestyle(ds), label=display,
                    marker=marker, markevery=10, markersize=6,
                    markeredgecolor="0.15", markeredgewidth=0.5)
        ax.legend(loc="upper right")
        ax.set_yscale("log")
        ax.set_title(model_display_name(model))
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.3, which="both")

    axes[0].set_ylabel("training loss (log scale)")
    # Only claim the dashed-v2 convention when a v2 run is actually on this figure -- Gemma's v2
    # runs predate per-step loss logging, so the earlier fixed subtitle described a line style
    # that appears nowhere on the chart.
    datasets_shown = sorted(loss_df["dataset_short"].dropna().unique())
    style_note = (" (solid = v5 / dashed = v2)" if len(datasets_shown) > 1 else "")
    titled(fig, f"Training loss by step, one panel per model{style_note}",
           "Every run converged cleanly — including the ones whose output was unusable. "
           "This is a training-process check, not a quality signal.",
           y=1.06)
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
        run_meta = df.drop_duplicates(subset=["run_id"])[["run_id", "dataset_version", "epochs", "steps"]]
        plot_loss_by_run(loss_df, args.out_dir / "loss_by_run.png", run_meta=run_meta,
                          label_lookup=_run_label_lookup(df))
        written += [args.out_dir / "loss_by_run.png", args.out_dir / "loss_by_run.pdf"]

    print(f"Wrote {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
