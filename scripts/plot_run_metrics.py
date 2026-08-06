"""Plot fine-tune run-log trends (per-label CER, JSON-parse-failure rate) from one or more
structured CSV companions to the narrative run logs (eval/gemma_finetune_runs.md,
eval/qwen_finetune_runs.md).

Each model's runs live in their own CSV/MD pair (not merged into one file, since they track
different models with partly different columns) -- this script is what combines them into
one comparison chart, the "CER & visualize" proof artifact requested directly by the mentor,
not eyeballed from prose.

CLI:
    python scripts/plot_run_metrics.py eval/gemma_finetune_runs.csv --out-dir docs/figures
    python scripts/plot_run_metrics.py eval/gemma_finetune_runs.csv eval/qwen_finetune_runs.csv \
        --out-dir docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _report_style import (
    REPORT_DPI,
    apply_report_style,
    dataset_short_label,
    model_display_name,
    run_display_label,
)

_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v"]


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
    """One panel per model, one line per label within it, CER vs. run (skipping rows with no
    computable CER, e.g. a run where every validation row failed to parse). Split into
    per-model panels rather than one shared axis for two reasons: (1) a single Qwen outlier
    (Page-Furniture CER 5.18 on v5-run1) would otherwise compress every other, more relevant
    point into an unreadable cluster near zero on a shared y-axis; (2) each model's own panel
    can legibly label its x-axis with dataset version + epoch count (see _run_label_lookup)
    without a combined axis getting overcrowded."""
    labeled = df.dropna(subset=["label_cer"])
    label_lookup = _run_label_lookup(df)
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5.5), squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        model_data = labeled[labeled["model"] == model]
        if model_data.empty:
            ax.text(0.5, 0.5, "no CER data yet", ha="center", va="center", transform=ax.transAxes)
        else:
            labels_present = sorted(model_data["label"].unique())
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color_by_label = {lbl: color_cycle[i % len(color_cycle)] for i, lbl in enumerate(labels_present)}
            # Order runs by DATASET VERSION first, then epoch count within it -- not epoch
            # count alone across both versions. v2 and v5 are different datasets (different
            # size/split logic), so interleaving e.g. v2/2ep, v5/3ep, v2/5ep would make it look
            # like a single controlled epoch sweep when two things are actually changing at
            # once between adjacent points.
            run_meta = model_data[["run_id", "dataset_version", "epochs"]].drop_duplicates()
            run_meta = run_meta.sort_values(["dataset_version", "epochs"])
            run_order = run_meta["run_id"].tolist()
            x_pos = {r: i for i, r in enumerate(run_order)}
            for label_idx, (label, group) in enumerate(model_data.groupby("label")):
                group = group.set_index("run_id").reindex(run_order).dropna(subset=["label_cer"]).reset_index()
                xs = [x_pos[r] for r in group["run_id"]]
                ax.plot(xs, group["label_cer"], marker="o", color=color_by_label[label], label=label)
                # Sample-size annotation on every point: a single-sample point (e.g. Gemma's
                # v5/5ep run, where 8/9 rows failed to parse and the one that did was missing
                # most of its regions) looks identical to a well-sampled point otherwise --
                # nothing on the line itself signals "this is one row, not sixteen." Offset
                # varies by label's position among this panel's labels, not a fixed (6, 6) --
                # near-tied CER values (e.g. Gemma's Table/Text both sitting near 0 on the same
                # run) would otherwise stack their annotations illegibly on top of each other.
                y_offset = 6 + 13 * (label_idx % 3)
                for x, y, n in zip(xs, group["label_cer"], group["label_n_matched"]):
                    if pd.notna(n):
                        ax.annotate(f"n={int(n)}", (x, y), textcoords="offset points",
                                    xytext=(6, y_offset), fontsize=6.5, color=color_by_label[label])
            ax.legend(fontsize=8, loc="best")
            ax.set_xticks(range(len(run_order)))
            ax.set_xticklabels([label_lookup.get(r, r) for r in run_order])
            prev_ds = None
            for i, run_id in enumerate(run_order):
                ds = run_meta.set_index("run_id").loc[run_id, "dataset_version"]
                if prev_ds is not None and ds != prev_ds:
                    ax.axvline(i - 0.5, color="0.75", linestyle=":", linewidth=1, zorder=0)
                prev_ds = ds
        ax.set_ylabel("CER (lower is better)")
        ax.set_title(model_display_name(model))
        ax.margins(y=0.18)

    # Auto-detected caption for same-model, same-epoch-count, different-step-count runs (e.g.
    # Qwen v5-run1 vs v5-run2: both "3 epochs", but GRAD_ACCUM=4 vs 2 gives 39 vs 78 steps --
    # a real different-config comparison, not a duplicate run). Labels already show step count,
    # but two points both reading "3ep" first is an easy thing to misread as redundant --
    # spell out the reason in words rather than relying on the reader to notice the step delta.
    notes = []
    for model, mgroup in df.groupby("model"):
        meta = mgroup[["run_id", "dataset_version", "epochs", "steps"]].drop_duplicates()
        for (ds, ep), sub in meta.groupby(["dataset_version", "epochs"]):
            steps_here = sorted(sub["steps"].dropna().unique())
            if len(steps_here) > 1:
                steps_str = " vs. ".join(str(int(s)) for s in steps_here)
                notes.append(f"{model_display_name(model)}'s {dataset_short_label(ds)}, "
                              f"{int(ep)}-epoch runs differ in step count ({steps_str} steps = "
                              f"different effective batch size), not duplicate configs")
    if notes:
        fig.text(0.5, -0.05, "Note: " + "; ".join(notes) + ".", ha="center", fontsize=8, style="italic")

    fig.suptitle("Per-label CER across fine-tune runs (one panel per model, own y-scale)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI)
    plt.close(fig)


def plot_parse_failure_rate_by_run(df: pd.DataFrame, out_path: Path) -> None:
    """One point per run (deduplicated by run_id -- json_parse_failures/n_validation_rows is
    a run-level number, repeated across that run's per-label rows in the flat CSV), one line
    per model so a Gemma/Qwen comparison stays visually separable. Runs are ordered by epoch
    count within each model (not alphabetically by run_id), and each dataset-version segment
    is separated by a vertical divider -- v2 and v5 are different datasets (different size,
    different split logic), not consecutive points on the same sweep, and a reader shouldn't
    be able to mistake the v2-run3 -> v5-run4 jump for a training-duration effect."""
    per_run = df.drop_duplicates(subset=["run_id"]).copy()
    label_lookup = _run_label_lookup(df)
    # Wide figure + rotated labels: up to 10 runs share this one axis (unlike the per-model-
    # panel CER chart), and each label now carries dataset+epochs+steps -- too much text to
    # fit horizontally at this density without either.
    fig, ax = plt.subplots(figsize=(max(10, 1.35 * len(per_run)), 6))
    if per_run.empty or per_run["n_validation_rows"].isna().all():
        ax.text(0.5, 0.5, "no run data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        per_run["failure_rate"] = per_run["json_parse_failures"] / per_run["n_validation_rows"]
        models = sorted(per_run["model"].unique())
        marker_by_model = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(models)}

        # Global x-order: group by model, then by dataset version, then by epoch count, so
        # each model's sweep reads left-to-right and version segments stay contiguous.
        per_run["dataset_short"] = per_run["dataset_version"].astype(str)
        ordered = per_run.sort_values(["model", "dataset_short", "epochs"])
        x_order = ordered["run_id"].tolist()
        x_pos = {r: i for i, r in enumerate(x_order)}

        for model, group in per_run.groupby("model"):
            group = group.set_index("run_id").reindex([r for r in x_order if r in group["run_id"].values])
            xs = [x_pos[r] for r in group.index]
            ax.plot(xs, group["failure_rate"], marker=marker_by_model[model], label=model_display_name(model))

        # Vertical dividers between dataset-version segments (any model).
        prev_ds = None
        for i, run_id in enumerate(x_order):
            ds = ordered.iloc[i]["dataset_short"]
            if prev_ds is not None and ds != prev_ds:
                ax.axvline(i - 0.5, color="0.75", linestyle=":", linewidth=1, zorder=0)
            prev_ds = ds

        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels([label_lookup.get(r, r) for r in x_order], rotation=35, ha="right")
        if len(models) > 1:
            ax.legend(fontsize=9, loc="best")
    ax.set_ylabel("JSON parse-failure rate")
    ax.set_xlabel(None)
    ax.set_title("JSON parse-failure rate across fine-tune runs")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI)
    plt.close(fig)


def plot_loss_by_run(loss_df: pd.DataFrame, out_path: Path, label_lookup: dict[str, str] | None = None) -> None:
    """One line per (model, run) from eval/loss_history.csv's long-format {model, run, step,
    loss} rows -- the actual per-step training-loss curve, not just the start/end summary
    logged in the narrative run logs. This is a convergence sanity check, not a quality signal:
    a clean drop here does not imply good generalization (Gemma v2-run3's loss dropped just as
    cleanly as every other run despite 9/9 eval JSON-parse failures -- see
    eval/gemma_finetune_runs.md). Use alongside plot_cer_by_run/plot_parse_failure_rate_by_run,
    never in place of them."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if loss_df.empty:
        ax.text(0.5, 0.5, "no loss history yet", ha="center", va="center", transform=ax.transAxes)
    else:
        label_lookup = label_lookup or {}
        run_ids = sorted(loss_df["run_id"].unique())
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        linestyle_by_model = {}
        color_by_run = {r: color_cycle[i % len(color_cycle)] for i, r in enumerate(run_ids)}
        for run_id, group in loss_df.groupby("run_id"):
            model = run_id.split("/")[0]
            linestyle_by_model.setdefault(model, _LINESTYLES[len(linestyle_by_model) % len(_LINESTYLES)])
            group = group.sort_values("step")
            display = label_lookup.get(run_id, run_id).replace("\n", ", ")
            ax.plot(group["step"], group["loss"], color=color_by_run[run_id],
                    linestyle=linestyle_by_model[model], label=f"{model_display_name(model)} — {display}")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.set_yscale("log")
    ax.set_ylabel("training loss (log scale)")
    ax.set_xlabel("training step")
    ax.set_title("Training loss by step, across fine-tune runs")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI)
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
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"),
                        help="Folder to write PNG charts into (default docs/figures)")
    args = parser.parse_args()

    apply_report_style()
    df = load_combined(args.csvs, args.labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cer_by_run(df, args.out_dir / "cer_by_run.png")
    plot_parse_failure_rate_by_run(df, args.out_dir / "parse_failure_rate_by_run.png")
    written = [args.out_dir / "cer_by_run.png", args.out_dir / "parse_failure_rate_by_run.png"]

    if args.loss_csv.is_file():
        loss_df = pd.read_csv(args.loss_csv)
        loss_df["run_id"] = loss_df["model"] + "/" + loss_df["run"].astype(str)
        plot_loss_by_run(loss_df, args.out_dir / "loss_by_run.png", label_lookup=_run_label_lookup(df))
        written.append(args.out_dir / "loss_by_run.png")

    print(f"Wrote {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
