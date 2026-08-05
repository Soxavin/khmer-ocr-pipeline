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


def plot_cer_by_run(df: pd.DataFrame, out_path: Path) -> None:
    """One line per (model, label), CER vs. run, skipping rows with no computable CER (e.g.
    a run where every validation row failed to parse). Color is consistent per label across
    models; linestyle distinguishes model, so a Gemma/Qwen comparison at the same label is
    still readable at a glance."""
    labeled = df.dropna(subset=["label_cer"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if labeled.empty:
        ax.text(0.5, 0.5, "no CER data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        models = sorted(labeled["model"].unique())
        linestyle_by_model = {m: _LINESTYLES[i % len(_LINESTYLES)] for i, m in enumerate(models)}
        labels_present = sorted(labeled["label"].unique())
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_by_label = {lbl: color_cycle[i % len(color_cycle)] for i, lbl in enumerate(labels_present)}
        for (model, label), group in labeled.groupby(["model", "label"]):
            group = group.sort_values("date") if "date" in group else group
            ax.plot(group["run_id"], group["label_cer"], marker="o",
                    linestyle=linestyle_by_model[model], color=color_by_label[label],
                    label=f"{model} — {label}")
        ax.legend(fontsize=8, loc="best")
    ax.set_ylabel("CER (lower is better)")
    ax.set_xlabel("run")
    ax.set_title("Per-label CER across fine-tune runs")
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_parse_failure_rate_by_run(df: pd.DataFrame, out_path: Path) -> None:
    """One point per run (deduplicated by run_id -- json_parse_failures/n_validation_rows is
    a run-level number, repeated across that run's per-label rows in the flat CSV), one line
    per model so a Gemma/Qwen comparison stays visually separable."""
    per_run = df.drop_duplicates(subset=["run_id"]).copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if per_run.empty or per_run["n_validation_rows"].isna().all():
        ax.text(0.5, 0.5, "no run data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        per_run["failure_rate"] = per_run["json_parse_failures"] / per_run["n_validation_rows"]
        models = sorted(per_run["model"].unique())
        marker_by_model = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(models)}
        for model, group in per_run.groupby("model"):
            group = group.sort_values("date") if "date" in group else group
            ax.plot(group["run_id"], group["failure_rate"], marker=marker_by_model[model], label=model)
        if len(models) > 1:
            ax.legend(fontsize=8, loc="best")
    ax.set_ylabel("JSON parse-failure rate")
    ax.set_xlabel("run")
    ax.set_title("JSON parse-failure rate across fine-tune runs")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fine-tune run-log trends from one or more CSVs.")
    parser.add_argument("csvs", type=Path, nargs="+",
                        help="One or more structured run-log CSVs, e.g. eval/gemma_finetune_runs.csv "
                             "eval/qwen_finetune_runs.csv")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Model name per CSV, same order as csvs (default: inferred from "
                             "filename, stripping a trailing '_finetune_runs')")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"),
                        help="Folder to write PNG charts into (default docs/figures)")
    args = parser.parse_args()

    df = load_combined(args.csvs, args.labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cer_by_run(df, args.out_dir / "cer_by_run.png")
    plot_parse_failure_rate_by_run(df, args.out_dir / "parse_failure_rate_by_run.png")
    print(f"Wrote {args.out_dir / 'cer_by_run.png'} and {args.out_dir / 'parse_failure_rate_by_run.png'}")


if __name__ == "__main__":
    main()
