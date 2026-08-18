"""Plot a training run's learning curve from its metrics.csv.

Usage:
    uv run python experiments/khmer_crnn/plot_metrics.py experiments/khmer_crnn/runs/<run>

Writes learning_curve.png next to the metrics.csv: validation CER and training
loss vs epoch on a twin axis.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, don't open a window
import matplotlib.pyplot as plt


def plot_run(run_dir: str) -> Path:
    """Read <run_dir>/metrics.csv and save a val_cer + train_loss learning curve."""
    run = Path(run_dir)
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    epochs = [int(r["epoch"]) for r in rows]
    val_cer = [float(r["val_cer"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, val_cer, "o-", color="tab:red", label="val CER")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation CER", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(1.0, ls=":", color="grey", lw=1)  # blank-collapse ceiling

    ax2 = ax1.twinx()
    ax2.plot(epochs, loss, "s--", color="tab:blue", alpha=0.6, label="train loss")
    ax2.set_ylabel("train loss (CTC)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # annotate the epoch where CER first drops below 1.0 (the breakout)
    below = [e for e, c in zip(epochs, val_cer) if c < 1.0]
    if below:
        b = below[0]
        ax1.annotate(f"breakout @ epoch {b}", xy=(b, 1.0), xytext=(b + 1, 0.8),
                     arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    ax1.set_title(f"Learning curve — {run.name} (best CER {min(val_cer):.3f})")
    fig.tight_layout()
    out = run / "learning_curve.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    plot_run(sys.argv[1] if len(sys.argv) > 1 else "experiments/khmer_crnn/runs/converge1")
