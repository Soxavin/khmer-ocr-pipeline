"""Shared matplotlib styling for report-facing figures (docs/figures/*.png), used by
plot_run_metrics.py and plot_real_doc_comparison.py so every chart in the report looks like
part of one consistent set rather than default-matplotlib one-offs. Print/embed-quality
concerns specifically: legible font sizes at report scale (not just on-screen), a fixed high
DPI so images stay crisp when scaled down into a document, and consistent color/grid choices.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt

REPORT_DPI = 200

# Report-facing display names for internal model keys ('gemma', 'qwen', as used throughout
# eval/*.csv and this project's own file-naming) -- used consistently across every chart so a
# report reader never sees the raw lowercase internal key.
MODEL_DISPLAY_NAME = {
    "gemma": "Gemma 4 E2B",
    "qwen": "Qwen3.5-0.8B",
    "ocr_pipeline_surya": "OCR pipeline (Surya)",
}


def model_display_name(model: str) -> str:
    return MODEL_DISPLAY_NAME.get(model, model)

# Muted, colorblind-friendlier alternative to matplotlib's saturated default tab10 cycle --
# same hue ordering (so existing model/label-to-color assignments don't visually shuffle),
# just desaturated/darkened slightly for a less "default chart library" look in print.
_PALETTE = [
    "#3B6FA0", "#D97F35", "#4C9F70", "#B94A48",
    "#7A5DA0", "#8C6248", "#C767A8", "#6B6B6B",
]


def apply_report_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,          # on-screen/preview only; savefig dpi is set per-call
        "savefig.dpi": REPORT_DPI,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=_PALETTE),
        "savefig.bbox": "tight",
    })


_DATASET_VERSION_RE = re.compile(r"-v(\d+)$")


def dataset_short_label(dataset_version: str) -> str:
    """'Soxavin/ardb-sft-v5' -> 'v5', 'Soxavin/ardb-gemma-sft-v2' -> 'v2'."""
    m = _DATASET_VERSION_RE.search(str(dataset_version))
    return f"v{m.group(1)}" if m else str(dataset_version)


def run_display_label(dataset_version: str, epochs, steps=None) -> str:
    """A run label a report reader can understand with no cross-referencing: dataset version +
    epoch count + step count, e.g. 'v5\\n3ep, 78 steps'. Model identity is conveyed separately
    (color/legend/panel), so it's deliberately not repeated here. Step count is included, not
    just epochs, because two runs can share the same (dataset, epochs) pair while differing in
    effective batch size (e.g. Qwen v5-run1 and v5-run2 are both '3 epochs' but 39 vs. 78
    steps) -- dropping it produces indistinguishable duplicate labels on a chart, not just a
    cosmetic gap."""
    ds = dataset_short_label(dataset_version)
    try:
        ep = int(float(epochs))
        ep_part = f"{ep}ep"
    except (TypeError, ValueError):
        return ds
    try:
        step_part = f", {int(float(steps))} steps"
    except (TypeError, ValueError):
        step_part = ""
    return f"{ds}\n{ep_part}{step_part}"
