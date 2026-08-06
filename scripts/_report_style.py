"""Shared matplotlib styling for report-facing figures (docs/figures/*.png), used by
plot_run_metrics.py and plot_real_doc_comparison.py so every chart in the report looks like
part of one consistent set rather than default-matplotlib one-offs. Print/embed-quality
concerns specifically: legible font sizes at report scale (not just on-screen), a fixed high
DPI so images stay crisp when scaled down into a document, and consistent color/grid choices.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DPI = 300

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

# Okabe-Ito palette (Okabe & Ito, 2008) -- the standard peer-reviewed colorblind-safe
# categorical palette, not a manual desaturation guess. Distinguishable under the common
# forms of color vision deficiency (protanopia/deuteranopia/tritanopia).
_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

# Distinct marker shapes, cycled alongside color -- color-vision-deficient AND true-grayscale
# rendering both collapse hue distinctions that a colorblind-safe palette alone doesn't fully
# protect against (grayscale keeps only luminance, not hue), so shape carries the same
# information redundantly.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Fixed color/marker per approach or model, used consistently across every chart in the
# fine-tune-eval set (real_doc_comparison.png, parse_failure_rate_by_run.png) -- so "blue"
# always means Gemma and "green" always means Qwen regardless of which figure a reader is
# looking at, instead of each script picking colors independently by cycle position.
MODEL_COLOR = {
    "ocr_pipeline_surya": _PALETTE[0],
    "gemma": _PALETTE[1],
    "qwen": _PALETTE[2],
}
MODEL_MARKER = {
    "ocr_pipeline_surya": "D",
    "gemma": "o",
    "qwen": "s",
}


def model_color(model: str) -> str:
    return MODEL_COLOR.get(model, _PALETTE[-1])


def model_marker(model: str) -> str:
    return MODEL_MARKER.get(model, "o")


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


def save_all_formats(fig, png_path) -> list[Path]:
    """Saves a PNG (report DPI) and a same-named PDF (vector, infinite-resolution) alongside
    it -- a report that gets resized or printed has a lossless option available, not just the
    fixed-DPI raster. Returns both paths written."""
    png_path = Path(png_path)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=REPORT_DPI)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]


_DATASET_VERSION_RE = re.compile(r"-v(\d+)$")


def dataset_short_label(dataset_version: str) -> str:
    """'Soxavin/ardb-sft-v5' -> 'v5', 'Soxavin/ardb-gemma-sft-v2' -> 'v2'."""
    m = _DATASET_VERSION_RE.search(str(dataset_version))
    return f"v{m.group(1)}" if m else str(dataset_version)


# Dataset-version encoding shared across the epoch-sweep charts (parse_failure_rate_by_run,
# cer_by_run, loss_by_run) -- linestyle rather than color, since color in those charts is
# already spoken for (model, content-label, or per-run identity) and a reader only ever needs
# to tell "v2 vs v5" apart, not rank them on a scale.
DATASET_LINESTYLE = {"v2": "--", "v5": "-"}


def dataset_linestyle(dataset_version: str) -> str:
    return DATASET_LINESTYLE.get(dataset_short_label(dataset_version), "-")


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
