"""Shared matplotlib styling for report-facing figures (docs/figures/*.png), used by
plot_run_metrics.py and plot_real_doc_comparison.py so every chart in the report looks like
part of one consistent set rather than default-matplotlib one-offs. Print/embed-quality
concerns specifically: legible font sizes at report scale (not just on-screen), a fixed high
DPI so images stay crisp when scaled down into a document, and consistent color/grid choices.

These figures have TWO consumption contexts, and the type scale here is set for the harder of
the two: a projected presentation slide read from across a room for 30-60 seconds with no
surrounding prose, rather than a report page a reader can lean into. A single set of files
serves both -- a report embeds these PNGs scaled down to page/column width, which shrinks
type further, so sizing for the projector improves the report render too. Two separate
"slide" and "report" exports were deliberately not built: it would double the artifact count
and give the README two of everything to explain, for a figure set whose only real difference
would be a scale factor.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DPI = 300

# Named sizes for the text that isn't covered by rcParams (annotations, footnotes, subtitles).
# Centralised so every chart's small print has the same weight -- these were previously
# per-chart literals that had drifted (6.5 in one script, 7 and 8 in another), which read as
# three different typographic systems across four figures that are meant to be one set.
ANNOTATION_SIZE = 9.5   # per-point labels (n=, step counts, bar values)
NOTE_SIZE = 10          # footnote line under a figure
SUBTITLE_SIZE = 11.5    # the factual one-line finding under a figure title

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


# Bar-chart counterpart to MODEL_MARKER. On the line charts, marker shape is what carries model
# identity once color is gone; a bar has no marker, and an actual grayscale conversion of
# real_doc_comparison.png showed why that matters -- Okabe-Ito's orange (L*~0.75) and sky blue
# (L*~0.72) are nearly the same luminance, so the OCR-pipeline and Gemma bars, and worse their
# legend swatches, desaturated into the same gray. Hatch fill restores a non-color distinction
# that a photocopier, a grayscale print, and full color-vision deficiency all preserve.
MODEL_HATCH = {
    "ocr_pipeline_surya": "",     # solid -- the baseline reads as the plain reference bar
    "gemma": "//",
    "qwen": "xx",
}


def model_hatch(model: str) -> str:
    return MODEL_HATCH.get(model, "")


# Fixed color/marker per content-region label, used by the per-label CER chart. Assigned from a
# fixed dict rather than by cycle position so a panel that happens to be missing a label (Qwen
# never produced a scorable "Text" region) can't shift every subsequent label onto a different
# color than the neighbouring panel gives it.
#
# Note the deliberate omission of Okabe-Ito's yellow (#F0E442), which the earlier cycle-position
# assignment landed on for "Text": at L*=0.90 it fails a colorblind-palette lightness-band check
# outright and sits at 1.29:1 contrast against white -- effectively invisible as a thin line on a
# projector, and the dark marker edge the code already added only rescued the markers, not the
# line between them. Reddish-purple (#CC79A7, also Okabe-Ito) replaces it: 2.98:1, inside the
# band. Its worst-case CVD separation against the green is ΔE 7.6 (deuteranopia), which is legal
# only with a secondary encoding -- the per-label marker shapes below are exactly that, so it
# must stay paired with them.
CONTENT_LABEL_COLOR = {
    "Page-Furniture": _PALETTE[0],   # #E69F00 orange
    "Section-Header": _PALETTE[1],   # #56B4E9 sky blue
    "Table": _PALETTE[2],            # #009E73 green
    "Text": _PALETTE[6],             # #CC79A7 reddish purple
}
CONTENT_LABEL_MARKER = {
    "Page-Furniture": "o",
    "Section-Header": "s",
    "Table": "^",
    "Text": "D",
}


def content_label_color(label: str, fallback_index: int = 0) -> str:
    return CONTENT_LABEL_COLOR.get(label, _PALETTE[(fallback_index + 4) % len(_PALETTE)])


def content_label_marker(label: str, fallback_index: int = 0) -> str:
    return CONTENT_LABEL_MARKER.get(label, MARKERS[fallback_index % len(MARKERS)])


# Epoch count is ORDINAL, not categorical, so the epoch-keyed loss chart uses a sequential
# light->dark ramp (single hue) rather than three unrelated categorical hues: "more epochs =
# darker" is readable without consulting the legend, and lightness order survives grayscale
# printing on its own. Deliberately a different hue family from MODEL_COLOR so a reader moving
# between slides never reads "orange" as meaning a model on one chart and an epoch count on the
# next. Marker shape (see MARKERS usage in the loss chart) carries the same distinction
# redundantly.
_EPOCH_RAMP = ["#6BAED6", "#3182BD", "#08519C"]


def epoch_color(epochs, all_epochs: list) -> str:
    """Position of `epochs` within the sorted sweep -> a step on the light-to-dark ramp."""
    ordered = sorted({float(e) for e in all_epochs})
    try:
        i = ordered.index(float(epochs))
    except ValueError:
        return _EPOCH_RAMP[-1]
    if len(ordered) == 1:
        return _EPOCH_RAMP[-1]
    # Spread however many epoch values exist across the ramp's endpoints.
    pos = round(i * (len(_EPOCH_RAMP) - 1) / (len(ordered) - 1))
    return _EPOCH_RAMP[pos]


# The epoch count that is the non-monotonic optimum for BOTH models independently -- the single
# most citable result of the fine-tune arc, so the epoch-axis charts mark it rather than leaving
# a viewer to infer it from the shape of two V-curves in the ~40 seconds a slide is on screen.
HIGHLIGHT_EPOCH = 3


def highlight_epoch(ax, epoch: float = HIGHLIGHT_EPOCH, half_width: float = 0.42,
                    caption: str | None = None) -> None:
    """Shades a narrow neutral band at `epoch` on an epoch-axis chart, optionally captioned just
    inside the top of the axes. Neutral gray on purpose: a hue here would read as a fourth data
    series, and gray survives both grayscale printing and color-vision deficiency without
    competing with the model/label colors. This is a pointer at a value on the x-axis, not a
    statistical annotation -- it deliberately carries no interval/confidence semantics, since
    every point behind it is a single-seed run."""
    ax.axvspan(epoch - half_width, epoch + half_width, color="#9A9A9A", alpha=0.16,
               zorder=0, linewidth=0)
    if caption:
        ax.text(epoch, 0.985, caption, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=ANNOTATION_SIZE, color="0.3", style="italic")


def titled(fig, title: str, subtitle: str | None = None, y: float = 1.0) -> None:
    """One title treatment shared by every figure in the set: a bold title that stays neutral and
    descriptive (so the PNG is still honest if reused outside this narrative), with the actual
    finding carried by an optional smaller subtitle underneath.

    The subtitle exists for the presentation case specifically: on a slide there is no README and
    no surrounding prose, so a chart whose title is purely descriptive leaves the viewer to derive
    the point unaided in well under a minute. Subtitle text must stay strictly factual about what
    the plotted data shows -- state the observed pattern, not its significance."""
    fig.suptitle(title, y=y, fontweight="bold")
    if subtitle:
        # Placed relative to the same top anchor as the title so the two move together when a
        # caller adjusts `y` for a taller/shorter figure.
        fig.text(0.5, y - 0.045, subtitle, ha="center", va="top", fontsize=SUBTITLE_SIZE,
                 color="0.28")


def apply_report_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,          # on-screen/preview only; savefig dpi is set per-call
        "savefig.dpi": REPORT_DPI,
        # Type scale is set for a projected slide, not a report page -- see module docstring.
        # Every size below is ~15-25% up from the earlier report-only scale.
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "figure.titlesize": 17,
        "legend.fontsize": 11,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=_PALETTE),
        "savefig.bbox": "tight",
        "lines.linewidth": 2.2,     # readable as a line, not a hairline, when projected
        "lines.markersize": 9,
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
