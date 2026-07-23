"""Kiri OCR recognizer wrapper — per-cell Otsu binarization + CTC recognition.

Provides the thin integration layer between the vendored Kiri model and the
surya_kiri hybrid engine. Exposes four public functions:

  otsu_cell(rgb)                  — Otsu-threshold a cell crop, auto-inverting dark backgrounds
  recognize_cell(crop_rgb)        — Full pipeline for a single crop: Otsu → Kiri CTC decode → clean text
  recognize_cells(crops_rgb)      — Batched equivalent of recognize_cell over a list of crops
  recognize_cells_conf(crops_rgb) — Same as recognize_cells, also returning per-cell confidence

Model weights are lazy-loaded on first call and cached for the process lifetime.
"""
from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .kiri_vendor.loader import load_kiri_model
from .kiri_vendor.model import preprocess_pil

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

_OTSU_INVERT_MEAN_THRESHOLD = 127  # mean below this → invert binary (auto-polarity)
_BATCH_SIZE = 64  # cap on cells stacked into a single model forward pass

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_kiri: Optional[tuple] = None
_kiri_load_failed = False  # cache load failure so we don't re-download per cell


def _get_kiri():
    """Lazy-load the Kiri recognizer (model + cfg + tokenizer), cached process-wide.

    A load failure is cached too: without this, a bad load would re-trigger the
    HuggingFace download for every one of the ~240 cells on a page.
    """
    global _kiri, _kiri_load_failed
    if _kiri_load_failed:
        raise RuntimeError("Kiri model load previously failed; not retrying this run.")
    if _kiri is None:
        try:
            _kiri = load_kiri_model(device="cpu", verbose=True)
        except Exception:
            _kiri_load_failed = True
            raise
    return _kiri


def reset_kiri_failure() -> None:
    """Clear the load-failure latch so the next recognize call retries the model
    load once. Called at the start of each pipeline run so a transient first-run
    failure (e.g. an HF network blip) doesn't disable Kiri for the whole process
    lifetime. The within-run latch itself is kept (it prevents ~240 repeated slow
    download attempts per page)."""
    global _kiri_load_failed
    _kiri_load_failed = False


def _emit_kiri_warning(msg: str, sink: Optional[list]) -> None:
    """Route a Kiri failure to *sink* (so pipeline callers can collect it into
    SuryaResult.warnings) or, when no sink is given, to warnings.warn (unchanged
    behavior for standalone callers)."""
    if sink is not None:
        sink.append(msg)
    else:
        warnings.warn(msg)


# ---------------------------------------------------------------------------
# Otsu helper
# ---------------------------------------------------------------------------

def otsu_cell(rgb: np.ndarray) -> np.ndarray:
    """Binarize an RGB cell crop with Otsu thresholding + auto-polarity.

    If the mean pixel value of the binarized image is below *OTSU_INVERT_MEAN_THRESHOLD*
    (i.e. the background came out dark), the image is inverted so the background
    stays white — this handles both dark-on-light and light-on-dark table cells.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if binary.mean() < _OTSU_INVERT_MEAN_THRESHOLD:
        binary = 255 - binary
    return binary


# ---------------------------------------------------------------------------
# Wide-cell chunking
# ---------------------------------------------------------------------------
# Kiri is a fixed-input line recognizer: ResizeKeepRatioPadNoCrop scales a crop
# by HEIGHT to IMG_H and then `crop((0, 0, IMG_W, IMG_H))` — so every pixel past
# IMG_W/IMG_H (~13.3:1) was silently thrown away before the model ran. Measured
# on the MoC gas notification: 21% of cells exceeded the cap and a 73-char cell
# returned 42 chars. Splitting over-wide cells into fitting chunks lets the
# recognizer see the whole cell.

# A cut must land in whitespace: Khmer stacks and connects glyphs, so slicing
# mid-glyph corrupts it in BOTH chunks. Search this fraction of a chunk's width
# either side of the ideal boundary for a blank gutter.
_CUT_SNAP_WINDOW_FRAC = 0.25
_INK_MAX_VALUE = 127  # binarized ink is 0, background 255


def _widest_gap_center(ink: np.ndarray, lo: int, hi: int) -> Optional[int]:
    """Center of the widest zero-ink run within [lo, hi), or None if fully inked."""
    best_len, best_center = 0, None
    run_start = None
    for x in range(lo, hi + 1):
        blank = x < hi and ink[x] == 0
        if blank:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            if x - run_start > best_len:
                best_len, best_center = x - run_start, (run_start + x) // 2
            run_start = None
    return best_center


def _split_wide_binary(binary: np.ndarray, max_aspect: float) -> list[np.ndarray]:
    """Split a binarized cell into chunks that each fit within *max_aspect*.

    Returns ``[binary]`` unchanged (same object) when it already fits, so the
    common case costs nothing. Cuts are snapped to the widest blank gutter near
    each ideal boundary; a cell with no gutter (solid ink) falls back to a hard
    cut, which still beats silently discarding the overflow."""
    h, w = binary.shape[:2]
    if h <= 0 or w <= 0 or w <= h * max_aspect:
        return [binary]

    n_chunks = int(np.ceil(w / (h * max_aspect)))
    ink = (binary <= _INK_MAX_VALUE).sum(axis=0)
    window = max(1, int(w / n_chunks * _CUT_SNAP_WINDOW_FRAC))

    cuts: list[int] = []
    for k in range(1, n_chunks):
        target = round(k * w / n_chunks)
        lo, hi = max(0, target - window), min(w, target + window)
        center = _widest_gap_center(ink, lo, hi)
        # Cuts must stay strictly increasing, else a chunk would be empty.
        cut = center if center is not None else target
        if cuts and cut <= cuts[-1]:
            cut = min(w - 1, cuts[-1] + 1)
        cuts.append(cut)

    bounds = [0, *cuts, w]
    return [binary[:, a:b] for a, b in zip(bounds, bounds[1:]) if b > a]


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognize_cell(crop_rgb: np.ndarray) -> str:
    """Recognize text in a single table-cell crop.

    Pipeline: Otsu binarization → Kiri CTC decode (fast path) → clean text.
    Delegates to `recognize_cells` so there is a single recognition code path.
    """
    return recognize_cells([crop_rgb])[0]


def recognize_cells(crops_rgb: list[np.ndarray],
                    warning_sink: Optional[list] = None) -> list[str]:
    """Recognize text for a batch of table-cell crops in one or few model passes.

    Equivalent to ``[recognize_cell(c) for c in crops_rgb]`` but batches the
    Otsu-binarized crops into ``(N,1,IMG_H,IMG_W)`` tensors (chunked by
    `_BATCH_SIZE`) so `encode`/`ctc_head` run once per chunk instead of once
    per cell. Returns one string per input crop, in order. Any failure is routed
    to *warning_sink* (or warnings.warn when None) and the affected crops fall
    back to `""` rather than raising.
    """
    return [t for t, _ in recognize_cells_conf(crops_rgb, warning_sink)]


def recognize_cells_conf(crops_rgb: list[np.ndarray],
                         warning_sink: Optional[list] = None) -> list[tuple[str, float]]:
    """Recognize text + confidence for a batch of table-cell crops.

    Same batched pipeline as `recognize_cells` (Otsu → Kiri CTC decode → clean
    text), but each result is paired with a confidence score: the mean of the
    max softmax probability over the non-blank predicted timesteps for that
    cell (0.0 if every timestep predicted blank). Returns one (text, conf) pair
    per input crop, in order. Any failure is appended to *warning_sink* when
    given (so the engine can collect it into SuryaResult.warnings), else raised
    via warnings.warn; affected crops fall back to ("", 0.0) rather than raising.
    """
    if not crops_rgb:
        return []

    try:
        model, cfg, tokenizer = _get_kiri()
    except Exception as exc:
        # Total recognizer unavailability (load failed): every cell comes back
        # empty — surface that distinctly rather than as generic low confidence.
        _emit_kiri_warning(
            f"Kiri recognizer unavailable: {exc} — table cells left empty",
            warning_sink,
        )
        return [("", 0.0)] * len(crops_rgb)

    # Binarize once, then split any cell too wide for the model's fixed input.
    # `pieces` is the flat list actually fed to the model; `owners[i]` says which
    # input cell piece i belongs to, so batching still spans whole batches rather
    # than being confined within a cell.
    max_aspect = cfg.IMG_W / cfg.IMG_H
    pieces: list[np.ndarray] = []
    owners: list[int] = []
    chunked_cells = 0
    for idx, crop_rgb in enumerate(crops_rgb):
        parts = _split_wide_binary(otsu_cell(crop_rgb), max_aspect)
        if len(parts) > 1:
            chunked_cells += 1
        for part in parts:
            pieces.append(part)
            owners.append(idx)
    if chunked_cells:
        _emit_kiri_warning(
            f"Kiri: {chunked_cells} cell(s) exceeded the recognizer's "
            f"{max_aspect:.1f}:1 input and were split into chunks to be read in full",
            warning_sink,
        )

    piece_results: list[tuple[str, float]] = [("", 0.0)] * len(pieces)
    for start in range(0, len(pieces), _BATCH_SIZE):
        chunk = pieces[start:start + _BATCH_SIZE]
        try:
            tensors = [preprocess_pil(cfg, Image.fromarray(binary)) for binary in chunk]
            batch = torch.cat(tensors, dim=0)

            with torch.inference_mode():
                mem = model.encode(batch)
                logits = model.ctc_head(mem)
                pred_ids = logits.argmax(dim=-1)

            probs = F.softmax(logits, dim=-1).cpu()
            maxp = probs.max(dim=-1).values          # (N, T)
            ids = pred_ids.cpu()

            for i in range(len(chunk)):
                row = ids[i].tolist()
                text = tokenizer.decode_ctc(row)
                if cfg.COLLAPSE_WHITESPACE:
                    text = " ".join(text.split())
                # Post-processing: strip whitespace, trailing dots (Kiri often
                # emits a '.' after numbers).
                text = text.strip().rstrip(".").strip()

                nb = [maxp[i, t].item() for t, pid in enumerate(row) if pid != tokenizer.blank_id]
                conf = float(sum(nb) / len(nb)) if nb else 0.0

                piece_results[start + i] = (text, conf)
        except Exception as exc:
            _emit_kiri_warning(f"Kiri recognition failed on a cell batch: {exc}", warning_sink)
            # Affected pieces fall back to ("", 0.0) (already the default).

    # Reassemble: pieces of a cell are left-to-right, so joining in order
    # reconstructs the line. Confidence is the MINIMUM across a cell's pieces — a
    # cell is only as trustworthy as its worst fragment, and taking the mean would
    # let one clean chunk mask a garbled one.
    texts: list[list[str]] = [[] for _ in crops_rgb]
    confs: list[list[float]] = [[] for _ in crops_rgb]
    for (text, conf), owner in zip(piece_results, owners):
        if text:
            texts[owner].append(text)
        confs[owner].append(conf)
    return [
        (" ".join(parts).strip(), min(cs) if cs else 0.0)
        for parts, cs in zip(texts, confs)
    ]
