"""Automatic engine routing — a deterministic confidence circuit-breaker.

Runs ``surya_kiri`` (the dense-Khmer / ARDB specialist and the common case) and,
only if its OWN per-cell confidence says it is struggling on this document, falls
back to ``surya`` (whose VLM handles wide / legacy-font tables that Kiri's per-cell
CTC cannot). Registered as ``OCR_ENGINE=auto``.

This is NOT a learned "AI router": it is a state machine keyed on one measured
signal — the fraction of table cells surya_kiri reports below ``CELL_CONF_LOW``.
It is deliberately *reactive* (surya_kiri runs to completion before failover), a
one-directional accuracy-over-latency trade-off appropriate for a single-user
desktop tool: the second pass only runs on the rare document that fails.

Cutoff basis (PROJECT_LOG §2.57, measured on the 7-page golden set): the worst
ARDB page reports a 0.222 low-confidence fraction while still winning, and the
legacy/scanned budget page reports 0.539 (median cell confidence 0.000 — it fails
*unconfidently*). ``_FALLBACK_LOW_CONF_FRACTION`` sits between them, biased toward
NOT falling back, because a false fallback on an ARDB doc actively regresses
accuracy (surya is worse than surya_kiri there).

Known ceiling: the signal catches *unconfident* failure only. A document where
surya_kiri is *confidently wrong* would not trigger the fallback — accepted, since
that mode is absent from our measured failures (budget's median confidence is 0).
"""
from __future__ import annotations

from typing import Callable, Optional

from ..models import PreprocessResult, SuryaResult
from ..model_config import CELL_CONF_LOW
from .surya import run_surya
from .surya_kiri_engine import run_surya_kiri

# Fraction of table cells below CELL_CONF_LOW above which surya_kiri is deemed to
# be failing this document and surya takes over. Between the measured ARDB max
# (0.222) and budget (0.539); see module docstring / §2.57. NOT tuned to the two
# docs exactly — chosen with margin on both sides.
_FALLBACK_LOW_CONF_FRACTION = 0.40


def _low_conf_fraction(result: SuryaResult) -> float:
    """Fraction of table cells (pooled over all pages) below CELL_CONF_LOW.

    Document-level, not per-page: a doc that is half-failing still crosses the
    cutoff. Returns 0.0 when the result has no confidence-bearing table cells
    (e.g. a text-only page), so such a doc never triggers a fallback."""
    confs = [
        c["confidence"]
        for page in result.pages
        for table in page.tables
        for c in table["cells"]
        if "confidence" in c
    ]
    if not confs:
        return 0.0
    low = sum(1 for c in confs if c < CELL_CONF_LOW)
    return low / len(confs)


def run_auto(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
    on_step: Optional[Callable[[str], None]] = None,
) -> SuryaResult:
    """Route a document to surya_kiri or surya.

    Two-stage router. First a PRE-FLIGHT check on the source geometry: a low-
    resolution raster scan goes straight to surya without running surya_kiri at
    all. §2.75 proved Kiri's per-cell confidence cannot detect this case — it
    reports 0.222 low-conf on the ARDB page it wins and 0.231 on the moc_gas scan
    it destroys, 0.009 apart — but the scan is knowable from the PDF before any
    inference (see ingest.page_is_scanned). So it is caught for free rather than
    run-then-measured.

    Otherwise the reactive path: run surya_kiri; if its low-confidence cell
    fraction exceeds ``_FALLBACK_LOW_CONF_FRACTION`` re-run with surya. Either way
    a machine-readable ``[AutoRouter] …`` note records the decision."""
    # Pre-flight: bypass the per-cell recognizer on a low-res scan. getattr keeps
    # older PreprocessResults (no flag) on the reactive path — absence = unknown.
    if getattr(result, "low_res_scan", False):
        surya = run_surya(result, on_page=on_page, on_step=on_step)
        surya.warnings.append(
            "[AutoRouter] pre-flight surya (low-res scan) | "
            "per-cell recognition unreliable below the scan-density threshold"
        )
        return surya

    kiri = run_surya_kiri(result, on_page=on_page, on_step=on_step)
    frac = _low_conf_fraction(kiri)
    cutoff = _FALLBACK_LOW_CONF_FRACTION

    if frac > cutoff:
        surya = run_surya(result, on_page=on_page, on_step=on_step)
        surya.warnings.append(
            f"[AutoRouter] fallback surya_kiri->surya | frac={frac:.3f} cutoff={cutoff:.3f}"
        )
        return surya

    kiri.warnings.append(
        f"[AutoRouter] kept surya_kiri | frac={frac:.3f} cutoff={cutoff:.3f}"
    )
    return kiri
