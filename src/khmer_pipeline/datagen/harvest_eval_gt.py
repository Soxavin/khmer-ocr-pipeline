"""Template-mapped evaluation GT for rigid born-digital document series.

The ARDB dailies are the same form every day: measured across all 30 corpus PDFs,
each page yields exactly ONE table shape and ONE Khmer label sequence — only the
numbers change. That rigidity is what makes cheap GT possible.

Why the obvious approach fails, and what this does instead:

* ``find_tables().extract()`` does NOT give the logical grid — on ARDB p2 it
  returns 27x18 where the true table is 27x9 (columns are split/staggered), and
  neither empty-column pruning (→12) nor x-centre clustering (→15) recovers it.
* The text layer also SCRAMBLES Khmer (``រតីរ៉ស់`` for ``ត្រីរ៉ស់``), so its
  label text cannot be trusted as GT at all.
* But numbers extract verbatim, and the template never moves.

So the column mapping is derived ONCE against a hand-verified page (measured:
all 9 GT columns located, the 6 numeric ones matching 26/27 rows exactly) and
replayed across the series. Per page: **numbers come from that date's text layer,
Khmer labels are carried from the verified template** (they are byte-identical
across dates, so there is nothing to transcribe — which also keeps this within
the rule that Khmer GT only ever comes from human-verified text).

Scope, stated honestly: extra pages add NUMERIC samples only. The labels and
layout are identical across dates, so they are not independent samples of Khmer
recognition or table structure — see the `gt_source` tag each page carries.
"""
from __future__ import annotations

from collections import Counter

_GT_SOURCE = "textlayer_template_mapped"


def _norm(v) -> str:
    return (v or "").strip() if isinstance(v, (str, type(None))) else str(v).strip()


def derive_column_mapping(
    gt_grid: list[list[str]], raw_grid: list[list]
) -> dict[int, int]:
    """Map each trusted-GT column index to the raw text-layer column holding it.

    Columns whose text survives extraction (numbers, digits) are located by exact
    match voting across rows. Khmer columns scramble and never match, so they are
    filled by monotonic interpolation between their located neighbours — the
    template's column order is fixed, so position is enough to carry labels into
    the right slot."""
    n_gt = len(gt_grid[0]) if gt_grid else 0
    located: dict[int, int] = {}
    for gc in range(n_gt):
        votes: Counter = Counter()
        for r in range(min(len(gt_grid), len(raw_grid))):
            want = _norm(gt_grid[r][gc]) if gc < len(gt_grid[r]) else ""
            if not want:
                continue
            for rc, val in enumerate(raw_grid[r]):
                if _norm(val) and _norm(val) == want:
                    votes[rc] += 1
        if votes:
            located[gc] = votes.most_common(1)[0][0]

    # Fill unmatched (scrambled-Khmer) columns strictly between their neighbours so
    # the mapping stays monotonic — a crossed mapping would put labels in the wrong
    # column, which is worse than leaving the cell to the template.
    mapping: dict[int, int] = {}
    for gc in range(n_gt):
        if gc in located:
            mapping[gc] = located[gc]
            continue
        prev = max((c for c in located if c < gc), default=None)
        nxt = min((c for c in located if c > gc), default=None)
        lo = located[prev] + 1 if prev is not None else 0
        hi = located[nxt] - 1 if nxt is not None else (len(raw_grid[0]) - 1 if raw_grid else 0)
        mapping[gc] = max(0, min(lo, hi)) if hi >= lo else lo
    return mapping


def apply_mapping(
    template_gt: list[list[str]], raw_grid: list[list], mapping: dict[int, int]
) -> list[list[str]]:
    """Build one page's GT: numbers from *raw_grid*, Khmer labels from *template_gt*.

    A raw grid whose shape differs from the template violates the rigid-template
    assumption this method rests on, so it raises rather than emit wrong GT."""
    t_rows, t_cols = len(template_gt), len(template_gt[0]) if template_gt else 0
    if len(raw_grid) != t_rows or (raw_grid and len(raw_grid[0]) < max(mapping.values(), default=0) + 1):
        raise ValueError(
            f"raw grid shape {len(raw_grid)}x{len(raw_grid[0]) if raw_grid else 0} "
            f"does not match the template ({t_rows} rows); refusing to map")

    out: list[list[str]] = []
    for r in range(t_rows):
        row: list[str] = []
        for c in range(t_cols):
            template_val = _norm(template_gt[r][c]) if c < len(template_gt[r]) else ""
            rc = mapping.get(c)
            raw_val = _norm(raw_grid[r][rc]) if rc is not None and rc < len(raw_grid[r]) else ""
            # Trust the text layer only where it round-trips (numbers/digits);
            # otherwise keep the verified template label, which is identical
            # across dates by construction.
            row.append(raw_val if raw_val and raw_val == template_val else (raw_val if _is_datalike(raw_val, template_val) else template_val))
        out.append(row)
    return out


def _is_datalike(raw_val: str, template_val: str) -> bool:
    """True when the raw cell should override the template: it varies per date and
    the text layer renders it faithfully. Numbers (any script) qualify; Khmer
    label text does not, because extraction scrambles it."""
    if not raw_val:
        return False
    khmer = sum(1 for ch in raw_val if "ក" <= ch <= "៓")
    return khmer == 0  # digits/punctuation/latin survive extraction; Khmer does not


def numeric_fidelity(
    gt_grid: list[list[str]], raw_grid: list[list], mapping: dict[int, int]
) -> dict[int, float]:
    """Per-GT-column fraction of rows where the mapped raw cell equals the trusted
    GT. The trust signal for the mapping: numeric columns should be ~1.0, Khmer
    columns near 0 (scrambled) — anything else means the mapping slipped."""
    rates: dict[int, float] = {}
    for gc, rc in mapping.items():
        hits = total = 0
        for r in range(min(len(gt_grid), len(raw_grid))):
            want = _norm(gt_grid[r][gc]) if gc < len(gt_grid[r]) else ""
            if not want:
                continue
            total += 1
            got = _norm(raw_grid[r][rc]) if rc < len(raw_grid[r]) else ""
            if got == want:
                hits += 1
        rates[gc] = hits / total if total else 0.0
    return rates
