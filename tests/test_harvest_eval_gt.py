"""Tests for template-mapped evaluation-GT harvesting (harvest_eval_gt.py).

The ARDB dailies are a rigid template: every doc renders the same shape and the
same Khmer label sequence, and only the numbers change day to day. So the
raw-text-layer → GT-grid mapping is derived ONCE against a trusted hand-verified
page and replayed over the rest of the corpus.
"""
from __future__ import annotations

import pytest

from khmer_pipeline.datagen import harvest_eval_gt as h


# Raw text-layer grid: wider than the GT (find_tables splits columns), Khmer
# scrambled, numbers correct — exactly the measured ARDB shape in miniature.
RAW = [
    ["", "២៣", "", "", "ស្គ្រាំ", "", "៛/x", "360", "500"],
    ["២៤", "", "", "ទ ា", "", "", "៛/y", "350", "480"],
]
# Trusted GT for the same page: narrower, Khmer correct.
GT = [
    ["២៣", "ពងមាន់", "៛/គ្រាប់", "360", "500"],
    ["២៤", "ពងទា", "៛/គ.ក", "350", "480"],
]


def test_derive_mapping_finds_numeric_columns_by_exact_match():
    m = h.derive_column_mapping(GT, RAW)
    # GT cols 3,4 are the numbers — they appear verbatim in raw cols 7,8.
    assert m[3] == 7
    assert m[4] == 8


def test_derive_mapping_covers_every_gt_column():
    m = h.derive_column_mapping(GT, RAW)
    assert set(m) == set(range(len(GT[0])))


def test_scrambled_khmer_columns_still_get_a_position():
    # Khmer text never matches (scrambled), so the column is located by
    # elimination/position rather than by text equality — but it must be found,
    # otherwise the carried-over labels would land in the wrong column.
    m = h.derive_column_mapping(GT, RAW)
    assert m[1] is not None and m[2] is not None


def test_apply_mapping_takes_numbers_from_raw_and_khmer_from_template():
    """The whole mechanism: numbers are per-date (from this doc's text layer),
    Khmer labels are identical across dates (carried from the verified GT)."""
    raw_other_day = [
        ["", "២៣", "", "", "ស្គ្រាំ", "", "៛/x", "999", "111"],
        ["២៤", "", "", "ទ ា", "", "", "៛/y", "222", "333"],
    ]
    m = h.derive_column_mapping(GT, RAW)
    out = h.apply_mapping(GT, raw_other_day, m)
    assert out[0][3] == "999" and out[0][4] == "111"   # numbers: this date's
    assert out[1][3] == "222" and out[1][4] == "333"
    assert out[0][1] == "ពងមាន់"                        # Khmer: from template
    assert out[0][2] == "៛/គ្រាប់"


def test_apply_mapping_preserves_shape():
    m = h.derive_column_mapping(GT, RAW)
    out = h.apply_mapping(GT, RAW, m)
    assert len(out) == len(GT)
    assert all(len(r) == len(GT[0]) for r in out)


def test_shape_mismatch_is_rejected_not_silently_mapped():
    """A doc whose raw shape differs from the template breaks the rigid-template
    assumption the whole method rests on — it must refuse, not emit wrong GT."""
    m = h.derive_column_mapping(GT, RAW)
    odd = [["only", "three", "cols"]]
    with pytest.raises(ValueError, match="shape"):
        h.apply_mapping(GT, odd, m)


def test_roundtrip_on_the_template_page_reproduces_trusted_gt():
    # Replaying the mapping on the very page it was derived from must return
    # that page's trusted GT — the correctness anchor for the whole method.
    m = h.derive_column_mapping(GT, RAW)
    assert h.apply_mapping(GT, RAW, m) == GT


def test_numeric_fidelity_reports_per_column_match_rate():
    # The trust signal: how often the mapped raw column equals the trusted GT.
    m = h.derive_column_mapping(GT, RAW)
    rates = h.numeric_fidelity(GT, RAW, m)
    assert rates[3] == pytest.approx(1.0)
    assert rates[4] == pytest.approx(1.0)
