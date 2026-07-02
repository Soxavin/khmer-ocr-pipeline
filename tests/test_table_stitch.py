from __future__ import annotations
from khmer_pipeline.engines.table_stitch import merge_table_regions, merge_table_rowbands

# Real page-2 fragmentation: one table shattered into a 2 row-band x 4 col-group
# grid of 8 boxes that tile the contiguous rectangle (30,220)-(1970,1960).
_PAGE2 = [
    (30, 220, 770, 680), (30, 750, 770, 1960),
    (810, 220, 1180, 680), (1210, 220, 1580, 680), (1610, 220, 1970, 680),
    (810, 750, 1180, 1960), (1210, 750, 1580, 1960), (1610, 750, 1970, 1960),
]


def test_empty():
    assert merge_table_regions([]) == []


def test_single_unchanged():
    assert merge_table_regions([(10, 10, 100, 100)]) == [(10, 10, 100, 100)]


def test_page2_eight_fragments_merge_to_one():
    out = merge_table_regions(_PAGE2)
    assert out == [(30, 220, 1970, 1960)]


def test_horizontally_adjacent_columns_merge():
    # two column boxes, small x-gap, same y-band -> one
    boxes = [(0, 0, 100, 500), (110, 0, 210, 500)]
    assert merge_table_regions(boxes) == [(0, 0, 210, 500)]


def test_far_apart_in_x_stay_separate():
    # large x-gap (> bridgeable) -> two tables
    boxes = [(0, 0, 100, 500), (900, 0, 1000, 500)]
    out = merge_table_regions(boxes)
    assert len(out) == 2
    assert set(out) == {(0, 0, 100, 500), (900, 0, 1000, 500)}


def test_far_apart_in_y_stay_separate():
    # stacked tables with a large vertical gap -> two tables
    boxes = [(0, 0, 500, 100), (0, 900, 500, 1000)]
    out = merge_table_regions(boxes)
    assert len(out) == 2


def test_idempotent():
    once = merge_table_regions(_PAGE2)
    assert merge_table_regions(once) == once


def test_output_sorted_top_to_bottom_then_left():
    boxes = [(0, 900, 500, 1000), (0, 0, 500, 100)]
    out = merge_table_regions(boxes)
    assert out == sorted(out, key=lambda b: (b[1], b[0]))


# --- row-band variant: merge into full-width strips per Y-band, not one giant box ---

def test_rowband_page2_eight_fragments_to_two_strips():
    out = merge_table_rowbands(_PAGE2)
    assert out == [(30, 220, 1970, 680), (30, 750, 1970, 1960)]


def test_rowband_empty():
    assert merge_table_rowbands([]) == []


def test_rowband_single_unchanged():
    assert merge_table_rowbands([(10, 10, 100, 100)]) == [(10, 10, 100, 100)]


def test_rowband_same_band_far_in_x_merge_to_full_width():
    # same Y-band, far apart in X -> one full-width strip (X distance ignored)
    boxes = [(0, 0, 100, 500), (900, 0, 1000, 500)]
    assert merge_table_rowbands(boxes) == [(0, 0, 1000, 500)]


def test_rowband_distinct_y_bands_stay_separate():
    boxes = [(0, 0, 500, 100), (0, 900, 500, 1000)]
    out = merge_table_rowbands(boxes)
    assert len(out) == 2


def test_rowband_idempotent():
    once = merge_table_rowbands(_PAGE2)
    assert merge_table_rowbands(once) == once
