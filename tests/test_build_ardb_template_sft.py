from __future__ import annotations

from pathlib import Path

import pytest

from khmer_pipeline.datagen.build_ardb_template_sft import (
    _classify_era_text,
    _crop_pixels,
    _is_section_row,
    _normalize_two_row_header,
    _split_merged_row_number,
    _union_bbox,
    build_corpus,
    build_page,
    build_title_text,
    load_coco_header_furniture_boxes,
    load_coco_table_boxes,
    load_templates,
    load_title_template,
    parse_filename_date,
    parse_header_date,
    substitute_header,
    substitute_row,
)

_GT_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "real"
_COCO_HF_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "ardb_layout_coco_v1_hf"
_RETAIL_ONLY_STEM = "តម្លៃទំនិញមួយចំនួន_នៅរាជធានីភ្នំពេញថ្ងៃទី_០៦_០៧_កុម្ភៈ_២០២៤"

# Synthetic ASCII fixtures for pure structural-logic tests (no Khmer needed to exercise
# the algorithm itself — real Khmer alignment is covered by the load_templates integration
# test against the actual verified GT).
_TEMPLATE_ROW_BOTH = ["1", "item-a", "unit", "ws1", "rt1", "ws2", "rt2", "pctws", "pctrt"]
_TEMPLATE_ROW_RETAIL_ONLY = ["2", "item-b", "unit", "", "rt1", "", "rt2", "", "pctrt"]
_TEMPLATE_SECTION_ROW = ["SECTION A", "", "", "", "", "", "", "", ""]


class TestSubstituteRow:
    def test_fills_all_populated_slots_in_order(self):
        raw = ["1", None, "item-a", None, "unit", None, "20,000", "23,000", "20,000",
               "23,000", "0.00%", "0.00%"]
        row = substitute_row(_TEMPLATE_ROW_BOTH, raw)
        assert row == ["1", "item-a", "unit", "20,000", "23,000", "20,000", "23,000",
                        "0.00%", "0.00%"]

    def test_skips_unpopulated_slots(self):
        raw = ["2", "item-b", "unit", "43,000", "42,000", "-2.33%"]
        row = substitute_row(_TEMPLATE_ROW_RETAIL_ONLY, raw)
        assert row == ["2", "item-b", "unit", "", "43,000", "", "42,000", "", "-2.33%"]

    def test_row_number_mismatch_quarantines(self):
        raw = ["99", "item-a", "unit", "20,000", "23,000", "20,000", "23,000", "0.00%", "0.00%"]
        assert substitute_row(_TEMPLATE_ROW_BOTH, raw) is None

    def test_value_count_mismatch_quarantines(self):
        # template expects 6 numeric values (both wholesale+retail), raw only has 3
        raw = ["1", "item-a", "unit", "20,000", "23,000", "0.00%"]
        assert substitute_row(_TEMPLATE_ROW_BOTH, raw) is None

    def test_malformed_number_quarantines(self):
        raw = ["2", "item-b", "unit", "7,8000", "42,000", "-2.33%"]  # bad comma grouping
        assert substitute_row(_TEMPLATE_ROW_RETAIL_ONLY, raw) is None

    def test_narrower_price_cols_for_retail_only_era(self):
        # Era A's 6-col layout: only 3 price/pct slots (indices 3,4,5), not the 9-col
        # wholesale+retail layout's 6.
        template = ["1", "item-a", "unit", "rt1", "rt2", "pct"]
        raw = ["1", "item-a", "unit", "20,000", "23,000", "0.00%"]
        row = substitute_row(template, raw, price_cols=(3, 4, 5))
        assert row == ["1", "item-a", "unit", "20,000", "23,000", "0.00%"]


class TestSubstituteHeader:
    def test_swaps_both_dates(self):
        template = ["no", "name", "unit", "01-01-26 x", "01-01-26 y", "02-01-26 x",
                    "02-01-26 y", "chg", "chg"]
        row = substitute_header(template, ["30-06-26", "01-07-26"])
        assert row[3] == "30-06-26 x"
        assert row[5] == "01-07-26 x"

    def test_missing_dates_returns_none(self):
        assert substitute_header(["no", "name"], ["30-06-26", "01-07-26"]) is None


class TestIsSectionRow:
    def test_section_row_detected(self):
        assert _is_section_row(_TEMPLATE_SECTION_ROW)

    def test_data_row_not_section(self):
        assert not _is_section_row(_TEMPLATE_ROW_BOTH)


class TestBuildPage:
    def test_aligned_continuation_page(self):
        template = [_TEMPLATE_SECTION_ROW, _TEMPLATE_ROW_BOTH]
        raw = [
            ["SECTION A"],
            ["1", "item-a", "unit", "20,000", "23,000", "20,000", "23,000", "0.00%", "0.00%"],
        ]
        out = build_page(template, raw, has_header=False)
        assert out is not None
        assert out[1][3] == "20,000"

    def test_missing_rows_quarantines(self):
        template = [_TEMPLATE_ROW_BOTH, _TEMPLATE_ROW_RETAIL_ONLY]
        raw = [["1", "item-a", "unit", "20,000", "23,000", "20,000", "23,000", "0.00%", "0.00%"]]
        assert build_page(template, raw, has_header=False) is None

    def test_trailing_footer_rows_are_truncated_not_quarantined(self):
        template = [_TEMPLATE_ROW_BOTH]
        raw = [
            ["1", "item-a", "unit", "20,000", "23,000", "20,000", "23,000", "0.00%", "0.00%"],
            ["Source: some footer note"],
            [],
        ]
        out = build_page(template, raw, has_header=False)
        assert out is not None and len(out) == 1

    def test_header_rows_param_controls_body_start(self):
        # Simulates block-extracted rows with extra leading header noise (e.g. a title
        # block ahead of the date row) beyond the find_tables()-shaped "always 2" default.
        template = [["no", "name", "unit", "01-01-26 x", "02-01-26 x", "chg"], _TEMPLATE_SECTION_ROW]
        raw = [
            ["some title block"],
            ["01-01-26", "02-01-26"],
            ["labels block"],
            ["SECTION A"],
        ]
        out = build_page(template, raw, has_header=True, header_rows=3)
        assert out is not None
        assert out[1] == _TEMPLATE_SECTION_ROW


@pytest.mark.skipif(not _GT_DIR.is_dir(), reason="verified GT fixtures not present")
class TestLoadTemplatesIntegration:
    def test_loads_three_pages_with_expected_shape(self):
        templates = load_templates(_GT_DIR)
        assert len(templates) == 3
        for page in templates:
            assert all(len(row) == 9 for row in page)
        # page 0's header row carries two embedded dates
        assert len(substitute_header(templates[0][0], ["01-01-26", "02-01-26"]) or []) == 9

    def test_retail_only_era_header_is_merged_to_one_row(self):
        # Era A's GT stores the header as 2 physical rows (dates on row 0, "លក់រាយ"
        # sub-labels on row 1) — load_templates must merge them into one, like Era B's
        # already-single-row GT convention, so build_page's header logic works unchanged.
        templates = load_templates(_GT_DIR, stem=_RETAIL_ONLY_STEM)
        assert len(templates[0][0]) == 6  # 6-col retail-only layout, not 9
        assert "លក់រាយ" in templates[0][0][3]
        assert templates[0][1][0]  # row 1 is now real content (the section row), not a header remnant


class TestNormalizeTwoRowHeader:
    def test_merges_date_row_and_sublabel_row(self):
        rows = [
            ["no", "name", "unit", "01-01-26", "02-01-26", "chg"],
            ["", "", "", "retail", "retail", "%"],
            ["1", "item-a", "unit", "20,000", "23,000", "0.00%"],
        ]
        merged = _normalize_two_row_header(rows)
        assert merged[0] == ["no", "name", "unit", "01-01-26 retail", "02-01-26 retail", "chg %"]
        assert merged[1] == ["1", "item-a", "unit", "20,000", "23,000", "0.00%"]

    def test_already_single_row_header_is_unchanged(self):
        rows = [
            ["no", "name", "unit", "01-01-26 x", "02-01-26 x", "chg"],
            ["1", "item-a", "unit", "20,000", "23,000", "0.00%"],
        ]
        assert _normalize_two_row_header(rows) == rows


class TestSplitMergedRowNumber:
    def test_splits_merged_number_and_name(self):
        # Recurring PDF rendering quirk: row-number and item-name render with no newline
        # between them on certain rows.
        assert _split_merged_row_number(["១៣ មាន់ស្រែ (សាច់)", "unit", "1,000"]) == (
            ["១៣", "មាន់ស្រែ (សាច់)", "unit", "1,000"])

    def test_normal_already_separate_row_is_unchanged(self):
        lines = ["១៣", "មាន់ស្រែ (សាច់)", "unit", "1,000"]
        assert _split_merged_row_number(lines) == lines

    def test_section_row_without_a_number_is_unchanged(self):
        lines = ["ក/-តម្លៃត្រីសាច់បន្លែ"]
        assert _split_merged_row_number(lines) == lines

    def test_empty_list_is_unchanged(self):
        assert _split_merged_row_number([]) == []


class TestClassifyEraText:
    def test_wholesale_marker_present(self):
        assert _classify_era_text("some text បោះដុំ more text") == "wholesale_retail"

    def test_wholesale_marker_absent(self):
        assert _classify_era_text("retail only text, no marker here") == "retail_only"

    def test_wholesale_marker_with_stray_inserted_space(self):
        # Confirmed on real 2025 wholesale+retail docs: the marker itself can render with
        # a stray space mid-word ("ប ោះដុំ" instead of "បោះដុំ").
        assert _classify_era_text("some text ប ោះដុំ more text") == "wholesale_retail"


class TestCropPixels:
    def test_crops_with_padding(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        crop = _crop_pixels(img, (10, 10, 20, 20), pad=2)
        assert crop.size == (24, 24)  # 20+2+2

    def test_clamps_to_image_bounds(self):
        from PIL import Image
        img = Image.new("RGB", (50, 50))
        crop = _crop_pixels(img, (45, 45, 20, 20), pad=6)
        assert crop.size == (11, 11)  # clamped to the 50x50 image from x0=39,y0=39

    def test_degenerate_bbox_returns_none(self):
        from PIL import Image
        img = Image.new("RGB", (50, 50))
        assert _crop_pixels(img, (10, 10, 0, 0), pad=0) is None


@pytest.mark.skipif(not _COCO_HF_DIR.is_dir(), reason="packaged COCO layout dataset not present")
class TestLoadCocoTableBoxes:
    def test_returns_a_box_per_page(self):
        boxes = load_coco_table_boxes(_COCO_HF_DIR)
        assert boxes
        (source, page), bbox = next(iter(boxes.items()))
        assert isinstance(source, str) and isinstance(page, int)
        assert len(bbox) == 4


class TestParseFilenameDate:
    def test_parses_day_month_year(self):
        assert parse_filename_date("foo-ប្រចាំថ្ងៃ-26.01.26.pdf") == (26, 1, 26)

    def test_no_match_returns_none(self):
        assert parse_filename_date("not_a_dated_file.pdf") is None


class TestParseHeaderDate:
    def test_uses_second_date(self):
        assert parse_header_date(["08-06-26", "09-06-26"]) == (9, 6, 26)

    def test_fewer_than_two_dates_returns_none(self):
        assert parse_header_date(["09-06-26"]) is None
        assert parse_header_date([]) is None


class TestBuildTitleText:
    def test_fills_day_and_month(self):
        template = "prefix ថ្ងៃទី{day} ខែ{month} suffix"
        assert build_title_text(template, day=9, month=6) == "prefix ថ្ងៃទី៩ ខែមិថុនា suffix"

    def test_unverified_month_returns_none(self):
        template = "prefix ថ្ងៃទី{day} ខែ{month} suffix"
        assert build_title_text(template, day=1, month=13) is None  # no 13th month


class TestUnionBbox:
    def test_unions_two_boxes(self):
        # [x, y, w, h]
        assert _union_bbox([(10, 10, 20, 20), (10, 40, 20, 10)]) == (10, 10, 20, 40)


@pytest.mark.skipif(not _GT_DIR.is_dir(), reason="verified GT fixtures not present")
class TestLoadTitleTemplate:
    def test_has_day_and_month_placeholders(self):
        template = load_title_template(_GT_DIR)
        assert "{day}" in template and "{month}" in template
        assert "ថ្ងៃទី{day}" in template


@pytest.mark.skipif(not _COCO_HF_DIR.is_dir(), reason="packaged COCO layout dataset not present")
class TestLoadCocoHeaderFurnitureBoxes:
    def test_returns_boxes_for_known_document(self):
        headers, furniture = load_coco_header_furniture_boxes(_COCO_HF_DIR)
        assert headers and furniture
        (src, bbox) = next(iter(headers.items()))
        assert isinstance(src, str) and len(bbox) == 4
        (src, bbox) = next(iter(furniture.items()))
        assert isinstance(src, str) and len(bbox) == 4


@pytest.mark.skipif(not _GT_DIR.is_dir(), reason="verified GT fixtures not present")
class TestBuildCorpus:
    def test_frozen_eval_stems_are_excluded(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        with pytest.raises(ValueError):
            build_corpus(corpus_dir, tmp_path / "out", _GT_DIR)
