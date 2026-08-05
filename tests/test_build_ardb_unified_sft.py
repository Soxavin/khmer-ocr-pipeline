from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from khmer_pipeline.datagen.build_ardb_template_sft import (
    _RETAIL_ONLY_GT_STEM,
    load_templates,
    load_title_template,
)
from khmer_pipeline.datagen.build_ardb_unified_sft import (
    _FOOTER_TEXT,
    build_corpus,
    build_document,
    build_region_texts,
    load_page_regions,
)

_GT_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "real"
_COCO_HF_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "ardb_layout_coco_v1_hf"
_COCO_V3_HF_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "ardb_layout_coco_v3_hf"
_CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "ardb_daily"
_RETAIL_ONLY_PDF = _CORPUS_DIR / f"{_RETAIL_ONLY_GT_STEM}.pdf"

# A page-0-shaped region set: Picture, 2x Page-Furniture (top/bottom), Section-Header, Table.
_PAGE0_REGIONS = [
    ("Picture", [15, 15, 90, 90]),
    ("Page-Furniture", [21, 91, 51, 266]),   # top (smaller y1) -> bank name line
    ("Page-Furniture", [53, 91, 87, 286]),   # bottom -> tagline line
    ("Section-Header", [889, 17, 983, 385]),
    ("Table", [111, 15, 885, 984]),
]


class TestBuildRegionTexts:
    def test_happy_path_assigns_and_sorts(self):
        out = build_region_texts(_PAGE0_REGIONS, table_text="| a | b |", title_text="title text")
        assert out is not None
        # sorted by box_2d[0] ascending
        assert [r["box_2d"][0] for r in out] == sorted(r["box_2d"][0] for r in out)
        by_label = {r["label"]: r for r in out if r["label"] != "Page-Furniture"}
        assert by_label["Picture"]["text"] == ""
        assert by_label["Table"]["text"] == "| a | b |"
        assert by_label["Section-Header"]["text"] == "title text"

    def test_furniture_split_top_bottom(self):
        out = build_region_texts(_PAGE0_REGIONS, table_text="t", title_text="ti")
        furniture = sorted([r for r in out if r["label"] == "Page-Furniture"],
                           key=lambda r: r["box_2d"][0])
        assert furniture[0]["text"] == "ធនាគារ ARDB"
        assert furniture[1]["text"] == "ដើម្បីកសិករនិងអភិវឌ្ឍន៍សេដ្ឋកិច្ចសង្គម"

    def test_text_label_gets_footer_text(self):
        regions = [("Text", [900, 15, 980, 385]), ("Table", [111, 15, 885, 984])]
        out = build_region_texts(regions, table_text="t", title_text=None)
        by_label = {r["label"]: r for r in out}
        assert by_label["Text"]["text"] == _FOOTER_TEXT

    def test_footer_text_has_no_stray_escaped_backslash(self):
        # Regression: a double-escaped "\\n" (literal backslash + n, not a real newline)
        # previously sat mid-word between "១ " and "ផ្សារ...", corrupting every Text-region
        # training row. A real newline has no backslash character in the string at all.
        assert "\\" not in _FOOTER_TEXT

    def test_unknown_label_gets_empty_text(self):
        regions = [("SomeFutureCategory", [0, 0, 10, 10]), ("Table", [111, 15, 885, 984])]
        out = build_region_texts(regions, table_text="t", title_text=None)
        by_label = {r["label"]: r for r in out}
        assert by_label["SomeFutureCategory"]["text"] == ""

    def test_table_present_but_text_none_quarantines_whole_page(self):
        assert build_region_texts(_PAGE0_REGIONS, table_text=None, title_text="ti") is None

    def test_header_present_but_title_none_quarantines_whole_page(self):
        assert build_region_texts(_PAGE0_REGIONS, table_text="t", title_text=None) is None

    def test_furniture_count_mismatch_quarantines(self):
        regions = [("Page-Furniture", [10, 10, 20, 20]), ("Table", [111, 15, 885, 984])]
        assert build_region_texts(regions, table_text="t", title_text=None) is None

    def test_no_table_no_header_is_fine(self):
        regions = [("Picture", [0, 0, 10, 10])]
        out = build_region_texts(regions, table_text=None, title_text=None)
        assert out == [{"box_2d": [0, 0, 10, 10], "label": "Picture", "text": ""}]


@pytest.mark.skipif(not _GT_DIR.is_dir(), reason="verified GT not present")
class TestFooterTextMatchesGT:
    def test_matches_09_06_26_page3_ground_truth_exactly(self):
        gt = json.loads((_GT_DIR / (
            "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26"
            "_p3_ground_truth.json")).read_text())
        assert _FOOTER_TEXT == "\n".join(gt["paragraphs"][-3:])


@pytest.mark.skipif(not _COCO_HF_DIR.is_dir(), reason="packaged COCO layout dataset not present")
class TestLoadPageRegions:
    def test_groups_by_source_and_page(self):
        pages = load_page_regions(_COCO_HF_DIR)
        assert pages
        (source, page_idx), data = next(iter(pages.items()))
        assert isinstance(source, str) and isinstance(page_idx, int)
        assert set(data) == {"split", "doc_id", "image", "regions"}
        assert all(len(box) == 4 for _, box in data["regions"])

    def test_page0_has_five_boxes_across_four_labels(self):
        pages = load_page_regions(_COCO_HF_DIR)
        p0 = [data for (_, page_idx), data in pages.items() if page_idx == 0]
        assert p0
        regions = p0[0]["regions"]
        assert len(regions) == 5
        assert {label for label, _ in regions} == {
            "Picture", "Page-Furniture", "Section-Header", "Table"}


@pytest.mark.skipif(not (_COCO_HF_DIR.is_dir() and _CORPUS_DIR.is_dir() and _GT_DIR.is_dir()),
                    reason="full corpus + COCO + verified GT not all present")
class TestBuildCorpusIntegration:
    def test_builds_joint_rows_with_valid_json_targets(self, tmp_path):
        totals = build_corpus(_COCO_HF_DIR, _CORPUS_DIR.parent, _GT_DIR, tmp_path)
        assert totals["pages_ok"] > 0
        assert totals["skipped_docs"] >= 2  # the 2 frozen eval docs
        found_any = False
        for split in ("train", "validation", "test"):
            jsonl = tmp_path / split / "pairs.jsonl"
            if not jsonl.is_dir() and jsonl.exists():
                for line in jsonl.read_text().splitlines():
                    row = json.loads(line)
                    assert "09.06.26" not in row["source"] and "15.06.26" not in row["source"]
                    regions = json.loads(row["text"])
                    assert isinstance(regions, list) and regions
                    for r in regions:
                        assert set(r) == {"box_2d", "label", "text"}
                    found_any = True
        assert found_any

    def test_frozen_eval_stems_excluded(self, tmp_path):
        totals = build_corpus(_COCO_HF_DIR, _CORPUS_DIR.parent, _GT_DIR, tmp_path)
        assert totals["skipped_docs"] >= 2


@pytest.mark.skipif(not (_RETAIL_ONLY_PDF.is_file() and _GT_DIR.is_dir()),
                    reason="retail_only anchor PDF or verified GT not present")
class TestBuildDocumentEraDispatch:
    """build_document must be called with the retail_only template/price_cols for a
    retail_only-era document, not the wholesale_retail default — otherwise the 6-col era
    silently misaligns against the 9-col template (see build_ardb_template_sft.py's own
    era-keyed build_corpus, which this mirrors)."""

    def test_retail_only_doc_parses_against_its_own_template(self, tmp_path):
        templates = load_templates(_GT_DIR, stem=_RETAIL_ONLY_GT_STEM)
        title_template = load_title_template(_GT_DIR, stem=_RETAIL_ONLY_GT_STEM)
        # A Table region is required so build_region_texts actually checks table_text
        # against None (an empty regions list short-circuits to "fine" regardless of
        # table_text, per test_no_table_no_header_is_fine above).
        pages = {0: {"regions": [("Table", [111, 15, 885, 984])], "doc_id": "doc_test",
                    "split": "train", "image": Image.new("RGB", (10, 10))}}
        counts = build_document(_RETAIL_ONLY_PDF, _RETAIL_ONLY_PDF.name, templates,
                                title_template, pages, "doc_test", "train", tmp_path,
                                price_cols=(3, 4, 5))
        assert counts["pages_ok"] == 1
        assert counts["pages_quarantined"] == 0

    def test_retail_only_doc_quarantines_against_wrong_era_template(self, tmp_path):
        # Using the default (wholesale_retail) template/price_cols against a retail_only
        # doc should fail to align -- this is the bug this era-dispatch fix prevents.
        templates = load_templates(_GT_DIR)
        title_template = load_title_template(_GT_DIR)
        pages = {0: {"regions": [("Table", [111, 15, 885, 984])], "doc_id": "doc_test",
                    "split": "train", "image": Image.new("RGB", (10, 10))}}
        counts = build_document(_RETAIL_ONLY_PDF, _RETAIL_ONLY_PDF.name, templates,
                                title_template, pages, "doc_test", "train", tmp_path)
        assert counts["pages_quarantined"] == 1
        assert counts["pages_ok"] == 0


@pytest.mark.skipif(not (_COCO_V3_HF_DIR.is_dir() and _CORPUS_DIR.is_dir() and _GT_DIR.is_dir()),
                    reason="v3 COCO layout dataset not packaged yet")
class TestBuildCorpusEraDispatchIntegration:
    """Same as TestBuildCorpusIntegration but against the v3 dataset, which (unlike v1)
    actually includes retail_only-era documents -- proves the era dispatch wired into
    build_corpus works end-to-end, not just in isolation."""

    def test_retail_only_docs_produce_ok_pages(self, tmp_path):
        totals = build_corpus(_COCO_V3_HF_DIR, _CORPUS_DIR.parent, _GT_DIR, tmp_path)
        assert totals["pages_ok"] > 0
        # Sanity: quarantine rate should stay well under half -- a systemic era mismatch
        # would quarantine most/all of the 16 retail_only docs' pages.
        total_pages = totals["pages_ok"] + totals["pages_quarantined"]
        assert totals["pages_ok"] / total_pages > 0.5
