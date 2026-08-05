from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from khmer_pipeline.datagen.pseudo_label_layout import (
    CLASS_NAMES,
    PageBoxes,
    _cluster_by_date,
    assign_splits,
    assign_splits_by_date_cluster,
    assign_splits_by_date_cluster_stratified,
    map_surya_label,
    write_coco,
)


class TestMapSuryaLabel:
    def test_core_labels(self):
        assert map_surya_label("Table") == "Table"
        assert map_surya_label("Text") == "Text"
        assert map_surya_label("SectionHeader") == "Section-Header"
        assert map_surya_label("PageHeader") == "Page-Furniture"
        assert map_surya_label("PageFooter") == "Page-Furniture"
        assert map_surya_label("Picture") == "Picture"
        assert map_surya_label("Figure") == "Picture"

    def test_textlike_labels_fold_to_text(self):
        for label in ("ListItem", "Caption", "Footnote", "Formula", "Form"):
            assert map_surya_label(label) == "Text"

    def test_unknown_label_falls_back_to_text(self):
        assert map_surya_label("SomethingNew") == "Text"

    def test_all_mapped_names_are_known_classes(self):
        for label in ("Table", "Text", "SectionHeader", "PageHeader", "Picture", "Unknown"):
            assert map_surya_label(label) in CLASS_NAMES


class TestAssignSplits:
    def test_split_is_by_document_and_covers_all(self):
        docs = [f"doc_{i}.pdf" for i in range(20)]
        splits = assign_splits(docs, seed=0)
        assert set(splits) == set(docs)
        assert set(splits.values()) <= {"train", "valid", "test"}
        # every split non-empty at 20 docs with 0.8/0.1/0.1
        assert {"train", "valid", "test"} == set(splits.values())

    def test_deterministic_under_seed(self):
        docs = [f"doc_{i}.pdf" for i in range(15)]
        assert assign_splits(docs, seed=7) == assign_splits(docs, seed=7)
        assert assign_splits(docs, seed=7) != assign_splits(docs, seed=8)

    def test_order_independent(self):
        docs = [f"doc_{i}.pdf" for i in range(12)]
        assert assign_splits(docs, seed=3) == assign_splits(list(reversed(docs)), seed=3)

    def test_tiny_corpus_all_train(self):
        # with 1-2 docs everything lands in train (never an empty train split)
        assert set(assign_splits(["a.pdf"], seed=0).values()) == {"train"}


class TestAssignSplitsByDateCluster:
    def test_adjacent_dates_never_straddle_splits(self):
        # 5 consecutive-day docs plus one far-away doc, repeated across many seeds --
        # the close-together cluster must always land in one split together.
        close = {f"doc_{i}.pdf": date(2026, 6, i + 1) for i in range(5)}
        doc_dates = {**close, "far.pdf": date(2026, 1, 1)}
        for seed in range(10):
            splits = assign_splits_by_date_cluster(doc_dates, seed=seed, window_days=3)
            assert len({splits[name] for name in close}) == 1

    def test_undated_docs_never_merged(self):
        # two undated docs must never be forced into the same cluster/split just
        # because they're both undated -- each is evaluated independently.
        doc_dates = {"a.pdf": None, "b.pdf": None, "c.pdf": date(2026, 1, 1)}
        clusters = _cluster_by_date(doc_dates, window_days=3)
        undated_clusters = [c for c in clusters if set(c) & {"a.pdf", "b.pdf"}]
        assert all(len(c) == 1 for c in undated_clusters)

    def test_covers_all_docs_and_valid_split_names(self):
        doc_dates = {f"doc_{i}.pdf": date(2026, 1, 1 + i) for i in range(20)}
        splits = assign_splits_by_date_cluster(doc_dates, seed=0)
        assert set(splits) == set(doc_dates)
        assert set(splits.values()) <= {"train", "valid", "test"}

    def test_deterministic_under_seed(self):
        doc_dates = {f"doc_{i}.pdf": date(2026, 1, 1) + timedelta(days=i * 5) for i in range(15)}
        assert (assign_splits_by_date_cluster(doc_dates, seed=7)
               == assign_splits_by_date_cluster(doc_dates, seed=7))

    def test_far_apart_dates_are_not_clustered(self):
        # documents a year apart should behave like independent docs, not one cluster,
        # so they can freely land in different splits.
        doc_dates = {"jan.pdf": date(2025, 1, 1), "dec.pdf": date(2025, 12, 31)}
        clusters = _cluster_by_date(doc_dates, window_days=3)
        assert len(clusters) == 2

    def test_window_boundary_is_inclusive(self):
        doc_dates = {"a.pdf": date(2026, 1, 1), "b.pdf": date(2026, 1, 4)}
        clusters = _cluster_by_date(doc_dates, window_days=3)
        assert len(clusters) == 1  # exactly 3 days apart, within the window

    def test_window_boundary_excludes_one_day_over(self):
        doc_dates = {"a.pdf": date(2026, 1, 1), "b.pdf": date(2026, 1, 5)}
        clusters = _cluster_by_date(doc_dates, window_days=3)
        assert len(clusters) == 2  # 4 days apart, outside the window


class TestAssignSplitsByDateClusterStratified:
    def _dates_and_eras(self, n_a: int, n_b: int) -> tuple[dict[str, date], dict[str, str]]:
        # eras never share a date, so no cluster can accidentally span both eras
        dates = {f"a_{i}.pdf": date(2022, 1, 1) + timedelta(days=i * 10) for i in range(n_a)}
        dates.update({f"b_{i}.pdf": date(2026, 1, 1) + timedelta(days=i * 10) for i in range(n_b)})
        eras = {name: "retail_only" for name in dates if name.startswith("a_")}
        eras.update({name: "wholesale_retail" for name in dates if name.startswith("b_")})
        return dates, eras

    def test_every_era_represented_in_all_splits_when_enough_clusters(self):
        doc_dates, doc_eras = self._dates_and_eras(n_a=16, n_b=21)
        splits = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=0)
        for era in ("retail_only", "wholesale_retail"):
            era_splits = {splits[name] for name in doc_eras if doc_eras[name] == era}
            assert era_splits == {"train", "valid", "test"}

    def test_single_cluster_era_lands_in_train_only(self):
        doc_dates, doc_eras = self._dates_and_eras(n_a=1, n_b=20)
        splits = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=0)
        assert splits["a_0.pdf"] == "train"

    def test_two_cluster_era_gets_valid_but_not_forced_test(self):
        doc_dates, doc_eras = self._dates_and_eras(n_a=2, n_b=20)
        splits = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=0)
        a_splits = {splits["a_0.pdf"], splits["a_1.pdf"]}
        assert "valid" in a_splits
        assert "test" not in a_splits  # only 1 cluster left after the valid floor -- stays train

    def test_mixed_era_cluster_raises(self):
        doc_dates = {"x.pdf": date(2026, 1, 1), "y.pdf": date(2026, 1, 1)}
        doc_eras = {"x.pdf": "retail_only", "y.pdf": "wholesale_retail"}
        with pytest.raises(ValueError):
            assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=0, window_days=3)

    def test_covers_all_docs(self):
        doc_dates, doc_eras = self._dates_and_eras(n_a=16, n_b=21)
        splits = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=0)
        assert set(splits) == set(doc_dates)

    def test_deterministic_under_seed(self):
        doc_dates, doc_eras = self._dates_and_eras(n_a=16, n_b=21)
        a = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=7)
        b = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=7)
        assert a == b

    def test_adjacent_dates_within_an_era_never_straddle_splits(self):
        close = {f"a_{i}.pdf": date(2026, 6, i + 1) for i in range(5)}
        doc_dates = {**close, "a_far.pdf": date(2022, 1, 1)}
        doc_eras = {name: "retail_only" for name in doc_dates}
        for seed in range(10):
            splits = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=seed, window_days=3)
            assert len({splits[name] for name in close}) == 1


class TestWriteCoco:
    def test_schema_and_xywh(self, tmp_path):
        pages = [
            PageBoxes(image_name="doc_000_p0.png", width=1000, height=1400,
                      boxes=[("Table", (100.0, 200.0, 600.0, 900.0), 0.98),
                             ("Text", (50.0, 50.0, 950.0, 150.0), 0.90)]),
            PageBoxes(image_name="doc_000_p1.png", width=1000, height=1400, boxes=[]),
        ]
        out = tmp_path / "_annotations.coco.json"
        write_coco(pages, out)
        coco = json.loads(out.read_text())

        assert {c["name"] for c in coco["categories"]} == set(CLASS_NAMES)
        assert [im["file_name"] for im in coco["images"]] == ["doc_000_p0.png", "doc_000_p1.png"]
        assert len(coco["annotations"]) == 2

        ann = coco["annotations"][0]
        # COCO bbox is [x, y, w, h], not [x0, y0, x1, y1]
        assert ann["bbox"] == [100.0, 200.0, 500.0, 700.0]
        assert ann["area"] == pytest.approx(500.0 * 700.0)
        assert ann["iscrowd"] == 0
        cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
        assert cat_by_id[ann["category_id"]] == "Table"
        img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
        assert img_by_id[ann["image_id"]] == "doc_000_p0.png"

    def test_annotation_ids_unique(self, tmp_path):
        pages = [
            PageBoxes(image_name=f"p{i}.png", width=100, height=100,
                      boxes=[("Text", (0.0, 0.0, 10.0, 10.0), 1.0)] * 3)
            for i in range(2)
        ]
        out = tmp_path / "coco.json"
        write_coco(pages, out)
        coco = json.loads(out.read_text())
        ids = [a["id"] for a in coco["annotations"]]
        assert len(ids) == len(set(ids)) == 6
