from __future__ import annotations

import json

import pytest

from khmer_pipeline.datagen.pseudo_label_layout import (
    CLASS_NAMES,
    PageBoxes,
    assign_splits,
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
