from __future__ import annotations

import json
from pathlib import Path

import pytest

from khmer_pipeline.datagen.build_layout_detection_sft import build, coco_box_to_gemma, convert_row

_COCO_HF_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "ardb_layout_coco_v1_hf"


class TestCocoBoxToGemma:
    def test_full_image_box_normalizes_to_0_1000(self):
        assert coco_box_to_gemma([0, 0, 2000, 2000], width=2000, height=2000) == [0, 0, 1000, 1000]

    def test_quarter_box(self):
        assert coco_box_to_gemma([0, 0, 1000, 500], width=2000, height=2000) == [0, 0, 250, 500]

    def test_order_is_y1_x1_y2_x2(self):
        # x=100 (narrow width offset), y=200 (taller height offset) -> y should map from height
        box = coco_box_to_gemma([100, 200, 10, 10], width=1000, height=2000)
        assert box[0] == 100  # y1 from height=2000
        assert box[1] == 100  # x1 from width=1000


class TestConvertRow:
    def test_builds_expected_json_shape(self):
        row = {
            "width": 2000, "height": 2000,
            "objects": {"category": ["Table", "Picture"],
                       "bbox": [[0, 0, 1000, 1000], [1000, 1000, 500, 500]]},
        }
        boxes = json.loads(convert_row(row))
        assert boxes == [
            {"box_2d": [0, 0, 500, 500], "label": "Table"},
            {"box_2d": [500, 500, 750, 750], "label": "Picture"},
        ]


@pytest.mark.skipif(not _COCO_HF_DIR.is_dir(), reason="packaged COCO layout dataset not present")
class TestBuildIntegration:
    def test_frozen_eval_docs_excluded_and_output_written(self, tmp_path):
        counts = build(_COCO_HF_DIR, tmp_path)
        # COCO totals 91 pages across 30 docs (not an exact 3/doc average); this just
        # pins the measured count after excluding the 2 frozen docs' pages.
        assert sum(counts.values()) == 85
        for split, n in counts.items():
            if n == 0:
                continue
            lines = (tmp_path / split / "pairs.jsonl").read_text().splitlines()
            assert len(lines) == n
            row = json.loads(lines[0])
            assert set(row) == {"image", "instruction", "text", "doc_id", "source"}
            assert "09.06.26" not in row["source"] and "15.06.26" not in row["source"]
            json.loads(row["text"])  # target must itself be valid JSON
