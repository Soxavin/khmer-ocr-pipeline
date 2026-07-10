from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from khmer_pipeline.datagen.harvest_table_gt import (
    build_row_lexicon,
    cap_empty_cells,
    grid_to_markdown,
    harvest_corpus,
    is_numeric_text,
    khmer_order_valid,
    passes_numeric_qa,
    prune_empty_columns,
)


class TestGridToMarkdown:
    def test_basic_pipe_table(self):
        md = grid_to_markdown([["h1", "h2"], ["a", "1"], ["b", "2"]])
        lines = md.splitlines()
        assert lines[0] == "| h1 | h2 |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| a | 1 |"
        assert lines[3] == "| b | 2 |"

    def test_pipe_in_cell_is_escaped(self):
        md = grid_to_markdown([["a|b", "c"]])
        assert "a\\|b" in md and md.count("|") >= 3

    def test_ragged_rows_padded(self):
        md = grid_to_markdown([["h1", "h2", "h3"], ["only-one"]])
        last = md.splitlines()[-1]
        assert last == "| only-one |  |  |"

    def test_none_cells_become_empty(self):
        md = grid_to_markdown([[None, "x"]])
        assert md.splitlines()[0] == "|  | x |"


class TestKhmerOrderValid:
    def test_valid_khmer_passes(self):
        for t in ("៛/គ.ក", "ស្ពៃខៀវ", "តារាងតម្លៃ", "ត្រី", ""):
            assert khmer_order_valid(t), t

    def test_scrambled_khmer_fails(self):
        # real scrambled cells from the ARDB text layer: stray marks at token starts
        for t in ("ម្សបខៀវ\nៃ", "សបត្ កដីរោែ ់\nោ", "មខទនំ ញិ\nុ"):
            assert not khmer_order_valid(t), t

    def test_non_khmer_passes(self):
        assert khmer_order_valid("1,200")
        assert khmer_order_valid("kg / day")


class TestPruneEmptyColumns:
    def test_drops_all_empty_columns(self):
        grid = [["a", "", "b", ""], ["c", "", "d", ""]]
        assert prune_empty_columns(grid) == [["a", "b"], ["c", "d"]]

    def test_keeps_partially_filled_columns(self):
        grid = [["a", "", ""], ["b", "x", ""]]
        assert prune_empty_columns(grid) == [["a", ""], ["b", "x"]]

    def test_ragged_rows_ok(self):
        grid = [["a", "", "z"], ["b"]]
        assert prune_empty_columns(grid) == [["a", "z"], ["b", ""]]


class TestBuildRowLexicon:
    GT = [
        ["២៣", "ពងមាន់", "៛/គ្រាប់", "360", "500", "0.00%"],
        ["២៤", "ពងទា", "៛/គ្រាប់", "350", "500", "-2.86%"],
    ]
    # same rows as the text layer extracts them: numbers clean, Khmer scrambled
    TL = [
        ["២៣", "ពងមន់ា", "៛/គ្រាប់", "360", "500", "0.00%"],
        ["២៤", "ពងទ\nា", "៛/គ្រាប់", "350", "500", "-2.86%"],
    ]

    def test_maps_scrambled_to_verified(self):
        lex = build_row_lexicon(self.GT, self.TL)
        assert lex["ពងមន់ា"] == "ពងមាន់"
        assert lex["ពងទា"] == "ពងទា"  # whitespace-stripped key

    def test_row_without_unique_fingerprint_skipped(self):
        gt = [["x", "1", "2"], ["y", "1", "2"]]  # identical fingerprints
        tl = [["x2", "1", "2"], ["y2", "1", "2"]]
        assert build_row_lexicon(gt, tl) == {}

    def test_khmer_cell_count_mismatch_skipped(self):
        gt = [["ពងមាន់", "ក", "360", "500"]]
        tl = [["ពងមន់ា", "360", "500"]]  # one Khmer cell fewer → unsafe to zip
        assert build_row_lexicon(gt, tl) == {}

    def test_conflicting_mappings_dropped(self):
        gt = [["ពងមាន់", "111", "222"], ["ពងស", "333", "444"]]
        tl = [["ដូចគ្នា", "111", "222"], ["ដូចគ្នា", "333", "444"]]
        assert build_row_lexicon(gt, tl) == {}


class TestNumericHelpers:
    def test_is_numeric_text(self):
        for t in ("1,234", "-2.86%", "360", "7,800", "0.00%", "១២៣"):
            assert is_numeric_text(t), t
        for t in ("៛/គ.ក", "abc", "12 kg", ""):
            assert not is_numeric_text(t), t

    def test_passes_numeric_qa_rejects_malformed(self):
        # Stage-4 malformed patterns: bad comma grouping, ≥2-digit integer percent
        assert not passes_numeric_qa("1,2345")
        assert not passes_numeric_qa("294%")
        assert passes_numeric_qa("2.94%")
        assert passes_numeric_qa("1,234")


class TestCapEmptyCells:
    def test_caps_ratio(self):
        cells = [{"text": ""}] * 80 + [{"text": "x"}] * 20
        kept = cap_empty_cells(cells, max_empty_ratio=0.25)
        n_empty = sum(1 for c in kept if not c["text"])
        assert n_empty <= len(kept) * 0.25 + 1
        assert sum(1 for c in kept if c["text"]) == 20  # non-empties all survive

    def test_no_cap_needed(self):
        cells = [{"text": "a"}, {"text": ""}, {"text": "b"}, {"text": "c"}]
        assert cap_empty_cells(cells, max_empty_ratio=0.5) == cells


def _ruled_table_pdf(tmp_path: Path) -> Path:
    """Draw a 3x2 ruled table with digits/latin text so find_tables detects it."""
    doc = fitz.open()
    page = doc.new_page()
    x0, y0, col_w, row_h, cols, rows = 72, 72, 120, 30, 2, 3
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * row_h), (x0 + cols * col_w, y0 + r * row_h))
    for c in range(cols + 1):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + rows * row_h))
    texts = [["item", "price"], ["rice", "1,200"], ["fish", "2.94%"]]
    for r, row in enumerate(texts):
        for c, t in enumerate(row):
            page.insert_text((x0 + c * col_w + 8, y0 + r * row_h + 20), t, fontsize=11)
    # filler paragraph so the doc passes the substantial-text-layer gate
    page.insert_text((72, 400), "market price bulletin " * 8, fontsize=10)
    out = tmp_path / "table_doc.pdf"
    doc.save(str(out))
    doc.close()
    return out


class TestHarvestCorpus:
    def test_end_to_end_on_drawn_table(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _ruled_table_pdf(corpus)
        out = tmp_path / "out"
        counts = harvest_corpus(corpus, out, exclude_stems=[])

        assert counts["sft_pairs"] == 1
        assert counts["recognition_pairs"] >= 5  # 6 filled cells, maybe minus QA/empties

        # single doc → everything lands in train
        rec_jsonl = out / "recognition" / "train" / "pairs.jsonl"
        rows = [json.loads(l) for l in rec_jsonl.read_text().splitlines()]
        texts = {r["text"] for r in rows if r["text"]}
        assert {"rice", "1,200", "2.94%"} <= texts
        for r in rows:
            assert (out / "recognition" / "train" / r["image"]).exists()

        sft_rows = [json.loads(l) for l in
                    (out / "sft" / "train" / "pairs.jsonl").read_text().splitlines()]
        assert "| rice | 1,200 |" in sft_rows[0]["markdown"]
        assert (out / "sft" / "train" / sft_rows[0]["image"]).exists()

    def test_exclude_stems(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _ruled_table_pdf(corpus)
        out = tmp_path / "out"
        counts = harvest_corpus(corpus, out, exclude_stems=["table_doc"])
        assert counts["sft_pairs"] == 0 and counts["recognition_pairs"] == 0
