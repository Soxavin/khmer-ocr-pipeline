from __future__ import annotations
import csv
import io
from unittest.mock import MagicMock
from khmer_pipeline.export import export
from khmer_pipeline.models import ExportResult, PostprocessResult, CorrectedPageResult


def _make_page(page_index=0, tables=None, qwen_used=False, corrected_text="", correction_diff="", raw_ocr_text=""):
    p = MagicMock(spec=CorrectedPageResult)
    p.page_index = page_index
    p.tables = tables or []
    p.qwen_used = qwen_used
    p.corrected_text = corrected_text
    p.correction_diff = correction_diff
    p.raw_ocr_text = raw_ocr_text
    return p


def _make_result(source_name="ardb_sample.pdf", pages=None):
    r = MagicMock(spec=PostprocessResult)
    r.source_name = source_name
    r.pages = pages or []
    return r


_CELL = {"row_id": 0, "col_id": 0, "text_lines": [{"text": chr(0x179F) + chr(0x17D2), "bbox": [0, 0, 10, 10]}], "bbox": [0, 0, 10, 10]}
_TABLE = {"rows": [{"row_id": 0}], "cols": [{"col_id": 0}], "cells": [_CELL], "image_bbox": [0, 0, 100, 100]}


def test_export_returns_export_result():
    assert isinstance(export(_make_result()), ExportResult)


def test_document_json_has_required_keys():
    doc = export(_make_result()).document_json
    for key in ("source_name", "extracted_at", "page_count", "pages"):
        assert key in doc


def test_document_json_page_count_matches():
    doc = export(_make_result(pages=[_make_page(0), _make_page(1)])).document_json
    assert doc["page_count"] == 2


def test_document_json_pages_have_required_keys():
    doc = export(_make_result(pages=[_make_page(0)])).document_json
    for key in ("page_index", "qwen_used", "corrected_text", "tables"):
        assert key in doc["pages"][0]


def _grid_table(grid):
    cells = [{"row_id": r, "col_id": c, "text_lines": ([{"text": t}] if t else []), "bbox": []}
             for r, row in enumerate(grid) for c, t in enumerate(row)]
    n = max((len(r) for r in grid), default=0)
    return {"rows": [{"row_id": i} for i in range(len(grid))],
            "cols": [{"col_id": j} for j in range(n)], "cells": cells}


def test_stitch_pages_merges_continuation_into_one_csv():
    H = ["a", "b"]
    pages = [_make_page(0, tables=[_grid_table([H, ["1", "x"]])]),
             _make_page(1, tables=[_grid_table([H, ["2", "y"]])])]
    res = export(_make_result(pages=pages), stitch_pages=True)
    assert len(res.tables_csv) == 1
    assert res.tables_csv[0][0] == "ardb_sample_table1"
    assert res.document_json["document_tables"][0]["source_pages"] == [0, 1]


def test_no_stitch_keeps_per_page_tables():
    H = ["a", "b"]
    pages = [_make_page(0, tables=[_grid_table([H, ["1", "x"]])]),
             _make_page(1, tables=[_grid_table([H, ["2", "y"]])])]
    res = export(_make_result(pages=pages), stitch_pages=False)
    assert len(res.tables_csv) == 2
    assert "document_tables" not in res.document_json


def test_table_id_naming_convention():
    result = export(_make_result(source_name="ardb_sample.pdf", pages=[_make_page(0, tables=[_TABLE])]))
    assert result.tables_csv[0][0] == "ardb_sample_page1_table1"


def test_tables_csv_length_matches_total_tables():
    pages = [_make_page(0, tables=[_TABLE]), _make_page(1, tables=[_TABLE])]
    assert len(export(_make_result(pages=pages)).tables_csv) == 2


def test_csv_is_utf8_bom():
    _, csv_string = export(_make_result(pages=[_make_page(0, tables=[_TABLE])])).tables_csv[0]
    assert csv_string.startswith("﻿")


def test_csv_cell_text_joined_from_text_lines():
    cell = {"row_id": 0, "col_id": 0, "text_lines": [{"text": chr(0x179F), "bbox": [0, 0, 5, 5]}, {"text": chr(0x1780), "bbox": [5, 0, 10, 5]}], "bbox": [0, 0, 10, 10]}
    table = {"rows": [{}], "cols": [{}], "cells": [cell], "image_bbox": [0, 0, 100, 100]}
    _, csv_string = export(_make_result(pages=[_make_page(0, tables=[table])])).tables_csv[0]
    rows = list(csv.reader(io.StringIO(csv_string.lstrip("﻿"))))
    assert rows[0][0] == chr(0x179F) + " " + chr(0x1780)


def test_empty_table_produces_empty_csv():
    table = {"rows": [], "cols": [], "cells": [], "image_bbox": [0, 0, 100, 100]}
    _, csv_string = export(_make_result(pages=[_make_page(0, tables=[table])])).tables_csv[0]
    rows = [r for r in csv.reader(io.StringIO(csv_string.lstrip("﻿"))) if any(r)]
    assert rows == []


def test_no_tables_produces_empty_csv_list():
    assert export(_make_result(pages=[_make_page(0)])).tables_csv == []


def test_raw_ocr_text_not_in_json():
    doc = export(_make_result(pages=[_make_page(0, raw_ocr_text="raw")])).document_json
    assert "raw_ocr_text" not in str(doc)


def test_correction_diff_not_in_json():
    doc = export(_make_result(pages=[_make_page(0, correction_diff="--- a\n+++ b")])).document_json
    assert "correction_diff" not in str(doc)


def test_khmer_numerals_converted_when_flag_set():
    cell = {"row_id": 0, "col_id": 0,
            "text_lines": [{"text": "១២,០០០", "bbox": [0, 0, 10, 10]}],
            "bbox": [0, 0, 10, 10]}
    table = {"rows": [{}], "cols": [{}], "cells": [cell], "image_bbox": [0, 0, 100, 100]}
    result = export(_make_result(pages=[_make_page(0, tables=[table])]),
                     convert_numerals=True)
    _, csv_string = result.tables_csv[0]
    rows = list(csv.reader(io.StringIO(csv_string.lstrip("﻿"))))
    assert rows[0][0] == "12,000"


def test_khmer_numerals_preserved_when_flag_not_set():
    cell = {"row_id": 0, "col_id": 0,
            "text_lines": [{"text": "១២,០០០", "bbox": [0, 0, 10, 10]}],
            "bbox": [0, 0, 10, 10]}
    table = {"rows": [{}], "cols": [{}], "cells": [cell], "image_bbox": [0, 0, 100, 100]}
    result = export(_make_result(pages=[_make_page(0, tables=[table])]),
                     convert_numerals=False)
    _, csv_string = result.tables_csv[0]
    rows = list(csv.reader(io.StringIO(csv_string.lstrip("﻿"))))
    assert rows[0][0] == "១២,០០០"


def test_json_not_affected_by_convert_numerals():
    cell = {"row_id": 0, "col_id": 0,
            "text_lines": [{"text": "១២,០០០", "bbox": [0, 0, 10, 10]}],
            "bbox": [0, 0, 10, 10]}
    table = {"rows": [{}], "cols": [{}], "cells": [cell], "image_bbox": [0, 0, 100, 100]}
    result = export(_make_result(pages=[_make_page(0, tables=[table])]),
                     convert_numerals=True)
    # JSON should still have original Khmer numerals
    json_str = str(result.document_json)
    assert "១២,០០០" in json_str


def test_cell_missing_row_id_defaults_to_zero():
    cell = {
        "col_id": 0,
        "text_lines": [{"text": "x", "bbox": [0, 0, 5, 5]}],
        "bbox": [0, 0, 5, 5],
    }
    table = {"rows": [{}], "cols": [{}], "cells": [cell], "image_bbox": [0, 0, 100, 100]}
    result = export(_make_result(pages=[_make_page(0, tables=[table])]))

    _, csv_string = result.tables_csv[0]
    rows = list(csv.reader(io.StringIO(csv_string.lstrip("﻿"))))
    assert rows[0][0] == "x"

    doc = result.document_json
    assert doc["pages"][0]["tables"][0]["cells"][0]["row"] == 0


def test_consistent_table_not_repaired():
    from khmer_pipeline.export import _validate_and_repair_table
    table = {
        "rows": [{"row_id": 0}, {"row_id": 1}],
        "cols": [{"col_id": 0}, {"col_id": 1}],
        "cells": [
            {"row_id": 0, "col_id": 0, "text_lines": [], "bbox": [0, 0, 10, 10]},
            {"row_id": 0, "col_id": 1, "text_lines": [], "bbox": [10, 0, 20, 10]},
            {"row_id": 1, "col_id": 0, "text_lines": [], "bbox": [0, 10, 10, 20]},
            {"row_id": 1, "col_id": 1, "text_lines": [], "bbox": [10, 10, 20, 20]},
        ],
        "image_bbox": [0, 0, 20, 20],
    }
    repaired, was_repaired = _validate_and_repair_table(table)
    assert was_repaired is False
    assert repaired is table


def test_inconsistent_table_repaired():
    from khmer_pipeline.export import _validate_and_repair_table
    table = {
        "rows": [{"row_id": 0}, {"row_id": 1}],
        "cols": [{"col_id": 0}, {"col_id": 1}],
        "cells": [
            {"row_id": 0, "col_id": 0, "text_lines": [{"text": "a"}], "bbox": [0, 0, 10, 10]},
            {"row_id": 0, "col_id": 1, "text_lines": [{"text": "b"}], "bbox": [10, 0, 20, 10]},
            {"row_id": 1, "col_id": 0, "text_lines": [{"text": "c"}], "bbox": [0, 10, 10, 20]},
        ],
        "image_bbox": [0, 0, 20, 20],
    }
    repaired, was_repaired = _validate_and_repair_table(table)
    assert was_repaired is True
    assert repaired["was_repaired"] is True

    from collections import Counter
    rows: dict[int, set] = {}
    for c in repaired["cells"]:
        rows.setdefault(c.get("row_id", 0), set()).add(c.get("col_id", 0))
    col_counts = {len(cols) for cols in rows.values()}
    assert len(col_counts) == 1  # all rows now have the same column count


def test_repaired_table_exports_without_error():
    table = {
        "rows": [{"row_id": 0}, {"row_id": 1}],
        "cols": [{"col_id": 0}, {"col_id": 1}],
        "cells": [
            {"row_id": 0, "col_id": 0, "text_lines": [{"text": "a"}], "bbox": [0, 0, 10, 10]},
            {"row_id": 0, "col_id": 1, "text_lines": [{"text": "b"}], "bbox": [10, 0, 20, 10]},
            {"row_id": 1, "col_id": 0, "text_lines": [{"text": "c"}], "bbox": [0, 10, 10, 20]},
        ],
        "image_bbox": [0, 0, 20, 20],
    }
    result = export(_make_result(pages=[_make_page(0, tables=[table])]), repair_tables=True)
    table_id, csv_string = result.tables_csv[0]
    rows = list(csv.reader(io.StringIO(csv_string.lstrip("﻿"))))
    assert len(rows) == 2
    assert all(len(r) == 2 for r in rows)
    assert result.document_json["pages"][0]["tables"][0]["was_repaired"] is True


def test_empty_table_not_repaired():
    from khmer_pipeline.export import _validate_and_repair_table
    table = {"rows": [], "cols": [], "cells": [], "image_bbox": [0, 0, 0, 0]}
    repaired, was_repaired = _validate_and_repair_table(table)
    assert was_repaired is False
    assert repaired is table


def test_table_not_repaired_by_default():
    table = {
        "rows": [{"row_id": 0}, {"row_id": 1}],
        "cols": [{"col_id": 0}, {"col_id": 1}],
        "cells": [
            {"row_id": 0, "col_id": 0, "text_lines": [{"text": "a"}], "bbox": [0, 0, 10, 10]},
            {"row_id": 0, "col_id": 1, "text_lines": [{"text": "b"}], "bbox": [10, 0, 20, 10]},
            {"row_id": 1, "col_id": 0, "text_lines": [{"text": "c"}], "bbox": [0, 10, 10, 20]},
        ],
        "image_bbox": [0, 0, 20, 20],
    }
    result = export(_make_result(pages=[_make_page(0, tables=[table])]))
    assert result.document_json["pages"][0]["tables"][0]["was_repaired"] is False
    # Unrepaired table still has 3 cells (no padding added)
    assert len(result.document_json["pages"][0]["tables"][0]["cells"]) == 3
