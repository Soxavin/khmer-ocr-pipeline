"""Validate engine behaviour on a genuine no-table (text-only) page.

§2.18 worried the hybrid engine fabricates tables on text pages, but that was
tested on a mislabelled page (p3 is really a table). This runs both engines on a
true text page — CambodiaBudgetExecutioninApr-2024.pdf p2 — and reports whether
hybrid hallucinates a table (Tables_Found > 0) and how its text CER compares to
Surya's. Born-digital text layer (PyMuPDF) is the ground truth.

    uv run python scripts/eval_notable_page.py
"""
from __future__ import annotations
from pathlib import Path
import fitz
import numpy as np

from khmer_pipeline.models import PreprocessResult
from khmer_pipeline.surya import run_surya
from khmer_pipeline.hybrid_engine import run_hybrid
from khmer_pipeline.evaluate_structure import evaluate_document

_PDF = Path("sample_data/CambodiaBudgetExecutioninApr-2024.pdf")
_PAGE = 1  # 0-indexed → page 2
_DPI = 200
_GT_OUT = Path("eval/datasets/real/CambodiaBudgetExecutioninApr-2024_p2_textlayer_gt.txt")


def main() -> int:
    if not _PDF.exists():
        print(f"PDF not found: {_PDF}")
        return 1
    doc = fitz.open(_PDF)
    page = doc[_PAGE]
    gt_text = page.get_text().strip()
    mat = fitz.Matrix(_DPI / 72, _DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
    doc.close()

    _GT_OUT.write_text(gt_text, encoding="utf-8")
    print(f"page 2 text-layer GT: {len(gt_text)} chars  (saved {_GT_OUT.name})")

    pre = PreprocessResult(source_name="cambodiabudget_p2", page_images=[arr], dpi=_DPI, page_count=1)
    gt = {"paragraphs": [gt_text], "tables": [], "footer": ""}

    for name, eng in [("surya", run_surya), ("hybrid_rowband", run_hybrid)]:
        res = eng(pre)
        p = res.pages[0]
        tables_found = len(p.tables)
        cells = sum(len(t.get("cells", [])) for t in p.tables)
        dc = evaluate_document(p.ocr_text, p.tables, gt)["document_cer"]
        print(f"\n{name}: Tables_Found={tables_found} (phantom if >0)  table_cells={cells}  "
              f"ocr_chars={len(p.ocr_text)}  Document_CER={dc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
