from __future__ import annotations
import html as _html_mod
import warnings
from html.parser import HTMLParser
from typing import Any, Callable, Optional
from PIL import Image
from .models import PreprocessResult, SuryaResult, SuryaPageResult
from .model_config import CONFIDENCE_LOW

_manager = None
_layout_pred = None
_rec_pred = None
_table_pred = None


def models_loaded() -> bool:
    return _manager is not None


def preload_models() -> None:
    _get_predictors()


def _get_predictors():
    global _manager, _layout_pred, _rec_pred, _table_pred
    if _manager is None:
        from surya.inference import SuryaInferenceManager
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.table_rec import TableRecPredictor
        _manager = SuryaInferenceManager()
        _layout_pred = LayoutPredictor(manager=_manager)
        _rec_pred = RecognitionPredictor(manager=_manager)
        _table_pred = TableRecPredictor(manager=_manager)
    return _layout_pred, _rec_pred, _table_pred


def run_surya(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    layout_pred, rec_pred, table_pred = _get_predictors()
    pil_images = [Image.fromarray(img) for img in result.page_images]
    total = len(pil_images)
    pages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for idx, pil_img in enumerate(pil_images):
            if on_page is not None:
                on_page(idx, total)
            pages.append(_process_page(idx, pil_img, layout_pred, rec_pred, table_pred))
        collected_warnings = [str(w.message) for w in caught]
    return SuryaResult(source_name=result.source_name, pages=pages, warnings=collected_warnings)


class _TagStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(p.strip() for p in self._parts if p.strip())


def _html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    stripper = _TagStripper()
    stripper.feed(_html_mod.unescape(html_str))
    return stripper.get_text()


def _process_page(
    page_index: int,
    pil_img: Image.Image,
    layout_pred,
    rec_pred,
    table_pred,
) -> SuryaPageResult:
    try:
        layout_result = layout_pred([pil_img])[0]

        if layout_result.error:
            warnings.warn(f"Layout failed on page {page_index}; returning empty result.")
            return SuryaPageResult(page_index=page_index, text_blocks=[], tables=[], ocr_text="")

        # OCR all text blocks in one batched call; Surya 0.20 handles Table/skip labels internally
        text_blocks: list[dict] = []
        try:
            page_ocr = rec_pred([pil_img], layout_results=[layout_result])[0]
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")
            page_ocr = None

        if page_ocr is not None:
            for block in page_ocr.blocks:
                if block.skipped or block.error:
                    continue
                text = _html_to_text(block.html)
                if not text:
                    continue
                text_blocks.append({
                    "text": text,
                    "bbox": list(block.bbox),
                    "polygon": block.polygon,
                    "confidence": block.confidence or 0.0,
                    "label": block.label,
                    "region_label": block.label,
                    "reading_order": block.reading_order,
                })

        # Blocks from Surya 0.20 are pre-ordered by layout reading order;
        # sort again defensively in case of fallback-path interleaving
        text_blocks.sort(key=lambda b: b.get("reading_order", 0))
        ocr_text = "\n\n".join(b["text"] for b in text_blocks if b.get("text"))

        # Table recognition
        table_bboxes = [b for b in layout_result.bboxes if b.label == "Table"]
        tables: list[dict] = []
        for b in table_bboxes:
            crop = pil_img.crop(tuple(int(v) for v in b.bbox))
            try:
                t = table_pred([crop])[0]
            except Exception as e:
                warnings.warn(f"Table recognition failed on page {page_index}: {e}")
                continue

            tbl = _serialize_table(t)

            # Per-cell OCR: batch all cell crops in a single manager call
            if t.cells:
                cell_crops = [crop.crop(tuple(int(v) for v in c.bbox)) for c in t.cells]
                try:
                    cell_ocrs = rec_pred(cell_crops)
                    for cell_dict, cell_ocr in zip(tbl["cells"], cell_ocrs):
                        combined = " ".join(
                            _html_to_text(blk.html)
                            for blk in cell_ocr.blocks
                            if not blk.skipped and not blk.error and blk.html
                        ).strip()
                        cell_dict["text_lines"] = (
                            [{"text": combined, "bbox": cell_dict["bbox"]}] if combined else []
                        )
                except Exception as e:
                    warnings.warn(f"Cell OCR failed: {e}")

            cells_before = tbl["cells"]
            tbl["cells"] = _filter_phantom_cells(cells_before, tbl["image_bbox"])
            removed = len(cells_before) - len(tbl["cells"])
            if removed:
                warnings.warn(
                    f"Page {page_index}: removed {removed} phantom cell(s) "
                    f"(outside table bounds) from table {len(tables)}."
                )
            tbl["bbox"] = list(b.bbox)
            tables.append(tbl)

        low_conf_blocks = sum(
            1 for b in text_blocks if (b.get("confidence") or 0.0) < CONFIDENCE_LOW
        )
        if low_conf_blocks:
            warnings.warn(
                f"Page {page_index}: {low_conf_blocks} text block(s) have low OCR "
                f"confidence (<{CONFIDENCE_LOW})."
            )

        return SuryaPageResult(
            page_index=page_index,
            text_blocks=text_blocks,
            tables=tables,
            ocr_text=ocr_text,
        )

    except Exception as e:
        warnings.warn(f"Critical failure processing page {page_index}: {e}")
        return SuryaPageResult(
            page_index=page_index,
            text_blocks=[],
            tables=[],
            ocr_text="",
        )


def _serialize_table(t) -> dict[str, Any]:
    cells = []
    for c in t.cells:
        cell_dict = c.model_dump()
        cell_dict["text_lines"] = []  # populated by per-cell OCR step
        cells.append(cell_dict)
    return {
        "rows": [r.model_dump() for r in t.rows],
        "cols": [col.model_dump() for col in t.cols],
        "cells": cells,
        "image_bbox": list(t.image_bbox),
    }


def _filter_phantom_cells(cells: list[dict], parent_bbox: list[float]) -> list[dict]:
    px0, py0, px1, py1 = parent_bbox
    return [
        c for c in cells
        if not (c["bbox"][2] <= px0 or c["bbox"][0] >= px1 or
                c["bbox"][3] <= py0 or c["bbox"][1] >= py1)
    ]
