from __future__ import annotations
import warnings as _warnings
from typing import Any, Callable, Optional
from ..models import PreprocessResult, SuryaPageResult, SuryaResult

_TESSERACT_LANG = "khm"


def _import_pytesseract():
    # Lazy import + clear error so the module stays importable when
    # pytesseract (or the system tesseract binary) is not installed.
    try:
        import pytesseract
    except ImportError as e:
        raise ImportError(
            "pytesseract is required for the Tesseract engine. "
            "Install with: brew install tesseract tesseract-lang && "
            "uv add 'pytesseract>=0.3,<0.4'"
        ) from e
    return pytesseract


def _dicts_to_words(data: dict) -> list[dict[str, Any]]:
    # Tesseract returns parallel lists per word; re-pack into one dict per word.
    n = len(data.get("text", []))
    words: list[dict[str, Any]] = []
    for i in range(n):
        words.append({
            "text": data["text"][i] or "",
            "conf": int(data["conf"][i]),
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "block_num": int(data["block_num"][i]),
            "par_num": int(data["par_num"][i]),
            "line_num": int(data["line_num"][i]),
        })
    return words


def _build_text_blocks(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Group words by (block_num, par_num, line_num) → one text_block per line.
    # Mirrors Surya's text_block dict shape exactly (same 7 keys).
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for w in words:
        key = (w["block_num"], w["par_num"], w["line_num"])
        groups.setdefault(key, []).append(w)

    blocks: list[dict[str, Any]] = []
    for key in sorted(groups.keys()):
        line_words = sorted(groups[key], key=lambda w: w["left"])
        texts = [w["text"] for w in line_words if w["text"].strip()]
        if not texts:
            continue
        left = min(w["left"] for w in line_words)
        top = min(w["top"] for w in line_words)
        right = max(w["left"] + w["width"] for w in line_words)
        bottom = max(w["top"] + w["height"] for w in line_words)
        confs = [w["conf"] / 100.0 for w in line_words if w["conf"] != -1]
        confidence = sum(confs) / len(confs) if confs else 0.0
        blocks.append({
            "text": " ".join(texts),
            "bbox": [left, top, right, bottom],
            "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
            "confidence": confidence,
            "label": "Text",
            "region_label": "Text",
        })
    for i, b in enumerate(blocks):
        b["reading_order"] = i
    return blocks


def run_tesseract(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    pytesseract = _import_pytesseract()
    from PIL import Image  # project dep, always available

    pages: list[SuryaPageResult] = []
    total = len(result.page_images)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        for idx, img in enumerate(result.page_images):
            try:
                if on_page is not None:
                    on_page(idx, total)
                pil_img = Image.fromarray(img)
                data = pytesseract.image_to_data(
                    pil_img,
                    lang=_TESSERACT_LANG,
                    output_type=pytesseract.Output.DICT,
                )
                blocks = _build_text_blocks(_dicts_to_words(data))
                ocr_text = "\n".join(b["text"] for b in blocks)
                pages.append(SuryaPageResult(
                    page_index=idx,
                    text_blocks=blocks,
                    # Tesseract yields no table structure; downstream table
                    # metrics will be empty for this engine — documented, not a bug.
                    tables=[],
                    ocr_text=ocr_text,
                ))
            except Exception as e:
                _warnings.warn(f"Page {idx + 1} failed in Tesseract engine: {e}")
                pages.append(SuryaPageResult(
                    page_index=idx, text_blocks=[], tables=[], ocr_text=""
                ))
        collected_warnings = [str(w.message) for w in caught]
    return SuryaResult(
        source_name=result.source_name,
        pages=pages,
        warnings=collected_warnings,
    )
