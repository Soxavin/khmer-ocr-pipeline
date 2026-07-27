"""Gemini cloud OCR engine — a labelled, opt-in alternative to the local engines.

Sends each page image to Google's Gemini API and asks for the same layout+HTML
contract as dots.ocr, so tables flow through the production HTML parser and
`colspan`/`rowspan` survive (Surya 2 dropped them; §2.40). Registered as
``OCR_ENGINE=gemini`` and surfaced in the UI under a **Cloud** group.

Data note: on Google's free tier, prompts/outputs may be used to improve their
models and seen by reviewers. The engine emits an audit warning naming what was
sent, and the UI labels it "Cloud — not for confidential documents".

Behaviour contract:
  - FAIL FAST on setup: no API key, or the SDK not installed, raises before any
    network call, with an actionable message (the runner shows it as run_error).
  - FAIL SOFT per page: a page's API error (one 429 retry first), or unparseable
    output, leaves that page empty and the run continues — one bad page in a
    multi-page document never discards the others.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models import PreprocessResult, SuryaResult, SuryaPageResult, Table, TextBlock
from .surya import _parse_html_table_with_spans, _build_table_from_grid

# Overridable without a redeploy. Default is the full Flash (quality on the free
# tier); set GEMINI_MODEL=gemini-flash-lite-latest for higher quota / lower latency.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# Retry once on a rate-limit (429) with this pause; free tier is ~10-15 req/min.
_RATE_LIMIT_BACKOFF_S = 2.0

# Categories that are NOT tables become TextBlocks. dots.ocr's taxonomy, which we
# reuse for the prompt; anything unrecognised is still kept as a text block.
_TABLE_CATEGORY = "Table"

# Same layout contract as scripts/spike_gemini.py — tables as HTML with spans.
_PROMPT = (
    "Please output the layout information from this document image, including each "
    "layout element's bbox, its category, and the corresponding text content within "
    "the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', "
    "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', "
    "'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: omit the text field.\n"
    "    - Formula: LaTeX.\n"
    "    - Table: HTML. Preserve colspan and rowspan attributes exactly as printed.\n"
    "    - All others: Markdown.\n\n"
    "4. Constraints:\n"
    "    - Output the original text with no translation.\n"
    "    - Transcribe Khmer script exactly as printed; do not romanize or paraphrase.\n"
    "    - Sort all elements in human reading order.\n\n"
    "5. Final Output: a single JSON array, with no markdown fences.\n"
)


def _make_client(api_key: str):
    """Build a genai client. Isolated so tests can monkeypatch it (no network).

    Raises ImportError if the optional SDK is absent — caught and re-raised as an
    actionable RuntimeError by the caller."""
    from google import genai
    return genai.Client(api_key=api_key)


def _png_bytes(img: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _normalize_bbox(raw) -> list[float]:
    """Coerce Gemini's bbox to a flat [x1, y1, x2, y2], or [] if unusable.

    Gemini is inconsistent: sometimes a flat 4-number box, sometimes a nested
    polygon [[x, y], ...] (seen live on the first real call). Nested forms
    collapse to their bounding rect. bbox is not load-bearing for scoring (the
    grid comes from the HTML), so anything malformed degrades to []."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return []
    if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in raw):
        try:
            xs = [float(p[0]) for p in raw]
            ys = [float(p[1]) for p in raw]
        except (TypeError, ValueError):
            return []
        return [min(xs), min(ys), max(xs), max(ys)]
    try:
        flat = [float(v) for v in raw]
    except (TypeError, ValueError):
        return []
    return flat if len(flat) == 4 else []


def _parse_layout(raw: str) -> list[dict] | None:
    """Decode the layout array; tolerate a ```json fence or a trailing tail, but
    never bracket-slice (bbox values contain brackets)."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value, _ = json.JSONDecoder().raw_decode(s)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, list):
                return v
        return None
    return value if isinstance(value, list) else None


def _elements_to_page(elements: list[dict], page_index: int) -> SuryaPageResult:
    """Split a layout array into TextBlocks + Tables (via the shared HTML parser)."""
    text_blocks: list[TextBlock] = []
    tables: list[Table] = []
    pooled: list[str] = []
    order = 0
    for el in elements:
        # Gemini varies its key names run-to-run: "category"/"bbox" on one call,
        # "label"/"box" on the next (observed live on gemini-3.6-flash). Accept both.
        category = el.get("category") or el.get("label")
        text = el.get("text") or ""
        bbox = _normalize_bbox(el.get("bbox") if el.get("bbox") is not None else el.get("box"))
        if category == _TABLE_CATEGORY:
            grid_map, spans = _parse_html_table_with_spans(text)
            table = _build_table_from_grid(grid_map, text, bbox)
            # _build_table_from_grid keeps only text; carry the parsed colspans onto
            # the anchor cells so merged headers survive to export — the §2.40
            # capability Surya 2's own table-rec can no longer emit.
            if spans:
                for cell in table["cells"]:
                    span = spans.get((cell["row_id"], cell["col_id"]))
                    if span and span > 1:
                        cell["col_span"] = span
            tables.append(table)
            pooled.extend(
                t["text"] for c in table["cells"] for t in (c.get("text_lines") or [])
                if t.get("text")
            )
        elif category == "Picture":
            continue  # per the prompt, pictures carry no text
        else:
            block: TextBlock = {"text": text, "bbox": bbox,
                                "label": category or "Text", "reading_order": order}
            text_blocks.append(block)
            order += 1
            if text:
                pooled.append(text)
    return SuryaPageResult(page_index=page_index, text_blocks=text_blocks,
                           tables=tables, ocr_text="\n".join(pooled))


def _generate(client, img_bytes: bytes) -> str:
    """One Gemini call for one page image; one 429 retry with backoff."""
    from google.genai import types
    contents = [types.Part.from_bytes(data=img_bytes, mime_type="image/png"), _PROMPT]
    config = types.GenerateContentConfig(temperature=0.0)
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
    except Exception as exc:  # noqa: BLE001 — one retry only on rate-limit
        if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
            time.sleep(_RATE_LIMIT_BACKOFF_S)
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents,
                                                   config=config)
        else:
            raise
    return resp.text or ""


def run_gemini(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
    on_step: Optional[Callable[[str], None]] = None,
) -> SuryaResult:
    """Run Gemini over every page image, returning the standard SuryaResult.

    Fails fast if the API key or SDK is missing; fails soft on a per-page API or
    parse error (empty page + warning). Always appends an audit warning naming
    the model and page count sent to Google."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini engine selected but no GEMINI_API_KEY (or GOOGLE_API_KEY) is set. "
            "See docs/ENGINE_EVALUATION.md §7."
        )
    try:
        client = _make_client(api_key)
    except ImportError as exc:
        raise RuntimeError(
            f"Gemini engine needs the google-genai SDK ({exc}). "
            "Install it: uv add --optional eval-extras google-genai"
        ) from exc

    total = len(result.page_images)
    warns: list[str] = [
        f"[Cloud] {total} page(s) sent to Google Gemini ({GEMINI_MODEL}) — "
        "not for confidential documents"
    ]
    pages: list[SuryaPageResult] = []
    for idx, img in enumerate(result.page_images):
        if on_page is not None:
            on_page(idx, total)
        if on_step is not None:
            on_step("cloud")
        try:
            raw = _generate(client, _png_bytes(img))
        except Exception as exc:  # noqa: BLE001 — fail soft: one page can't kill the run
            warns.append(f"Gemini failed on page {idx + 1}: {exc!r} — page left empty")
            pages.append(SuryaPageResult(page_index=idx, text_blocks=[], tables=[], ocr_text=""))
            continue
        elements = _parse_layout(raw)
        if elements is None:
            warns.append(f"Gemini output for page {idx + 1} was not parseable JSON — page left empty")
            pages.append(SuryaPageResult(page_index=idx, text_blocks=[], tables=[], ocr_text=""))
            continue
        pages.append(_elements_to_page(elements, idx))

    for w in warns:
        warnings.warn(w)
    return SuryaResult(source_name=result.source_name, pages=pages, warnings=warns)
