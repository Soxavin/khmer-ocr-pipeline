from __future__ import annotations
import io
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageSequence

from .models import IngestResult

MAX_PAGES = 50
DEFAULT_DPI = 200

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def ingest(source: bytes, source_name: str, dpi: int = DEFAULT_DPI,
           page_indices: list[int] | None = None) -> IngestResult:
    """Rasterize a PDF or image to normalized RGB page images.

    `page_indices` (PDF only): 0-based page indices to render; None renders every
    page (default). Out-of-range indices raise ValueError. The MAX_PAGES limit
    applies to the number of pages actually rendered, not the document length —
    so a large PDF is fine as long as few pages are selected. Ignored for image
    inputs (single page). Returned page_index values are 0-based within the
    rendered set."""
    suffix = Path(source_name).suffix.lower()
    if suffix == ".pdf":
        return _ingest_pdf(source, source_name, dpi, page_indices)
    if suffix in _IMAGE_SUFFIXES:
        return _ingest_image(source, source_name)
    raise ValueError(f"Unsupported file type: {suffix!r}. Expected PDF or image.")


def _ingest_pdf(data: bytes, source_name: str, dpi: int,
                page_indices: list[int] | None = None) -> IngestResult:
    try:
        doc_cm = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open PDF '{source_name}': {e}") from e
    with doc_cm as doc:
        doc_len = len(doc)
        if page_indices is None:
            indices = list(range(doc_len))
        else:
            for i in page_indices:
                if i < 0 or i >= doc_len:
                    raise ValueError(
                        f"Page index {i} out of range for document with {doc_len} page(s)."
                    )
            indices = list(page_indices)
        # Limit applies to pages actually rendered, so a huge PDF is fine when the
        # analyst only selected a few pages.
        if len(indices) > MAX_PAGES:
            raise ValueError(
                f"Requested {len(indices)} pages; limit is {MAX_PAGES} for this prototype."
            )
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        images: list[np.ndarray] = []
        for i in indices:
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(arr.copy())
    return IngestResult(
        source_name=source_name,
        page_images=images,
        dpi=dpi,
        page_count=len(images),
    )


def _ingest_image(data: bytes, source_name: str) -> IngestResult:
    # Iterate every frame so multi-frame TIFFs (fax-style multi-page scans) are
    # fully ingested rather than silently truncated to frame 0. Single-frame
    # images yield exactly one page, byte-identical to a plain convert("RGB").
    img = Image.open(io.BytesIO(data))
    images: list[np.ndarray] = []
    for frame in ImageSequence.Iterator(img):
        images.append(np.array(frame.convert("RGB"), dtype=np.uint8))
        if len(images) > MAX_PAGES:
            raise ValueError(
                f"Image has more than {MAX_PAGES} frames; limit is {MAX_PAGES} for this prototype."
            )
    return IngestResult(
        source_name=source_name,
        page_images=images,
        dpi=0,
        page_count=len(images),
    )
