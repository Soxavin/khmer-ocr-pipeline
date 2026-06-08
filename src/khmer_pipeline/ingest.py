from __future__ import annotations
import io
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

from .models import IngestResult

MAX_PAGES = 50
DEFAULT_DPI = 200

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def ingest(source: bytes, source_name: str, dpi: int = DEFAULT_DPI) -> IngestResult:
    suffix = Path(source_name).suffix.lower()
    if suffix == ".pdf":
        return _ingest_pdf(source, source_name, dpi)
    if suffix in _IMAGE_SUFFIXES:
        return _ingest_image(source, source_name)
    raise ValueError(f"Unsupported file type: {suffix!r}. Expected PDF or image.")


def _ingest_pdf(data: bytes, source_name: str, dpi: int) -> IngestResult:
    try:
        doc_cm = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open PDF '{source_name}': {e}") from e
    with doc_cm as doc:
        page_count = len(doc)
        if page_count > MAX_PAGES:
            raise ValueError(
                f"Document has {page_count} pages; limit is {MAX_PAGES} for this prototype."
            )
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        images: list[np.ndarray] = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(arr.copy())
    return IngestResult(
        source_name=source_name,
        page_images=images,
        dpi=dpi,
        page_count=page_count,
    )


def _ingest_image(data: bytes, source_name: str) -> IngestResult:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return IngestResult(
        source_name=source_name,
        page_images=[arr],
        dpi=0,
        page_count=1,
    )
