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

# Auto-DPI: a clean, high-density source needs no more than 200 DPI (enough for
# Khmer table OCR, and faster / lighter on memory); a faint or low-resolution
# scan is rendered at 300 so each glyph carries more pixels for the recognizer.
# The threshold is the source's own native density (embedded-image px per inch).
_AUTO_DPI_CLEAN = 200
_AUTO_DPI_FALLBACK = 300
# Below this native density a source reads as a low-res/faint scan. 250 sits
# comfortably under a true 300-DPI scan and above the ~150-DPI faxed range.
_AUTO_DPI_DENSITY_THRESHOLD = 250.0


# A raster must cover at least this fraction of the page to BE the page — below
# it the image is decoration (masthead, logo, stamp) sitting on a vector page.
# ARDB bulletins embed a 499x142 logo whose density reads ~50 DPI; treating that
# as the page's resolution made every document look like a faint scan.
_SCAN_MIN_PAGE_COVERAGE = 0.5


def _page_raster_coverage(page) -> tuple[float, float | None]:
    """(fraction of the page covered by the largest raster, that raster's density).

    Density is px per inch across the image's PLACED width, which is the real
    resolution the renderer can recover — not the raw pixel count over the page
    width, which conflates a small dense logo with a full-page scan."""
    best_area, best_density = 0.0, None
    page_area = abs(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0, None
    for img in page.get_images(full=True):
        try:
            rect = page.get_image_bbox(img)
        except Exception:
            continue  # images without a resolvable placement can't be judged
        area = abs(rect.width * rect.height)
        if area <= best_area or rect.width <= 0:
            continue
        best_area = area
        px_w = img[2]  # get_images tuple: (xref, smask, width, …)
        best_density = (px_w / (rect.width / 72.0)) if px_w else None
    return best_area / page_area, best_density


def page_is_scanned(page) -> bool:
    """True when the page IS a raster scan rather than a vector page with images.

    Requires a raster covering most of the page. Used both to pick a render DPI
    and, by the engine router, to keep low-resolution scans away from per-cell
    recognizers that cannot resolve Khmer diacritics at that density."""
    coverage, _ = _page_raster_coverage(page)
    return coverage >= _SCAN_MIN_PAGE_COVERAGE


def _page_native_density(page) -> float | None:
    """Native density (px per inch) of a page-covering scan, else None.

    None means "nothing to upscale": either a vector/born-digital page, or one
    whose only rasters are decoration."""
    coverage, density = _page_raster_coverage(page)
    return density if coverage >= _SCAN_MIN_PAGE_COVERAGE else None


def resolve_auto_dpi(source: bytes, source_name: str) -> int:
    """Pick a render DPI for the 'auto' setting by inspecting the source.

    200 for clean, high-density PDFs and for images (ingested at native pixels
    regardless); 300 for faint/low-resolution scans, where more pixels per glyph
    aids Khmer OCR. The worst (lowest-density) page drives the decision — one
    faint page is enough to warrant the higher DPI. Unreadable metadata biases to
    300 (accuracy over speed)."""
    if Path(source_name).suffix.lower() != ".pdf":
        return _AUTO_DPI_CLEAN
    try:
        with fitz.open(stream=source, filetype="pdf") as doc:
            densities = [d for page in doc if (d := _page_native_density(page)) is not None]
    except Exception:
        return _AUTO_DPI_FALLBACK
    if not densities:
        return _AUTO_DPI_CLEAN  # vector/born-digital: nothing to upscale
    return _AUTO_DPI_CLEAN if min(densities) >= _AUTO_DPI_DENSITY_THRESHOLD else _AUTO_DPI_FALLBACK


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
        low_res_scan = False
        for i in indices:
            page = doc.load_page(i)
            density = _page_native_density(page)  # None unless a page-covering scan
            if density is not None and density < _AUTO_DPI_DENSITY_THRESHOLD:
                low_res_scan = True
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(arr.copy())
    return IngestResult(
        source_name=source_name,
        page_images=images,
        dpi=dpi,
        page_count=len(images),
        low_res_scan=low_res_scan,
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
