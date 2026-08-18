"""Upload identity and probing, shared by both UIs.

`webapp/api.py` (REST, backing the React workspace) and `webapp/main.py` (the
NiceGUI fallback) each used to carry their own copy of this logic, and the copies
had silently diverged: `upload_id` was `md5[:12]` in the API and `md5[:8]` in
NiceGUI, so the *same file* uploaded through the two UIs produced two different
ids -- and therefore two different `Settings.settings_key` values, meaning cached
results could never be shared between them. `probe_pages` was duplicated verbatim,
one copy a nested closure, so a change to the accepted-suffix rule in one would
have made the two UIs disagree on page counts.

Anything both UIs must agree on belongs here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

# Suffixes the pipeline accepts. Drives the upload widgets' `accept` attribute and
# the PDF-vs-image branch in probe_pages. The React side keeps its own copy in
# frontend/src/lib/uploads.ts -- keep the two in step (a cross-language constant
# would mean serving this from /api/meta).
ACCEPTED_SUFFIXES: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

_ID_LEN = 12


def accept_attr() -> str:
    """The comma-separated suffix list for an upload widget's `accept`."""
    return ",".join(ACCEPTED_SUFFIXES)


def upload_id(data: bytes) -> str:
    """Content-addressed id for an uploaded document.

    Content-addressed, so re-uploading the same bytes reuses the cached run.
    Feeds `Settings.settings_key`, so both UIs must derive it identically."""
    return hashlib.md5(data).hexdigest()[:_ID_LEN]


def probe_pages(name: str, data: bytes) -> int:
    """Page count for PDFs (0 = unreadable), 1 for images."""
    if Path(name).suffix.lower() != ".pdf":
        return 1
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return len(doc)
    except Exception:
        return 0
