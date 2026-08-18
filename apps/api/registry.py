"""Process-global document registry for the REST API (React frontend).

Unlike the NiceGUI UI's per-connection `AppState`, the registry survives page
reloads — the React app is refresh-safe by design. Single-analyst localhost
tool: no auth, and a single **global run lock** (one GPU; a second concurrent
run is refused with 409 by the API layer).
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .state import Document

_documents: dict[str, Document] = {}
# Settings JSON of the last run per document (staleness comparison client-side).
_last_run_settings: dict[str, dict] = {}
# Held for the duration of any pipeline run (GPU is a singleton resource).
run_lock = asyncio.Lock()


def add(doc: Document) -> Document:
    """Register `doc`, deduping by content id (existing instance wins)."""
    return _documents.setdefault(doc.upload_id, doc)


def get(doc_id: str) -> Optional[Document]:
    return _documents.get(doc_id)


def remove(doc_id: str) -> bool:
    _last_run_settings.pop(doc_id, None)
    return _documents.pop(doc_id, None) is not None


def all_documents() -> list[Document]:
    return list(_documents.values())


def clear() -> None:
    _documents.clear()
    _last_run_settings.clear()


def set_last_run_settings(doc_id: str, settings: dict) -> None:
    _last_run_settings[doc_id] = settings


def last_run_settings(doc_id: str) -> Optional[dict]:
    return _last_run_settings.get(doc_id)
