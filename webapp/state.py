"""Per-client application state for the NiceGUI review UI.

`AppState` (one per browser connection) holds the shared `Settings` and a list of
`Document`s — one per uploaded file — so a batch of files can be reviewed sequentially,
each with its own results, edits, pagination, and cell↔image selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .settings import Settings


@dataclass
class Progress:
    """Mutable holder the OCR page-callback (running in a worker thread) writes to and a
    UI timer reads from on the event loop. Simple scalar writes only — GIL-atomic."""
    stage: str = ""
    page: int = 0
    total: int = 0
    fraction: float = 0.0
    active: bool = False
    # Set by the Stop button or on client disconnect; the runner checks it at every
    # stage boundary and inside the per-page OCR callback (page-granular abort).
    cancel_requested: bool = False


@dataclass
class Document:
    """One uploaded file and everything derived from it."""
    upload_name: str
    upload_bytes: bytes
    upload_id: str
    doc_page_count: int = 0

    ingest_result: Any = None
    preprocess_result: Any = None
    surya_result: Any = None
    postprocess_result: Any = None
    export_result: Any = None

    stage_times: dict[str, float] = field(default_factory=dict)
    last_key: str | None = None
    run_error: str | None = None

    progress: Progress = field(default_factory=Progress)
    current_page_idx: int = 0
    edited_tables: dict[str, list[list[str]]] = field(default_factory=dict)
    edited_text: dict[int, str] = field(default_factory=dict)
    # table_id → analyst marked the table verified (React review workflow).
    reviewed: dict[str, bool] = field(default_factory=dict)
    selected: tuple | None = None  # (table_id, row, col) currently linked cell↔box

    @property
    def has_results(self) -> bool:
        return self.export_result is not None

    def results_are_stale(self, settings: Settings) -> bool:
        """Results reflect `last_key`; stale once the sidebar changes the settings key."""
        return (
            self.has_results
            and self.last_key is not None
            and self.last_key != settings.settings_key(self.upload_id)
        )

    def reset_run(self) -> None:
        """Clear only run + edit state (keep the upload) — the analogue of app.py's
        `_reset_run_state`, so 'Retry' re-runs the same upload."""
        self.ingest_result = None
        self.preprocess_result = None
        self.surya_result = None
        self.postprocess_result = None
        self.export_result = None
        self.stage_times = {}
        self.last_key = None
        self.run_error = None
        self.progress = Progress()
        self.current_page_idx = 0
        self.edited_tables = {}
        self.reviewed = {}
        self.edited_text = {}
        self.selected = None


@dataclass
class AppState:
    settings: Settings = field(default_factory=Settings)
    documents: list[Document] = field(default_factory=list)
    active: int = 0

    def doc(self) -> Document | None:
        if not self.documents:
            return None
        self.active = max(0, min(self.active, len(self.documents) - 1))
        return self.documents[self.active]

    def add_document(self, name: str, data: bytes, upload_id: str, page_count: int) -> None:
        """Append an uploaded file (deduped by content id) and make it active."""
        for i, d in enumerate(self.documents):
            if d.upload_id == upload_id:
                self.active = i
                return
        self.documents.append(Document(name, data, upload_id, page_count))
        self.active = len(self.documents) - 1

    def clear_documents(self) -> None:
        self.documents = []
        self.active = 0
