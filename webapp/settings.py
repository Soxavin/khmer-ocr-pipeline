"""Extraction settings for the NiceGUI review UI.

`Settings` mirrors the sidebar controls of the Streamlit `app.py` one-for-one so the two
UIs drive the pipeline identically. `settings_key` reproduces app.py's re-run guard: a
change to any field that affects the pipeline yields a new key, which tells the runner the
cached results are stale and a re-run is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from khmer_pipeline.model_config import ANOMALY_THRESHOLD


@dataclass
class Settings:
    # "auto" inspects each document's density to pick 200 or 300 (see
    # ingest.resolve_auto_dpi); an explicit int forces that render DPI.
    dpi: "int | str" = "auto"
    page_scope: str = "all"          # "all" | "single" | "range" | "list"
    page_num: int = 1
    page_start: int = 1
    page_end: int = 5
    # Disjoint page picks from the grid overview (1-based, like the fields above).
    page_list: list[int] = field(default_factory=list)

    remove_stamps: bool = True
    sharpen: bool = True
    normalise: bool = True
    deskew: bool = True
    normalise_table_backgrounds: bool = True

    # "auto" routes per document (surya_kiri, falling back to surya on low
    # confidence) and matched the best manual engine on all 7 GT pages (§2.57).
    ocr_engine_key: str = "auto"     # "auto" | "surya" | "surya_kiri" | "surya_kiri_vlm"
    tables_only: bool = False

    enable_qwen: bool = False
    anomaly_threshold: float = ANOMALY_THRESHOLD

    repair_tables: bool = False
    stitch_pages: bool = True
    convert_numerals: bool = False

    show_layout: bool = True
    overlay_mode: str = "Region type"  # "Region type" | "Confidence"

    @property
    def invalid_range(self) -> bool:
        """True when Page-range mode has 'To' before 'From' — a run must be blocked."""
        return self.page_scope == "range" and int(self.page_end) < int(self.page_start)

    def page_indices(self, doc_page_count: int) -> list[int] | None:
        """0-based page indices to rasterize, or None for all pages. Clamped to the
        document length, mirroring app.py's selection logic."""
        if self.page_scope == "single":
            idx = max(0, min(int(self.page_num) - 1, max(0, doc_page_count - 1)))
            return [idx]
        if self.page_scope == "range":
            start = max(0, int(self.page_start) - 1)
            # Settings persist across uploads, so a range can start past THIS
            # document's end. Without the clamp, max(start + 1, end) below forced a
            # phantom index through and ingest was asked for a page that does not
            # exist (the same defect as the frontend's pagesFromSettings).
            if doc_page_count:
                start = min(start, doc_page_count - 1)
            end = min(int(self.page_end), doc_page_count) if doc_page_count else int(self.page_end)
            return list(range(start, max(start + 1, end)))
        if self.page_scope == "list" and self.page_list:
            limit = doc_page_count if doc_page_count else max(self.page_list)
            picked = sorted({int(p) - 1 for p in self.page_list if 1 <= int(p) <= limit})
            return picked or None  # every pick clamped away → defensively all pages
        return None

    def settings_key(self, upload_id: str) -> str:
        """Stable signature of everything that changes pipeline output. Matches the
        role of app.py's `settings_key` f-string."""
        if self.page_scope == "single":
            page_part = f"page_{self.page_num}"
        elif self.page_scope == "range":
            page_part = f"range_{self.page_start}_{self.page_end}"
        elif self.page_scope == "list" and self.page_list:
            page_part = "list_" + "_".join(str(p) for p in sorted(set(self.page_list)))
        else:
            page_part = "all"
        return "_".join(str(x) for x in (
            upload_id, self.dpi, page_part, self.remove_stamps, self.sharpen,
            self.normalise, self.enable_qwen, self.convert_numerals, self.repair_tables,
            self.stitch_pages, self.anomaly_threshold, self.deskew,
            self.normalise_table_backgrounds, self.ocr_engine_key,
        ))
