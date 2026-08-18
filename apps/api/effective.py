"""What a run's "Auto" choices actually resolved to.

Auto DPI and the Auto engine router both decide at run time, from the document
itself. Without reporting the outcome, "Auto" is an unauditable black box in the
UI — the analyst cannot tell 200 from 300, or which recognizer read the page.
"""
from __future__ import annotations

from typing import Any, Optional

# Decision record appended by khmer_pipeline.engines.auto_engine.run_auto:
#   "[AutoRouter] kept surya_kiri | frac=... cutoff=..."
#   "[AutoRouter] fallback surya_kiri->surya | frac=... cutoff=..."
_ROUTER_PREFIX = "[AutoRouter]"


def effective_engine(requested: Optional[str], warnings: list[str]) -> Optional[str]:
    """The engine key that actually read the pages, or None when not yet decided.

    `requested` is the run's `ocr_engine_key`; `warnings` the run's warning list.
    An explicit engine resolves to itself; "auto" is answered only by the router's
    own note, never guessed."""
    for w in warnings:
        if not w.startswith(_ROUTER_PREFIX):
            continue
        body = w[len(_ROUTER_PREFIX):].split("|")[0].strip()
        if "->" in body:
            return body.split("->")[-1].strip()
        if body.startswith("kept "):
            return body[len("kept "):].strip()
    if requested and requested != "auto":
        return requested
    return None


def effective_dpi(doc: Any) -> Optional[int]:
    """The concrete render DPI this document was rasterized at, or None pre-ingest."""
    dpi = getattr(doc.ingest_result, "dpi", None)
    return int(dpi) if isinstance(dpi, (int, float)) and dpi > 0 else None
