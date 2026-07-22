"""Guard against scoring an engine against ground truth its own model drafted.

Some GT in `eval/datasets/` is LLM-drafted and then human-verified against pixels
(see `scripts/review_scan_gt.py`). That is legitimate GT — but scoring a
Gemini-backed engine against Gemini-drafted GT is circular even after
verification: both sides share the same failure modes, so agreement overstates
accuracy. Human verification reduces this but cannot remove it, because a
verifier confirms what is on the page and is least likely to catch precisely the
cells where the drafting model was confidently wrong.

Comparison is by model FAMILY, not exact checkpoint — the shared weights are what
create the correlation, so "gemini-2.5-pro" drafting and "gemini-3-flash" scoring
still count as circular.
"""
from __future__ import annotations

# Substrings that identify a model family in a free-text provenance string.
# Ordered longest-first within a family so a more specific token wins.
_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "gemini": ("gemini",),
    "openai": ("gpt-4", "gpt4", "gpt-5", "gpt", "openai", "o3", "o4"),
    "anthropic": ("claude", "anthropic", "opus", "sonnet", "haiku"),
    "qwen": ("qwen",),
    "mistral": ("mistral", "pixtral"),
    "internvl": ("internvl",),
}

# Stage-3 OCR engines backed by a hosted model. Local engines are absent by
# design: an unknown key returns None, so a new local engine can never
# accidentally trip the guard.
_ENGINE_FAMILIES: dict[str, str] = {
    "gemini": "gemini",
    "mistral_ocr": "mistral",
}


def _sniff_family(text: str) -> str | None:
    lowered = text.lower()
    for family, markers in _FAMILY_MARKERS.items():
        if any(marker in lowered for marker in markers):
            return family
    return None


def drafting_model_family(gt: dict) -> str | None:
    """Model family that drafted this GT, or None if it is model-free.

    Prefers the explicit `gt_drafted_by` field; falls back to sniffing the older
    free-text `gt_source` (e.g. "gemini_draft_human_verified") so GT written
    before the field existed is still protected."""
    drafted_by = gt.get("gt_drafted_by")
    if drafted_by:
        return _sniff_family(str(drafted_by))
    source = gt.get("gt_source")
    if source:
        return _sniff_family(str(source))
    return None


def engine_model_family(engine_key: str) -> str | None:
    """Model family behind an OCR engine key, or None for local/unknown engines."""
    return _ENGINE_FAMILIES.get(engine_key)


def is_circular(engine_key: str, gt: dict) -> bool:
    """True when this engine and this GT share a model family (scoring is circular)."""
    engine_family = engine_model_family(engine_key)
    if engine_family is None:
        return False
    return engine_family == drafting_model_family(gt)


def circularity_note(engine_key: str, gt: dict) -> str | None:
    """Human-readable warning when scoring would be circular, else None."""
    if not is_circular(engine_key, gt):
        return None
    family = engine_model_family(engine_key)
    return (
        f"CIRCULAR: engine {engine_key!r} and this ground truth are both {family}-derived "
        f"(gt_drafted_by/gt_source={gt.get('gt_drafted_by') or gt.get('gt_source')!r}). "
        f"The score will be optimistic and is not comparable with other engines."
    )
