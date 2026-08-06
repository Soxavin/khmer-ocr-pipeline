"""Qwen3.5-0.8B ARDB fine-tune OCR engine — exposed in the UI as a labeled
"trial" engine, deliberately included BECAUSE it is unreliable.

Every Qwen ARDB config tested so far fails to produce valid structured output
on most real documents (eval/qwen_finetune_runs.md). Unlike a normal
experimental engine (whose risk is being slower or narrower), this one's risk
is unreliable OUTPUT — that is the point: the user wants to see the actual
garbled/failed generation live, not read about it in eval logs. See
webapp/api.py's _ENGINES entry (`trial: True`) and the "Trial" chip in
SettingsDrawer.tsx for how this is surfaced honestly rather than hidden.

Runs on top of `unsloth/Qwen3.5-0.8B`, in the same isolated-subprocess pattern
as gemma_ardb_engine.py (Qwen3.5's fine-tune notebook needed torch==2.10.0 +
causal_conv1d built --no-binary — a different, also-incompatible pin set from
the main project's transformers<5.0). Registered as ``qwen_ardb``. Table
regions are markdown pipe tables in this fine-tune's output (not HTML, unlike
Gemma) — see finetune_ardb/parsing.py's markdown_table_to_grid.

Behaviour contract (differs from gemma_ardb_engine.py in ONE deliberate way):
  - FAIL SOFT per page: a subprocess error or unparseable output leaves that
    page's structured fields (tables/text_blocks) empty and the run continues.
  - Unlike Gemma, an UNPARSEABLE-JSON page preserves the raw model text in
    `ocr_text` rather than discarding it — this flows through
    postprocess.py's untouched `raw_ocr_text` field to the review UI's
    raw-output fallback (PageTextPanel.tsx), so a failed generation is VISIBLE,
    not blank. A subprocess-level failure (no model output at all, e.g. a
    dependency error) has no raw text to preserve and stays empty.
  - Always emits an audit warning naming the model.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

from ..models import PreprocessResult, SuryaResult, SuryaPageResult
from .finetune_ardb.subprocess_runner import FineTuneConfig, run_isolated_inference
from .finetune_ardb.parsing import parse_regions
from .finetune_ardb.transform import regions_to_page

# The most-tested Qwen adapter — chosen deliberately despite its unreliability
# (see module docstring). Not the "best" by eval score; there isn't one that
# passes. One hardcoded adapter id, same pattern as Gemma: no epoch-variant
# picker.
_ADAPTER_REPO_ID = "Soxavin/qwen35-ardb-lora-v5-e3-ga2"

# Distinct on_step marker (not one of Surya's layout/text/tables steps) so the
# frontend can show engine-appropriate "may take several minutes" copy instead
# of nothing — see RunControls.tsx's SUB_STEPS.
_STEP_SLOW_FINETUNE = "finetune_slow"

_CONFIG = FineTuneConfig(
    base_model_id="unsloth/Qwen3.5-0.8B",
    adapter_repo_id=_ADAPTER_REPO_ID,
    infer_script="src/khmer_pipeline/engines/finetune_ardb/qwen_ardb_infer.py",
    # torch==2.10.0 + causal_conv1d --no-binary: the exact combination
    # scripts/colab_qwen35_finetune.ipynb's install cell documents as required
    # for this model's Gated DeltaNet kernels (transformers itself stays at the
    # official 5.2.0 pin, confirmed unnecessary to bump once the real cause —
    # a mismatched prebuilt causal_conv1d wheel — was found).
    extra_pins=["transformers>=5.0,<6", "peft>=0.13,<1", "torch==2.10.0"],
    table_text_format="markdown",
    # Generation here is known-slow and may hit its token cap without ever
    # producing a clean stop — well beyond Gemma's 600s default (per the user:
    # "generation can legitimately take several minutes per page... needs its
    # own expectation-setting rather than reusing whatever timeout assumptions
    # exist for the OCR pipeline or Gemma").
    timeout_s=1200,
)


def run_qwen_ardb(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
    on_step: Optional[Callable[[str], None]] = None,
) -> SuryaResult:
    """Run the Qwen3.5-0.8B ARDB fine-tune over every page image, returning the
    standard SuryaResult. Fails soft on a per-page subprocess or parse error;
    never raises mid-run. Unlike gemma_ardb_engine.py, an unparseable-JSON page
    preserves the raw model text in `ocr_text` (see module docstring) instead
    of discarding it — the expected, not exceptional, outcome for this engine."""
    total = len(result.page_images)
    warns: list[str] = [
        f"[Trial] {total} page(s) run through the Qwen3.5 ARDB fine-tune "
        f"({_CONFIG.adapter_repo_id}) — frequently fails to produce valid "
        "structured output; shown for comparison, not for real extraction"
    ]
    pages: list[SuryaPageResult] = []
    for idx, img in enumerate(result.page_images):
        if on_page is not None:
            on_page(idx, total)
        if on_step is not None:
            on_step(_STEP_SLOW_FINETUNE)
        try:
            raw = run_isolated_inference(img, _CONFIG)
        except Exception as exc:  # noqa: BLE001 — fail soft: one page can't kill the run
            warns.append(f"Qwen ARDB fine-tune failed on page {idx + 1}: {exc!r} — page left empty")
            pages.append(SuryaPageResult(page_index=idx, text_blocks=[], tables=[], ocr_text=""))
            continue
        regions = parse_regions(raw)
        if regions is None:
            warns.append(f"Qwen ARDB output for page {idx + 1} was not parseable JSON — showing raw model output")
            # ocr_text=raw (not ""): the whole point of exposing this engine is
            # seeing the actual failure, not a blank page. Flows unmodified
            # through postprocess.py's raw_ocr_text to the review UI's
            # raw-output fallback (see PageTextPanel.tsx).
            pages.append(SuryaPageResult(page_index=idx, text_blocks=[], tables=[], ocr_text=raw))
            continue
        h, w = img.shape[0], img.shape[1]
        pages.append(regions_to_page(
            regions, page_index=idx, table_text_format=_CONFIG.table_text_format,
            page_w=w, page_h=h,
        ))

    for w in warns:
        warnings.warn(w)
    return SuryaResult(source_name=result.source_name, pages=pages, warnings=warns)
