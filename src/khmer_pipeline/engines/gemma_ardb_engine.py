"""Gemma 4 E2B ARDB fine-tune OCR engine — an experimental, opt-in alternative
to the local Surya-based engines.

Runs the LoRA fine-tune `Soxavin/gemma4-e2b-ardb-lora-v5-e3` (the one adapter
that has cleared real-document evaluation; see eval/gemma_finetune_runs.md) on
top of `unsloth/gemma-4-E2B-it`, in an isolated subprocess (Gemma 4 needs
transformers>=5.5, which conflicts with Surya's <5.0 pin — see
finetune_ardb/subprocess_runner.py). Registered as ``gemma_ardb`` and surfaced
in the UI's Labs/Experimental subsection, never selected by `auto`.

Behaviour contract (mirrors gemini_engine.py):
  - FAIL SOFT per page: a subprocess error or unparseable output leaves that
    page empty and the run continues — one bad page never discards the others.
  - Always emits an audit warning naming the model, so a run's provenance is
    traceable even when it succeeds.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

from ..models import PreprocessResult, SuryaResult, SuryaPageResult
from .finetune_ardb.subprocess_runner import FineTuneConfig, InferenceCancelled, run_isolated_inference
from .finetune_ardb.parsing import parse_regions
from .finetune_ardb.transform import regions_to_page

_CONFIG = FineTuneConfig(
    base_model_id="unsloth/gemma-4-E2B-it",
    adapter_repo_id="Soxavin/gemma4-e2b-ardb-lora-v5-e3",
    infer_script="src/khmer_pipeline/engines/finetune_ardb/gemma_ardb_infer.py",
    extra_pins=["transformers>=5.5,<6", "peft>=0.13,<1", "torch"],
    table_text_format="html",
)


def run_gemma_ardb(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
    on_step: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> SuryaResult:
    """Run the Gemma 4 E2B ARDB fine-tune over every page image, returning the
    standard SuryaResult. Fails soft on a per-page subprocess or parse error
    (empty page + warning); never raises mid-run. `is_cancelled`, if given, is
    polled while a page's subprocess is running — on True the subprocess is
    killed and InferenceCancelled propagates out uncaught (a user Stop, not a
    per-page failure to fail soft on)."""
    total = len(result.page_images)
    warns: list[str] = [
        f"[Experimental] {total} page(s) run through the Gemma 4 E2B ARDB "
        f"fine-tune ({_CONFIG.adapter_repo_id}) — under active evaluation, "
        "not production-recommended"
    ]
    pages: list[SuryaPageResult] = []
    for idx, img in enumerate(result.page_images):
        if on_page is not None:
            on_page(idx, total)
        if on_step is not None:
            on_step("gemma_ardb")
        try:
            raw = run_isolated_inference(img, _CONFIG, is_cancelled=is_cancelled)
        except InferenceCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — fail soft: one page can't kill the run
            warns.append(f"Gemma ARDB fine-tune failed on page {idx + 1}: {exc!r} — page left empty")
            pages.append(SuryaPageResult(page_index=idx, text_blocks=[], tables=[], ocr_text=""))
            continue
        regions = parse_regions(raw)
        if regions is None:
            warns.append(f"Gemma ARDB output for page {idx + 1} was not parseable JSON — page left empty")
            pages.append(SuryaPageResult(page_index=idx, text_blocks=[], tables=[], ocr_text=""))
            continue
        h, w = img.shape[0], img.shape[1]
        pages.append(regions_to_page(
            regions, page_index=idx, table_text_format=_CONFIG.table_text_format,
            page_w=w, page_h=h,
        ))

    for w in warns:
        warnings.warn(w)
    return SuryaResult(source_name=result.source_name, pages=pages, warnings=warns)
