"""Gemma 4 E2B ARDB fine-tune OCR engine — an experimental, opt-in alternative
to the local Surya-based engines.

Runs the LoRA fine-tune `Soxavin/gemma4-e2b-ardb-lora-v5-e3` (the one adapter
that has cleared real-document evaluation; see eval/gemma_finetune_runs.md),
in an isolated subprocess (Gemma 4 needs transformers>=5.5, which conflicts
with Surya's <5.0 pin — see finetune_ardb/subprocess_runner.py). Registered
as ``gemma_ardb`` and surfaced in the UI's Labs/Experimental subsection,
never selected by `auto`.

`base_model_id` below points at a PRE-MERGED checkpoint
(`Soxavin/gemma4-e2b-ardb-merged-v5-e3`), not the bare `unsloth/gemma-4-E2B-it`
+ this adapter loaded separately — loading the adapter via plain
`transformers`+`peft` outside an Unsloth environment fails, since
`unsloth/gemma-4-E2B-it` uses Unsloth's own layer classes that stock `peft`
doesn't recognize as LoRA-injectable. See gemma_ardb_infer.py's docstring and
the "Merge for local inference" cell in
scripts/colab_gemma4_e2b_finetune.ipynb, which produced the merged repo.
`adapter_repo_id` is kept as the original LoRA repo purely for provenance in
warning messages — it identifies which fine-tune this is, even though the
merged checkpoint is what's actually loaded.

Behaviour contract (mirrors qwen_ardb_engine.py — updated after the merged
checkpoint fixed the load-time crash but exposed a real quality problem: the
model's output frequently doesn't follow its trained schema):
  - FAIL SOFT per page: a subprocess error leaves that page empty (no model
    output exists to show); the run continues either way — one bad page never
    discards the others.
  - An UNPARSEABLE-JSON page preserves the raw model text in `ocr_text`
    rather than discarding it, same as qwen_ardb_engine.py — flows through
    postprocess.py's `raw_ocr_text` to the review UI's raw-output fallback
    (PageTextPanel.tsx), so a bad generation is visible, not blank.
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
    base_model_id="Soxavin/gemma4-e2b-ardb-merged-v5-e3",
    adapter_repo_id="Soxavin/gemma4-e2b-ardb-lora-v5-e3",
    infer_script="src/khmer_pipeline/engines/finetune_ardb/gemma_ardb_infer.py",
    # peft dropped: base_model_id is a pre-merged checkpoint, no adapter attachment
    # step needed at inference time (see module docstring).
    # pillow/torchvision are NOT optional extras: the infer script imports PIL, and
    # transformers imports torchvision while building a vision AutoProcessor. Both
    # used to leak in from the project venv, which is exactly the mixed-install bug
    # `uv run --isolated` now prevents (see subprocess_runner.py) — a sealed env has
    # to declare everything it actually uses.
    extra_pins=["transformers>=5.5,<6", "torch", "torchvision", "pillow"],
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
            warns.append(f"Gemma ARDB output for page {idx + 1} was not parseable JSON — showing raw model output")
            # ocr_text=raw (not ""): preserve what the model actually produced —
            # flows through to the review UI's raw-output fallback (see
            # qwen_ardb_engine.py, which established this pattern).
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
