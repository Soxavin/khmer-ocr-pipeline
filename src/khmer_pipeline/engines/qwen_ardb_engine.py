"""Qwen3.5-0.8B ARDB fine-tune OCR engine — built and tested end-to-end, but
NOT YET exposed in the UI (see webapp/api.py's _ENGINES).

Every Qwen ARDB config tested so far fails to produce usable output on real
documents (eval/qwen_finetune_runs.md) — unlike gemma_ardb_engine.py's adapter,
none has cleared real-doc evaluation. Shipping it selectable now would let an
analyst pick a config that visibly breaks, defeating the point of "experiment
and see how it performs." This module exists so the shared infrastructure
(subprocess runner, parsing, transform) is proven against BOTH models, and so
flipping Qwen on later — once a config is named — is a one-line addition to
`webapp/api.py`'s _ENGINES, not new engineering. `_ADAPTER_REPO_ID` below is a
placeholder; update it when a config passes eval.

Runs on top of `unsloth/Qwen3.5-0.8B`, in the same isolated-subprocess pattern
as gemma_ardb_engine.py (Qwen3.5's fine-tune notebook needed torch==2.10.0 +
causal_conv1d built --no-binary — a different, also-incompatible pin set from
the main project's transformers<5.0). Registered as ``qwen_ardb``. Table
regions are markdown pipe tables in this fine-tune's output (not HTML, unlike
Gemma) — see finetune_ardb/parsing.py's markdown_table_to_grid.

Behaviour contract (mirrors gemma_ardb_engine.py / gemini_engine.py):
  - FAIL SOFT per page: a subprocess error or unparseable output leaves that
    page empty and the run continues.
  - Always emits an audit warning naming the model.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

from ..models import PreprocessResult, SuryaResult, SuryaPageResult
from .finetune_ardb.subprocess_runner import FineTuneConfig, run_isolated_inference
from .finetune_ardb.parsing import parse_regions
from .finetune_ardb.transform import regions_to_page

# No config has passed real-doc eval yet (see module docstring) — this repo id
# is a deliberate placeholder, not a real pushed adapter. Update it, and add
# this engine's entry to webapp/api.py's _ENGINES, once one does.
_ADAPTER_REPO_ID = "Soxavin/qwen35-ardb-lora-PLACEHOLDER-not-yet-selected"

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
)


def run_qwen_ardb(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
    on_step: Optional[Callable[[str], None]] = None,
) -> SuryaResult:
    """Run the Qwen3.5-0.8B ARDB fine-tune over every page image, returning the
    standard SuryaResult. Fails soft on a per-page subprocess or parse error
    (empty page + warning); never raises mid-run. Not reachable from the UI
    yet — see module docstring."""
    total = len(result.page_images)
    warns: list[str] = [
        f"[Experimental] {total} page(s) run through the Qwen3.5 ARDB fine-tune "
        f"({_CONFIG.adapter_repo_id}) — under active evaluation, no config has "
        "cleared real-document eval yet"
    ]
    pages: list[SuryaPageResult] = []
    for idx, img in enumerate(result.page_images):
        if on_page is not None:
            on_page(idx, total)
        if on_step is not None:
            on_step("qwen_ardb")
        try:
            raw = run_isolated_inference(img, _CONFIG)
        except Exception as exc:  # noqa: BLE001 — fail soft: one page can't kill the run
            warns.append(f"Qwen ARDB fine-tune failed on page {idx + 1}: {exc!r} — page left empty")
            pages.append(SuryaPageResult(page_index=idx, text_blocks=[], tables=[], ocr_text=""))
            continue
        regions = parse_regions(raw)
        if regions is None:
            warns.append(f"Qwen ARDB output for page {idx + 1} was not parseable JSON — page left empty")
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
