"""Isolated-subprocess inference for the Qwen3.5-0.8B ARDB LoRA fine-tune.

Same isolated-subprocess contract as gemma_ardb_infer.py (see that module's
docstring) — the specific package pins differ (Qwen3.5's fine-tune notebook
needed torch==2.10.0 and causal_conv1d built --no-binary; see
qwen_ardb_engine.py's FineTuneConfig), but the invocation shape and
loader-object naming ("tokenizer", per the Colab notebook — Qwen3.5's
processor plays the same apply_chat_template role Gemma's `processor` does)
are otherwise the same. Standalone by design: never imports from
`khmer_pipeline`, since it runs in a separate, intentionally-incompatible
environment.

NOTE: no Qwen ARDB config has passed real-document evaluation yet (see
eval/qwen_finetune_runs.md) — this script and qwen_ardb_engine.py are built
and tested end-to-end, but the engine is intentionally NOT exposed in
apps/api/api.py's _ENGINES, so it is not selectable from the UI. Flip it on by
adding one entry there once a config is named.
"""
from __future__ import annotations

import argparse
import sys

# Same unified page-understanding instruction Gemma's fine-tune was trained on
# (datagen/build_ardb_unified_sft.py's _UNIFIED_INSTRUCTION) — both models share
# one instruction/output contract; only the Table region's text format differs
# (Gemma: HTML, Qwen: markdown — see qwen_ardb_engine.py's table_text_format).
_INSTRUCTION = (
    "Extract every layout region on this page. Output a JSON list of "
    'objects, each {"box_2d": [y1, x1, y2, x2], "label": category, "text": ...} (text is '
    "the region's transcribed content, or an empty string if it has none, e.g. a photo), "
    "with box_2d normalized to a 0-1000 grid. Categories: Table, Text, Section-Header, "
    "Page-Furniture, Picture."
)

_MAX_NEW_TOKENS = 4096


def _load_cached_first(fn, *args, **kwargs):
    """Try the local HF cache first (local_files_only=True skips the network
    round-trip entirely — including the HEAD request from_pretrained normally
    makes to revalidate the cache, which has been observed to fail on this
    machine with an intermittent DNS error even though the model and adapter
    are already fully downloaded). Falls back to a normal, network-enabled
    call on a cache miss (first-ever run, or a repo that genuinely changed)."""
    try:
        return fn(*args, local_files_only=True, **kwargs)
    except OSError:
        return fn(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    tokenizer = _load_cached_first(AutoProcessor.from_pretrained, args.base_model)
    base = _load_cached_first(
        AutoModelForImageTextToText.from_pretrained, args.base_model,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = _load_cached_first(PeftModel.from_pretrained, base, args.adapter)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": _INSTRUCTION},
    ]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=_MAX_NEW_TOKENS, no_repeat_ngram_size=4,
            use_cache=True, do_sample=False,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
