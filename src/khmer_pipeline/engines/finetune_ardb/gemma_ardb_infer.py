"""Isolated-subprocess inference for the Gemma 4 E2B ARDB LoRA fine-tune.

Run OUTSIDE the project's main venv (Gemma 4 needs transformers>=5.5, which
conflicts with Surya's pin) via `subprocess_runner.run_isolated_inference`,
which invokes this script as:

    uv run --no-project --with "transformers>=5.5,<6" --with "peft>=0.13,<1" ... \\
        python gemma_ardb_infer.py --base-model <id> --adapter <id> --image <path>

Prints the model's raw generation (the fine-tune's JSON region list, possibly
truncated/malformed — parsing/repair happens back in the main process via
`finetune_ardb.parsing.parse_regions`) to stdout. Standalone by design: this
process never imports from `khmer_pipeline`, since it runs in a separate,
intentionally-incompatible environment.
"""
from __future__ import annotations

import argparse
import sys

# Matches the unified page-understanding instruction the fine-tune was trained
# on (datagen/build_ardb_unified_sft.py's _UNIFIED_INSTRUCTION) — inference must
# use the identical prompt the model saw during training.
_INSTRUCTION = (
    "Extract every layout region on this page. Output a JSON list of "
    'objects, each {"box_2d": [y1, x1, y2, x2], "label": category, "text": ...} (text is '
    "the region's transcribed content, or an empty string if it has none, e.g. a photo), "
    "with box_2d normalized to a 0-1000 grid. Categories: Table, Text, Section-Header, "
    "Page-Furniture, Picture."
)

_MAX_NEW_TOKENS = 4096


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

    processor = AutoProcessor.from_pretrained(args.base_model)
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": _INSTRUCTION},
    ]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=_MAX_NEW_TOKENS, no_repeat_ngram_size=4,
            use_cache=True, do_sample=False,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
