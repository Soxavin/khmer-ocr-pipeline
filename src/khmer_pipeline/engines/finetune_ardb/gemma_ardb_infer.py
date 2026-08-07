"""Isolated-subprocess inference for the Gemma 4 E2B ARDB fine-tune.

Run OUTSIDE the project's main venv (Gemma 4 needs transformers>=5.5, which
conflicts with Surya's pin) via `subprocess_runner.run_isolated_inference`,
which invokes this script as:

    uv run --no-project --with "transformers>=5.5,<6" ... \\
        python gemma_ardb_infer.py --base-model <id> --adapter <id> --image <path>

`--base-model` points at a PRE-MERGED checkpoint (`Soxavin/gemma4-e2b-ardb-merged-v5-e3`,
produced by the "Merge for local inference" cell in
scripts/colab_gemma4_e2b_finetune.ipynb), not the bare `unsloth/gemma-4-E2B-it` base + a
separate LoRA adapter. `--adapter` is still accepted (subprocess_runner always passes it —
shared contract with qwen_ardb_infer.py) but unused: loading the original LoRA adapter here
via plain `transformers` + `peft` fails, because `unsloth/gemma-4-E2B-it` uses Unsloth's own
layer classes (e.g. `Gemma4ClippableLinear`) instead of stock `torch.nn.Linear`, which stock
`peft` doesn't recognize as LoRA-injectable (`ValueError: Target module
Gemma4ClippableLinear(...) is not supported`). Merging once, in an Unsloth environment
(Colab), sidesteps that entirely — the merged repo loads with plain `from_pretrained`, no
PEFT, no Unsloth-compatibility issue.

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
    parser.add_argument("--adapter", required=True)  # unused — see module docstring
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # device_map="auto" lets accelerate decide placement across GPU/CPU/disk; on
    # this single-GPU Mac it can choose disk offload. Not needed for correctness
    # now that there's no PEFT injection step, but loading straight onto the one
    # real device is still simpler/faster than letting accelerate guess.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
    ).to(device)
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
