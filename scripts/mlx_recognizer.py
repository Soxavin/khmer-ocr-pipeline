"""Run Qwen2.5-VL-7B (4-bit, MLX) locally on the real eval pages -> predictions.json.

The recognizer A/B's third contender, run fully local on Apple Silicon. mlx-vlm needs
transformers>=5.1 but Surya pins <5.0, so this MUST run ISOLATED from the project env:

    uv run --no-project --with "mlx-vlm>=0.6.3,<0.7" python scripts/mlx_recognizer.py

Reads eval/datasets/real/*.png that have a matching *_ground_truth.json and writes
predictions.json = {"<image_file.png>": "<recognized text>"}. Score it in the main env
with the identical recognition metric used for Surya/Tesseract:

    uv run python scripts/eval_recognizers.py --predictions predictions.json --name qwen2.5-vl-7b-mlx
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
REAL = Path("eval/datasets/real")
OUT = Path("predictions.json")
MAX_TOKENS = 4096
# Greedy decoding (mlx-vlm default temp=0) with no penalty collapses into repetition
# loops on these pages; a repetition penalty breaks the loop deterministically.
REPETITION_PENALTY = 1.4
# NOTE: asking for "Markdown table syntax" made the 4-bit model collapse into an
# empty-grid repetition loop (| | | ...). Plain-text transcription avoids that and
# also removes the markdown-vs-GT scoring bias.
PROMPT = (
    "Read this image and transcribe ALL of its text exactly as written, in natural "
    "top-to-bottom, left-to-right reading order. The text is Khmer (ភាសាខ្មែរ) and "
    "includes numbers. Output plain text only — no tables, no Markdown, no formatting, "
    "no translation, no commentary. Put each line or row on its own line."
)


def main() -> int:
    pngs = [p for p in sorted(glob.glob(str(REAL / "*.png")))
            if Path(p).with_name(Path(p).name.replace(".png", "_ground_truth.json")).exists()]
    if not pngs:
        print(f"No scored PNGs in {REAL}")
        return 1

    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)

    preds = {}
    for path in pngs:
        name = Path(path).name
        prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
        res = generate(model, processor, prompt, image=[path], max_tokens=MAX_TOKENS,
                       repetition_penalty=REPETITION_PENALTY, verbose=False)
        preds[name] = res.text
        print(f"{name}: {len(res.text)} chars")

    OUT.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(preds)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
