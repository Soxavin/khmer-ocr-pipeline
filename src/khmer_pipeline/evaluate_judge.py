from __future__ import annotations
import base64
import json
import os
import sys
import warnings
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

# Required env var: OPENAI_API_KEY
_SYSTEM_PROMPT = (
    "You are an expert Khmer OCR evaluator. I will provide an image of a Khmer "
    "document and the text extracted by an OCR pipeline. Evaluate the accuracy based on:\n"
    "1. Hallucinations: Text in the output that is NOT in the image.\n"
    "2. Omissions: Text clearly visible in the image that is MISSING from the output.\n"
    "3. Character Accuracy: Especially check for correct rendering of stacked Khmer "
    "consonants (cheung) and ligatures.\n"
    "Respond STRICTLY in JSON format matching this schema:\n"
    "{\n"
    '  "overall_score": <integer 0-100>,\n'
    '  "estimated_cer_percent": <integer 0-100>,\n'
    '  "hallucinated_words": [<list of strings>],\n'
    '  "omitted_words": [<list of strings>],\n'
    '  "reasoning": "<brief 1-2 sentence explanation>"\n'
    "}"
)

_MAX_RESPONSE_TOKENS = 1000

_FALLBACK: dict = {
    "overall_score": 0,
    "estimated_cer_percent": 100,
    "hallucinated_words": [],
    "omitted_words": [],
    "reasoning": "Evaluation API failed",
}


def evaluate_ocr_quality(
    image_path: str,
    extracted_text: str,
    model: str = "gpt-4o",
) -> dict:
    if OpenAI is None:
        warnings.warn("openai package not installed; evaluation unavailable")
        return _FALLBACK.copy()

    try:
        suffix = Path(image_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        image_data = base64.b64encode(Path(image_path).read_bytes()).decode()
        data_uri = f"data:{mime};base64,{image_data}"

        client = OpenAI()  # reads OPENAI_API_KEY from environment
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": f"Extracted text:\n{extracted_text}"},
                    ],
                },
            ],
            max_tokens=_MAX_RESPONSE_TOKENS,
        )

        raw = response.choices[0].message.content or ""
        # Strip markdown code fences the LLM may wrap around the JSON
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        warnings.warn("Evaluation API returned unexpected JSON structure")
        return _FALLBACK.copy()

    except Exception as exc:
        warnings.warn(f"Evaluation API failed: {exc}")
        return _FALLBACK.copy()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m khmer_pipeline.evaluate_judge <image_path> <extracted_text_or_txt_file>")
        sys.exit(1)

    image_path = sys.argv[1]
    text_arg = sys.argv[2]

    # Allow passing a .txt file for long extracted text to avoid shell escaping issues
    if Path(text_arg).is_file() and text_arg.endswith(".txt"):
        extracted_text = Path(text_arg).read_text(encoding="utf-8")
    else:
        extracted_text = text_arg

    result = evaluate_ocr_quality(image_path, extracted_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
