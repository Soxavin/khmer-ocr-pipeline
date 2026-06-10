from __future__ import annotations
import difflib
import warnings

try:
    from mlx_lm import generate
except ImportError:
    generate = None  # type: ignore[assignment]

from .models import SuryaResult, SuryaPageResult, PostprocessResult, CorrectedPageResult

# ---------------------------------------------------------------------------
# Rule table — deliberately empty. Add targeted pairs only after review.
# ---------------------------------------------------------------------------
RULE_BASED_CORRECTIONS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Qwen2.5-VL module-level singletons (lazy-loaded on first use)
# ---------------------------------------------------------------------------
_qwen_model = None
_qwen_tokenizer = None


def _get_qwen():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        from mlx_lm import load
        _qwen_model, _qwen_tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    return _qwen_model, _qwen_tokenizer


# ---------------------------------------------------------------------------
# Correction layers
# ---------------------------------------------------------------------------

def _apply_rules(text: str) -> str:
    for wrong, correct in RULE_BASED_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text


def _detect_errors(text: str) -> bool:
    # Check A: foreign script characters
    for ch in text:
        cp = ord(ch)
        if (0x0D80 <= cp <= 0x0DFF or   # Sinhala
                0x0E00 <= cp <= 0x0E7F or   # Thai
                0x0E80 <= cp <= 0x0EFF or   # Lao
                0x1000 <= cp <= 0x109F or   # Myanmar
                0x0600 <= cp <= 0x06FF or   # Arabic
                0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
                0x3040 <= cp <= 0x30FF):    # Hiragana/Katakana
            return True

    # Check B: Arabic numerals present but no Khmer numerals, only when Khmer text exists.
    # Pure Latin/ASCII strings (dates, headers) must not trigger even with many digits.
    has_khmer = any(0x1780 <= ord(ch) <= 0x17FF for ch in text)
    arabic_count = sum(1 for ch in text if 0x30 <= ord(ch) <= 0x39)
    khmer_count = sum(1 for ch in text if 0x17E0 <= ord(ch) <= 0x17E9)
    if has_khmer and arabic_count > 5 and khmer_count == 0:
        return True

    return False


def _qwen_correct(text: str) -> str:
    prompt = (
        "You are correcting Khmer OCR errors in Khmer financial document text.\n"
        "The document is a daily market price table from Cambodia.\n"
        "Fix misread characters, wrong scripts, and missing diacritics.\n"
        "Return only the corrected Khmer text with no explanation.\n\n"
        "Example 1:\n"
        "Wrong: \"មើន្គារំ ជពរជស\"\n"
        "Correct: \"ធនាគារ ARDB\"\n\n"
        "Example 2:\n"
        "Wrong: \"ពា សាច់ជ្រូករស់\"\n"
        "Correct: \"៣ សាច់ជ្រូករស់\"\n\n"
        "Example 3:\n"
        "Wrong: \"មាន្រ​ ​ ​ ​ ​ ​ ​ ​\"\n"
        "Correct: \"មាន់\"\n\n"
        f"Now correct this text:\n"
        f"Wrong: \"{text}\"\n"
        "Correct:"
    )
    if generate is None:
        warnings.warn("mlx_lm not installed; Qwen correction unavailable")
        return text
    try:
        model, tokenizer = _get_qwen()
        return generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
    except Exception as e:
        warnings.warn(f"Qwen correction failed: {e}")
        return text


def _build_diff(raw: str, corrected: str) -> str:
    diff = difflib.ndiff(raw.splitlines(), corrected.splitlines())
    return "\n".join(diff)


def _correct_page(page: SuryaPageResult, skip_qwen: bool = False) -> CorrectedPageResult:
    raw = page.ocr_text
    after_rules = _apply_rules(raw)
    if not skip_qwen and _detect_errors(after_rules):
        corrected = _qwen_correct(after_rules)
        qwen_used = True
    else:
        corrected = after_rules
        qwen_used = False
    diff = _build_diff(raw, corrected)
    return CorrectedPageResult(
        page_index=page.page_index,
        text_blocks=page.text_blocks,
        tables=page.tables,
        raw_ocr_text=raw,
        corrected_text=corrected,
        correction_diff=diff,
        qwen_used=qwen_used,
    )


def postprocess(result: SuryaResult, skip_qwen: bool = False) -> PostprocessResult:
    return PostprocessResult(
        source_name=result.source_name,
        pages=[_correct_page(page, skip_qwen=skip_qwen) for page in result.pages],
    )
