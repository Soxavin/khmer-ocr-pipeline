from __future__ import annotations
import difflib
import warnings
import unicodedata

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
    text = unicodedata.normalize("NFC", text)
    for wrong, correct in RULE_BASED_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text


ANOMALY_THRESHOLD: float = 0.15  # tunable — proportion of foreign-script chars that triggers correction


def _is_foreign_script(ch: str) -> bool:
    """Returns True if character belongs to a script that should not appear
    in Khmer financial documents."""
    cp = ord(ch)
    return (
        0x0D80 <= cp <= 0x0DFF or  # Sinhala
        0x0E80 <= cp <= 0x0EFF or  # Lao
        0x0E00 <= cp <= 0x0E7F or  # Thai
        0x1000 <= cp <= 0x109F or  # Myanmar
        0x0600 <= cp <= 0x06FF or  # Arabic script (not numerals)
        0x4E00 <= cp <= 0x9FFF or  # CJK Unified Ideographs
        0x3040 <= cp <= 0x30FF     # Hiragana/Katakana
    )


def _anomaly_score(text: str) -> float:
    """Returns a score from 0.0 (clean) to 1.0 (highly anomalous).
    Score is the proportion of characters from wrong scripts.
    ANOMALY_THRESHOLD = 0.15 is tunable."""
    if not text.strip():
        return 0.0
    total = len(text)
    foreign_count = sum(1 for ch in text if _is_foreign_script(ch))
    return foreign_count / total


def _detect_errors(text: str) -> bool:
    """Thin wrapper kept for backward compatibility with existing tests."""
    return _anomaly_score(text) >= ANOMALY_THRESHOLD


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
    raw = page.ocr_text  # always copied unchanged into raw_ocr_text

    # Process each text block individually
    corrected_block_texts = []
    qwen_used = False
    for block in page.text_blocks:
        block_text = _apply_rules(block.get("text", ""))
        if not skip_qwen and _anomaly_score(block_text) >= ANOMALY_THRESHOLD:
            block_text = _qwen_correct(block_text)
            qwen_used = True
        corrected_block_texts.append(block_text)

    # Rebuild corrected_text from corrected blocks
    # If page has no text blocks (table-only page), fall back to rule-corrected ocr_text
    if corrected_block_texts:
        corrected_text = "\n\n".join(t for t in corrected_block_texts if t)
    else:
        corrected_text = _apply_rules(raw)

    diff = _build_diff(raw, corrected_text)
    return CorrectedPageResult(
        page_index=page.page_index,
        text_blocks=page.text_blocks,  # unchanged
        tables=page.tables,
        raw_ocr_text=raw,
        corrected_text=corrected_text,
        correction_diff=diff,
        qwen_used=qwen_used,
    )


def postprocess(result: SuryaResult, skip_qwen: bool = False) -> PostprocessResult:
    return PostprocessResult(
        source_name=result.source_name,
        pages=[_correct_page(page, skip_qwen=skip_qwen) for page in result.pages],
    )
