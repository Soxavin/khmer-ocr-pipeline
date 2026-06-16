from __future__ import annotations
import difflib
import json
import warnings
import unicodedata

try:
    from mlx_lm import generate
except ImportError:
    generate = None  # type: ignore[assignment]

from .models import SuryaResult, SuryaPageResult, PostprocessResult, CorrectedPageResult
from .model_config import ANOMALY_THRESHOLD, STAGE4_MODEL_PATH
from .memory import clear_device_cache

# ---------------------------------------------------------------------------
# Rule table — deliberately empty. Add targeted pairs only after review.
# ---------------------------------------------------------------------------
RULE_BASED_CORRECTIONS: dict[str, str] = {}

_BATCH_MIN_TOKENS = 512
_BATCH_TOKENS_PER_STRING = 100

# ---------------------------------------------------------------------------
# Qwen2.5-VL module-level singletons (lazy-loaded on first use)
# ---------------------------------------------------------------------------
_qwen_model = None
_qwen_tokenizer = None

def _get_qwen():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        try:
            from mlx_lm import load
            _qwen_model, _qwen_tokenizer = load(STAGE4_MODEL_PATH)
        except Exception as e:
            # Crash-proofing: If it fails to load, warn and return None 
            # so we don't keep trying and crashing the pipeline.
            warnings.warn(f"Failed to load Qwen model: {e}. Disabling Qwen fallback for this run.")
            return None, None
    return _qwen_model, _qwen_tokenizer

def qwen_loaded() -> bool:
    return _qwen_model is not None

# ---------------------------------------------------------------------------
# Correction layers
# ---------------------------------------------------------------------------
def _apply_rules(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for wrong, correct in RULE_BASED_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text

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

# ---------------------------------------------------------------------------
# BATCHED VLM CORRECTION (Speed Optimization)
# ---------------------------------------------------------------------------
def _qwen_correct_batch(texts: list[str]) -> list[str]:
    # Single prompt for all anomalous blocks on a page; falls back to originals on parse failure.
    if not texts:
        return []
        
    prompt = (
        "You are correcting Khmer OCR errors in a list of strings from a Cambodian financial document.\n"
        "Fix misread characters, wrong scripts, and missing diacritics.\n"
        "Return ONLY a valid JSON array of strings containing the corrected text. "
        "Do not include any explanations, markdown formatting, or code blocks.\n\n"
        f"Input list: {json.dumps(texts, ensure_ascii=False)}\n\n"
        "Output JSON array:"
    )
    
    if generate is None:
        warnings.warn("mlx_lm not installed; Qwen correction unavailable")
        return texts
        
    model, tokenizer = _get_qwen()
    if model is None:
        return texts  # Model failed to load previously, skip gracefully
        
    try:
        max_tokens = max(_BATCH_MIN_TOKENS, len(texts) * _BATCH_TOKENS_PER_STRING)
        raw_output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        
        # Clean up common LLM markdown quirks
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()
        
        parsed = json.loads(cleaned_output)
        
        # Validate structure: must be a list of strings of the exact same length
        if isinstance(parsed, list) and len(parsed) == len(texts) and all(isinstance(x, str) for x in parsed):
            return parsed
        else:
            warnings.warn(f"Qwen batch correction returned invalid structure. Falling back to original texts.")
            return texts
            
    except json.JSONDecodeError:
        warnings.warn("Qwen batch correction failed to parse JSON. Falling back to original texts.")
        return texts
    except Exception as e:
        warnings.warn(f"Qwen batch correction failed: {e}")
        return texts

def _build_diff(raw: str, corrected: str) -> str:
    diff = difflib.ndiff(raw.splitlines(), corrected.splitlines())
    return "\n".join(diff)

# ---------------------------------------------------------------------------
# Page & Pipeline Orchestration
# ---------------------------------------------------------------------------
def _correct_page(
    page: SuryaPageResult,
    skip_qwen: bool = False,
    anomaly_threshold: float = ANOMALY_THRESHOLD,
) -> CorrectedPageResult:
    raw = page.ocr_text  # always copied unchanged into raw_ocr_text
    
    corrected_block_texts = []
    qwen_used = False
    indices_needing_qwen = []
    
    # 1. FAST PASS: Apply rules and identify anomalies
    for block in page.text_blocks:
        block_text = _apply_rules(block.get("text", ""))
        corrected_block_texts.append(block_text)
        
        if not skip_qwen and _anomaly_score(block_text) >= anomaly_threshold:
            indices_needing_qwen.append(len(corrected_block_texts) - 1)
            
    # 2. BATCHED VLM PASS: Call the heavy model exactly ONCE per page (if needed)
    if indices_needing_qwen:
        batch_texts = [corrected_block_texts[i] for i in indices_needing_qwen]
        batch_corrected = _qwen_correct_batch(batch_texts)
        
        # Map the corrected texts back to their original indices
        for idx, corrected_text in zip(indices_needing_qwen, batch_corrected):
            if corrected_text != corrected_block_texts[idx]:
                corrected_block_texts[idx] = corrected_text
                qwen_used = True

    # 3. Rebuild corrected_text from corrected blocks
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

def postprocess(
    result: SuryaResult,
    skip_qwen: bool = False,
    anomaly_threshold: float = ANOMALY_THRESHOLD,
) -> PostprocessResult:
    pages = []
    for page in result.pages:
        corrected_page = _correct_page(
            page, 
            skip_qwen=skip_qwen, 
            anomaly_threshold=anomaly_threshold
        )
        pages.append(corrected_page)
        
        # CRITICAL FOR 24GB RAM: Clear memory after every page that uses the heavy VLM
        if corrected_page.qwen_used:
            clear_device_cache()
            
    return PostprocessResult(
        source_name=result.source_name,
        pages=pages,
    )