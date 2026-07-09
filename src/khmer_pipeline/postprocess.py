from __future__ import annotations
import difflib
import json
import re
import warnings

try:
    from mlx_lm import generate
except ImportError:
    generate = None  # type: ignore[assignment]

from .models import SuryaResult, SuryaPageResult, PostprocessResult, CorrectedPageResult
from .model_config import ANOMALY_THRESHOLD, CONFIDENCE_LOW, STAGE4_MODEL_PATH
from .utils.memory import clear_device_cache
from .utils.khmer_normalize import normalize_khmer

# ---------------------------------------------------------------------------
# Rule table — deliberately empty. Add targeted pairs only after review.
# ---------------------------------------------------------------------------
RULE_BASED_CORRECTIONS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# GDDE-DOMAIN cell rules (PROJECT_LOG §2.33/§2.34) — full-cell pattern fixes for
# measured, systematic OCR confusions on financial-table cells. These are DOMAIN
# rules, deliberately kept out of the script-level Unicode normalizer
# (utils/khmer_normalize.py). Anti-overfit contract: each rule fires only on a
# full-cell match of a corrupt form that is not plausible Khmer text, so it
# cannot alter other document types (gate: §2.34 — identity on the synthetic
# benchmark, measured lift on the real docs).
# ---------------------------------------------------------------------------
_RIEL_UNIT_RULES: list[tuple[re.Pattern[str], str]] = [
    # ៛ misread as អ/#/វ before a known riel-per-unit suffix
    # (§2.33: អគ.ក 115×, #គ.ក 10×, អគ្រាប់ 4×; §2.27's Surya-era វ/គ.ក).
    (re.compile(r"^[អ#វ]/?(គ\.ក|គ្រាប់|ផ្លែ)$"), r"៛/\1"),
    # Same, with the '.' in គ.ក additionally misread as ៈ or : (§2.33: អគៈក 5×).
    (re.compile(r"^[អ#វ]/?គ[ៈ:]\.?ក$"), "៛/គ.ក"),
]

# Percent-shaped cell (optionally signed, one decimal separator, trailing %),
# Khmer or Arabic digits — used to fold stray Khmer digits in percent cells
# (§2.33: 0.00% → ០.00%/០.០០%, ~25×/doc). Script-generic, not layout-specific.
# Khmer digits ០-៩ are the contiguous block U+17E0–U+17E9.
_PERCENT_SHAPE_RE = re.compile(r"^[+-]?[0-9០-៩]+(?:[.,][0-9០-៩]+)?%$")

# Khmer→Arabic digit fold. Canonical copy lives in export._KHMER_TO_ARABIC;
# duplicated (with this note) to keep postprocess out of export's import chain.
_KHMER_TO_ARABIC = {
    "០": "0", "១": "1", "២": "2", "៣": "3", "៤": "4",
    "៥": "5", "៦": "6", "៧": "7", "៨": "8", "៩": "9",
}

# Malformed-number detection (§2.33/§2.35 dot-drop & digit-duplication:
# 2.94%→294%, 8.33%→8333%, 7,800→7,8000). Matching cells are FLAGGED for
# analyst review (confidence capped below CONFIDENCE_LOW) — digits are NEVER
# rewritten. The percent rule flags any ≥2-digit integer percent (no decimal
# separator): in these docs %-values carry decimals, so an integer form is a
# likely dot-drop; single-digit percents (5%) stay unflagged.
_MALFORMED_COMMA_RE = re.compile(r"\d,\d{4}")          # comma followed by 4+ digits
_MALFORMED_PERCENT_RE = re.compile(r"^[+-]?\d{2,}%$")  # ≥2-digit integer percent
_MALFORMED_CONF_CAP = 0.4  # < CONFIDENCE_LOW → shows red in the UI confidence view

# Gridline-noise strip (§2.35): Kiri reads a cell's border line as text — the
# measured noise on empty cells is pipe-dominated ('|' 15×/10× per real doc).
# Conservative: only pipe-BEARING junk is emptied; a bare '-' may be a legit
# "no data" placeholder in other document types and survives.
_CELL_NOISE_CHARS = set("|-_—– ")


def _strip_cell_noise(text: str) -> str:
    """Empty a cell whose text is only gridline junk (pipes/dashes incl. a '|')."""
    if "|" in text and all(ch in _CELL_NOISE_CHARS for ch in text):
        return ""
    return text


def _strip_foreign_scripts(text: str) -> tuple[str, int]:
    """Remove characters from scripts that cannot appear in these documents
    (product constraint: Khmer or English only — §2.35). Returns
    (cleaned_text, n_removed); callers warn when n_removed > 0."""
    kept = [ch for ch in text if not _is_foreign_script(ch)]
    n_removed = len(text) - len(kept)
    if not n_removed:
        return text, 0
    # collapse the doubled spaces the removal leaves behind (keep newlines)
    return re.sub(r" {2,}", " ", "".join(kept)).strip(" "), n_removed


def _apply_cell_rules(text: str) -> str:
    """Apply the GDDE-domain full-cell corrections (riel-prefix repair, percent
    Khmer-digit fold) to one table-cell string. Identity for any other cell."""
    for pat, repl in _RIEL_UNIT_RULES:
        if pat.match(text):
            return pat.sub(repl, text)
    if "%" in text and _PERCENT_SHAPE_RE.match(text) and any(ch in _KHMER_TO_ARABIC for ch in text):
        return "".join(_KHMER_TO_ARABIC.get(ch, ch) for ch in text)
    return text


def _is_malformed_number(text: str) -> bool:
    """True if the cell text matches a known digit-duplication artifact pattern."""
    return bool(_MALFORMED_COMMA_RE.search(text) or _MALFORMED_PERCENT_RE.match(text))

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
    """Return True if the Qwen correction model is already loaded into memory."""
    return _qwen_model is not None

# ---------------------------------------------------------------------------
# Correction layers
# ---------------------------------------------------------------------------
def _apply_rules(text: str) -> str:
    # deterministic Khmer Unicode normalization (NFC, format-char strip,
    # canonical reorder, dup collapse, whitespace) then targeted exact-pair fixes
    text = normalize_khmer(text)
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
    Score is the proportion of foreign-script characters among the
    NON-WHITESPACE characters (whitespace would otherwise dilute the ratio).
    ANOMALY_THRESHOLD = 0.15 is tunable."""
    non_ws = sum(1 for ch in text if not ch.isspace())
    if non_ws == 0:
        return 0.0
    foreign_count = sum(1 for ch in text if _is_foreign_script(ch))
    return foreign_count / non_ws

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

def _normalize_table(table: dict, page_index: int, table_index: int,
                     warning_sink: list[str]) -> dict:
    """Return a NEW table dict (with NEW cell/text_line dicts) whose cell texts
    are Khmer-normalized (NFC, ZWSP/BOM strip, dup-diacritic collapse) and then
    passed through the GDDE-domain cell rules (`_apply_cell_rules`). The input
    table — and the SuryaPageResult it belongs to — is never mutated, so export's
    in-place table repair operates only on these Stage-4-owned copies.

    Cells whose text matches a malformed-number pattern (dot-drop /
    digit-duplication artifacts) are flagged: confidence capped to
    `_MALFORMED_CONF_CAP` (set if absent) and a warning appended to
    *warning_sink* — digits are never rewritten. Foreign-script characters are
    scrubbed (one aggregated warning per table); pipe-only gridline noise is
    emptied. Order: normalize → scrub → domain rules → noise strip → flag."""
    new_cells = []
    foreign_removed = 0
    for cell in table.get("cells", []):
        new_cell = dict(cell)
        text_lines = cell.get("text_lines")
        if text_lines:
            new_lines = []
            for tl in text_lines:
                if tl.get("text") is None:
                    new_lines.append(dict(tl))
                    continue
                text, n = _strip_foreign_scripts(normalize_khmer(tl["text"]))
                foreign_removed += n
                text = _strip_cell_noise(_apply_cell_rules(text))
                new_lines.append({**tl, "text": text})
            new_cell["text_lines"] = new_lines
        cell_text = " ".join(
            t["text"] for t in (new_cell.get("text_lines") or []) if t.get("text")
        ).strip()
        if cell_text and _is_malformed_number(cell_text):
            new_cell["confidence"] = min(
                new_cell.get("confidence", _MALFORMED_CONF_CAP), _MALFORMED_CONF_CAP
            )
            warning_sink.append(
                f"Page {page_index + 1}, table {table_index + 1}: cell "
                f"(row {cell.get('row_id', 0) + 1}, col {cell.get('col_id', 0) + 1}) "
                f"looks like a malformed number ({cell_text!r}) — flagged low-confidence; "
                f"verify against the page image."
            )
        new_cells.append(new_cell)
    if foreign_removed:
        warning_sink.append(
            f"Page {page_index + 1}, table {table_index + 1}: removed {foreign_removed} "
            f"foreign-script character(s) from table cells (output is Khmer/English only)."
        )
    new_table = dict(table)
    new_table["cells"] = new_cells
    return new_table


def _build_diff(raw: str, corrected: str) -> str:
    diff = difflib.ndiff(raw.splitlines(), corrected.splitlines())
    return "\n".join(diff)

# ---------------------------------------------------------------------------
# Page & Pipeline Orchestration
# ---------------------------------------------------------------------------
def _correct_page(
    page: SuryaPageResult,
    skip_qwen: bool = True,  # Qwen is opt-in; deterministic normalizer always runs
    anomaly_threshold: float = ANOMALY_THRESHOLD,
    warning_sink: list[str] | None = None,
) -> CorrectedPageResult:
    if warning_sink is None:
        warning_sink = []
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

    # 2b. Foreign-script scrub — LAST text step, after anomaly detection/Qwen so
    # their routing semantics are unchanged (product constraint: Khmer/English only).
    page_foreign_removed = 0
    for i, t in enumerate(corrected_block_texts):
        cleaned, n = _strip_foreign_scripts(t)
        if n:
            corrected_block_texts[i] = cleaned
            page_foreign_removed += n

    # 3. Rebuild corrected_text from corrected blocks
    if corrected_block_texts:
        corrected_text = "\n\n".join(t for t in corrected_block_texts if t)
    else:
        corrected_text, page_foreign_removed = _strip_foreign_scripts(_apply_rules(raw))

    if page_foreign_removed:
        warning_sink.append(
            f"Page {page.page_index + 1}: removed {page_foreign_removed} foreign-script "
            f"character(s) from page text (output is Khmer/English only)."
        )

    diff = _build_diff(raw, corrected_text)
    return CorrectedPageResult(
        page_index=page.page_index,
        text_blocks=page.text_blocks,  # unchanged
        # Copy-on-write: normalized + rule-corrected cell text in NEW dicts so
        # the input SuryaPageResult.tables stay byte-identical (no aliasing).
        tables=[_normalize_table(t, page.page_index, t_idx, warning_sink)
                for t_idx, t in enumerate(page.tables)],
        raw_ocr_text=raw,
        corrected_text=corrected_text,
        correction_diff=diff,
        qwen_used=qwen_used,
    )

def postprocess(
    result: SuryaResult,
    skip_qwen: bool = True,  # Qwen is opt-in; deterministic normalizer always runs
    anomaly_threshold: float = ANOMALY_THRESHOLD,
) -> PostprocessResult:
    """Apply deterministic Khmer normalization to every page's text blocks, optionally
    escalating anomalous blocks to the Qwen VLM for correction (`skip_qwen=False`).
    Returns a `PostprocessResult` with per-page raw/corrected text, diffs, and any
    Stage-4 warnings (e.g. malformed-number flags) in `.warnings`."""
    pages = []
    stage_warnings: list[str] = []
    for page in result.pages:
        corrected_page = _correct_page(
            page,
            skip_qwen=skip_qwen,
            anomaly_threshold=anomaly_threshold,
            warning_sink=stage_warnings,
        )
        pages.append(corrected_page)

        # CRITICAL FOR 24GB RAM: Clear memory after every page that uses the heavy VLM
        if corrected_page.qwen_used:
            clear_device_cache()

    return PostprocessResult(
        source_name=result.source_name,
        pages=pages,
        warnings=stage_warnings,
    )