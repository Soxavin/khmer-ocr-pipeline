from __future__ import annotations
import re
import unicodedata

# Deterministic, 100%-local Khmer Unicode normalizer. Replaces the old no-op
# Stage-4 deterministic layer. Pure functions: ASCII/Latin/digits/Khmer numerals
# pass through untouched; only Khmer orthographic clusters are touched.

# Base = consonants (U+1780–U+17A2) + independent vowels (U+17A3–U+17B3)
_BASE_LO, _BASE_HI = 0x1780, 0x17B3
# Combining marks = U+17B4–U+17D3 (minus coeng) plus U+17DD (atthacan)
_MARK_LO, _MARK_HI = 0x17B4, 0x17D3
_COENG = "្"           # subscript marker; pairs with the following base
_ATTHACAN = 0x17DD

# Invisible/format chars that are OCR noise — safe to drop. ZWNJ (U+200C) and
# ZWJ (U+200D) are deliberately NOT here: they can be meaningful in Khmer shaping.
_STRIP_CHARS = {chr(0x200B), chr(0xFEFF), chr(0x00AD)}  # ZWSP, BOM/ZWNBSP, soft hyphen

_HSPACE_RE = re.compile(r"[^\S\n]+")  # runs of horizontal whitespace, keep \n


def _is_base(ch: str) -> bool:
    return _BASE_LO <= ord(ch) <= _BASE_HI


def _is_mark(ch: str) -> bool:
    cp = ord(ch)
    if ch == _COENG:
        return False  # handled as its own unit
    return (_MARK_LO <= cp <= _MARK_HI) or cp == _ATTHACAN


def _is_combining(ch: str) -> bool:
    return ch == _COENG or _is_mark(ch)


def _mark_rank(ch: str) -> int:
    cp = ord(ch)
    if cp == 0x17CC:            # robat
        return 2
    if cp in (0x17C9, 0x17CA):  # consonant shifters (muusikatoan / triisap)
        return 3
    if 0x17B4 <= cp <= 0x17C5:  # dependent vowels
        return 4
    return 5                    # remaining signs


def _strip_format(text: str) -> str:
    return "".join(ch for ch in text if ch not in _STRIP_CHARS)


def _reorder(text: str) -> str:
    # Within each cluster (base + following marks), stable-sort marks into the
    # canonical order. A coeng + its base form one atomic unit (rank 1).
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _is_base(ch):
            base = ch
            i += 1
            units: list[tuple[int, str]] = []
            while i < n:
                c = text[i]
                if c == _COENG and i + 1 < n and _is_base(text[i + 1]):
                    units.append((1, c + text[i + 1]))
                    i += 2
                elif c == _COENG:
                    units.append((1, c))  # lone coeng (no base) — keep
                    i += 1
                elif _is_mark(c):
                    units.append((_mark_rank(c), c))
                    i += 1
                else:
                    break
            units.sort(key=lambda u: u[0])  # stable
            out.append(base + "".join(s for _, s in units))
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _collapse_dup(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if out and ch == out[-1] and _is_combining(ch):
            continue
        out.append(ch)
    return "".join(out)


def _tidy_ws(text: str) -> str:
    return _HSPACE_RE.sub(" ", text).strip()


def normalize_khmer(text: str, reorder: bool = False) -> str:
    # reorder (canonical cluster reordering) is opt-in: benchmark-validated as
    # neutral on current OCR output (Surya already emits canonical order), so it
    # is off by default and reserved for legacy/scanned docs with mis-ordered Khmer.
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _strip_format(text)
    if reorder:
        text = _reorder(text)
    text = _collapse_dup(text)
    text = _tidy_ws(text)
    return text
