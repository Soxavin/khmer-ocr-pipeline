from __future__ import annotations
import unicodedata
from khmer_pipeline.khmer_normalize import normalize_khmer

# --- character constants (chr resolves at module load) ---
_KO = chr(0x1780)        # ក base consonant
_KHO = chr(0x1781)       # ខ base consonant
_SO = chr(0x179F)        # ស base consonant
_TO = chr(0x178F)        # ត base consonant
_MO = chr(0x1798)        # ម base consonant
_RO = chr(0x179A)        # រ base consonant
_COENG = chr(0x17D2)     # ្ subscript marker
_AA = chr(0x17B6)        # ា dependent vowel (rank 4)
_AE = chr(0x17C2)        # ែ dependent vowel (rank 4)
_U = chr(0x17BB)         # ុ dependent vowel (rank 4)
_NIKAHIT = chr(0x17C6)   # ំ sign (rank 5)

_ZWSP = chr(0x200B)
_ZWNJ = chr(0x200C)
_ZWJ = chr(0x200D)
_BOM = chr(0xFEFF)
_SHY = chr(0x00AD)       # soft hyphen

# A correctly-ordered Khmer word: ខ ្ម ែ រ
_KHMER_WORD = _KHO + _COENG + _MO + _AE + _RO  # "ខ្មែរ"


# --- empty / passthrough ---

def test_empty_string():
    assert normalize_khmer("") == ""


def test_latin_and_digits_untouched():
    # financial content must survive verbatim
    s = "CP ARDB 03-06-26 12,000 0.00%"
    assert normalize_khmer(s) == s


def test_khmer_numerals_untouched():
    s = chr(0x17E3) + chr(0x17E4) + chr(0x17E5)  # ៣៤៥
    assert normalize_khmer(s) == s


# --- rule 1: NFC ---

def test_nfc_applied():
    nfd = "កា"
    assert normalize_khmer(nfd) == unicodedata.normalize("NFC", nfd)


def test_idempotent():
    s = _ZWSP + _SO + _AA + _COENG + _TO + "  text  " + _NIKAHIT + _U
    once = normalize_khmer(s)
    assert normalize_khmer(once) == once


# --- rule 2: strip noise format chars ---

def test_strip_zwsp():
    assert normalize_khmer(_KHMER_WORD + _ZWSP + "test") == _KHMER_WORD + "test"


def test_strip_bom():
    assert normalize_khmer(_BOM + "abc") == "abc"


def test_strip_soft_hyphen():
    assert normalize_khmer("a" + _SHY + "b") == "ab"


def test_zwnj_zwj_preserved():
    # these can be meaningful in Khmer rendering — not stripped by default
    s = _KO + _ZWNJ + _KHO
    assert _ZWNJ in normalize_khmer(s)
    s2 = _KO + _ZWJ + _KHO
    assert _ZWJ in normalize_khmer(s2)


# --- rule 3: collapse duplicate combining marks ---

def test_collapse_duplicate_sign():
    assert normalize_khmer(_KO + _NIKAHIT + _NIKAHIT + _KHO) == _KO + _NIKAHIT + _KHO


def test_collapse_duplicate_coeng():
    # a doubled coeng marker is an OCR artifact
    assert normalize_khmer(_KO + _COENG + _COENG + _TO) == _KO + _COENG + _TO


def test_does_not_collapse_repeated_base_consonants():
    # identical base letters are valid (e.g. នន) — must not be collapsed
    s = _KO + _KO
    assert normalize_khmer(s) == s


# --- rule 4: whitespace tidy ---

def test_collapse_spaces():
    assert normalize_khmer("a   b") == "a b"


def test_strip_and_tabs():
    assert normalize_khmer("  a\tb  ") == "a b"


def test_newlines_preserved():
    assert normalize_khmer("a\nb") == "a\nb"


# --- rule 5: canonical cluster reordering ---

def test_reorder_off_by_default():
    # reorder is opt-in; a scrambled cluster is left as-is unless requested
    scrambled = _KO + _NIKAHIT + _U
    assert normalize_khmer(scrambled) == scrambled


def test_reorder_vowel_before_sign():
    # sign (rank 5) typed before vowel (rank 4) -> vowel first
    scrambled = _KO + _NIKAHIT + _U
    assert normalize_khmer(scrambled, reorder=True) == _KO + _U + _NIKAHIT


def test_reorder_coeng_before_vowel():
    # vowel typed before coeng-unit -> coeng-unit first
    scrambled = _SO + _AA + _COENG + _TO
    assert normalize_khmer(scrambled, reorder=True) == _SO + _COENG + _TO + _AA


def test_already_canonical_unchanged():
    assert normalize_khmer(_KHMER_WORD, reorder=True) == _KHMER_WORD


def test_coeng_unit_kept_together():
    # the consonant after a coeng must stay attached to it through reordering
    s = _SO + _COENG + _TO + _AA  # already canonical: ស ្ត ា
    assert normalize_khmer(s, reorder=True) == s
