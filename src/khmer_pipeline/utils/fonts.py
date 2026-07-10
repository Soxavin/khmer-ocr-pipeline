from __future__ import annotations
import base64
from pathlib import Path

# Vendored OFL Khmer fonts (see fonts/MANIFEST.txt). Embedded as base64 data URIs
# so synthetic-data generation is fully offline and deterministic — no live
# fonts.googleapis.com dependency. Works with Playwright's about:blank origin
# (page.set_content) without any file-access flags.
# parents[3] = repo root (this file lives at src/khmer_pipeline/utils/fonts.py;
# the restructure that added utils/ silently broke the old parents[2])
_FONTS_DIR = Path(__file__).resolve().parents[3] / "fonts"

# family -> (filename, css font-weight descriptor). Variable fonts cover a range.
_FONT_FILES: dict[str, tuple[str, str]] = {
    "Noto Sans Khmer": ("NotoSansKhmer-Variable.ttf", "100 900"),
    "Battambang": ("Battambang-Regular.ttf", "400"),
    "Hanuman": ("Hanuman-Variable.ttf", "100 900"),
    "Moul": ("Moul-Regular.ttf", "400"),
    "Fasthand": ("Fasthand-Regular.ttf", "400"),
}


def _face(family: str) -> str:
    filename, weight = _FONT_FILES[family]
    data = (_FONTS_DIR / filename).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return (
        "@font-face {\n"
        f"  font-family: '{family}';\n"
        "  font-style: normal;\n"
        f"  font-weight: {weight};\n"
        f"  src: url(data:font/ttf;base64,{b64}) format('truetype');\n"
        "}"
    )


def font_face_style_tag(family: str | None = None) -> str:
    # family=None embeds all vendored fonts; otherwise just the one requested.
    families = [family] if family is not None else list(_FONT_FILES)
    faces = "\n".join(_face(f) for f in families)
    return f"<style>\n{faces}\n</style>"
