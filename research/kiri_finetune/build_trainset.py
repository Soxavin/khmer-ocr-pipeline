"""Assemble the Track B Kiri fine-tune training set from three sources.

1. REAL cell crops from the dataset factory (eval/datasets/table_gt_v1/recognition)
   — referenced in place, not copied.
2. seanghay/khmer-hanuman-100k subsample — general Khmer so the model doesn't
   forget the language while we drill the failure cases.
3. TARGETED SYNTHETIC lines rendered with Playwright + vendored Khmer fonts
   (correct shaping) for the measured error classes: ៛-unit strings, decimal
   numbers/percents, and empty/gridline negatives (PIL, no text to shape).

Output: <out>/labels_{train,val}.jsonl with {"image": <path relative to repo root>,
"text", "origin": real|hanuman|synthetic|empty} + materialized PNGs for 2 and 3.

Usage (repo root):
    uv run python experiments/kiri_finetune/build_trainset.py \
        --out experiments/kiri_finetune/trainset [--hanuman 15000] [--synthetic 3000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

_REAL_DIR = _REPO / "eval/datasets/table_gt_v1/recognition"
_HANUMAN_REPO = "seanghay/khmer-hanuman-100k"

# --- targeted synthetic curriculum (measured error classes, §2.33/2.35/2.36) ---
_RIEL_UNITS = ["៛/គ.ក", "៛/គ្រាប់", "៛/ផ្លែ", "៛/លីត្រ", "៛/ដើម", "៛/កញ្ចប់", "៛"]
_FONTS = ["Noto Sans Khmer", "Battambang", "Hanuman"]  # body-text fonts, not display


def _synthetic_texts(n: int, rng: random.Random) -> list[str]:
    """Text lines for the error-class curriculum: riel units, decimal percents,
    grouped and multi-decimal numbers, Khmer digits."""
    texts: list[str] = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.30:
            texts.append(rng.choice(_RIEL_UNITS))
        elif kind < 0.55:  # decimal percents incl. signs (the dot-drop class)
            texts.append(f"{rng.choice(['', '-', '+'])}{rng.randint(0, 99)}.{rng.randint(0, 99):02d}%")
        elif kind < 0.75:  # grouped thousands (comma placement)
            texts.append(f"{rng.randint(1, 999):,}".replace(",", "") if rng.random() < 0.2
                         else f"{rng.randint(1000, 99000000):,}")
        elif kind < 0.90:  # long decimals (the budget-table class: 27.39622)
            texts.append(f"{rng.randint(0, 999)}.{rng.randint(0, 99999):05d}")
        else:  # Khmer digits
            texts.append("".join("០១២៣៤៥៦៧៨៩"[int(d)] for d in str(rng.randint(1, 9999))))
    return texts


def _render_lines_playwright(texts: list[str], out_dir: Path, rng: random.Random) -> list[dict]:
    """Render each text as a small cell-like PNG via Chromium (correct Khmer shaping)."""
    from playwright.sync_api import sync_playwright
    from khmer_pipeline.utils.fonts import font_face_style_tag

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 640, "height": 80})
        for i, text in enumerate(texts):
            font = rng.choice(_FONTS)
            size = rng.randint(14, 22)
            bg, fg = rng.choice([("#fff", "#000"), ("#f4f4f4", "#111"),
                                 ("#2e8b57", "#fff"), ("#fdf6e3", "#333")])
            page.set_content(
                f"<html><head>{font_face_style_tag(font)}<style>"
                f"body{{margin:0;background:{bg};}}"
                f"div{{font-family:'{font}';font-size:{size}px;color:{fg};"
                f"padding:{rng.randint(2, 8)}px {rng.randint(4, 14)}px;display:inline-block;}}"
                f"</style></head><body><div>{text}</div></body></html>")
            el = page.locator("div")
            name = f"syn_{i:05d}.png"
            el.screenshot(path=str(out_dir / name))
            rows.append({"image": str((out_dir / name).relative_to(_REPO)),
                         "text": text, "origin": "synthetic"})
        browser.close()
    return rows


def _render_empty_negatives(n: int, out_dir: Path, rng: random.Random) -> list[dict]:
    """Blank / gridline-only crops that must decode to '' (the §2.35 pipe-noise class)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i in range(n):
        w, h = rng.randint(60, 400), rng.randint(24, 60)
        bg = rng.choice([255, 250, 244, 235])
        img = Image.new("L", (w, h), bg)
        draw = ImageDraw.Draw(img)
        if rng.random() < 0.7:  # cell borders / stray gridlines
            gray = rng.randint(120, 200)
            if rng.random() < 0.5:
                draw.line([(0, h - 1), (w, h - 1)], fill=gray)
            if rng.random() < 0.5:
                draw.line([(rng.choice([0, w - 1]), 0), (rng.choice([0, w - 1]), h)], fill=gray)
        name = f"empty_{i:05d}.png"
        img.save(out_dir / name)
        rows.append({"image": str((out_dir / name).relative_to(_REPO)),
                     "text": "", "origin": "empty"})
    return rows


def _real_rows(split: str) -> list[dict]:
    jsonl = _REAL_DIR / split / "pairs.jsonl"
    rows = []
    for line in jsonl.read_text().splitlines():
        r = json.loads(line)
        rows.append({"image": str((_REAL_DIR / split / r["image"]).relative_to(_REPO)),
                     "text": r["text"], "origin": "real"})
    return rows


def _hanuman_rows(n: int, out_dir: Path, rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(_HANUMAN_REPO, split="train", streaming=True)
    rows: list[dict] = []
    text_key = None
    for i, ex in enumerate(ds):
        if i >= n:
            break
        if text_key is None:
            text_key = next(k for k in ex if k != "image")
        name = f"han_{i:06d}.png"
        ex["image"].save(out_dir / name)
        rows.append({"image": str((out_dir / name).relative_to(_REPO)),
                     "text": str(ex[text_key]), "origin": "hanuman"})
    return rows


def _correction_rows(corrections_dir: Path, only_flags: set[str] | None = None) -> list[dict]:
    """Rows from HITL-captured analyst corrections (src/khmer_pipeline/corrections.py).

    These are the highest-value pairs in the set: verified human labels for real
    long-tail failures. `only_flags` filters on the nested validate.py taxonomy
    (e.g. {"sequence_illegal"}) — the reason provenance is stored structured."""
    # Resolve here rather than trusting the caller: the emitted "image" paths are
    # repo-relative, so a relative corrections_dir would break relative_to(_REPO).
    corrections_dir = (corrections_dir if corrections_dir.is_absolute()
                       else _REPO / corrections_dir)
    jsonl = corrections_dir / "corrections.jsonl"
    if not jsonl.exists():
        print(f"corrections: none at {jsonl}")
        return []
    rows = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if only_flags and not (set(r.get("provenance", {}).get("flags", [])) & only_flags):
            continue
        rows.append({"image": str((corrections_dir / r["image"]).relative_to(_REPO)),
                     "text": r["text"], "origin": "correction"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the Kiri fine-tune training set.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hanuman", type=int, default=15000, help="hanuman-100k subsample (0=skip)")
    parser.add_argument("--synthetic", type=int, default=3000, help="targeted synthetic lines")
    parser.add_argument("--empty", type=int, default=800, help="empty-cell negatives")
    parser.add_argument("--corrections", type=Path, default=None,
                        help="HITL corrections dir (contains corrections.jsonl + crops/)")
    parser.add_argument("--corrections-flags", default=None,
                        help="comma-separated validate.py flags to keep, e.g. sequence_illegal,digit_mixed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = args.out if args.out.is_absolute() else _REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    train_rows = _real_rows("train")
    val_rows = _real_rows("validation")
    print(f"real: {len(train_rows)} train / {len(val_rows)} val")

    if args.synthetic:
        texts = _synthetic_texts(args.synthetic, rng)
        syn = _render_lines_playwright(texts, out / "synthetic", rng)
        cut = max(1, len(syn) // 20)  # hold out 5% for val
        train_rows += syn[cut:]
        val_rows += syn[:cut]
        print(f"synthetic: {len(syn)} rendered")
    if args.empty:
        emp = _render_empty_negatives(args.empty, out / "empty", rng)
        cut = max(1, len(emp) // 20)
        train_rows += emp[cut:]
        val_rows += emp[:cut]
        print(f"empty negatives: {len(emp)}")
    if args.hanuman:
        han = _hanuman_rows(args.hanuman, out / "hanuman", rng)
        train_rows += han  # val stays real+targeted: that's what we're graded on
        print(f"hanuman: {len(han)}")
    if args.corrections:
        cdir = args.corrections if args.corrections.is_absolute() else _REPO / args.corrections
        flags = set(args.corrections_flags.split(",")) if args.corrections_flags else None
        cor = _correction_rows(cdir, flags)
        # All corrections go to TRAIN: they are the errors we want fixed, and the
        # val split must stay the fixed real+targeted set we are graded on.
        train_rows += cor
        print(f"corrections: {len(cor)}" + (f" (flags={sorted(flags)})" if flags else ""))

    rng.shuffle(train_rows)
    for name, rows in (("labels_train.jsonl", train_rows), ("labels_val.jsonl", val_rows)):
        (out / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"Done: {len(train_rows)} train / {len(val_rows)} val → {out}")


if __name__ == "__main__":
    main()
