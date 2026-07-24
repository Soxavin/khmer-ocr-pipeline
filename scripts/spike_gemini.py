"""Gemini as an OCR challenger — run one page and score it against local GT.

Zero local compute. Asks Gemini for the SAME output contract as dots.ocr (layout
JSON, tables as HTML with colspan/rowspan preserved), so the same parser scores
both. Needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment and the
`eval-extras` group installed (`uv add --optional eval-extras google-genai`).

    uv run --extra eval-extras python scripts/spike_gemini.py \
        --image eval/datasets/real/CambodiaBudgetExecutioninApr-2024_p3.png

Circularity: the moc_gas GT is gemini_draft_human_verified, so scoring Gemini
against it is not valid — gt_provenance.is_circular refuses that pairing and this
spike prints the warning and skips scoring. Use budget/ARDB pages, whose GT is
model-free (PDF text layer) or independently drafted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from khmer_pipeline.engines.surya import _parse_html_table_with_spans, _build_table_from_grid  # noqa: E402
from khmer_pipeline.evaluation.evaluate_structure import evaluate_table, gt_table_grid  # noqa: E402
from khmer_pipeline.evaluation.gt_provenance import circularity_note  # noqa: E402

_PROMPT = """Please output the layout information from this document image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML. Preserve colspan and rowspan attributes exactly as they appear.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - Transcribe Khmer script exactly as printed. Do not romanize or paraphrase.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON array, with no markdown fences.
"""

# The moc_gas GT is Gemini-drafted; the guard keys on this engine name.
_ENGINE_KEY = "gemini"


def _find_gt(image: Path) -> Path | None:
    p = image.with_name(image.stem + "_ground_truth.json")
    return p if p.exists() else None


def _parse_layout(raw: str) -> list[dict] | None:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value, _ = json.JSONDecoder().raw_decode(s)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, list):
                return v
        return None
    return value if isinstance(value, list) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path,
                    default=_REPO / "eval/datasets/real/CambodiaBudgetExecutioninApr-2024_p3.png")
    # A stable alias by default: specific versions (e.g. gemini-2.5-flash) get
    # retired for new keys and 404. Pin a concrete version for a benchmark once
    # --list-models shows what this key can reach.
    ap.add_argument("--model", default="gemini-flash-latest",
                    help="a Flash model or alias; run --list-models to see valid names")
    ap.add_argument("--list-models", action="store_true",
                    help="print the models this key can call, then exit")
    ap.add_argument("--out", type=Path, default=None, help="where to save raw output")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("SKIP: set GEMINI_API_KEY (see docs/ENGINE_EVALUATION.md §7)")
        return 2
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("SKIP: uv add --optional eval-extras google-genai")
        return 2

    client = genai.Client(api_key=key)

    def _list_generate_models() -> list[str]:
        names = []
        for mdl in client.models.list():
            actions = getattr(mdl, "supported_actions", None) or \
                getattr(mdl, "supported_generation_methods", None) or []
            if "generateContent" in actions and "flash" in (mdl.name or "").lower():
                names.append(mdl.name)
        return names

    if args.list_models:
        print("Flash models this key can call:")
        for n in _list_generate_models():
            print("  ", n)
        return 0

    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=args.model,
            contents=[types.Part.from_bytes(data=args.image.read_bytes(), mime_type="image/png"),
                      _PROMPT],
            config=types.GenerateContentConfig(temperature=0.0),
        )
    except Exception as exc:
        if "NOT_FOUND" in str(exc) or "404" in str(exc):
            print(f"Model {args.model!r} is not available to this key. Available Flash models:")
            for n in _list_generate_models():
                print("  ", n)
            print("Re-run with --model <one of the above>.")
            return 1
        raise
    raw = resp.text or ""
    print(f"{args.model}: {time.perf_counter()-t0:.0f}s, {len(raw)} chars")
    out = args.out or args.image.with_suffix(".gemini.json")
    out.write_text(raw, encoding="utf-8")

    elements = _parse_layout(raw)
    if elements is None:
        print("VERDICT: output was not parseable JSON — see", out)
        return 1
    cats: dict = {}
    tables = []
    for el in elements:
        cats[el.get("category")] = cats.get(el.get("category"), 0) + 1
        if el.get("category") == "Table":
            tables.append(el)
    spans = sum((el.get("text") or "").count("colspan") + (el.get("text") or "").count("rowspan")
                for el in tables)
    print(f"elements={len(elements)} categories={cats} span_attrs={spans}")

    gt_path = _find_gt(args.image)
    if gt_path is None:
        print("(no GT beside the image — parsed OK, not scored)")
        return 0
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    note = circularity_note(_ENGINE_KEY, gt)
    if note:
        print(f"NOT SCORED — {note}")
        return 0

    pred_tables = []
    for el in tables:
        grid_map, _ = _parse_html_table_with_spans(el.get("text") or "")
        pred_tables.append(_build_table_from_grid(grid_map, el.get("text") or "",
                                                  [float(v) for v in (el.get("bbox") or [0, 0, 0, 0])]))
    m = evaluate_table(pred_tables, gt_table_grid(gt))
    scope = gt.get("scoring_scope")
    cell = "—" if scope == "numeric_and_structure" else f"{m['cell_accuracy']:.3f}"
    khmer = "—" if scope == "numeric_and_structure" else f"{m['khmer_cell_accuracy']:.3f}"
    print(f"SCORED vs GT ({gt_path.name}): cell={cell} numeric={m['numeric_cell_accuracy']:.3f} "
          f"khmer={khmer} rowaln={m['row_alignment_rate']:.3f} colaln={m['col_alignment_rate']:.3f} "
          f"pred={m['pred_rows']}x{m['pred_cols']} gt={m['gt_rows']}x{m['gt_cols']}"
          + (f" scope={scope}" if scope else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
