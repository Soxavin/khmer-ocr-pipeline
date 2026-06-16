from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

# NOTE: Run `uv run playwright install chromium` once before first use.
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None  # type: ignore[assignment]

_FONTS = ["Noto Sans Khmer", "Battambang", "Hanuman", "Moul", "Fasthand"]
_DEFAULT_COUNT = 3
_VIEWPORT_WIDTH = 900   # px — wide enough for 4-col tables without wrapping

_TABLE_TEMPLATES: list[dict] = [
    {
        "title": "តារាងតម្លៃទីផ្សារប្រចាំថ្ងៃ",  # Daily Market Price Table
        "data": [
            ["លរ", "ផលិតផល", "តម្លៃ (រៀល)", "ការប្រែប្រួល"],
            ["១", "សាច់ជ្រូក", "12,000", "+5%"],
            ["២", "មាន់", "8,500", "-2%"],
            ["៣", "ត្រី", "6,000", "0%"],
            ["៤", "ស្ករ", "3,200", "+1%"],
            ["៥", "ប្រេង", "4,500", "+3%"],
            ["៦", "អង្ករ", "2,800", "0%"],
        ],
    },
    {
        "title": "អត្រាប្តូររូបិយប័ណ្ណ",  # Currency Exchange Rates
        "data": [
            ["ធនាគារ", "USD → KHR", "THB → KHR", "ចេញ​ / ទទួល"],
            ["ធនាគារ ARDB", "4,085", "115", "ចេញ"],
            ["ABA Bank", "4,090", "116", "ចេញ"],
            ["ACLEDA Bank", "4,080", "114", "ចេញ"],
            ["ធនាគារ ARDB", "4,095", "117", "ទទួល"],
            ["ABA Bank", "4,100", "118", "ទទួល"],
        ],
    },
    {
        "title": "សង្ខេបចំណាយប្រចាំខែ",  # Monthly Expense Summary
        "data": [
            ["លរ", "ប្រភេទចំណាយ", "ចំនួន (រៀល)", "កំណត់ចំណាំ"],
            ["១", "ជួលការិយាល័យ", "500,000", "ខែ មិថុនា"],
            ["២", "ប្រាក់ខែបុគ្គលិក", "1,200,000", "៥ នាក់"],
            ["៣", "ភ្លើង / ទឹក", "85,000", ""],
            ["៤", "ទំនិញ​ / ភ្នាក់ងារ", "320,000", ""],
            ["៥", "ថ្លៃដឹក", "45,000", ""],
            ["", "សរុប", "2,150,000", ""],
        ],
    },
]

_GOOGLE_FONTS_LINK = (
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Battambang&family=Hanuman&family=Moul&family=Fasthand"
    '&family=Noto+Sans+Khmer&display=swap" rel="stylesheet">'
)


def _build_html(font_family: str, title: str, data: list[list[str]]) -> str:
    ncols = len(data[0])

    header_cells = "".join(f"<th>{cell}</th>" for cell in data[0])
    body_rows = ""
    for row in data[1:]:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_rows += f"<tr>{cells}</tr>\n"

    return f"""<!DOCTYPE html>
<html lang="km">
<head>
<meta charset="UTF-8">
{_GOOGLE_FONTS_LINK}
<style>
  body {{
    font-family: '{font_family}', sans-serif;
    margin: 20px;
    background: #ffffff;
  }}
  table {{
    font-family: '{font_family}', sans-serif;
    border-collapse: collapse;
    font-size: 15px;
    min-width: 500px;
  }}
  th, td {{
    border: 1px solid #ccc;
    padding: 8px 12px;
    text-align: left;
  }}
  thead tr:first-child th {{
    background-color: #1a5276;
    color: #ffffff;
    text-align: center;
    font-size: 16px;
  }}
  thead tr:last-child th {{
    background-color: #2e86c1;
    color: #ffffff;
  }}
  tbody tr:nth-child(even) {{
    background-color: #f2f2f2;
  }}
  tbody tr:nth-child(odd) {{
    background-color: #ffffff;
  }}
</style>
</head>
<body>
<table>
  <thead>
    <tr><th colspan="{ncols}">{title}</th></tr>
    <tr>{header_cells}</tr>
  </thead>
  <tbody>
{body_rows}  </tbody>
</table>
</body>
</html>"""


def generate_synthetic_table(
    output_dir: Path,
    table_index: int,
    font_family: str = "Battambang",
) -> tuple[Path, Path]:
    if sync_playwright is None:
        raise ImportError(
            "playwright is not installed. "
            "Run: uv add 'playwright>=1.44,<2.0' && uv run playwright install chromium"
        )

    tmpl = _TABLE_TEMPLATES[table_index % len(_TABLE_TEMPLATES)]
    html = _build_html(font_family, tmpl["title"], tmpl["data"])

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_font = font_family.replace(" ", "_")
    img_path = output_dir / f"table_{table_index}_{safe_font}.png"
    json_path = output_dir / f"table_{table_index}_{safe_font}_ground_truth.json"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": _VIEWPORT_WIDTH, "height": 800})
        page.set_content(html, wait_until="networkidle")  # ensures Google Fonts load
        page.locator("table").screenshot(path=str(img_path))
        browser.close()

    # Merged title row: value in col-0, empty strings for remaining columns
    ncols = len(tmpl["data"][0])
    full_data = [[tmpl["title"]] + [""] * (ncols - 1)] + tmpl["data"]
    ground_truth = {
        "font_family": font_family,
        "table_index": table_index,
        "template": tmpl["title"],
        "data": full_data,
    }
    json_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")

    return img_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Khmer table images for font benchmarking.")
    parser.add_argument("--output-dir", default="./synthetic_data", help="Directory for output files")
    parser.add_argument("--font", default="all", help="Font family name, or 'all' to generate for all fonts")
    parser.add_argument("--count", type=int, default=_DEFAULT_COUNT, help="Number of tables per font")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fonts = _FONTS if args.font == "all" else [args.font]
    total = 0

    for font in fonts:
        for i in range(args.count):
            try:
                img_path, json_path = generate_synthetic_table(output_dir, i, font)
                print(f"Generated: {img_path}")
                print(f"          + {json_path}")
                total += 1
            except Exception as exc:
                warnings.warn(f"Failed to generate table {i} for font '{font}': {exc}")

    print(f"\nDone. Generated {total} image(s) in {output_dir}/")
