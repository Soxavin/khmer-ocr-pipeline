"""Generate full-page A4 Khmer government/financial document images for benchmark testing.

Produces *_ground_truth.json + *.png pairs that run_benchmark.py can pick up.

CLI:
    uv run python -m khmer_pipeline.generate_synthetic_documents \\
        [--output-dir ./synthetic_documents] [--font all] [--count 3]
"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

_FONTS = ["Noto Sans Khmer", "Battambang", "Hanuman", "Moul", "Fasthand"]

_FONT_URL_PARAMS: dict[str, str] = {
    "Noto Sans Khmer": "Noto+Sans+Khmer",
    "Battambang": "Battambang",
    "Hanuman": "Hanuman",
    "Moul": "Moul",
    "Fasthand": "Fasthand",
}

_DOCUMENT_TEMPLATES: list[dict] = [
    {
        "document_type": "market_report",
        "org_header": "ព្រះរាជាណាចក្រកម្ពុជា\nជាតិ សាសនា ព្រះមហាក្សត្រ",
        "org_name": "ក្រសួងពាណិជ្ជកម្ម",
        "title": "របាយការណ៍តម្លៃទីផ្សារប្រចាំថ្ងៃ",
        "body_paragraphs": [
            "ក្រសួងពាណិជ្ជកម្ម សូមជម្រាបជូនដំណឹងអំពីតម្លៃទំនិញប្រចាំថ្ងៃ ក្នុងរាជធានីភ្នំពេញ។",
            "តារាងខាងក្រោមបង្ហាញពីតម្លៃមធ្យមនៃផលិតផលកសិកម្ម ដែលត្រូវបានប្រមូលពីទីផ្សារចំនួន ៥ ក្នុងរាជធានី។",
        ],
        "table_title": "តារាងតម្លៃទីផ្សារ",
        "table_headers": ["លរ", "ផលិតផល", "តម្លៃ (រៀល/គ.ក)", "ការប្រែប្រួល"],
        "table_rows": [
            ["១", "អង្ករសរបស់ខ្មែរ", "4,000", "+50"],
            ["២", "ត្រីស្រស់", "12,000", "-200"],
            ["៣", "បន្លែស្រស់", "3,500", "+100"],
            ["៤", "ដំឡូងមី", "2,800", "+80"],
            ["៥", "ល្ង", "18,000", "-300"],
        ],
        "footer": "ភ្នំពេញ, ថ្ងៃទី ១៥ ខែ មិថុនា ឆ្នាំ ២០២៦\nអ្នកស្ដីទីប្រធានក្រសួង",
    },
    {
        "document_type": "currency_report",
        "org_header": "ព្រះរាជាណាចក្រកម្ពុជា\nជាតិ សាសនា ព្រះមហាក្សត្រ",
        "org_name": "ធនាគារជាតិនៃកម្ពុជា",
        "title": "តារាងអត្រាប្តូរប្រាក់ប្រចាំថ្ងៃ",
        "body_paragraphs": [
            "ធនាគារជាតិនៃកម្ពុជា សូមជម្រាបជូនអំពីអត្រាប្តូរប្រាក់ជាផ្លូវការ ដោយផ្អែកលើទីផ្សារអន្តរជាតិ។",
            "អត្រាប្តូរប្រាក់ខាងក្រោម ត្រូវចូលជាធរមាន ចាប់ពីថ្ងៃទី ០១ ដល់ ថ្ងៃទី ៣០ ខែ មិថុនា ឆ្នាំ ២០២៦ ។",
        ],
        "table_title": "អត្រាប្តូរប្រាក់",
        "table_headers": ["ធនាគារ", "USD → KHR (ចូល)", "USD → KHR (ចេញ)", "THB → KHR"],
        "table_rows": [
            ["ABA", "4,015", "4,035", "112"],
            ["ACLEDA", "4,010", "4,030", "111"],
            ["NBC", "4,020", "4,040", "113"],
            ["Canadia", "4,012", "4,032", "111.5"],
            ["Vattanac", "4,018", "4,038", "112.5"],
        ],
        "footer": "ភ្នំពេញ, ថ្ងៃទី ០១ ខែ មិថុនា ឆ្នាំ ២០២៦\nអគ្គទេសាភិបាលធនាគារជាតិ",
    },
    {
        "document_type": "expense_report",
        "org_header": "ព្រះរាជាណាចក្រកម្ពុជា\nជាតិ សាសនា ព្រះមហាក្សត្រ",
        "org_name": "ក្រសួងសេដ្ឋកិច្ច និងហិរញ្ញវត្ថុ",
        "title": "សេចក្តីសង្ខេបចំណាយប្រចាំខែ",
        "body_paragraphs": [
            "ក្រសួងសេដ្ឋកិច្ច និងហិរញ្ញវត្ថុ សូមចងក្រងរបាយការណ៍ចំណាយ ស្របតាមគោលនយោបាយថវិការហិរញ្ញវត្ថុជាតិ។",
            "ការចំណាយខាងក្រោម ត្រូវបានអនុម័តដោយ គណៈរដ្ឋមន្ត្រី ហើយអ្នកពាក់ព័ន្ធទាំងអស់ ត្រូវគោរពតាមក្របខណ្ឌថវិការ ។",
        ],
        "table_title": "សង្ខេបចំណាយ",
        "table_headers": ["លរ", "ប្រភេទចំណាយ", "ចំនួនទឹកប្រាក់ (USD)", "កំណត់ចំណាំ"],
        "table_rows": [
            ["១", "ប្រាក់ខែបុគ្គលិក", "150,000", "ស្ថានភាព ៖ ធម្មតា"],
            ["២", "ថ្លៃដំណើរការ", "25,000", "ស្ថានភាព ៖ ធម្មតា"],
            ["៣", "ថ្លៃការិយាល័យ", "8,500", "ស្ថានភាព ៖ ស្រប"],
            ["៤", "ថ្លៃជួលអគារ", "12,000", "ស្ថានភាព ៖ ស្រប"],
            ["៥", "ចំណាយផ្សេងៗ", "5,200", "ស្ថានភាព ៖ ត្រូវពិនិត្យ"],
        ],
        "footer": "ភ្នំពេញ, ថ្ងៃទី ៣០ ខែ មិថុនា ឆ្នាំ ២០២៦\nរដ្ឋមន្ត្រីក្រសួងសេដ្ឋកិច្ច",
    },
]

_VIEWPORT_WIDTH = 1000
_VIEWPORT_HEIGHT = 1400

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="km">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family={font_url_param}&display=swap" rel="stylesheet">
<style>
  body {{
    margin: 0;
    padding: 30px;
    background: #c8c8c8;
    display: flex;
    justify-content: center;
    font-family: '{font_name}', sans-serif;
  }}
  .a4-page {{
    width: 800px;
    min-height: 1130px;
    background: white;
    padding: 60px;
    box-sizing: border-box;
  }}
  .org-header {{
    text-align: center;
    font-size: 13px;
    line-height: 1.8;
    color: #333;
    margin-bottom: 4px;
    white-space: pre-line;
  }}
  .org-name {{
    text-align: center;
    font-size: 14px;
    font-weight: bold;
    color: #1a3a6b;
    margin-bottom: 16px;
  }}
  .divider {{
    border: none;
    border-top: 2px solid #1a3a6b;
    margin: 12px 0 20px 0;
  }}
  .doc-title {{
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: #1a3a6b;
    margin-bottom: 20px;
    text-decoration: underline;
  }}
  .body-para {{
    font-size: 14px;
    line-height: 1.9;
    color: #222;
    margin-bottom: 12px;
    text-align: justify;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
    margin-bottom: 30px;
    font-size: 13px;
  }}
  .data-table th {{
    background-color: #1a5276;
    color: white;
    padding: 9px 12px;
    text-align: center;
    border: 1px solid #ccc;
  }}
  .data-table td {{
    padding: 8px 12px;
    border: 1px solid #ccc;
    text-align: center;
  }}
  .data-table tr:nth-child(even) td {{
    background-color: #f2f2f2;
  }}
  .table-title-row th {{
    background-color: #2e86c1;
    font-size: 14px;
  }}
  .footer {{
    margin-top: 40px;
    text-align: right;
    font-size: 13px;
    line-height: 1.9;
    color: #444;
    white-space: pre-line;
  }}
</style>
</head>
<body>
<div class="a4-page">
  <div class="org-header">{org_header}</div>
  <div class="org-name">{org_name}</div>
  <hr class="divider">
  <div class="doc-title">{title}</div>
  {body_paragraphs_html}
  <table class="data-table">
    <thead>
      <tr class="table-title-row"><th colspan="{col_count}">{table_title}</th></tr>
      <tr>{header_cells}</tr>
    </thead>
    <tbody>
      {body_rows}
    </tbody>
  </table>
  <div class="footer">{footer}</div>
</div>
</body>
</html>"""


def _build_html(tmpl: dict, font: str, rows: list[list[str]]) -> str:
    font_url_param = _FONT_URL_PARAMS.get(font, font.replace(" ", "+"))
    headers = tmpl["table_headers"]
    col_count = len(headers)

    body_paragraphs_html = "\n  ".join(
        f'<div class="body-para">{p}</div>' for p in tmpl["body_paragraphs"]
    )
    header_cells = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = "\n      ".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )

    return _HTML_TEMPLATE.format(
        font_url_param=font_url_param,
        font_name=font,
        org_header=tmpl["org_header"],
        org_name=tmpl["org_name"],
        title=tmpl["title"],
        body_paragraphs_html=body_paragraphs_html,
        col_count=col_count,
        table_title=tmpl["table_title"],
        header_cells=header_cells,
        body_rows=body_rows,
        footer=tmpl["footer"],
    )


def _render_document(
    tmpl: dict, font: str, row_count: int, output_dir: Path, doc_index: int
) -> None:
    font_slug = font.replace(" ", "_")
    img_path = output_dir / f"doc_{doc_index}_{font_slug}.png"
    gt_path = output_dir / f"doc_{doc_index}_{font_slug}_ground_truth.json"

    rows = tmpl["table_rows"][:row_count]
    html = _build_html(tmpl, font, rows)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT})
        page.set_content(html, wait_until="networkidle")
        page.evaluate("document.fonts.ready")  # Playwright auto-awaits the returned promise
        if not page.evaluate(f'document.fonts.check(\'16px "{font}"\')'):
            raise RuntimeError(
                f"Font '{font}' did not load — aborting to avoid a fallback-font image."
            )
        page.locator(".a4-page").screenshot(path=str(img_path))
        browser.close()

    gt = {
        "font_family": font,
        "template": tmpl["document_type"],
        "document_type": tmpl["document_type"],
        "paragraphs": [
            tmpl["org_header"],
            tmpl["org_name"],
            tmpl["title"],
            *tmpl["body_paragraphs"],
        ],
        "tables": [{"data": [tmpl["table_headers"]] + rows}],
        "footer": tmpl["footer"],
    }
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {img_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full-page A4 Khmer document images for benchmark testing."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval/datasets/synthetic_documents"),
        help="Directory to write PNG + JSON pairs (default: eval/datasets/synthetic_documents)",
    )
    parser.add_argument(
        "--font", default="all",
        help="Font name or 'all' to sweep all 5 fonts (default: all)",
    )
    parser.add_argument(
        "--count", type=int, default=3,
        help="Number of table rows per document, 1–5 (default: 3)",
    )
    args = parser.parse_args()

    fonts = _FONTS if args.font == "all" else [args.font]
    row_count = max(1, min(5, args.count))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for font in fonts:
        print(f"\nFont: {font}")
        for doc_index, tmpl in enumerate(_DOCUMENT_TEMPLATES):
            _render_document(tmpl, font, row_count, args.output_dir, doc_index)
            total += 1

    print(f"\nDone. {total} document images written to {args.output_dir}/")


if __name__ == "__main__":
    main()
