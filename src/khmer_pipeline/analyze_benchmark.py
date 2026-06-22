import csv
import sys
from collections import defaultdict
from pathlib import Path

_RUNS_ROOT = Path("eval/runs")


def _load_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        p = Path(p)
        # if a directory, read results.csv inside it
        if p.is_dir():
            csv_path = p / "results.csv"
        else:
            csv_path = p
        if not csv_path.exists():
            print(f"Note: {csv_path} not found — skipping.")
            continue
        rows.extend(csv.DictReader(csv_path.open(encoding="utf-8")))
    return rows


def _latest_run_dir() -> Path | None:
    if not _RUNS_ROOT.exists():
        return None
    subdirs = [d for d in _RUNS_ROOT.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "No results."

    lines = []

    by_engine: dict[str, list] = defaultdict(list)
    by_font: dict[str, list] = defaultdict(list)
    by_template: dict[str, list] = defaultdict(list)
    by_dataset: dict[str, list] = defaultdict(list)
    for r in rows:
        by_engine[r.get("Engine", "")].append(r)
        by_font[r.get("Font", "")].append(r)
        by_template[r.get("Template", "")].append(r)
        by_dataset[r.get("Dataset", "")].append(r)

    def avg(items: list, key: str) -> float:
        vals = [float(i[key]) for i in items if i.get(key)]
        return sum(vals) / len(vals) if vals else 0.0

    def total(items: list, key: str) -> int:
        vals = [int(float(i[key])) for i in items if i.get(key)]
        return sum(vals)

    def ratio(items: list, found_key: str, expected_key: str) -> float:
        found_vals = [float(i[found_key]) for i in items if i.get(found_key)]
        exp_vals = [float(i[expected_key]) for i in items if i.get(expected_key)]
        if not found_vals or not exp_vals:
            return 0.0
        return sum(found_vals) / sum(exp_vals)

    header = (
        f"{'Group':<28} {'CellAcc':>8} {'ContentRec':>10} {'TableCER':>9}"
        f" {'TextCER':>8} {'DocCER':>8} {'TabRatio':>9} {'ParaRec':>8} {'ParaLeak':>9}"
    )

    # per-Engine summary for model-vs-model comparison
    lines.append(f"\n=== Per-Engine Summary ({len(rows)} images) ===")
    eng_header = f"{'Engine':<28} {'CellAcc':>8} {'TableCER':>9} {'TextCER':>8} {'DocCER':>8}"
    lines.append(eng_header)
    for name, items in sorted(by_engine.items()):
        lines.append(
            f"{name:<28}"
            f" {avg(items, 'Cell_Accuracy'):>8.3f}"
            f" {avg(items, 'Table_CER'):>9.3f}"
            f" {avg(items, 'Text_CER'):>8.3f}"
            f" {avg(items, 'Document_CER'):>8.3f}"
        )

    def add_group_table(groups: dict[str, list], label: str) -> None:
        lines.append(f"\n=== Per-{label} Summary ({len(rows)} images) ===")
        lines.append(header)
        for name, items in sorted(groups.items()):
            lines.append(
                f"{name:<28}"
                f" {avg(items, 'Cell_Accuracy'):>8.3f}"
                f" {avg(items, 'Cell_Content_Recall'):>10.3f}"
                f" {avg(items, 'Table_CER'):>9.3f}"
                f" {avg(items, 'Text_CER'):>8.3f}"
                f" {avg(items, 'Document_CER'):>8.3f}"
                f" {ratio(items, 'Tables_Found', 'Tables_Expected'):>9.3f}"
                f" {avg(items, 'Paragraph_Recall'):>8.3f}"
                f" {total(items, 'Paragraph_Leak'):>9}"
            )

    add_group_table(by_font, "Font")
    add_group_table(by_template, "Template")
    add_group_table(by_dataset, "Dataset")

    # best/worst by Cell_Accuracy
    scored = [r for r in rows if r.get("Cell_Accuracy")]
    if scored:
        best = max(scored, key=lambda r: float(r["Cell_Accuracy"]))
        worst = min(scored, key=lambda r: float(r["Cell_Accuracy"]))
        lines.append(
            f"\n=== Best & Worst by Cell_Accuracy ===\n"
            f"Best:  {best['Image_File']} | Font: {best.get('Font', '')} | Template: {best.get('Template', '')}"
            f" | Cell_Accuracy: {best['Cell_Accuracy']}\n"
            f"Worst: {worst['Image_File']} | Font: {worst.get('Font', '')} | Template: {worst.get('Template', '')}"
            f" | Cell_Accuracy: {worst['Cell_Accuracy']}"
        )

    cer_scored = [r for r in rows if r.get("Table_CER")]
    if cer_scored:
        lowest = min(cer_scored, key=lambda r: float(r["Table_CER"]))
        lines.append(
            f"\n=== Lowest Table_CER ===\n"
            f"{lowest['Image_File']} | Font: {lowest.get('Font', '')} | Template: {lowest.get('Template', '')}"
            f" | Table_CER: {lowest['Table_CER']}"
        )

    return "\n".join(lines)


def analyze(paths) -> None:
    if not paths:
        latest = _latest_run_dir()
        if latest is None:
            print("No results")
            return
        paths = [latest]

    rows = _load_rows(paths)
    print(summarize(rows))


if __name__ == "__main__":
    input_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    analyze(input_paths)
