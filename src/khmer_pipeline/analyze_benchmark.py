import csv
import sys
from collections import defaultdict
from pathlib import Path


def analyze(csv_paths: list[Path]) -> None:
    rows = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"Note: {csv_path} not found — skipping.")
            continue
        rows.extend(csv.DictReader(csv_path.open(encoding="utf-8")))

    if not rows:
        print("No results")
        return

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
        f" {'TextCER':>8} {'TabRatio':>9} {'ParaRec':>8} {'ParaLeak':>9}"
    )

    def print_group_table(groups: dict[str, list], label: str) -> None:
        print(f"\n=== Per-{label} Summary ({len(rows)} images) ===")
        print(header)
        for name, items in sorted(groups.items()):
            print(
                f"{name:<28}"
                f" {avg(items, 'Cell_Accuracy'):>8.3f}"
                f" {avg(items, 'Cell_Content_Recall'):>10.3f}"
                f" {avg(items, 'Table_CER'):>9.3f}"
                f" {avg(items, 'Text_CER'):>8.3f}"
                f" {ratio(items, 'Tables_Found', 'Tables_Expected'):>9.3f}"
                f" {avg(items, 'Paragraph_Recall'):>8.3f}"
                f" {total(items, 'Paragraph_Leak'):>9}"
            )

    # per-Engine summary for model-vs-model comparison
    print(f"\n=== Per-Engine Summary ({len(rows)} images) ===")
    eng_header = f"{'Engine':<28} {'CellAcc':>8} {'TableCER':>9} {'TextCER':>8}"
    print(eng_header)
    for name, items in sorted(by_engine.items()):
        print(
            f"{name:<28}"
            f" {avg(items, 'Cell_Accuracy'):>8.3f}"
            f" {avg(items, 'Table_CER'):>9.3f}"
            f" {avg(items, 'Text_CER'):>8.3f}"
        )

    print_group_table(by_font, "Font")
    print_group_table(by_template, "Template")
    print_group_table(by_dataset, "Dataset")

    # best/worst by Cell_Accuracy
    scored = [r for r in rows if r.get("Cell_Accuracy")]
    if scored:
        best = max(scored, key=lambda r: float(r["Cell_Accuracy"]))
        worst = min(scored, key=lambda r: float(r["Cell_Accuracy"]))
        print(
            f"\n=== Best & Worst by Cell_Accuracy ===\n"
            f"Best:  {best['Image_File']} | Font: {best.get('Font', '')} | Template: {best.get('Template', '')}"
            f" | Cell_Accuracy: {best['Cell_Accuracy']}\n"
            f"Worst: {worst['Image_File']} | Font: {worst.get('Font', '')} | Template: {worst.get('Template', '')}"
            f" | Cell_Accuracy: {worst['Cell_Accuracy']}"
        )

    cer_scored = [r for r in rows if r.get("Table_CER")]
    if cer_scored:
        lowest = min(cer_scored, key=lambda r: float(r["Table_CER"]))
        print(
            f"\n=== Lowest Table_CER ===\n"
            f"{lowest['Image_File']} | Font: {lowest.get('Font', '')} | Template: {lowest.get('Template', '')}"
            f" | Table_CER: {lowest['Table_CER']}"
        )


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [Path("./benchmark_results.csv")]
    analyze(paths)
