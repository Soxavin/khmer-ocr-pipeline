import csv
import sys
from collections import defaultdict
from pathlib import Path


def analyze(csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        print("No results found.")
        return

    by_font: dict[str, list] = defaultdict(list)
    by_template: dict[str, list] = defaultdict(list)
    for r in rows:
        by_font[r["Font"]].append(r)
        by_template[r["Template"]].append(r)

    def avg(items: list, key: str) -> float:
        vals = [float(i[key]) for i in items if i[key]]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n=== Per-Font Summary ({len(rows)} images) ===")
    print(f"{'Font':<24} {'Avg Score':>10} {'Avg CER%':>10} {'Halluc':>8} {'Omiss':>8}")
    font_avgs = sorted(by_font.items(), key=lambda kv: avg(kv[1], "Overall_Score"), reverse=True)
    for font, items in font_avgs:
        print(
            f"{font:<24} {avg(items, 'Overall_Score'):>10.1f} {avg(items, 'Estimated_CER'):>10.1f}"
            f" {avg(items, 'Hallucinations_Count'):>8.1f} {avg(items, 'Omissions_Count'):>8.1f}"
        )

    print("\n=== Per-Template Summary ===")
    print(f"{'Template':<24} {'Avg Score':>10} {'Avg CER%':>10}")
    for tmpl, items in sorted(by_template.items(), key=lambda kv: avg(kv[1], "Overall_Score"), reverse=True):
        print(f"{tmpl:<24} {avg(items, 'Overall_Score'):>10.1f} {avg(items, 'Estimated_CER'):>10.1f}")

    best = max(rows, key=lambda r: float(r["Overall_Score"] or 0))
    print(
        f"\n=== Best Overall ===\n"
        f"Font: {best['Font']} | Template: {best['Template']} | "
        f"Score: {best['Overall_Score']} | CER: {best['Estimated_CER']}%"
    )


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./benchmark_results.csv")
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    analyze(path)
