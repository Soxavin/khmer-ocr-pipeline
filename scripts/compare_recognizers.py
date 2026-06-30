"""Combine per-engine recognition A/B results into one comparison table.

Reads every eval/runs/<ts>_recog_<engine>/recognition.csv (using only the
latest run dir per engine), pivots into one row per page / one column per
engine (cell = Recognition_CER), prints it, and writes the pivot to
eval/runs/<ts>_recog_compare/recognition_compare.csv.

Read-only with respect to the rest of the repo — only writes its own output
under eval/runs/.

    uv run python scripts/compare_recognizers.py
"""
from __future__ import annotations
import csv
import glob
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_RUNS = Path("eval/runs")


def _latest_run_per_engine() -> dict[str, Path]:
    """Map engine name -> its latest recognition.csv path."""
    candidates: dict[str, list[Path]] = defaultdict(list)
    for csv_path in sorted(glob.glob(str(_RUNS / "*_recog_*" / "recognition.csv"))):
        run_dir = Path(csv_path).parent
        if run_dir.name.endswith("_recog_compare"):
            continue
        rows = list(csv.DictReader(Path(csv_path).read_text(encoding="utf-8").splitlines()))
        if not rows:
            continue
        engine = rows[0]["Engine"]
        candidates[engine].append(run_dir)

    latest: dict[str, Path] = {}
    for engine, dirs in candidates.items():
        best = max(dirs, key=lambda d: d.name)
        latest[engine] = best / "recognition.csv"
    return latest


def _load_rows(latest: dict[str, Path]) -> dict[str, dict[str, float]]:
    """engine -> {Image_File: Recognition_CER}."""
    data: dict[str, dict[str, float]] = {}
    for engine, csv_path in latest.items():
        per_image: dict[str, float] = {}
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                per_image[row["Image_File"]] = float(row["Recognition_CER"])
        data[engine] = per_image
    return data


def _short_label(image_file: str, width: int = 28) -> str:
    """Shorten a (possibly long, Khmer) filename for display."""
    if len(image_file) <= width:
        return image_file
    return "..." + image_file[-(width - 3):]


def main() -> int:
    latest = _latest_run_per_engine()
    if not latest:
        print(f"No */recognition.csv found under {_RUNS}")
        return 1

    data = _load_rows(latest)
    engines = sorted(data)
    images = sorted({img for per_image in data.values() for img in per_image})

    if not images:
        print("No pages found across engine runs.")
        return 1

    # Build pivot rows (as strings for the CSV/printed table).
    pivot_rows: list[tuple[str, dict[str, str]]] = []
    for img in images:
        cells = {}
        for engine in engines:
            cerv = data[engine].get(img)
            cells[engine] = f"{cerv:.3f}" if cerv is not None else ""
        pivot_rows.append((img, cells))

    means: dict[str, str] = {}
    for engine in engines:
        values = list(data[engine].values())
        means[engine] = f"{(sum(values) / len(values)):.3f}" if values else ""

    # --- print readable aligned table ---
    label_width = max([len("Image_File")] + [len(_short_label(img)) for img in images] + [len("MEAN")])
    col_widths = {engine: max(len(engine), 5) for engine in engines}

    header = "Image_File".ljust(label_width) + "  " + "  ".join(
        engine.rjust(col_widths[engine]) for engine in engines
    )
    print(header)
    print("-" * len(header))
    for img, cells in pivot_rows:
        line = _short_label(img).ljust(label_width) + "  " + "  ".join(
            cells[engine].rjust(col_widths[engine]) for engine in engines
        )
        print(line)
    print("-" * len(header))
    mean_line = "MEAN".ljust(label_width) + "  " + "  ".join(
        means[engine].rjust(col_widths[engine]) for engine in engines
    )
    print(mean_line)

    # --- write compare CSV (full Image_File names, not shortened) ---
    run_dir = _RUNS / f"{datetime.now():%Y%m%d_%H%M%S}_recog_compare"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "recognition_compare.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_File", *engines])
        for img, cells in pivot_rows:
            w.writerow([img, *(cells[engine] for engine in engines)])
        w.writerow(["MEAN", *(means[engine] for engine in engines)])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
