"""Engine A/B on real financial documents, split by cell class.

Answers one question: when the frontend runs `auto` instead of `surya`, what
actually degrades — the numbers, the Khmer labels, or both? A single aggregate
accuracy hides that, so every run reports Cell/Numeric/Khmer accuracy side by side.

Targets come in two flavours (see `_TARGETS`): multi-page documents scored against
a stitched `_document_gt.json`, and single scanned pages scored against a per-page
`_ground_truth.json` with no stitching. The scanned pages are the hard cases — the
legacy budget table and the MoC gas bulletin — which the document-only harness
could not reach.

One engine per invocation (models are multi-GB; a fresh process per engine keeps
peak memory flat on the 24GB Mac and guarantees no cross-engine state leaks).
Each run appends a JSON file, so a long matrix is resumable and `--summarize`
can be re-run at any time. Every record also stores the PREDICTED GRID, so a
change to the metric is re-scored with `--rescore` in seconds instead of costing
another pass over the models.

    uv run python scripts/compare_engines_ab.py --engine surya --out eval/runs/ab_x
    uv run python scripts/compare_engines_ab.py --engine auto --target moc_gas_p1 --repeat 3 --out eval/runs/ab_x
    uv run python scripts/compare_engines_ab.py --rescore eval/runs/ab_x
    uv run python scripts/compare_engines_ab.py --summarize eval/runs/ab_x

Reuses the production path: the same `ingest`/`preprocess` as
scripts/eval_document.py, `merge_document_tables` for stitching, and
`evaluate_table` for scoring against the verified GT.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from khmer_pipeline.ingest import ingest
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.preprocess import preprocess, PreprocessConfig
from khmer_pipeline.engines.engine_registry import get_ocr_engine
from khmer_pipeline.engines.table_merge_pages import merge_document_tables
from khmer_pipeline.evaluation.evaluate_structure import (
    evaluate_table,
    gt_table_grid,
    pred_table_grid,
)
from khmer_pipeline.evaluation.gt_provenance import circularity_note, is_circular

_REAL_DIR = Path("eval/datasets/real")
_MOC_GAS_DIR = Path("eval/datasets/moc_gas")
_BUDGET_TEXTLAYER_DIR = Path("eval/datasets/budget_textlayer")


@dataclass(frozen=True)
class Target:
    """One scoreable document: where its pages live and how its GT is shaped.

    `mode="document"` ingests every `<stem>_p*.png`, stitches with
    `merge_document_tables` and scores against `<stem>_document_gt.json`.
    `mode="page"` ingests the single `<stem>.png` and scores that page's tables
    directly against `<stem>_ground_truth.json` — nothing to stitch."""
    directory: Path
    stem: str
    mode: str


# ARDB bulletins are the born-digital happy path; budget p3 and moc_gas p1 are
# scanned pages and the documents this investigation actually needs.
_TARGETS: dict[str, Target] = {
    "ardb0": Target(
        _REAL_DIR,
        "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26",
        "document",
    ),
    "ardb1": Target(
        _REAL_DIR,
        "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-15.06.26",
        "document",
    ),
    "budget_p3": Target(_REAL_DIR, "CambodiaBudgetExecutioninApr-2024_p3", "page"),
    "moc_gas_p1": Target(_MOC_GAS_DIR, "notification_no_2138_16.06.2026_p1", "page"),
}

# Text-layer-harvested budget pages (scripts/harvest_textlayer_gt.py): free,
# model-free numeric+structure GT. Khmer is blanked there, so only the metrics
# named in each file's scoring_scope are meaningful — see _SCOPE_MASKED_METRICS.
for _page in (4, 5, 6, 8, 9):
    _TARGETS[f"budget_p{_page}"] = Target(
        _BUDGET_TEXTLAYER_DIR, f"CambodiaBudgetExecutioninApr-2024_p{_page}", "page"
    )

# Metrics that a restricted-scope GT cannot support. Printed as "—" rather than
# as a number, so a masked score can never be mistaken for a measured one.
_SCOPE_MASKED_METRICS: dict[str, set[str]] = {
    "numeric_and_structure": {"cell_accuracy", "khmer_cell_accuracy", "table_cer"},
}

_ENGINES = ["surya", "auto", "surya_kiri_vlm", "surya_kiri"]
_METRICS = [
    ("cell_accuracy", "Cell_Acc"),
    ("numeric_cell_accuracy", "Numeric"),
    ("khmer_cell_accuracy", "Khmer"),
    # Script-INDEPENDENT: an engine with no Khmer can still win on structure, and
    # the bake-off needs to see that rather than reading it as a total failure.
    ("row_alignment_rate", "RowAlign"),
    ("table_cer", "CER"),
]


def _load_pages(target: Target) -> PreprocessResult:
    """Ingest + preprocess a target's page PNGs exactly as the product does."""
    if target.mode == "document":
        pngs = sorted(glob.glob(str(target.directory / f"{target.stem}_p*.png")))
    else:
        pngs = [str(target.directory / f"{target.stem}.png")]

    images = []
    for p in pngs:
        images.extend(ingest(Path(p).read_bytes(), Path(p).name, dpi=200).page_images)
    if not images:
        raise SystemExit(f"No page PNGs for target: {target.stem}")

    ing = IngestResult(
        source_name=target.stem, page_images=images, dpi=200, page_count=len(images)
    )
    # with_recognition_images: surya_kiri and the VLM hybrid read cells from the
    # geometric-only frame; omitting it silently costs them accuracy (§2.30).
    return preprocess(ing, PreprocessConfig(with_recognition_images=True))


def _gt_path(target: Target) -> Path:
    suffix = "_document_gt.json" if target.mode == "document" else "_ground_truth.json"
    return target.directory / f"{target.stem}{suffix}"


def _table_from_grid(grid: list[list[str]]) -> dict:
    """Rebuild a minimal predicted-table dict from a stored grid.

    Every cell is emitted, empty ones included, so `pred_table_grid` recovers the
    original row/column extent and `--rescore` round-trips exactly."""
    return {
        "cells": [
            {"row_id": r, "col_id": c, "text_lines": [{"text": value}]}
            for r, row in enumerate(grid)
            for c, value in enumerate(row)
        ]
    }


def _score(pred_grids: list[list[list[str]]], gt: dict) -> dict:
    """Score stored predicted grids against a loaded GT document."""
    return evaluate_table([_table_from_grid(g) for g in pred_grids], gt_table_grid(gt))


def run_one(engine_key: str, name: str, target: Target, repeat: int) -> dict:
    """Run one engine over one target and score it against the verified GT."""
    pre = _load_pages(target)

    t0 = time.perf_counter()
    result = get_ocr_engine(engine_key)(pre)
    seconds = time.perf_counter() - t0

    if target.mode == "document":
        pred_tables = merge_document_tables(result.pages)
    else:
        pred_tables = result.pages[0].tables

    gt = json.loads(_gt_path(target).read_text(encoding="utf-8"))
    pred_grids = [pred_table_grid(t) for t in pred_tables]

    return {
        "engine": engine_key,
        "target": name,
        "stem": target.stem,
        "mode": target.mode,
        "repeat": repeat,
        "seconds": round(seconds, 1),
        "pages": pre.page_count,
        "gt_provisional": bool(gt.get("needs_review_rows")),
        # Recorded on the result itself so a circular score can never be read
        # back later without its caveat attached.
        "gt_circular": is_circular(engine_key, gt),
        "scoring_scope": gt.get("scoring_scope"),
        "metrics": _score(pred_grids, gt),
        # Stored so a metric change is a re-score, not another model run.
        "pred_grids": pred_grids,
        # The router records its decision + measured fraction here; this is how we
        # see whether `auto` fell back to surya or kept kiri.
        "auto_router": [w for w in result.warnings if "[AutoRouter]" in w],
    }


def rescore(out_dir: Path) -> int:
    """Recompute every stored run's metrics from its saved grid, in place."""
    files = sorted(out_dir.glob("*.json"))
    if not files:
        print(f"No results in {out_dir}")
        return 1
    skipped = 0
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "pred_grids" not in rec:
            # Predates grid persistence — only a fresh run can recover it.
            print(f"  SKIP {f.name}: no stored grid")
            skipped += 1
            continue
        gt = json.loads(_gt_path(_TARGETS[rec["target"]]).read_text(encoding="utf-8"))
        before = rec["metrics"]["cell_accuracy"]
        rec["metrics"] = _score(rec["pred_grids"], gt)
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        after = rec["metrics"]["cell_accuracy"]
        flag = "" if abs(after - before) < 1e-9 else f"  (cell {before:.3f} -> {after:.3f})"
        print(f"  {f.name}{flag}")
    print(f"rescored {len(files) - skipped}/{len(files)} records in {out_dir}")
    return 0


def _fmt_row(label: str, cols: list[str], width: int = 22) -> str:
    return label.ljust(width) + "".join(c.rjust(14) for c in cols)


def _agg(values: list[float]) -> tuple[float, float]:
    """Median and spread (max-min) of a metric across repeat runs."""
    return statistics.median(values), (max(values) - min(values))


def summarize(out_dir: Path) -> int:
    """Print the comparison table from every JSON result in `out_dir`.

    Repeat runs of the same engine collapse into one row: the median, with a
    ±spread suffix whenever the runs disagree."""
    files = sorted(out_dir.glob("*.json"))
    if not files:
        print(f"No results in {out_dir}")
        return 1
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    by_target: dict[str, list[dict]] = {}
    for r in runs:
        by_target.setdefault(r.get("target", r["stem"]), []).append(r)

    for target_name, rs in sorted(by_target.items()):
        by_engine: dict[str, list[dict]] = {}
        for r in rs:
            by_engine.setdefault(r["engine"], []).append(r)

        print(f"\n=== {target_name}  ({rs[0]['stem']})")
        if any(r["gt_provisional"] for r in rs):
            print("    (PROVISIONAL — GT has unverified rows)")
        scope = rs[0].get("scoring_scope")
        if scope:
            print(f"    (scope={scope} — '—' columns are not measurable against this GT)")
        print(_fmt_row("engine", [label for _, label in _METRICS] + ["secs", "runs"]))
        print("-" * 96)

        for engine in sorted(by_engine, key=lambda e: _ENGINES.index(e) if e in _ENGINES else 99):
            group = by_engine[engine]
            masked = _SCOPE_MASKED_METRICS.get(group[0].get("scoring_scope") or "", set())
            cols = []
            spreads = []
            for key, _ in _METRICS:
                if key in masked:
                    cols.append("—")
                    continue
                median, spread = _agg([g["metrics"][key] for g in group])
                spreads.append(spread)
                cols.append(f"{median:.3f}" if spread == 0 else f"{median:.3f}±{spread:.3f}")
            secs = statistics.median([g["seconds"] for g in group])
            cols.append(f"{secs:.0f}")
            cols.append(str(len(group)))
            # Column count is the structural axis that separates engines; exact
            # grid match is reported too but is near-always False on real pages
            # (systematic ±1 row from header handling), so it leads with columns.
            n = len(group)
            cols_ok = sum(1 for g in group if g["metrics"].get("col_count_match"))
            shape_ok = sum(1 for g in group if g["metrics"].get("grid_shape_match"))
            label = engine + (" [CIRCULAR]" if any(g.get("gt_circular") for g in group) else "")
            print(_fmt_row(label, cols) + f"   cols {cols_ok}/{n}  shape {shape_ok}/{n}")
            if len(group) > 1 and all(s == 0 for s in spreads):
                print(f"    ^ identical across {len(group)} runs")

        n = rs[0]["metrics"]
        print(f"    cells: {n['cells_total']} total · {n['numeric_cells_total']} numeric "
              f"· {n['khmer_cells_total']} Khmer")
        for r in rs:
            for w in r["auto_router"]:
                print(f"    [{r['engine']} r{r.get('repeat', 1)}] {w}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=_ENGINES, help="engine to run (one per process)")
    ap.add_argument("--target", action="append", choices=list(_TARGETS),
                    help="target to score; repeatable (default: all)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per target, for measuring run-to-run spread (default 1)")
    ap.add_argument("--allow-circular", action="store_true",
                    help="score even when the engine's model family drafted the GT "
                         "(optimistic and not comparable; off by default)")
    ap.add_argument("--out", type=Path, help="directory to write JSON results")
    ap.add_argument("--rescore", type=Path, help="recompute metrics from stored grids")
    ap.add_argument("--summarize", type=Path, help="print the table for a results dir")
    args = ap.parse_args()

    if args.summarize:
        return summarize(args.summarize)
    if args.rescore:
        return rescore(args.rescore)
    if not args.engine or not args.out:
        ap.error("--engine and --out are required unless --summarize/--rescore is given")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in (args.target or list(_TARGETS)):
        target = _TARGETS[name]
        # Refuse a circular pairing BEFORE spending model time on it. --allow-circular
        # exists because such a run is still useful as a sanity check — but it must be
        # a deliberate choice, never a default that quietly inflates a benchmark.
        gt = json.loads(_gt_path(target).read_text(encoding="utf-8"))
        note = circularity_note(args.engine, gt)
        if note:
            if not args.allow_circular:
                print(f"SKIP {name}: {note}\n  (pass --allow-circular to run anyway)",
                      flush=True)
                continue
            print(f"WARNING {name}: {note}", flush=True)
        for i in range(1, args.repeat + 1):
            print(f"running {args.engine} on {name} (run {i}/{args.repeat}) …", flush=True)
            rec = run_one(args.engine, name, target, repeat=i)
            (args.out / f"{args.engine}__{name}__r{i}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            m = rec["metrics"]
            print(f"  cell={m['cell_accuracy']:.3f} numeric={m['numeric_cell_accuracy']:.3f} "
                  f"khmer={m['khmer_cell_accuracy']:.3f} cer={m['table_cer']:.3f} "
                  f"({rec['seconds']:.0f}s)", flush=True)
            for w in rec["auto_router"]:
                print(f"  {w}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
