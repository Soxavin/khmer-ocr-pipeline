# Code-Quality Audit (R0) — Khmer OCR Pipeline

*Read-only audit performed 2026-07-02 to scope the mentor's code-quality mandate. Feeds the surgical
refactor (R1), the **Deferred Technical Debt** backlog, and the **Proposed Directory Restructuring**
(both destined for `docs/REPORT.md`). No source was modified. Method: AST walk (docstring/typing
coverage) + targeted greps. See "How to re-measure" at the bottom.*

Related: approved sprint plan `~/.claude/plans/i-need-you-to-golden-summit.md`; `PROJECT_LOG.md` §2.25
(thesis pivot); `CLAUDE.md` (conventions — note the no-docstrings rule is being **reversed**, below).

---

## 1. Executive summary — mentor's list vs. reality

The mandate is **substantially smaller than the raw list implies**: two items are already met and one
is 90 % done.

| # | Mentor item | Status | Work remaining |
|---|---|---|---|
| 4c | `Hatchling` build backend | ✅ **Done** (`pyproject.toml:31–33`) | none |
| 3 | Modern `pytest` idioms | ✅ **Already satisfied** — 0 `TestCase` classes, 0 `self.assert`; every `unittest` reference is `unittest.mock` (idiomatic under pytest); tests are plain `def test_*` + `assert` | none |
| 4a | `from __future__ import annotations` | ✅ **~90 %** — 28/31 real `src/` files + all `scripts/` + both apps | 3 one-line adds: `analyze_benchmark.py`, `generate_synthetic_documents.py`, `model_config.py` |
| 4b | Docstrings on public funcs | ❌ Real — **3 of 62** public `src/` funcs documented | Trio: **14**; rest (**~45**) → debt |
| 2 | Export consistency | ❌ Real — `pandas` is **display-only (apps)**; CSV+BOM hack in `export.py`; 3 more modules hand-roll `csv` | `export.py` now; rest → debt |
| 1 | Data consistency (dataclass vs dict) | ❌ Real & **load-bearing** — cell/table/text_block are untyped `dict`s across ~8 modules | TypedDicts in trio now; full migration → debt |
| 5 | Dependency hygiene | ❌ Real findings (see §4) | small `pyproject` edits |

---

## 2. Confirmed surgical scope (R1)

**Trio = `surya.py` + `export.py` + `postprocess.py`** (swapped in over the mentor's `hybrid_engine.py`
example, since hybrid is demoted to a documented negative result — §2.25). Plus `app.py` / `lab.py`
(structural cleanup) and `models.py` (shared TypedDict home). **Must not break the 377 tests**; each
change gated by `uv run pytest -q --tb=short` + a manual Streamlit smoke-test.

- **`surya.py`** — docstrings for its 9 public funcs; annotate as the `Cell`/`Table`/`TextBlock`
  **producer**. (`future` already present.)
- **`export.py`** — docstrings (3 funcs); **standardize the CSV path**: preserve the UTF-8 BOM (Excel +
  Khmer needs it — `export.py:137`) via `utf-8-sig`, route XLSX through the same grid builder; annotate
  as heavy TypedDict **consumer** (23 `.get()`). **Guard byte-equivalence** with `test_export.py`
  (36 tests).
- **`postprocess.py`** — docstrings (2 funcs); TypedDict annotations.
- **`models.py`** — define `Cell` / `Table` / `TextBlock` `TypedDict`s (the shared schema).
- **`app.py` / `lab.py`** — functions are all `_`-private, so "public docstrings" barely applies; the
  high `.get()` counts (27 / 17) are mostly `st.session_state.get(...)` (idiomatic Streamlit), **not**
  the pipeline-dict problem. Their real work is the structural cleanup from the plan (`_run_stage`
  helper, shared `ui_helpers.py`, unified page-selection, session-state schema) — see the sprint plan.

**Rationale for TypedDict over dataclass now:** the cell/table dicts are the existing runtime
representation flowing through ~8 modules and serialized to/from JSON. `TypedDict` adds a checkable
schema with **zero runtime change** (annotations erased) — satisfies "typed & consistent" without the
all-or-nothing rewrite that would risk the 377 tests. Full dataclass migration is the documented
end-state (§5).

**Also update `CLAUDE.md`** in R1: reverse the "no docstrings in `src/`" rule to "docstrings on public
functions" so project conventions stop contradicting the mentor.

---

## 3. Per-axis findings

- **Docstrings (4b):** 62 public funcs in `src/`, only 3 documented (`harvest_ground_truth`,
  `inspect_pdf`, `memory` — 1 each). This reflects the old `CLAUDE.md` convention; the mentor reverses it.
- **`__future__` annotations (4a):** near-universal. Missing only in `analyze_benchmark.py`,
  `generate_synthetic_documents.py`, `model_config.py` (all trivial adds; none in the trio).
- **Test style (3):** 21 test files, all pytest-native (plain functions + `assert`; `@pytest.fixture`
  in 2). `unittest` appears **only** as `from unittest.mock import ...` — the standard way to mock
  under pytest. No migration needed.
- **Export (2):** `pandas` imported **only** in `app.py`/`lab.py` (for `st.data_editor` display). The
  `csv` module is used in `export.py` (+ manual BOM), and in `analyze_benchmark.py`, `run_benchmark.py`,
  `visualize_benchmark.py`. So there's no true pandas-vs-csv mix in one path; the fix is to centralize
  CSV writing (starting with `export.py`) and drop the manual BOM in favour of `utf-8-sig`.
- **Data consistency (1):** dict-access density (`.get()` count) — heaviest: `app.py` 27 (mostly
  session_state), `evaluate_structure.py` 23, `export.py` 23, `lab.py` 17, `analyze_benchmark.py` 16,
  `table_merge_pages.py` 13, `surya.py` 5. The pipeline's structured payloads (`tables`,
  `text_blocks`, cells) are `list[dict[str, Any]]` (`models.py:26–27, 41–42`).

---

## 4. Dependency hygiene (item 5)

- **`torch`** — imported directly (`device.py`, `memory.py`) but **not declared** in `pyproject.toml`
  (rides in transitively via `surya-ocr`). → **Add an explicit, pinned dependency.**
- **`openai`** — **declared but imported in ~1 file**, and the project is "100 % local." → **Verify the
  single usage and most likely remove it.**
- **`mlx`** (`mlx.core`) — imported directly while only `mlx-lm` is declared → consider an explicit
  (Mac-gated) `mlx` pin.
- **Intentional, leave as-is:** `transformers` (declared to pin the version even if only transitively
  imported — matches the project's ML-pinning rule); `mlx-vlm` (deliberately isolated via
  `uv run --no-project --with mlx-vlm`).

---

## 5. Deferred Technical Debt (→ REPORT section)

Out-of-surgical-scope, documented as a prioritized backlog:

1. **Docstrings** on ~45 public funcs across ~25 files (heaviest: `evaluate_structure.py` 11,
   `analyze_benchmark.py` 6, `table_stitch.py` 4, `backend_status.py`/`device.py`/`visualize_benchmark.py` 2).
2. **`from __future__ import annotations`** in the 3 remaining files (trivial; do opportunistically).
3. **TypedDict adoption** in the other heavy dict consumers: `evaluate_structure.py` (23),
   `table_merge_pages.py` (13), `table_stitch.py`, `hybrid_engine.py`.
4. **Centralize CSV writing** in `analyze_benchmark.py`, `run_benchmark.py`, `visualize_benchmark.py`
   (adopt `export.py`'s helper).
5. **Full `dict` → `dataclass` migration** for the cell/table/text_block payloads — one coordinated
   change across the ~8 modules in the table path (supersedes the interim TypedDicts).

---

## 6. Proposed Directory Restructuring (→ REPORT section; do NOT execute now)

`src/khmer_pipeline/` is **32 flat modules**, plus a separate top-level `scripts/` (10) and 2 root apps
(`app.py`, `lab.py`) — this flatness is what reads as "too many files / non-standard." Proposed grouping
(future work, high import/test-breakage risk → not executed):

| Sub-package | Modules |
|---|---|
| `engines/` | surya, tesseract_engine, hybrid_engine, engine_registry, protocols, layout_detect, slanet_structure, table_stitch, table_merge_pages |
| `pipeline/` | ingest, preprocess, postprocess, export, models, pipeline |
| `eval/` | run_benchmark, evaluate_structure, evaluate_judge, analyze_benchmark, visualize_benchmark |
| `datagen/` | generate_synthetic_tables, generate_synthetic_documents, generate_degraded, harvest_ground_truth, inspect_pdf |
| `cli/` | consolidate the 10 loose `scripts/*.py` + CLI entry points |
| (root utils) | device, memory, model_config, backend_status, fonts, khmer_normalize |

**Coherence smell to note:** overlap between `src/` eval modules and `scripts/` eval scripts (e.g.
`scripts/eval_document.py` / `eval_recognizers.py` / `eval_notable_page.py` vs `run_benchmark.py` +
`evaluate_structure.py`). **Only low-risk consolidation now if time permits:** move the loose
`scripts/*.py` into a unified `cli/` folder.

---

## Appendix — per-file raw data

`pub` = public funcs/methods · `doc` = how many have docstrings · `.get` = dict-access count ·
`fut` = has `from __future__ import annotations`.

### src/khmer_pipeline
| file | pub | doc | .get | fut |
|---|--:|--:|--:|:--|
| analyze_benchmark.py | 6 | 0 | 16 | ✗ |
| backend_status.py | 2 | 0 | 0 | ✓ |
| device.py | 2 | 0 | 3 | ✓ |
| engine_registry.py | 0 | 0 | 2 | ✓ |
| evaluate_judge.py | 1 | 0 | 0 | ✓ |
| evaluate_structure.py | 11 | 0 | 23 | ✓ |
| export.py | 3 | 0 | 23 | ✓ |
| fonts.py | 1 | 0 | 0 | ✓ |
| generate_degraded.py | 2 | 0 | 0 | ✓ |
| generate_synthetic_documents.py | 1 | 0 | 0 | ✗ |
| generate_synthetic_tables.py | 1 | 0 | 0 | ✓ |
| harvest_ground_truth.py | 2 | 1 | 0 | ✓ |
| hybrid_engine.py | 1 | 0 | 3 | ✓ |
| ingest.py | 1 | 0 | 0 | ✓ |
| inspect_pdf.py | 2 | 1 | 2 | ✓ |
| khmer_normalize.py | 1 | 0 | 0 | ✓ |
| layout_detect.py | 1 | 0 | 0 | ✓ |
| memory.py | 1 | 1 | 0 | ✓ |
| model_config.py | 0 | 0 | 0 | ✗ |
| models.py | 0 | 0 | 0 | ✓ |
| pipeline.py | 1 | 0 | 0 | ✓ |
| postprocess.py | 2 | 0 | 1 | ✓ |
| preprocess.py | 1 | 0 | 0 | ✓ |
| protocols.py | 0 | 0 | 0 | ✓ |
| run_benchmark.py | 1 | 0 | 2 | ✓ |
| slanet_structure.py | 1 | 0 | 0 | ✓ |
| surya.py | 9 | 0 | 5 | ✓ |
| table_merge_pages.py | 1 | 0 | 13 | ✓ |
| table_stitch.py | 4 | 0 | 0 | ✓ |
| tesseract_engine.py | 1 | 0 | 1 | ✓ |
| visualize_benchmark.py | 2 | 0 | 5 | ✓ |

### root apps
| file | pub | doc | .get | fut |
|---|--:|--:|--:|:--|
| app.py | 0 | 0 | 27 | ✓ |
| lab.py | 0 | 0 | 17 | ✓ |

*(`scripts/` — all 10 have `future`, 0 docstrings on their single public `main`; `tests/` — all 21
pytest-native, `future` present, docstrings not expected.)*

---

## How to re-measure
The audit was produced by an ad-hoc AST + grep pass (script kept in the session scratchpad, not
committed to avoid adding to the file count). To regenerate: walk every `.py`, `ast.parse` it, and for
each count public (non-`_`) `FunctionDef`s and how many have `ast.get_docstring`, whether the source
contains `from __future__ import annotations`, and the `.get(` occurrence count; separately grep for
`import pandas`, `import csv`, and `unittest`, and collect non-stdlib top-level imports to compare
against `pyproject.toml`.
