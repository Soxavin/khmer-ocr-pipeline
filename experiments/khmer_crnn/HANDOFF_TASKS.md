# Handoff task specs (paste-ready prompts for another AI)

Each block below is a **self-contained prompt** you can paste into another AI
agent (e.g. a Sonnet coding agent) working in this repo
(`/Users/vin/Internship/khmer-ocr-pipeline`). They assume the agent can read the
repo. Dispatch order suggestion: **#1** (highest leverage), then **#2**
(housekeeping), then **#3** (independent). Delete this file once the tasks land.

Repo-wide constraints every task must follow:
- Use **`uv`** for deps (`uv add`, pin ML upper bounds); follow `CLAUDE.md`.
- `src/khmer_pipeline/**` public functions get concise docstrings; `from __future__ import annotations`.
- **Never commit** `eval/datasets/` or `sample_data/` (gitignored sensitive government data).
- After touching `src/khmer_pipeline/**` or `app.py`, run:
  `uv run pytest -q --tb=short && python3 -m py_compile app.py $(find src/khmer_pipeline -name '*.py')`

---

## TASK #1 — Multi-font synthetic line-image generator + local-corpus loader

You are working in `/Users/vin/Internship/khmer-ocr-pipeline`. Read
`experiments/khmer_crnn/FINETUNING_PLAN.md` (§5), `experiments/khmer_crnn/train.py`,
`src/khmer_pipeline/utils/fonts.py`, and
`src/khmer_pipeline/datagen/generate_synthetic_documents.py` first.

**Goal:** produce a synthetic **multi-font Khmer line-image corpus** for training
a recognizer, and let the CRNN trainer consume it locally. Today the trainer only
reads the HF dataset `seanghay/khmer-hanuman-100k` (single font); this unblocks
the multi-font path.

**Deliverable A — the generator** `src/khmer_pipeline/datagen/generate_synthetic_lines.py`:
- Renders **single lines** of Khmer text, one image per line, in each of the **5
  vendored OFL fonts** (Noto Sans Khmer, Battambang, Hanuman, Moul, Fasthand).
- **Reuse** `khmer_pipeline.utils.fonts.font_face_style_tag(family)` and mirror
  the **Playwright** rendering pattern in `generate_synthetic_documents.py`
  (fonts inlined/offline, `document.fonts.ready`, font-loaded assertion). Render
  one line of text on a white background, **tight-crop to the text bounding box**,
  variable width.
- Output: a folder of PNGs + a **`manifest.jsonl`** with one record per line:
  `{"image": "<relative_png_path>", "text": "<string>", "font": "<family>"}`.
- CLI (argparse): `--output-dir`, `--count` (lines per font), `--fonts` (default
  all 5), `--text-source` (path to a UTF-8 file of Khmer strings, one per line;
  OR the literal `hanuman` to reuse the labels from `seanghay/khmer-hanuman-100k`),
  `--degrade` (optional; reuse `datagen/generate_degraded.py` to add scan-like
  noise), `--img-h` (render height), `--seed`.
- Follow the repo's datagen conventions (module docstring with CLI example,
  `from __future__ import annotations`, docstrings on public functions).

**Deliverable B — local-corpus loading in `experiments/khmer_crnn/train.py`:**
- Extend `load_rows()` so `--dataset` accepts **either** an HF dataset id (current
  behavior) **or a local path** to a folder containing `manifest.jsonl`. When
  local, yield the same `(text_col="text", img_col="image")` interface the rest
  of the trainer already expects (load the PNG via PIL). Keep changes minimal and
  backward-compatible; do not change the training loop.

**Tests:** add `tests/test_generate_synthetic_lines.py` — render a tiny corpus
(2–3 lines, 1 font) to a tmp dir, assert PNGs exist, `manifest.jsonl` parses, and
records have the three keys. Keep it fast (small counts). Do not require network
unless `--text-source hanuman` is used.

**Constraints:** don't break existing datagen or the 377-test suite; don't commit;
Playwright is already a dependency.

**Verify before finishing:**
1. `uv run python -m khmer_pipeline.datagen.generate_synthetic_lines --output-dir /tmp/lines_demo --count 5 --fonts all` → 25 PNGs + manifest.
2. `uv run python experiments/khmer_crnn/train.py --dataset /tmp/lines_demo --limit 25 --epochs 1 --max-steps 5` runs end-to-end on MPS.
3. `uv run pytest -q` still green.

**Report back:** files created/changed, the manifest schema, sample render
dimensions, whether the two verify steps + pytest passed (with real output), and
any assumptions you had to make. Do not overstate — if a step didn't complete,
say so.

---

## TASK #2 — Commit the CRNN experiment + PROJECT_LOG entry + gitignore

You are working in `/Users/vin/Internship/khmer-ocr-pipeline`. The Khmer CRNN
training exercise under `experiments/khmer_crnn/` is complete but uncommitted.
Read `experiments/khmer_crnn/FINDINGS.md` and the tail of `docs/PROJECT_LOG.md`
(to match its section numbering + house style) first.

**Deliverables:**
1. **`.gitignore`:** add `experiments/khmer_crnn/runs/` (training artifacts) and
   ensure `__pycache__/` is ignored. Do **not** ignore the source files.
2. **PROJECT_LOG entry:** append a new section (next number in sequence, matching
   the existing format) summarizing the exercise. It must cover:
   - Framing: **adapted the mentor's CUDA-oriented starter script** to our Mac
     (MPS) + research use-case — training a CRNN (ResNet+BiRNN+CTC) **from
     scratch** on `seanghay/khmer-hanuman-100k` (single font) to **learn the
     training loop and benchmark epoch time**. Not fine-tuning; single-font, so
     it won't read the real GDDE docs.
   - Adaptations made (portable device/MPS, MPS-CTC CPU fallback, leakage-safe
     split + train-only vocab, validation CER, per-epoch benchmarking, seeding/
     checkpoints, and the **CTC-feasibility check** that surfaced the 256px→long-
     line mismatch → widened to 1024px + length filter).
   - Results: benchmark ~121 s/epoch (ResNet34) / ~76 s (ResNet18); GRU≈LSTM for
     speed; convergence reaches **~3.4% CER (easy curriculum)** and **~4.6% CER
     (full sentence-length task)**; CTC blank-collapse breakout at epoch 3–4.
   - Pointers to `experiments/khmer_crnn/FINDINGS.md` and `FINETUNING_PLAN.md`.
   - Keep the neutral "adapted a starter script" framing — do NOT say the
     mentor's code was wrong/buggy.
3. **Commit on a branch** (e.g. `experiment/khmer-crnn-trainer`): create the
   branch, then commit `experiments/khmer_crnn/*.py`, `*.md`, `pyproject.toml`,
   `uv.lock`, `.gitignore`, and `docs/PROJECT_LOG.md`. Commit message ends with:
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. **Do NOT push.**

**Constraints:** never `git add` `experiments/khmer_crnn/runs/`, `eval/datasets/`,
or `sample_data/`. Do not force-push. Verify `uv run pytest -q` is still 377-green
before committing.

**Report back:** the branch name, exact files committed, the PROJECT_LOG section
number/title, and confirmation `runs/` is untracked + pytest green.

---

## TASK #3 — Track B: surgical app-layer refactor (mentor code-quality item)

You are working in `/Users/vin/Internship/khmer-ocr-pipeline`. Read `app.py`,
`CLAUDE.md`, and `docs/CODE_AUDIT.md` first. This is the last outstanding
code-quality item: `app.py`'s Streamlit stage-execution logic is repetitive.

**Goal (surgical — do not reorganize unrelated code):**
- Extract a single `_run_stage(...)` helper that captures the repeated
  run-a-pipeline-stage-and-surface-warnings pattern in `app.py`.
- Create `ui_helpers.py` for shared Streamlit helpers currently duplicated in
  `app.py` (and `lab.py` if clearly shared) — e.g. page-selection, warning
  display, download buttons. Unify the page-selection logic (the established
  `st.session_state.current_page_idx` pagination pattern — keep it, don't
  reintroduce a per-page loop).
- Add a short docstring/comment block documenting the `st.session_state` schema
  (keys + meaning) at the top of `app.py`.

**Constraints (from CLAUDE.md):**
- Streamlit width: use `width="stretch"`, never `use_container_width=True`.
- Engine execution goes through `engines/engine_registry.py` (`ACTIVE_OCR_ENGINE`
  / `ACTIVE_CORRECTION_ENGINE`) — don't import `run_surya`/`postprocess` directly.
- Surface pipeline issues via the existing `warnings.warn` → `SuryaResult.warnings`
  mechanism; don't add ad-hoc warning plumbing.
- Do NOT change behavior or reorganize unrelated code. There is no `test_app.py`;
  UI is verified by **manually running the app**.

**Verify:** `uv run pytest -q --tb=short && python3 -m py_compile app.py $(find src/khmer_pipeline -name '*.py')` stays green; then run `uv run streamlit run app.py` and confirm the stages, pagination, warnings, and Excel/CSV export still work. Clear `__pycache__` before the manual run if signatures changed.

**Report back:** the helpers extracted, what moved to `ui_helpers.py`, the
session_state schema you documented, pytest/py_compile result, and the manual
smoke-test outcome. Don't commit unless asked.

---

## TASK #4 — Prototype: Surya-detect + Kiri-recognize hybrid recognizer

You are working in `/Users/vin/Internship/khmer-ocr-pipeline`. Read
`experiments/khmer_crnn/FINETUNING_PLAN.md` **§3b and §7a**, `CONTEXT.md`
("Engine Swappability"), `src/khmer_pipeline/engines/engine_registry.py`,
`src/khmer_pipeline/engines/protocols.py`, and
`src/khmer_pipeline/engines/hybrid_engine.py` first.

**Background (established, do not re-investigate):** the recognition bottleneck on
the real GDDE market-price tables can be addressed by **mrrtmob/kiri-ocr** — a
bilingual EN+Khmer OCR (Apache-2.0). Its recognizer, called with
**`decode_method="fast"` (pure CTC)**, reads the real page `09.06.26_p2` — Khmer
product names, row numbers, `៛` units, and **all Arabic-numeral prices** — at
~99% confidence. (The default `"accurate"`/`"beam"` decoders duplicate digits —
do NOT use them.) The idea: let **Surya do detection/table-structure** (its
strength) and feed each cell crop to **Kiri's recognizer** for the text.

**Goal:** a working prototype recognizer engine that plugs into the existing
swappable-engine registry, plus an eval vs Surya-alone on the real docs.

**Deliverables:**
1. A **Kiri recognizer wrapper** (e.g. `src/khmer_pipeline/engines/kiri_recognizer.py`)
   exposing a function that takes a PIL/ndarray **cell crop** → returns recognized
   text, using `kiri_ocr` with `decode_method="fast"`, `device="cpu"`. Load the
   model once (module-level/lazy singleton). **We only need Kiri's *recognizer*,
   not its detector** — so avoid the DB detector path entirely (that's what pins
   `onnxruntime-gpu`, which has no macOS ARM wheels).
2. **Wire it as an OCR engine** behind `engine_registry` (an `OCR_ENGINE=surya_kiri`
   or similar): reuse Surya for layout + table **detection/structure** (as the
   hybrid engine already does), but route each detected cell/line crop through the
   Kiri recognizer instead of Surya's recognizer. Assemble into the same
   `SuryaResult`/table structures the pipeline expects (see `models.py`,
   `hybrid_engine.py` for the shape).
3. **Dependencies:** add `kiri-ocr` in a way that works on macOS ARM. The PyPI
   build is stale (dim-256 vs the HF checkpoint dim-384) and git main pins
   `onnxruntime-gpu`. Simplest robust path: install from git **without** its
   `onnxruntime-gpu` dep (or vendor the small recognizer + `preprocess_pil` +
   `model.py`), plus `torch`, `torchvision`, `safetensors`, `huggingface_hub`,
   `pillow`, `numpy`. Pin upper bounds (CLAUDE.md). Keep it in the **`experiments`
   optional group** or a new optional group — do NOT bloat the core pipeline env.
   Confirm exact install recipe works via `uv`.
4. **Eval:** run the new engine vs the Surya baseline on the real docs through the
   existing harness — `scripts/eval_document.py "<stem>" --preprocess` for both —
   and report Cell_Accuracy / Cell_Content_Recall / Table_CER side by side. Real
   docs + GT are gitignored under `eval/datasets/real/`.

**Constraints:** don't break the 377-test suite or `src/khmer_pipeline/**`
behavior for existing engines; the new engine is additive/opt-in. Real GDDE data
stays gitignored — never commit or upload it. Don't commit unless asked.

**Verify:**
1. Prototype reads `eval/datasets/real/…09.06.26_p2.png` cells correctly (spot-check
   prices vs GT — expect ~99% on Khmer + Arabic prices with `fast` decode).
2. `OCR_ENGINE=surya_kiri` eval vs `OCR_ENGINE=surya` on ≥1 real doc, metrics
   reported.
3. `uv run pytest -q` still green.

**Report back:** the engine wiring, the exact Kiri install recipe that works on
Mac, the head-to-head metrics table (Surya vs Surya+Kiri), any cells where Kiri
still errs (e.g. the `៛` unit occasionally → `!`/`អ`, and whether Surya's cropping
fixed the %-cells), and whether pytest passed. Be honest about regressions.
