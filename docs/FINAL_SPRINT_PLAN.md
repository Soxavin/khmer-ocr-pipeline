# Final Internship Sprint Plan (≈6 weeks, from 2026-06-20)

Durable copy of the approved roadmap for the final phase. Goal: a production-grade
(single-user), scientifically validated, 100% local Khmer financial-document OCR
pipeline with a thesis-ready evaluation and a polished Streamlit UI.

**Legend:** 🤖 = code task (Sonnet) · 🧑 = your task (collect/label/run/write).

**Decisions:** real docs are a mix of born-digital + scanned · golden set 3–8 docs ·
baseline = Tesseract `khm` (not PaddleOCR/RapidOCR — no Khmer support) ·
"production-grade" = polished, reliable single-user desktop app (no multi-user/Docker).

**Core approach:** reuse the existing `eval/` harness — real documents become a dataset
under `eval/datasets/real/` using the same `*_ground_truth.json` schema, so
`run_benchmark` / `evaluate_structure` / `analyze_benchmark` work unchanged.

---

## Audit summary (why this plan)

- **Good (thesis assets):** Surya 0.20 + llamacpp Metal; VLM-HTML single source of truth
  (removed perf trap + the two-grid index-join bug); deterministic free eval + manifests +
  PROJECT_LOG; engine registry.
- **Risks:** (1) synthetic numbers are an upper bound, not real performance; (2) the OpenCV
  preprocessing stack was never tested on the scans it's for; (3) **legacy non-Unicode Khmer
  encoding** could invalidate Unicode metrics — check first; (4) evaluator assumes single
  table/page.
- **Blind spots:** llama-server lifecycle/leaks; 24 GB memory ceiling on big scans;
  **born-digital text layer = free ground truth**; reproducibility depends on live Google Fonts.

---

## Phase 0 — Reality Check & Foundations (Week 1) — Risk #3; unblocks all
- 🤖 `inspect_pdf.py` — diagnose each real PDF: text layer?, Unicode-vs-legacy heuristic
  (Khmer block U+1780–17FF ratio), scanned/image-only?, raster DPI. Report + JSON.
- 🤖 `harvest_ground_truth.py` — born-digital → render page PNG + auto `*_ground_truth.json`
  (paragraphs from text layer, NFC; tables stub for manual fill) into `eval/datasets/real/`.
- 🤖 Document the real-dataset convention in `eval/README.md`.
- 🧑 Collect 3–8 real docs; run diagnostic. **Decision gate:** legacy-encoded share ⇒ headline
  finding (+ transcoding/limitation sub-task) before trusting CER.
- 🧑 Hand-verify harvested GT (digital) / hand-label GT (scanned).

## Phase 1 — Real-Document Evaluation (Weeks 2–3) — Risks #1,#2,#4; thesis core
- 🤖 Multi-table evaluation (match N predicted↔N GT tables; drop the `table[0]`-only assumption).
- 🤖 Preprocessing A/B for scans (raw vs full `PreprocessConfig`) to test the untested module.
- 🧑 Run benchmark on `eval/datasets/real`; analyze; write the **synthetic-vs-real gap**.

## Phase 2 — Tesseract Baseline (Weeks 3–4) — Path A
- 🤖 `tesseract_engine.py` (`OCREngine` protocol, `pytesseract` + `khm`); text_blocks from TSV;
  `tables=[]` (classic OCR has no structure — a finding). Register in `engine_registry`. Pin dep.
- 🤖 Scope comparison to text metrics (`Text_CER`, `Paragraph_Recall`, flattened content recall).
- 🧑 Run both engines on synthetic + real; `analyze_benchmark <surya_run> <tesseract_run>`.

## Phase 3 — Single-User Productionization & UI (Week 5) — Path C, timeboxed
- 🤖 `llama-server` lifecycle (clean start/stop, health check, launch/stop script).
- 🤖 UI polish (surface warnings/confidence, clearer table grid, model-loaded indicator, error states).
- 🧑/🤖 Memory stress check on a real multi-page scan; document practical page limit.
- 🤖 Reproducibility freeze (commit dataset snapshot or pin font versions + manifest).

## Phase 4 — Thesis Report & Presentation (Week 6) — Path D, seeded throughout
- 🤖 `docs/REPORT.md` assembled from PROJECT_LOG + manifests (problem, architecture/decisions,
  methodology, results synthetic+real+gap+Surya-vs-Tesseract, limitations, future work).
- 🤖 `analyze_benchmark --charts` (matplotlib) → per-font / per-engine / synthetic-vs-real figures.
- 🧑 Write narrative + presentation.

## Continuous
Update `PROJECT_LOG.md` per phase; cite results by run_id; Opus details each phase when reached,
Sonnet implements; stay within existing module boundaries + CLAUDE.md conventions.
