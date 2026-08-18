# tools/

One-off research and evaluation scripts used during development. They are **not** part of the core
pipeline (`src/khmer_pipeline/`) or the UI (`app.py`) — they support specific experiments and the
write-up in [`../docs/report/REPORT.md`](../docs/report/REPORT.md) / [`../docs/decisions/PROJECT_LOG.md`](../docs/decisions/PROJECT_LOG.md).
Most read real data from `eval/datasets/` (gitignored), so they won't run as-is without those inputs.

| Script | Purpose |
|--------|---------|
| `eval_recognizers.py` | Recognition A/B harness — per-page recognition CER for the local engines (swap via `OCR_ENGINE`) or an external model (`--predictions preds.json`). |
| `compare_recognizers.py` | Combine each engine's `recognition.csv` into one side-by-side comparison table. |
| `mlx_recognizer.py` | Run Qwen2.5-VL-7B (4-bit, MLX) locally on the eval pages → `predictions.json`. Runs in an isolated env (`uv run --no-project --with mlx-vlm`) to avoid the `transformers` version clash with Surya. |
| `colab_recognizer.ipynb` | Colab-notebook alternative for running a VLM on uploaded pages → `predictions.json` (scored locally with `eval_recognizers.py --predictions`). |
| `eval_document.py` | Document-level evaluation: run a whole multi-page doc → stitch tables → stitch sanity checks + scored metrics vs the verified document ground truth. |
| `eval_notable_page.py` | Validate engine behaviour on a genuine no-table (text-only) page (checks for phantom tables). |
| `draft_document_gt.py` | Restructure per-page ground truth into a single document-level GT JSON (a draft for manual verification). |
| `probe_rowstrip_recognition.py` | Early probe for the row-strip recognition idea (read each table row as a full-width strip). |
| `discover_slanet_api.py` | One-off to inspect the `rapid_table` / SLANet API surface while integrating the structure model. |
| `probe_layout_detectors.py` | Gate-first probe: does an alternative layout detector (DocLayout-YOLO / PP-DocLayout via `rapid_layout`, ONNX) see the dense GDDE table as one region where Surya fragments it into ~8 (PROJECT_LOG §2.12)? Detection only, no recognition. |
| `visualize_layout.py` | Per-page overlay of Surya (red) vs DocLayout-YOLO (green) table boxes across a whole document — makes the fragmentation and DocLayout's left-column clipping visible for manual verification (PROJECT_LOG §2.24). |
| `collect_documents.py` | Week-1 dataset collection helper: batch-download PDFs from a URL list into `corpus/`, then classify them for the dataset factory. |
| `compare_engines_ab.py` | Engine A/B on real documents split by cell class (Numeric vs Khmer vs overall) — answers what actually degrades when the frontend runs `auto` instead of `surya`. |
| `discover_slanet_api.py` | One-off to inspect the `rapid_table` / SLANet API surface while integrating the structure model. |
| `generate_ardb_eval_gt.py` | CLI driver for `khmer_pipeline.datagen.harvest_eval_gt` — template-maps evaluation GT across the ARDB daily series, staged to `eval/datasets/real_draft/`. **Reserves a set of documents already used as training data — see its docstring before running.** |
| `harvest_textlayer_gt.py` | Harvest evaluation GT for table numbers/structure directly from a born-digital PDF's text layer — free, unlimited, non-circular (no engine in the loop). |
| `package_layout_dataset.py` | Corrected Roboflow YOLOv8 export → COCO dataset + self-contained HF parquet upload folder (the `ardb-layout-coco-v1/v2` builder). |
| `package_sft_dataset.py` | Image+instruction+text SFT dataset (pairs.jsonl) → self-contained HF parquet upload folder, mirroring `package_layout_dataset.py`'s pattern. Schema-flexible (used for `ardb-gemma-sft-v2`, the unified builder's output). |
| `package_multiconfig_dataset.py` | Combines multiple already-packaged single-config HF folders into one multi-config repo folder (`data/<config>/*.parquet` per config) — built for the now-deprecated `Soxavin/ardb-gemma-sft-v1` (2-config) repo; superseded by v2's single flat schema, kept as valid infra for future multi-config needs. |
| `colab_layout_finetune.ipynb` | Fine-tunes DocLayout-YOLO on the human-corrected layout dataset (Track A). Free T4, ~10–20 min. |
| `colab_dots_ocr.ipynb` | Runs `rednote-hilab/dots.ocr` (full-page layout + tables-as-HTML) as a challenger engine test on CUDA. |
| `colab_gemma4_e2b_finetune.ipynb` | Fine-tunes `unsloth/gemma-4-E2B-it` (LoRA/QLoRA) on `Soxavin/ardb-gemma-sft-v2` — one unified task, full page image → JSON list of `{box_2d, label, text}` per region, in a single forward pass. `SMOKE_TEST` toggle runs 10 examples first before the full run. |
| `score_dots_predictions.py` | Scores `colab_dots_ocr.ipynb`'s predictions against local GT, reusing the production HTML-table parser so results are comparable to Surya's. |
| `spike_gemini.py` | Gemini as a zero-local-compute OCR challenger — same output contract as dots.ocr, scored by the same parser. Needs `GEMINI_API_KEY`. |
| `probe_cambodiabudget_fragmentation.py` | Ground-truth-free probe confirming preprocessing (crop-margins + resolution cap) collapses Surya's ~8-region table fragmentation to ~1 on the market-price bulletin template (PROJECT_LOG §2.26). |
| `probe_stamp_mask.py` | Quantifies what the red/blue stamp-removal colour mask actually erases (including coloured body text), deterministically rather than via a noisy OCR ablation. |
| `recall_taxonomy.py` | Classifies recall failures for a single document run by row-aligning predicted vs verified GT with the same difflib-based alignment `evaluate_structure` uses. |
| `review_scan_gt.py` | Cell-by-cell review sheet for OCR-drafted GT on a scanned (no text-layer) page — shows every cell's crop beside the drafted text so a human verifies against pixels, not just low-confidence cells. |
| `verify_corrections.py` | Verifies HITL correction crops (`--inspect`, read-only contact sheet) and demos the correction-capture loop. |
| `verify_eval_gt.py` | Visual verification for template-mapped evaluation GT — crops each drafted cell beside its value so a human confirms against pixels; the failure mode to hunt is a column slip. |

Run from the repo root, e.g.:

```bash
OCR_ENGINE=surya uv run python tools/eval_recognizers.py
uv run python tools/compare_recognizers.py
```
