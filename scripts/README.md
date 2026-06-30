# scripts/

One-off research and evaluation scripts used during development. They are **not** part of the core
pipeline (`src/khmer_pipeline/`) or the UI (`app.py`) — they support specific experiments and the
write-up in [`../docs/REPORT.md`](../docs/REPORT.md) / [`../docs/PROJECT_LOG.md`](../docs/PROJECT_LOG.md).
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

Run from the repo root, e.g.:

```bash
OCR_ENGINE=surya uv run python scripts/eval_recognizers.py
uv run python scripts/compare_recognizers.py
```
