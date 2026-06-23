# Operations Guide — Khmer OCR Pipeline (single-user desktop)

Practical notes for running the pipeline reliably on an Apple-Silicon Mac.

## OCR backend (llama-server)

Surya 0.20 runs its OCR/VLM through a C++ **`llama-server`** child process on the
Metal GPU. It is configured by environment variables set in `setup-metal-macos.sh`
(`SURYA_INFERENCE_BACKEND=llamacpp`, `SURYA_LLAMACPP_BINARY`, `LLAMA_CPP_NGL=99`,
`SURYA_INFERENCE_PARALLEL=1`, `SURYA_INFERENCE_KEEP_ALIVE=true`).

`KEEP_ALIVE=true` keeps the model resident in VRAM between calls (fast reruns), so
the `llama-server` process **stays alive** for the session.

### Start
```bash
source setup-metal-macos.sh      # sets env + reports the llama-server binary
uv run streamlit run app.py      # or: uv run python -m khmer_pipeline.pipeline ...
```

### Status
The Streamlit sidebar shows **🟢 OCR backend running** / **⚪ OCR backend not detected**.
From a shell: `pgrep -f llama-server`.

### Stop / crash recovery
Because the server is resident, a crash or an unclean exit can **orphan** it,
holding unified memory and a port. Stop it explicitly:
```bash
bash stop-metal-macos.sh         # SIGTERM, then SIGKILL fallback; reports PIDs
```
There is intentionally **no automatic kill-on-exit** in the CLI/benchmark — a blanket
kill would also terminate a server a concurrently-running Streamlit app is using.
Use the stop script when you're done, or if a previous run was orphaned.

## Memory ceiling (24 GB unified)

PyTorch (Surya, via llama-server) and MLX (Qwen, opt-in) share the 24 GB unified
memory. `clear_device_cache()` is called after each stage to release caches.

- The Streamlit app shows a soft warning when a job exceeds `_MEMORY_WARN_PAGES`
  (in `app.py`), as **effective pages = page_count × (DPI / 200)**. Current value: **12**.
- **Measured result (2026-06-23):** a **10-page born-digital PDF at 300 DPI**
  (`CambodiaBudgetExecutioninApr-2024.pdf`, effective ≈ 15) **completed without
  crashing**, but drove the machine into **heavy swapping** (~29 GB swap in use) and
  took **~72 min wall time — roughly 5× a healthy run**, i.e. it was thrashing. So
  ~10 pages at 300 DPI is at/over the comfortable ceiling on this 24 GB machine; at
  the default **200 DPI** the per-page raster is ~2.25× smaller and far lighter.
  `_MEMORY_WARN_PAGES = 12` warns with margin before the observed effective-15 thrash
  point (→ warns at ≥8 pages @ 300 DPI, ≥13 @ 200 DPI).
- **Caveats on the measurement (be honest):** memory is **per-page bounded**, not
  cumulative — Surya processes pages sequentially (`SURYA_INFERENCE_PARALLEL=1`) and
  `clear_device_cache()` runs after each page — so the real driver is **DPI/page
  resolution**, not page *count* alone. The exact peak RSS was not captured cleanly
  (sampler bug + the resident `llama-server` and dev environment also consume memory;
  macOS swap is cumulative). The conclusion rests on the unambiguous thrash signals
  (swap + 5× runtime), not a precise peak number. **Mitigation:** for large/high-DPI
  jobs, process in page ranges (sidebar → "Page range").

### Re-running the stress test (for a cleaner peak)
Because memory is per-page bounded, a **short** run (2–3 pages) captures the same peak
far faster. Drop a PDF into `sample_data/` (gitignored), then sample peak RSS of the
python + `llama-server` PIDs (`ps -o rss=`) plus `sysctl -n vm.swapusage` while:
```bash
source setup-metal-macos.sh
uv run python -m khmer_pipeline.pipeline sample_data/<doc>.pdf /tmp/out --dpi 300
```
Update `_MEMORY_WARN_PAGES` and the measured result above if the numbers shift.

## Reproducible synthetic data (offline fonts)

The synthetic generators render HTML with **vendored OFL Khmer fonts** embedded as
base64 `@font-face` (see `fonts/` and `fonts/MANIFEST.txt`) — no live
`fonts.googleapis.com` dependency, so datasets regenerate deterministically offline:
```bash
uv run python -m khmer_pipeline.generate_synthetic_tables --output-dir eval/datasets/synthetic_tables
uv run python -m khmer_pipeline.generate_synthetic_documents --output-dir eval/datasets/synthetic_documents
```
The generators abort (rather than emit a fallback-font image) if a font fails to load.

## Future work — why not Docker?

This app is **deliberately not containerized**. Its performance depends on local
Apple-Silicon acceleration: Surya runs on the **Metal** GPU and Qwen runs on **MLX**.
macOS containers run Linux in a VM with **no access to Metal**, and **MLX does not run
on Linux** — a container today would silently fall back to CPU (far slower, likely OOM
on the larger models). Containerization should be reconsidered only for a different
future architecture: a **Linux/CUDA multi-user server** deployment, which would use
different inference backends (no MLX) and is out of scope for this single-user thesis
project.
