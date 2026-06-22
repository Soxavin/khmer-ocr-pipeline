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
  (in `app.py`), scaled by DPI relative to the 200-DPI baseline.
- **Measured practical limit:** _TODO — set after the stress test on a large scanned
  PDF (see below)._ Until then `_MEMORY_WARN_PAGES` is a conservative provisional
  value. If a large job stalls or the machine starts swapping, split it into page
  ranges (sidebar → "Page range") and process in batches.

### Running the stress test
Drop a large multi-page **scanned** PDF into `sample_data/` (gitignored), then:
```bash
source setup-metal-macos.sh
# sample llama-server + python RSS while a CLI run processes increasing page counts
uv run python -m khmer_pipeline.pipeline sample_data/<big_scan>.pdf /tmp/out --dpi 300
```
Watch peak memory (`ps -o rss= -p <pid>` for the python + llama-server PIDs, plus
`memory_pressure`). Record the largest page count at the target DPI that completes
without heavy swapping, then set `_MEMORY_WARN_PAGES` and replace the TODO above.

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
