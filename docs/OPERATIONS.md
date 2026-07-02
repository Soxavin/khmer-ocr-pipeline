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

- The Streamlit app shows a soft "large job" warning when a job exceeds
  `_MEMORY_WARN_PAGES` (in `app.py`), as **effective pages = page_count × (DPI / 200)**.
  Current value: **25**. This is a "this will take a while" heads-up, **not** a measured
  memory ceiling.
- **Measured result (2026-06-23):** a **10-page born-digital PDF at 300 DPI**
  (`CambodiaBudgetExecutioninApr-2024.pdf`, effective ≈ 15) completed in **~7 min** with
  **peak combined RSS ≈ 2.1 GB** (python + `llama-server`) and only **+384 MB swap** added
  during the run — i.e. **no memory distress**; the machine was nowhere near the 24 GB
  ceiling. Memory is **per-page bounded** (Surya processes pages sequentially,
  `SURYA_INFERENCE_PARALLEL=1`, with `clear_device_cache()` after each page), so the real
  cost driver is **DPI/page resolution**, not page *count*.
  - *Correction:* an earlier run of this same job reported ~72 min / heavy swap — that was
    an artifact of the **laptop sleeping mid-run** (closed lid over lunch) plus a cumulative
    swap baseline, not thrashing. The clean re-run above (with `caffeinate`, lid open) is the
    valid measurement.
  - *Caveat:* process RSS undercounts the VLM weights resident in Metal/GPU-wired memory
    (loaded once, kept alive), but the negligible swap delta confirms no pressure. We did not
    push to the actual page ceiling — it's simply well above any realistic GDDE document.
- **Mitigation for very large/high-DPI jobs:** process in page ranges (sidebar →
  "Page range"); pages run sequentially so memory stays bounded regardless of count.

### Re-running the stress test
Memory is per-page bounded, so a **short** run (2–3 pages) captures the same peak quickly.
Use `caffeinate` and keep the lid open. Drop a PDF into `sample_data/` (gitignored) and
sample peak RSS of the python + `llama-server` PIDs (`ps -o rss=`) plus
`sysctl -n vm.swapusage` (before/after delta) while:
```bash
source setup-metal-macos.sh
caffeinate -dis uv run python -m khmer_pipeline.pipeline sample_data/<doc>.pdf /tmp/out --dpi 300
```
Update `_MEMORY_WARN_PAGES` and the measured result above only if the numbers shift materially.

## Reproducible synthetic data (offline fonts)

The synthetic generators render HTML with **vendored OFL Khmer fonts** embedded as
base64 `@font-face` (see `fonts/` and `fonts/MANIFEST.txt`) — no live
`fonts.googleapis.com` dependency, so datasets regenerate deterministically offline:
```bash
uv run python -m khmer_pipeline.datagen.generate_synthetic_tables --output-dir eval/datasets/synthetic_tables
uv run python -m khmer_pipeline.datagen.generate_synthetic_documents --output-dir eval/datasets/synthetic_documents
```
The generators abort (rather than emit a fallback-font image) if a font fails to load.

## Two deployment lanes (Mac native vs Docker)

The project runs in **two lanes**, and the compute device is auto-selected by
`src/khmer_pipeline/utils/device.py` (`configure_runtime()` → `TORCH_DEVICE`):

- **Mac (Apple Silicon) — run natively, no Docker.** Surya runs on the **Metal** GPU and
  the optional Qwen step on **MLX**. macOS containers run Linux in a VM with **no access
  to Metal**, and **MLX does not run on Linux**, so a container on a Mac would only fall
  back to CPU. Mac users therefore run directly (`source setup-metal-macos.sh`).
- **Linux / NVIDIA / cloud — use the `Dockerfile`.** The image installs the torch (CUDA)
  backend and system deps (`mlx` is auto-excluded by its platform marker). `docker run
  --gpus all` uses CUDA; without `--gpus` it falls back to CPU. This is the multi-user /
  server lane.

Both lanes share the same code; only the device backend differs. See the README
"Running with Docker" section for the exact build/run commands.
