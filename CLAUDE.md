# Khmer OCR Pipeline — Project Rules

See `CONTEXT.md` for architecture/data-flow orientation.

## Workflow
- After any change to `src/khmer_pipeline/**/*.py` or `app.py`, run:
  ```bash
  uv run pytest -q --tb=short && python3 -m py_compile app.py $(find src/khmer_pipeline -name '*.py')
  ```
- This project uses `uv` for dependency management — use `uv add <pkg>`,
  not pip/poetry, and pin upper bounds for ML libraries (e.g.
  `>=4.56.1,<5.0`). Open-ended pins have silently broken on past major
  releases of `transformers`/`surya-ocr`.
- For Streamlit UI changes, clear `__pycache__` and restart/reload the
  dev server after changing pipeline function signatures or installed
  packages — stale bytecode/imports otherwise mask the fix.
- This is a 24GB unified-memory M4 Pro Mac running PyTorch (Surya) and
  MLX (Qwen) in the same process. After any new heavy model
  invocation, call `clear_device_cache()` from
  `src/khmer_pipeline/utils/memory.py` (already wired in after each pipeline
  stage and after any page where Qwen runs) — use `mx.clear_cache()`,
  not the deprecated `mx.metal.clear_cache()`.
- Streamlit width: use `width="stretch"` on `st.image`/`st.button`/
  `st.download_button`. Never use the deprecated
  `use_container_width=True`.

## Code conventions
- `src/khmer_pipeline/**/*.py` modules require concise docstrings (1–3
  lines: what it does, key args, return) on **public** functions.
  Private `_`-prefixed helpers don't need them. Keep using short `#`
  comments for the non-obvious *why* (e.g. a non-obvious axis-swap, a
  workaround for a library bug). Match this convention in new code in
  these files.
- Tunable numeric thresholds get extracted as module-level
  `_UPPER_SNAKE_CASE` constants near their point of use (e.g.
  `_DESKEW_MIN_ANGLE_DEG`, `_TABLE_BG_MIN_VALUE`,
  `_TABLE_BG_MIN_SATURATION`) — don't leave magic numbers inline.
- New `PreprocessConfig` flags / pipeline options follow the established
  4-point pattern: dataclass field with a default, sidebar checkbox in
  `app.py`, `--no-<flag>` CLI argument in `pipeline.py`, and an entry
  appended to `app.py`'s `settings_key` f-string.
- Pipeline issues (low OCR confidence, phantom table cells, OCR/table
  failures, etc.) are surfaced via `warnings.warn(...)`, collected into
  `SuryaResult.warnings`, and already displayed in both `app.py`
  (`st.warning`) and `pipeline.py` (`WARNING:` prefix). Don't add new
  ad-hoc UI plumbing for warnings — extend this mechanism.
- Stage 3 (OCR) and Stage 4 (correction) execution functions are swapped via
  `engines/engine_registry.py` (see CONTEXT.md "Engine Swappability"). Orchestrators
  (`pipeline.py`, `app.py`) must import `ACTIVE_OCR_ENGINE` /
  `ACTIVE_CORRECTION_ENGINE` from there, never `run_surya`/`postprocess`
  directly. State-checking helpers (`models_loaded`, `preload_models`,
  `qwen_loaded`) are still imported directly from `engines/surya.py`/`postprocess.py`.
- Multi-page results in `app.py` use the established pagination
  pattern: `st.session_state.current_page_idx` (clamped to
  `[0, total_pages - 1]`, reset on new file upload), a "Jump to page"
  `st.selectbox`, and Previous/Next `st.button`s, each calling
  `st.rerun()` on change. Render one page at a time — don't reintroduce
  a per-page `for` loop over all results.

## Testing
- TDD: extend/add tests in `tests/test_<module>.py` mirroring
  `src/khmer_pipeline/<module>.py` *before* implementing.
- There is no `test_app.py` — UI-only changes in `app.py` are verified
  manually by running the Streamlit app, not via unit tests.

## Things not to do
- Don't add error handling, fallbacks, or feature flags for scenarios
  that can't occur — trust the dataclass contracts in `models.py`.
- Don't refactor or restructure unrelated code as part of a feature
  change (e.g. don't reorganize `app.py` while adding a sidebar option).
