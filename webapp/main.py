"""NiceGUI review UI — entrypoint.  Run: `uv run python -m webapp.main`.

Presentation layer over the Khmer OCR pipeline. Imports and calls the same functions as
the Streamlit `app.py`; all heavy work runs off the event loop via `webapp.runner`.
Supports a batch of uploaded documents, reviewed one at a time.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
from PIL import Image
from nicegui import ui

from khmer_pipeline.model_config import ANOMALY_THRESHOLD, CELL_CONF_LOW, CELL_CONF_MID
from khmer_pipeline.utils.backend_status import llama_server_running
from khmer_pipeline.export import grid_to_csv, tables_to_xlsx

from .runner import run_pipeline
from .state import AppState
from . import tables, components, downloads, edits
from . import api  # registers /api routes on nicegui.app (must precede ui.run)

_ENGINE_OPTIONS = {
    "surya": "Surya (fast — best all-round)",
    "surya_kiri": "Surya + Kiri (Khmer-text-heavy tables, slower)",
    "surya_kiri_vlm": "Surya + Kiri VLM (Surya structure + Kiri Khmer — slowest)",
}
_DPI_OPTIONS = [150, 200, 300]
_SCOPE_OPTIONS = {"all": "All pages", "single": "Single page", "range": "Page range"}
_MEMORY_WARN_PAGES = 15


@ui.page("/")
def index() -> None:
    state = AppState()
    s = state.settings

    ui.colors(primary="#1565C0")
    dark = ui.dark_mode()
    with ui.header(elevated=True).classes("items-center justify-between px-4 py-2"):
        ui.label("Khmer Document Extraction").classes("text-lg font-semibold")
        with ui.row().classes("items-center gap-3"):
            ui.label("GDDE · MEF").classes("text-xs opacity-70")
            ui.switch("Dark", on_change=lambda e: dark.enable() if e.value else dark.disable()).props("dense")

    ui.label("Upload one or more financial documents → review the extracted tables → download.").classes(
        "text-sm text-gray-500"
    )

    # ======================= Sidebar (settings) ==============================
    with ui.left_drawer(value=True).classes("bg-gray-50 dark:bg-neutral-900 w-80"):
        ui.label("Settings").classes("text-lg font-semibold")

        def _probe_pages(name: str, data: bytes) -> int:
            if Path(name).suffix.lower() != ".pdf":
                return 1
            try:
                with fitz.open(stream=data, filetype="pdf") as doc:
                    return len(doc)
            except Exception:
                return 0

        def _on_upload(e) -> None:
            data = e.content.read()
            state.add_document(e.name, data, hashlib.md5(data).hexdigest()[:8], _probe_pages(e.name, data))
            doc_selector.refresh()
            file_info.refresh()
            stale_banner.refresh()
            results.refresh()

        ui.upload(on_upload=_on_upload, auto_upload=True, multiple=True, max_files=20).props(
            'accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif"'
        ).classes("w-full")
        ui.button("Clear all documents", on_click=lambda: (
            state.clear_documents(), doc_selector.refresh(), file_info.refresh(),
            stale_banner.refresh(), results.refresh(),
        )).props("flat dense")

        ui.checkbox("Stitch multi-page tables").bind_value(s, "stitch_pages")
        ui.checkbox("Convert Khmer numerals").bind_value(s, "convert_numerals")

        with ui.expansion("⚙️ Advanced engine settings").classes("w-full"):
            ui.select(_DPI_OPTIONS, value=s.dpi, label="Scan quality (DPI)").bind_value(s, "dpi").classes("w-full")
            scope = ui.select(_SCOPE_OPTIONS, value=s.page_scope, label="Pages").bind_value(s, "page_scope").classes("w-full")
            with ui.row().bind_visibility_from(scope, "value", value="single"):
                ui.number("Page", value=1, min=1, format="%d").bind_value(s, "page_num")
            with ui.row().bind_visibility_from(scope, "value", value="range"):
                ui.number("From", value=1, min=1, format="%d").bind_value(s, "page_start")
                ui.number("To", value=5, min=1, format="%d").bind_value(s, "page_end")

            ui.label("Preprocessing").classes("font-medium mt-2")
            ui.checkbox("Remove colored stamps").bind_value(s, "remove_stamps")
            ui.checkbox("Sharpen text").bind_value(s, "sharpen")
            ui.checkbox("Enhance contrast").bind_value(s, "normalise")
            ui.checkbox("Deskew rotated scans").bind_value(s, "deskew")
            ui.checkbox("Normalise colored table backgrounds").bind_value(s, "normalise_table_backgrounds")

            ui.label("Extraction").classes("font-medium mt-2")
            ui.select(_ENGINE_OPTIONS, value=s.ocr_engine_key, label="OCR engine").bind_value(s, "ocr_engine_key").classes("w-full")
            ui.select({False: "Full extraction (text + tables)", True: "Tables only"},
                      value=s.tables_only, label="Extraction mode").bind_value(s, "tables_only").classes("w-full")

            ui.label("Post-processing").classes("font-medium mt-2")
            ui.checkbox("Qwen correction (experimental, slow)").bind_value(s, "enable_qwen")
            ui.number("Anomaly threshold", value=ANOMALY_THRESHOLD, min=0.0, max=1.0, step=0.01,
                      format="%.2f").bind_value(s, "anomaly_threshold").classes("w-full")

            ui.label("Export & overlay").classes("font-medium mt-2")
            ui.checkbox("Auto-repair inconsistent table grids").bind_value(s, "repair_tables")
            ui.checkbox("Show layout overlay").bind_value(s, "show_layout")
            ui.radio(["Region type", "Confidence"], value=s.overlay_mode).bind_value(s, "overlay_mode").props("inline")

        ui.separator()
        ui.label("🟢 OCR backend ready" if llama_server_running()
                 else "⚪ OCR backend idle — starts on first extraction").classes("text-xs text-gray-500")

    # ============================ Main column ================================
    def _select_doc(i: int) -> None:
        state.active = i
        file_info.refresh()
        stale_banner.refresh()
        results.refresh()

    @ui.refreshable
    def doc_selector() -> None:
        if len(state.documents) <= 1:
            return
        opts = {i: f"{'✓' if d.has_results else '•'} {i + 1}. {d.upload_name}"
                for i, d in enumerate(state.documents)}
        ui.select(opts, value=state.active, label="Document",
                  on_change=lambda e: _select_doc(e.value)).props("dense outlined").classes("w-full max-w-lg")

    doc_selector()

    @ui.refreshable
    def file_info() -> None:
        d = state.doc()
        if d is None:
            ui.label("Upload a document to get started.").classes("text-gray-500")
            return
        kb = round(len(d.upload_bytes) / 1024, 1)
        pages = d.doc_page_count if d.doc_page_count else "?"
        ui.label(f"File: {d.upload_name}  ·  {kb} KB  ·  {pages} page(s)")
        est = len(s.page_indices(d.doc_page_count) or range(d.doc_page_count or 1))
        if est * (s.dpi / 200.0) > _MEMORY_WARN_PAGES:
            ui.label("⚠️ Large job — processing may take several minutes; consider a smaller page range.").classes("text-amber-600 text-sm")

    file_info()

    @ui.refreshable
    def stale_banner() -> None:
        d = state.doc()
        if d is not None and d.results_are_stale(s):
            ui.label("⚠️ Settings changed since the last run — results below are out of date. Re-run to refresh.").classes(
                "text-amber-700 bg-amber-50 rounded px-2 py-1 text-sm"
            )

    stale_banner()

    with ui.row().classes("items-center gap-2"):
        progress_label = ui.label().classes("text-sm text-primary")
        stop_btn = ui.button("■ Stop", on_click=lambda: _stop_run()).props("flat dense color=negative")
        stop_btn.visible = False
    progress_bar = ui.linear_progress(value=0.0, show_value=False)
    progress_bar.visible = False

    def _stop_run() -> None:
        d = state.doc()
        if d is not None and d.progress.active:
            d.progress.cancel_requested = True
            ui.notify("Stopping — the run aborts after the current page.", type="warning")

    def _tick() -> None:
        d = state.doc()
        p = d.progress if d else None
        active = bool(p and p.active)
        progress_bar.visible = active
        stop_btn.visible = active
        if active:
            progress_bar.value = p.fraction
            progress_label.set_text(p.stage + (f"  (page {p.page}/{p.total})" if p.total else ""))
        else:
            progress_label.set_text("")

    ui.timer(0.1, _tick)

    # A closed/refreshed tab must not leave an orphaned all-pages run grinding in
    # a worker thread with no UI attached — cancel every in-flight run this
    # client owns. (Runs abort at the next page/stage boundary.)
    def _cancel_all_runs() -> None:
        for d in state.documents:
            if d.progress.active:
                d.progress.cancel_requested = True

    ui.context.client.on_disconnect(_cancel_all_runs)

    async def _run() -> None:
        d = state.doc()
        if d is None:
            ui.notify("Upload a document first.", type="warning")
            return
        if s.invalid_range:
            ui.notify("'To' page is before 'From' page — fix the range.", type="negative")
            return
        run_btn.disable()
        run_all_btn.disable()
        ok = await run_pipeline(d, s)
        run_btn.enable()
        run_all_btn.enable()
        if not ok:
            ui.notify(d.run_error or "Extraction failed.", type="negative")
        doc_selector.refresh()
        stale_banner.refresh()
        results.refresh()

    async def _run_all() -> None:
        if not state.documents:
            ui.notify("Upload a document first.", type="warning")
            return
        if s.invalid_range:
            ui.notify("'To' page is before 'From' page — fix the range.", type="negative")
            return
        run_btn.disable()
        run_all_btn.disable()
        for i, d in enumerate(state.documents):
            if d.has_results and not d.results_are_stale(s):
                continue
            state.active = i  # surface this doc's progress + results while it runs
            doc_selector.refresh()
            file_info.refresh()
            results.refresh()
            ok = await run_pipeline(d, s)
            if not ok:
                ui.notify(f"{d.upload_name}: {d.run_error}", type="negative")
                if "cancelled" in (d.run_error or "").lower():
                    break  # Stop means stop — don't start the next document
        run_btn.enable()
        run_all_btn.enable()
        doc_selector.refresh()
        stale_banner.refresh()
        results.refresh()

    def _retry() -> None:
        d = state.doc()
        if d is not None:
            d.reset_run()
        doc_selector.refresh()
        stale_banner.refresh()
        results.refresh()

    with ui.row().classes("items-center gap-2"):
        run_btn = ui.button("▶ Run Extraction", on_click=_run).props("color=primary")
        run_all_btn = ui.button("▶▶ Run all", on_click=_run_all).props("outline")
        ui.button("↺ Retry", on_click=_retry).props("flat")

    def _goto(d, idx: int) -> None:
        d.current_page_idx = idx
        d.selected = None
        results.refresh()

    async def _capture_edit(grid_el, d, table_id: str, ncols: int, original: list[list[str]]) -> None:
        rows = await grid_el.get_client_data()
        new_grid = [[str(row.get(f"c{c}", "") or "") for c in range(ncols)] for row in rows]
        if new_grid != original:
            d.edited_tables[table_id] = new_grid
        else:
            d.edited_tables.pop(table_id, None)

    def _reset_table(d, table_id: str) -> None:
        d.edited_tables.pop(table_id, None)
        results.refresh()

    # ---------------------------- results -----------------------------------
    @ui.refreshable
    def results() -> None:
        d = state.doc()
        if d is None:
            return
        if d.run_error and not d.has_results:
            ui.label(d.run_error).classes("text-red-600")
            return
        if not d.has_results:
            return

        doc_json = d.export_result.document_json
        proc_imgs = d.preprocess_result.page_images
        orig_imgs = d.ingest_result.page_images
        surya_pages = d.surya_result.pages
        post_pages = d.postprocess_result.pages
        total = len(proc_imgs)
        idx = max(0, min(d.current_page_idx, total - 1))
        d.current_page_idx = idx
        stitched = tables.is_stitched(doc_json)

        # overview
        total_tables = sum(len(p.tables) for p in surya_pages)
        warnings = list(d.surya_result.warnings) + list(getattr(d.postprocess_result, "warnings", []))
        with ui.row().classes("gap-8 mt-2"):
            for label, val in (("Pages", total), ("Tables detected", total_tables), ("Warnings", len(warnings))):
                with ui.column().classes("items-center"):
                    ui.label(str(val)).classes("text-2xl font-bold")
                    ui.label(label).classes("text-xs text-gray-500")
        if total_tables == 0:
            ui.label("No tables detected. Try higher DPI or confirm the document has tables.").classes("text-red-600")
        elif warnings:
            with ui.expansion(f"⚠️ Pipeline warnings ({len(warnings)})").classes("w-full"):
                for w in warnings:
                    ui.label(f"• {w}").classes("text-sm")

        # pagination
        with ui.row().classes("items-center gap-2 mt-2"):
            ui.button("⬅ Previous", on_click=lambda: _goto(d, idx - 1)).props("flat").set_enabled(idx > 0)
            ui.select({i: f"Page {i + 1}" for i in range(total)}, value=idx,
                      on_change=lambda e: _goto(d, e.value)).props("dense outlined")
            ui.button("Next ➡", on_click=lambda: _goto(d, idx + 1)).props("flat").set_enabled(idx < total - 1)

        surya_page = surya_pages[idx]
        post_page = post_pages[idx]

        # per-page quality banner
        low_conf = sum(1 for b in surya_page.text_blocks if (b.get("confidence") or 0.0) < CELL_CONF_LOW)
        with ui.row().classes("gap-6 text-sm text-gray-600"):
            ui.label(f"Text blocks: {len(surya_page.text_blocks)}")
            ui.label(f"Tables: {len(surya_page.tables)}")
            ui.label(f"Low-confidence: {low_conf}")
            ui.label(f"Qwen: {'Yes' if post_page.qwen_used else 'No'}")

        # find & replace across ALL document tables (fix a systematic OCR error at once)
        def _replace_all() -> None:
            find = (find_in.value or "").strip()
            if not find:
                ui.notify("Enter text to find.", type="warning")
                return
            changed, total_hits = edits.replace_across(downloads.final_tables(d), find, repl_in.value or "")
            d.edited_tables.update(changed)
            ui.notify(
                f"Replaced {total_hits} occurrence(s) across {len(changed)} table(s)." if total_hits
                else "No matches found.",
                type="positive" if total_hits else "info",
            )
            if total_hits:
                results.refresh()

        with ui.expansion("🔎 Find & replace across all tables").classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                find_in = ui.input("Find").props("dense outlined")
                repl_in = ui.input("Replace with").props("dense outlined")
                ui.button("Replace all", on_click=_replace_all).props("dense")
            ui.label("Applies to every extracted table (all pages), folding into your edits.").classes("text-xs text-gray-500")

        # Table↔image linking (unstitched only). The pipeline exposes only table-region
        # geometry — not per-cell — so linking highlights the whole table region: click a
        # cell → its table lights up on the page; click a table on the page → jump to its
        # grid. `d.selected` is (table_id, row, col); row/col drive the readout only.
        page_tables = tables.page_export_tables(doc_json, idx)
        page_blocks = tables.page_table_blocks(doc_json, idx)
        table_bboxes = {} if stitched else components.table_bbox_index(surya_page.tables, page_blocks)
        grids: dict[str, object] = {}

        def _base_overlay() -> str:
            if not s.show_layout:
                return ""
            return components.overlay_svg(
                surya_page.text_blocks,
                [{"bbox": t.get("bbox"), "label": "Table"} for t in surya_page.tables],
                s.overlay_mode,
            )

        def _render_overlay() -> None:
            svg = _base_overlay()
            tid = d.selected[0] if d.selected else None
            if tid in table_bboxes:
                svg += components.highlight_rect(table_bboxes[tid])
            image_el.set_content(svg)

        def _set_readout() -> None:
            if d.selected and d.selected[0] in table_bboxes:
                tid, r, c = d.selected
                if r is None:
                    readout.set_text(f"🔗 {tid} — jumped to this table")
                else:
                    grid = d.edited_tables.get(tid) or {t[0]: t[1] for t in page_tables}.get(tid, [])
                    val = grid[r][c] if r < len(grid) and c < len(grid[r]) else ""
                    readout.set_text(f"🔗 {tid} · row {r + 1}, col {c + 1} = “{val}” — table highlighted on page")
            elif table_bboxes:
                readout.set_text("Tip: click a table cell to highlight its table on the page, or click a table on the page to jump to its grid.")
            else:
                readout.set_text("")

        async def _on_image_click(e) -> None:
            x, y = e.image_x, e.image_y
            for tid, bbox in table_bboxes.items():
                x0, y0, x1, y1 = bbox[:4]
                if x0 <= x <= x1 and y0 <= y <= y1:
                    d.selected = (tid, None, None)
                    _render_overlay()
                    _set_readout()
                    ag = grids.get(tid)
                    if ag is not None:
                        ag.run_grid_method("ensureIndexVisible", 0, "top")
                    break

        def _on_cell_click(e, tid: str) -> None:
            args = e.args or {}
            col_id = args.get("colId") or ""
            try:
                c = int(col_id[1:])
            except (ValueError, IndexError):
                c = 0
            d.selected = (tid, args.get("rowIndex", 0), c)
            _render_overlay()
            _set_readout()

        # side-by-side
        with ui.row().classes("w-full gap-4 no-wrap items-start"):
            with ui.column().classes("w-2/5"):
                image_el = ui.interactive_image(
                    Image.fromarray(proc_imgs[idx]), content=_base_overlay(),
                    on_mouse=_on_image_click, events=["mousedown"], cross=True,
                ).classes("w-full border rounded")
                ui.label("🟢 high  🟡 medium  🔴 low" if s.overlay_mode == "Confidence" else "Layout regions").classes("text-xs text-gray-500")
                readout = ui.label().classes("text-xs text-primary")
                with ui.expansion("Original image").classes("w-full"):
                    ui.image(Image.fromarray(orig_imgs[idx])).classes("w-full")

            with ui.column().classes("w-3/5"):
                if not page_tables:
                    ui.label("No tables to review." if stitched else "No tables detected on this page.").classes("text-gray-500")
                if stitched:
                    ui.label("✏️ Review & edit tables — all pages (stitched)").classes("font-semibold")
                elif page_tables:
                    ui.label("✏️ Review & edit tables — this page").classes("font-semibold")

                for table_id, orig_grid, conf_grid in page_tables:
                    if not orig_grid:
                        continue
                    grid = d.edited_tables.get(table_id, orig_grid)
                    ncols = max((len(r) for r in grid), default=0)
                    ui.label(table_id).classes("font-mono text-xs mt-1")
                    if any(v is not None for row in conf_grid for v in row):
                        with ui.expansion("🔍 Confidence view (read-only)").classes("w-full"):
                            ui.html(components.conf_view_html(grid, conf_grid))
                            ui.label(f"🔴 below {CELL_CONF_LOW:.0%} · 🟡 {CELL_CONF_LOW:.0%}–{CELL_CONF_MID:.0%} · untinted ≥ {CELL_CONF_MID:.0%}").classes("text-xs text-gray-500")
                    ag = ui.aggrid({
                        "columnDefs": [{"headerName": f"Col {c + 1}", "field": f"c{c}", "editable": True} for c in range(ncols)],
                        "rowData": [{f"c{c}": (row[c] if c < len(row) else "") for c in range(ncols)} for row in grid],
                        "defaultColDef": {"resizable": True, "sortable": False, "wrapText": True, "autoHeight": True},
                    }).classes("w-full")
                    grids[table_id] = ag
                    ag.on("cellValueChanged",
                          lambda e, a=ag, tid=table_id, n=ncols, o=orig_grid: _capture_edit(a, d, tid, n, o))
                    ag.on("cellClicked", lambda e, tid=table_id: _on_cell_click(e, tid))
                    ui.button("↺ Reset table", on_click=lambda tid=table_id: _reset_table(d, tid)).props("flat dense")

        _set_readout()

        # text & processing details
        with ui.expansion("Text & processing details").classes("w-full mt-2"):
            if d.stage_times:
                ui.label("  ·  ".join(f"{k}: {v:.1f}s" for k, v in d.stage_times.items())).classes("text-xs text-gray-500")
            ui.label("OCR text").classes("font-medium mt-1")
            if s.tables_only:
                ui.label("Hidden in 'Tables only' mode.").classes("text-gray-500 text-sm")
            elif surya_page.ocr_text:
                ui.html(surya_page.ocr_text)
            else:
                ui.label("No OCR text on this page.").classes("text-gray-500 text-sm")
            ui.label("⚡ Qwen correction applied" if post_page.qwen_used else "✓ rule-based only").classes("font-medium mt-1")
            ta = ui.textarea("Corrected text (editable)", value=d.edited_text.get(idx, post_page.corrected_text)).classes("w-full").props("autogrow")

            def _capture_text(e, i=idx, base=post_page.corrected_text) -> None:
                if e.value != base:
                    d.edited_text[i] = e.value
                else:
                    d.edited_text.pop(i, None)
            ta.on_value_change(_capture_text)

        # ---------------------------- downloads ----------------------------
        stem = Path(d.upload_name).stem
        ui.separator().classes("mt-2")
        ui.label("Downloads").classes("text-lg font-semibold")
        ui.label("Reflect your edits — generated fresh on each click.").classes("text-xs text-gray-500 -mt-1")

        def _dl_json() -> None:
            ui.download(downloads.json_bytes(downloads.patched_document_json(d)), f"{stem}_extracted.json")

        def _dl_txt() -> None:
            dj = downloads.patched_document_json(d)
            ui.download(downloads.text_report(d, s, dj.get("pages", [])).encode("utf-8"), f"{stem}_extracted.txt")

        def _dl_xlsx() -> None:
            ui.download(tables_to_xlsx(downloads.final_tables(d), s.convert_numerals), f"{stem}_extracted.xlsx")

        def _dl_zip() -> None:
            dj = downloads.patched_document_json(d)
            txt = downloads.text_report(d, s, dj.get("pages", []))
            ui.download(downloads.zip_bundle(d, s, stem, dj, txt, downloads.final_tables(d)), f"{stem}_extracted.zip")

        ne_tables = downloads.nonempty_tables(downloads.final_tables(d))
        with ui.row().classes("gap-2 flex-wrap items-center"):
            ui.button("⬇ JSON", on_click=_dl_json).props("outline")
            ui.button("⬇ Text (.txt)", on_click=_dl_txt).props("outline")
            if ne_tables:
                ui.button("⬇ Excel (.xlsx)", on_click=_dl_xlsx).props("outline")
            ui.button("⬇ Everything (.zip)", on_click=_dl_zip).props("color=primary")

        if ne_tables:
            with ui.row().classes("gap-2 flex-wrap items-center"):
                ui.label("Per-table CSV:").classes("text-sm text-gray-500")
                for tid, _grid in ne_tables:
                    def _dl_csv(tid=tid) -> None:
                        cur = dict(downloads.final_tables(d)).get(tid, [])
                        ui.download(grid_to_csv(cur, s.convert_numerals).encode("utf-8-sig"), f"{tid}.csv")
                    ui.button(f"⬇ {tid}", on_click=_dl_csv).props("flat dense")
        skipped = len(downloads.final_tables(d)) - len(ne_tables)
        if skipped:
            ui.label(f"{skipped} empty table(s) excluded from downloads.").classes("text-xs text-gray-500")

    results()


api.mount_frontend()  # serve the built React app at /app when frontend/dist exists
ui.run(title="Khmer Document Extraction", port=8600, reload=False, show=False)
