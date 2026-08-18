# Khmer Document Extraction — frontend

React 19 + TypeScript + Tailwind. The primary UI for the Khmer OCR pipeline: upload a
scanned/PDF bulletin, run extraction, review and correct the recognized tables against the
page image, export JSON/CSV/Excel. Served at `/app` by the FastAPI backend in `../apps/api/`.

See the repo root's `docs/architecture/CONTEXT.md` for how this fits into the wider pipeline, and this
directory's `PRODUCT.md`/`DESIGN.md` for the product brief and design-token system (used by
the `/impeccable` design skill).

## Running it

Don't run `npm run dev` directly — start from the repo root instead, which also brings up the
backend this app talks to:

```bash
./dev.sh          # backend on :8600 + Vite HMR on :5173/app/ — use this for UI work
```

Vite proxies `/api` to `:8600` (`vite.config.ts`), so `:5173/app/` and `:8600/app` hit the same
backend and the same in-memory document state. Edits hot-reload instantly at `:5173`.

```bash
./dev.sh build     # rebuild apps/web/dist so :8600/app serves the real (non-HMR) bundle
./dev.sh restart   # force a fresh backend — required after editing apps/api/ or
                    # src/khmer_pipeline/, since a reused backend keeps serving old code
```

An already-running backend is normally *reused*, not restarted, because it holds the loaded
OCR models and the document registry — restarting drops every uploaded document and costs a
slow model reload.

## Checks

```bash
npx tsc -b        # typecheck (solution-style tsconfig — `tsc --noEmit` is a silent no-op here)
npx vitest run    # unit/component tests
npx oxlint        # lint
```

Run these before committing any frontend change; CI expects a clean `tsc -b` and `vitest run`.

## Architecture, in brief

Three-zone workspace under `src/components/`:

- `queue/` — the document queue rail.
- `viewer/` — zoom/pan page viewer with confidence/region overlays, two-way table↔image
  linking (`PageViewer.tsx`, `PageGrid.tsx`).
- `review/` — AG Grid table editing (undo/redo, diff view, verify, per-table CSV export).

`run/SettingsDrawer.tsx` and `run/RunControls.tsx` hold the extraction-settings panel and the
Upload → Run → Export primary action. Server state lives entirely in the backend
(`apps/api/registry.py`) — the app is refresh-safe; reloading the tab keeps the queue, results,
and in-progress edits. `api/client.ts` is the one place that talks to the backend.

Khmer text rendering uses a bundled Noto Sans Khmer variable font
(`src/assets/fonts/`, OFL-licensed) with a dedicated `.khmer-content` line-height and a
user-adjustable size (A−/A+, persisted to `localStorage`).

## i18n

UI strings live in `src/i18n.tsx` (English + Khmer side by side per key). New Khmer strings
are marked `// PROVISIONAL — flag for native review` at the point they're added, and tracked in
`../docs/i18n_km_review_prompt.md` — that file's own freshness marker shows which keys still
need a native speaker's review pass. Keep the two in sync when adding strings.

## Error handling

`src/components/ErrorBoundary.tsx` wraps the whole app at `main.tsx` — a render-time exception
anywhere shows a recovery screen (with the raw error under a "Technical details" toggle)
instead of a blank white page. It's deliberately dependency-light (no i18n, no app state) so
it can't itself be taken down by whatever it's catching.
