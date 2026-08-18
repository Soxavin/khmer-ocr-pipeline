import type { RunSettings } from '../api/types'

/** Which workflow stage the document canvas is rendering for: the raw upload
    before any run, or the review of a finished analysis. */
export type CanvasMode = 'pre-upload' | 'post-analysis'

/** The settings that describe WHICH pages to run. They belong to one document — a
    `page_start:2, page_end:4` from a 10-page bulletin is meaningless on a 3-page one
    — so they are stored per document (lib/perDocSettings.ts) and stripped from the
    GLOBAL last-used prefs, which seed documents that have no settings of their own. */
export const PAGE_SCOPE_KEYS = ['page_scope', 'page_num', 'page_start', 'page_end', 'page_list'] as const

/** The "all pages" starting point for a document with no scope of its own. Field
    values mirror the backend Settings defaults (apps/api/settings.py). */
export const PAGE_SCOPE_DEFAULTS: RunSettings = {
  page_scope: 'all',
  page_num: 1,
  page_start: 1,
  page_end: 5,
  page_list: [],
}

/** `settings` with every page-scope field removed — what gets persisted as the
    GLOBAL last-used prefs, so a range set on one document never seeds another. */
export function withoutPageScope(settings: RunSettings): RunSettings {
  const out = { ...settings }
  for (const k of PAGE_SCOPE_KEYS) delete out[k]
  return out
}

/** A page scope the backend would reject: an inverted range, or a single page past
    the end of the document. Mirrors the drawer's inline range/single errors so the
    launch controls can block rather than fire a run the API only 400s.
    `pageCount` 0 means "not known yet" and skips the single-page bound check. */
export function scopeInvalid(settings: RunSettings, pageCount: number): boolean {
  if (settings.page_scope === 'range') {
    return Number(settings.page_end ?? 5) < Number(settings.page_start ?? 1)
  }
  if (settings.page_scope === 'single') {
    return pageCount > 0 && Number(settings.page_num ?? 1) > pageCount
  }
  return false
}

/** Grid checkboxes derive from runSettings and encode back to the MINIMAL scope —
    one source of truth, no parallel selection state to desync. */
export function pagesFromSettings(s: RunSettings, pageCount: number): Set<number> {
  const scope = String(s.page_scope ?? 'all')
  const all = new Set(Array.from({ length: pageCount }, (_, i) => i))
  // Settings outlive the document they were set on (they persist across uploads),
  // so a scope can address pages this document does not have. Every branch clamps
  // to real pages: a phantom index renders a grid card with no image behind it and
  // encodes back into a run scope for a page that cannot be rasterized.
  if (pageCount <= 0) return new Set()
  const last = pageCount - 1
  if (scope === 'single') return new Set([Math.min(Math.max(0, Number(s.page_num ?? 1) - 1), last)])
  if (scope === 'range') {
    const start = Math.min(Math.max(0, Number(s.page_start ?? 1) - 1), last)
    const end = Math.min(Number(s.page_end ?? pageCount), pageCount)
    // max(1, …) keeps an inverted range (end < start) meaning "just the start
    // page" rather than nothing — but start is now guaranteed to be a real page.
    return new Set(Array.from({ length: Math.max(1, end - start) }, (_, i) => start + i))
  }
  if (scope === 'list') {
    const list = (s.page_list as number[] | undefined) ?? []
    const picked = new Set(list.map((p) => p - 1).filter((i) => i >= 0 && i < pageCount))
    return picked.size ? picked : all
  }
  return all
}

export function encodePages(picked: Set<number>, pageCount: number): Partial<RunSettings> {
  // Strict numeric sort — the default lexicographic sort would break contiguity math.
  const a = Array.from(picked).sort((x, y) => x - y)
  if (a.length === 0 || a.length === pageCount) return { page_scope: 'all' }
  if (a.length === 1) return { page_scope: 'single', page_num: a[0] + 1 }
  if (a[a.length - 1] - a[0] + 1 === a.length) {
    return { page_scope: 'range', page_start: a[0] + 1, page_end: a[a.length - 1] + 1 }
  }
  return { page_scope: 'list', page_list: a.map((i) => i + 1) }
}

/** Result index of document page `n`'s cleaned rendition, or -1 if it has none yet.

    Preprocessing finishes at stage 2, so cleaned pages exist long before the run
    ends and the grid can upgrade thumbnails mid-run. A page-scoped run only
    rasterizes the selected pages, so the mapping is positional in
    `processedPages` — never `n` itself. */
export function processedIndex(n: number, processedPages: number[] | undefined): number {
  if (!processedPages) return -1
  return processedPages.indexOf(n)
}

/** The 0-based document pages the grid overview renders, sorted ascending.
    Pre-upload shows the whole document; post-analysis shows ONLY the pages the
    finished run actually processed (its recorded settings), so pages excluded
    from the scope never render as empty frames. */
export function gridPages(mode: CanvasMode, pageCount: number, lastRun: RunSettings | null): number[] {
  if (mode === 'pre-upload' || lastRun === null) {
    return Array.from({ length: pageCount }, (_, i) => i)
  }
  return Array.from(pagesFromSettings(lastRun, pageCount)).sort((x, y) => x - y)
}
