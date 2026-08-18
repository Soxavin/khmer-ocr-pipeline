import type { DocSummary, Issue, Meta, Overview, PageData, RunSettings, RunStatus, Suggestion } from './types'

// A hung server must not leave promises pending forever: every JSON call gets a
// hard timeout (export downloads are navigations, not fetches — unaffected).
const withTimeout = (url: string, init?: RequestInit) =>
  fetch(url, { ...init, signal: AbortSignal.timeout(30_000) })

async function j<T>(req: Promise<Response>): Promise<T> {
  const res = await req
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* non-JSON error body: keep the HTTP status */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  meta: () => j<Meta>(withTimeout('/api/meta')),

  documents: () => j<{ documents: DocSummary[] }>(withTimeout('/api/documents')),
  upload: (files: File[]) => {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f))
    return j<{ documents: DocSummary[] }>(withTimeout('/api/documents', { method: 'POST', body: fd }))
  },
  remove: (id: string) => j<{ ok: boolean }>(withTimeout(`/api/documents/${id}`, { method: 'DELETE' })),
  clear: () => j<{ ok: boolean }>(withTimeout('/api/documents', { method: 'DELETE' })),

  run: (id: string, settings: RunSettings) =>
    j<{ started: boolean }>(
      withTimeout(`/api/documents/${id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      }),
    ),
  cancel: (id: string) => j<{ cancelling: boolean }>(withTimeout(`/api/documents/${id}/cancel`, { method: 'POST' })),
  status: (id: string) => j<RunStatus>(withTimeout(`/api/documents/${id}/status`)),
  suggest: (id: string) => j<Suggestion>(withTimeout(`/api/documents/${id}/suggest`)),

  overview: (id: string) => j<Overview>(withTimeout(`/api/documents/${id}/overview`)),
  putTable: (id: string, tableId: string, grid: string[][]) =>
    j<{ ok: boolean }>(
      withTimeout(`/api/documents/${id}/tables/${tableId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid }),
      }),
    ),
  resetTable: (id: string, tableId: string) =>
    j<{ ok: boolean }>(withTimeout(`/api/documents/${id}/tables/${tableId}`, { method: 'DELETE' })),
  lowconf: (id: string) => j<{ issues: Issue[] }>(withTimeout(`/api/documents/${id}/lowconf`)),
  review: (id: string, tableId: string, verified: boolean) =>
    j<{ ok: boolean }>(
      withTimeout(`/api/documents/${id}/review/${encodeURIComponent(tableId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verified }),
      }),
    ),
  replace: (id: string, find: string, replace: string) =>
    j<{ total: number; tables_changed: number }>(
      withTimeout(`/api/documents/${id}/replace`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ find, replace }),
      }),
    ),
  undoReplace: (id: string) =>
    j<{ ok: boolean }>(withTimeout(`/api/documents/${id}/replace/undo`, { method: 'POST' })),
  putPageText: (id: string, n: number, text: string) =>
    j<{ ok: boolean }>(
      withTimeout(`/api/documents/${id}/pages/${n}/text`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      }),
    ),
  page: (id: string, n: number) => j<PageData>(withTimeout(`/api/documents/${id}/pages/${n}`)),
  pageImageUrl: (id: string, n: number, variant: 'processed' | 'original' = 'processed') =>
    `/api/documents/${id}/pages/${n}/image?variant=${variant}`,
  // Raw page render BEFORE any run — analysts pick a page range by looking, not guessing.
  previewImageUrl: (id: string, n: number) => `/api/documents/${id}/preview/${n}`,
  // `combine` joins tables that continue across pages into one — an export
  // choice, never an extraction one (extraction stays per-page for linking).
  exportZipUrl: (id: string, combine = true) => `/api/documents/${id}/export/zip?combine=${combine}`,
  exportUrl: (id: string, fmt: 'json' | 'txt' | 'xlsx', combine = true) =>
    `/api/documents/${id}/export/${fmt}?combine=${combine}`,
  exportCsvUrl: (id: string, tableId: string) =>
    `/api/documents/${id}/export/csv/${encodeURIComponent(tableId)}`,
  exportAllUrl: (combine = true) => `/api/export/all.zip?combine=${combine}`,
}
