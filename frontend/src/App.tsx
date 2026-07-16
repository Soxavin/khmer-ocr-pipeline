import { useCallback, useEffect, useRef, useState } from 'react'
import { CircleHelp, FileUp, RotateCw, Settings2, ShieldCheck, TriangleAlert } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './api/client'
import type { RunSettings } from './api/types'
import { QueueRail } from './components/queue/QueueRail'
import { RunControls } from './components/run/RunControls'
import { SettingsDrawer } from './components/run/SettingsDrawer'
import { PageViewer } from './components/viewer/PageViewer'
import { TablesPanel } from './components/review/TablesPanel'
import { IssuesDrawer } from './components/review/IssuesDrawer'
import { useRunStatus } from './hooks/useRunStatus'
import { ICON, iconBtnCls, primaryBtnCls } from './ui'

/** Errors shown to analysts, not developers: translate transport noise. */
function friendlyError(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e)
  if (/failed to fetch|networkerror|load failed/i.test(msg)) {
    return 'Cannot reach the extraction server — check that it is still running.'
  }
  return msg.replace(/^(TypeError|Error):\s*/, '')
}

export default function App() {
  const qc = useQueryClient()
  const [activeId, setActiveId] = useState<string | null>(null)
  const [pageIdx, setPageIdx] = useState(0)
  const [selectedTable, setSelectedTable] = useState<string | null>(null)
  const [flashToken, setFlashToken] = useState<{ tid: string; n: number } | null>(null)
  const [focusCell, setFocusCell] = useState<{ tid: string; row: number; col: number; n: number } | null>(null)
  // Silently remember the analyst's last-used engine/settings (no presets UI).
  const [engine, setEngine] = useState(() => localStorage.getItem('engine') ?? 'surya')
  // Joining continuation tables is an EXPORT choice; extraction always stays
  // per-page so every row keeps a page image to verify against.
  const [combineExport, setCombineExport] = useState(() => localStorage.getItem('combineExport') !== 'false')
  const [batchRunning, setBatchRunning] = useState(false)
  const [drawer, setDrawer] = useState<'issues' | 'settings' | null>(null)
  const [showFind, setShowFind] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  const [issueIdx, setIssueIdx] = useState(-1)
  const [runSettings, setRunSettings] = useState<RunSettings>({})
  const [error, setError] = useState<string | null>(null)
  const uploadRef = useRef<HTMLInputElement>(null)

  const meta = useQuery({ queryKey: ['meta'], queryFn: api.meta, staleTime: 60_000 })
  const docs = useQuery({ queryKey: ['documents'], queryFn: api.documents })
  const documents = docs.data?.documents ?? []
  const active = documents.find((d) => d.id === activeId) ?? null
  const status = useRunStatus(active?.id ?? null)

  // Results-dependent queries only once a run has finished.
  const hasResults = (status.data?.has_results ?? false) && !(status.data?.active ?? false)
  const overview = useQuery({
    queryKey: ['overview', active?.id],
    queryFn: () => api.overview(active!.id),
    enabled: active !== null && hasResults,
  })
  const pageCount = overview.data?.pages ?? 0
  const page = useQuery({
    queryKey: ['page', active?.id, pageIdx],
    queryFn: () => api.page(active!.id, pageIdx),
    enabled: active !== null && hasResults && pageIdx < pageCount,
  })
  const lowconf = useQuery({
    queryKey: ['lowconf', active?.id],
    queryFn: () => api.lowconf(active!.id),
    enabled: active !== null && hasResults,
  })
  const issues = lowconf.data?.issues ?? []

  // Seed the Advanced drawer from server defaults, overlaid with last-used values.
  const defaults = meta.data?.defaults
  useEffect(() => {
    if (!defaults) return
    let last: RunSettings = {}
    try {
      last = JSON.parse(localStorage.getItem('runSettings') ?? '{}') as RunSettings
    } catch { /* stale/corrupt entry: fall back to defaults */ }
    setRunSettings((s) => ({ ...defaults, ...last, ...s }))
  }, [defaults])

  const jumpToIssue = useCallback(
    (idx: number) => {
      const it = issues[idx]
      if (!it) return
      setIssueIdx(idx)
      if (it.page !== null) setPageIdx(it.page)
      setSelectedTable(it.table_id)
      setFlashToken((f) => ({ tid: it.table_id, n: (f?.n ?? 0) + 1 }))
      setFocusCell((f) => ({ tid: it.table_id, row: it.row, col: it.col, n: (f?.n ?? 0) + 1 }))
    },
    [issues],
  )

  // Help dialog: real focus management (trap Tab inside, restore focus on close).
  const helpRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!showHelp) return
    const prev = document.activeElement as HTMLElement | null
    helpRef.current?.focus()
    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !helpRef.current) return
      const focusables = helpRef.current.querySelectorAll<HTMLElement>('button, [href], [tabindex]:not([tabindex="-1"])')
      if (!focusables.length) {
        e.preventDefault()
        return
      }
      const first = focusables[0]
      const last = focusables[focusables.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', trap)
    return () => {
      window.removeEventListener('keydown', trap)
      prev?.focus()
    }
  }, [showHelp])

  // When a run finishes, refresh everything derived from it.
  const wasActive = useRef(false)
  useEffect(() => {
    const nowActive = status.data?.active ?? false
    if (wasActive.current && !nowActive) {
      qc.invalidateQueries({ queryKey: ['documents'] })
      qc.invalidateQueries({ queryKey: ['overview'] })
      qc.invalidateQueries({ queryKey: ['page'] })
      qc.invalidateQueries({ queryKey: ['lowconf'] })
      setPageIdx(0)
      setIssueIdx(-1)
    }
    wasActive.current = nowActive
  }, [status.data?.active, qc])

  const upload = useMutation({
    mutationFn: api.upload,
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      if (res.documents.length) setActiveId(res.documents[res.documents.length - 1].id)
      setError(null)
    },
    onError: (e) => setError(friendlyError(e)),
  })
  const remove = useMutation({
    mutationFn: api.remove,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  })
  // Every run is per-page: stitching is applied at export instead, so the review
  // panel can always link a row back to the page image it came from.
  const runPayload: RunSettings = { ...runSettings, ocr_engine_key: engine, stitch_pages: false }

  const rememberSettings = useCallback(() => {
    localStorage.setItem('engine', engine)
    localStorage.setItem('runSettings', JSON.stringify(runSettings))
    localStorage.setItem('combineExport', String(combineExport))
  }, [engine, runSettings, combineExport])

  const run = useMutation({
    mutationFn: (id: string) => api.run(id, runPayload),
    onSuccess: () => {
      setError(null)
      rememberSettings()
      qc.invalidateQueries({ queryKey: ['status'] })
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (e) => setError(friendlyError(e)),
  })
  const cancel = useMutation({ mutationFn: (id: string) => api.cancel(id) })

  // "Run all": sequential client-side loop (single GPU — the server 409s overlap).
  const runAll = useCallback(async () => {
    setBatchRunning(true)
    rememberSettings()
    try {
      const targets = documents.filter((d) => d.status === 'queued' || d.status === 'error')
      for (const d of targets) {
        setActiveId(d.id)
        try {
          await api.run(d.id, runPayload)
        } catch (e) {
          setError(friendlyError(e))
          break
        }
        let sawActive = false
        for (let i = 0; i < 7200; i++) {
          await new Promise((r) => setTimeout(r, 1000))
          const st = await api.status(d.id)
          if (st.active) sawActive = true
          if (!st.active && (st.has_results || st.run_error || sawActive)) {
            if (st.run_error?.includes('cancelled')) {
              setBatchRunning(false)
              return // ■ Stop ends the whole batch, not just one document
            }
            break
          }
        }
        qc.invalidateQueries({ queryKey: ['documents'] })
      }
    } finally {
      setBatchRunning(false)
      qc.invalidateQueries({ queryKey: ['documents'] })
      qc.invalidateQueries({ queryKey: ['status'] })
    }
  }, [documents, engine, runSettings, qc, rememberSettings])

  // Keyboard: r runs, n/p step issues, ←/→ pages, Ctrl/Cmd-F find, Esc closes drawers.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
        e.preventDefault()
        setShowFind((s) => !s)
        return
      }
      if (target.closest('input,textarea,select,[contenteditable],.ag-cell')) return
      if (e.key === 'Escape') {
        setDrawer(null)
        setShowFind(false)
        setShowHelp(false)
      } else if (e.key === '?') {
        setShowHelp((h) => !h)
      } else if (e.key === 'r' && active && !status.data?.active && !run.isPending) {
        run.mutate(active.id)
      } else if (e.key === 'n' && issues.length) {
        jumpToIssue((issueIdx + 1) % issues.length)
      } else if (e.key === 'p' && issues.length) {
        jumpToIssue((issueIdx - 1 + issues.length) % issues.length)
      } else if (e.key === 'ArrowLeft' && pageIdx > 0) {
        setPageIdx(pageIdx - 1)
      } else if (e.key === 'ArrowRight' && pageIdx < pageCount - 1) {
        setPageIdx(pageIdx + 1)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [issues.length, issueIdx, pageIdx, pageCount, jumpToIssue, active, status.data?.active, run])

  const runError = status.data?.run_error ?? null
  const warnings = overview.data?.warnings ?? []

  // Staleness: current form vs the settings the visible results were made with.
  const lastRun = status.data?.last_run_settings ?? null
  const current: RunSettings = { ...runSettings, ocr_engine_key: engine, stitch_pages: false }
  // Key-order-insensitive: compare each field the last run recorded.
  const stale =
    hasResults &&
    lastRun !== null &&
    Object.keys(lastRun).some((k) => k in current && JSON.stringify(current[k]) !== JSON.stringify(lastRun[k]))

  return (
    <div className="flex h-full flex-col bg-slate-50 text-slate-800">
      {/* Top bar: identity left, run controls right — the one primary action lives here. */}
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2">
        <div className="flex items-baseline gap-3">
          <h1 className="text-base font-semibold text-primary">Khmer Document Extraction</h1>
          <span
            className={`inline-block h-2 w-2 self-center rounded-full ${meta.data?.backend_ready ? 'bg-conf-high' : 'bg-slate-300'}`}
            title={meta.data?.backend_ready
              ? 'AI text-correction backend is running'
              : 'AI text-correction backend not running (only needed when Qwen is enabled)'}
          />
        </div>
        <div className="flex items-center gap-2">
          {hasResults && (
            <button
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-semibold ${
                issues.length
                  ? 'bg-red-50 text-red-700 ring-1 ring-red-200 hover:bg-red-100'
                  : 'bg-green-50 text-green-800 ring-1 ring-green-200'
              }`}
              onClick={() => setDrawer((d) => (d === 'issues' ? null : 'issues'))}
              title="Low-confidence cells to review (n / p to step through)"
            >
              {issues.length ? (
                <>
                  <TriangleAlert size={14} aria-hidden />
                  Issues ({issues.length})
                </>
              ) : (
                <>
                  <ShieldCheck size={14} aria-hidden />
                  No issues
                </>
              )}
            </button>
          )}
          <RunControls
            engines={meta.data?.engines ?? []}
            engine={engine}
            onEngineChange={setEngine}
            status={status.data}
            docSelected={active !== null}
            onUploadClick={() => uploadRef.current?.click()}
            onRun={() => active && run.mutate(active.id)}
            onStop={() => active && cancel.mutate(active.id)}
            exportUrl={active && hasResults ? api.exportZipUrl(active.id, combineExport) : null}
            docId={active?.id ?? null}
            unverifiedTables={active ? Math.max(0, active.total_tables - active.reviewed_tables) : 0}
            openIssues={issues.length}
            combineExport={combineExport}
            onCombineChange={(v) => {
              setCombineExport(v)
              localStorage.setItem('combineExport', String(v))
            }}
            multiPage={pageCount > 1}
          />
          <button
            className={`${iconBtnCls} ${drawer === 'settings' ? 'bg-blue-50 text-primary' : ''}`}
            onClick={() => setDrawer((d) => (d === 'settings' ? null : 'settings'))}
            aria-label="Extraction settings"
            title="Extraction settings (advanced)"
          >
            <Settings2 size={ICON} aria-hidden />
          </button>
          <button
            className={iconBtnCls}
            onClick={() => setShowHelp(true)}
            aria-label="Keyboard shortcuts"
            title="Keyboard shortcuts (?)"
          >
            <CircleHelp size={ICON} aria-hidden />
          </button>
        </div>
      </header>
      {/* Hidden input backing the header's Upload state of the primary action. */}
      <input
        ref={uploadRef}
        type="file"
        multiple
        accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff"
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) upload.mutate(Array.from(e.target.files))
          e.target.value = ''
        }}
      />

      {/* ONE status line at a time, most urgent first (error → stale → warnings):
          stacked banners turn "what do I do now" into a scan. */}
      {error || runError ? (
        <div className="border-b border-red-200 bg-red-50 px-4 py-1.5 text-sm text-red-700">
          {error ?? runError}
        </div>
      ) : stale && !status.data?.active ? (
        <div className="flex items-center gap-3 border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-sm text-amber-800">
          <span>Settings have changed since these results were made — re-run to apply them.</span>
          <button className="inline-flex items-center gap-1 rounded border border-amber-400 px-2 py-0.5 text-xs font-medium hover:bg-amber-100"
                  onClick={() => active && run.mutate(active.id)}>
            <RotateCw size={12} aria-hidden />
            Re-run now
          </button>
        </div>
      ) : hasResults && warnings.length > 0 ? (
        <details className="border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-sm text-amber-900">
          <summary className="cursor-pointer">
            <TriangleAlert size={14} className="mr-1 inline-block align-[-2px]" aria-hidden />
            {warnings.length} pipeline warning{warnings.length === 1 ? '' : 's'}
          </summary>
          <ul className="mt-1 list-inside list-disc">
            {warnings.map((w, i) => {
              const m = /Page (\d+)/.exec(w)
              const target = m ? Number(m[1]) - 1 : null
              return (
                <li key={i}>
                  {w}
                  {target !== null && target < pageCount && (
                    <button className="ml-2 text-xs text-primary underline" onClick={() => setPageIdx(target)}>
                      view page
                    </button>
                  )}
                </li>
              )
            })}
          </ul>
        </details>
      ) : null}

      <main className="flex min-h-0 flex-1">
        <QueueRail
          documents={documents}
          activeId={active?.id ?? null}
          onSelect={(id) => {
            setActiveId(id)
            setPageIdx(0)
            setSelectedTable(null)
            setFlashToken(null)
            setFocusCell(null)
            setIssueIdx(-1)
          }}
          onUpload={(files) => upload.mutate(files)}
          onRemove={(id) => {
            remove.mutate(id)
            if (id === activeId) setActiveId(null)
          }}
          uploading={upload.isPending}
          onRunAll={runAll}
          batchRunning={batchRunning}
          exportAllUrl={documents.some((d) => d.status === 'done') ? api.exportAllUrl(combineExport) : null}
        />

        {active === null ? (
          <EmptyState onUpload={() => uploadRef.current?.click()} />
        ) : !hasResults ? (
          <div className="flex flex-1 items-center justify-center text-sm text-slate-500">
            {status.data?.active
              ? status.data.stage || 'Working…'
              : runError
                ? 'The last run did not finish — adjust and retry from the top bar.'
                : 'Ready — press “Run extraction” in the top bar.'}
          </div>
        ) : (
          <>
            <section className="flex min-w-0 flex-[3] flex-col border-r border-slate-200">
              {pageCount > 0 && (
                <PageViewer
                  docId={active.id}
                  pageIdx={pageIdx}
                  pageCount={pageCount}
                  onPageChange={(i) => {
                    setPageIdx(i)
                    setSelectedTable(null)
                    setFlashToken(null)
                  }}
                  page={page.data}
                  selectedTable={selectedTable}
                  onTableClick={(tid) => {
                    setSelectedTable(tid)
                    setFlashToken((f) => ({ tid, n: (f?.n ?? 0) + 1 }))
                  }}
                />
              )}
            </section>
            <section className="flex min-w-0 flex-[2] flex-col bg-slate-50">
              {page.data ? (
                <TablesPanel
                  docId={active.id}
                  pageIdx={pageIdx}
                  page={page.data}
                  selectedTable={selectedTable}
                  onSelectTable={setSelectedTable}
                  flashToken={flashToken}
                  focusCell={focusCell}
                  showFind={showFind}
                  onCloseFind={() => setShowFind(false)}
                />
              ) : (
                <p className="p-4 text-sm text-slate-400">Loading tables…</p>
              )}
            </section>
          </>
        )}

        {drawer === 'issues' && hasResults && (
          <IssuesDrawer
            issues={issues}
            currentIdx={issueIdx}
            onJump={jumpToIssue}
            onClose={() => setDrawer(null)}
          />
        )}
        {showHelp && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setShowHelp(false)}>
            <div
              ref={helpRef}
              role="dialog"
              aria-modal="true"
              aria-label="Keyboard shortcuts"
              tabIndex={-1}
              className="w-96 rounded-lg bg-white p-5 shadow-xl focus:outline-none"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="mb-3 text-sm font-semibold text-slate-800">Keyboard shortcuts</h2>
              <table className="w-full text-sm text-slate-600">
                <tbody>
                  {(
                    [
                      ['r', 'Run / re-run extraction'],
                      ['← / →', 'Previous / next page'],
                      ['n / p', 'Next / previous issue (low-confidence cell)'],
                      ['Ctrl/Cmd-F', 'Find & replace across all tables'],
                      ['Cmd-Z / Shift-Cmd-Z', 'Undo / redo in the focused table'],
                      ['Right-click a row', 'Insert or delete rows'],
                      ['Esc', 'Close panels'],
                      ['?', 'This overlay'],
                    ] as const
                  ).map(([k, desc]) => (
                    <tr key={k} className="border-b border-slate-100 last:border-0">
                      <td className="py-1.5 pr-3 font-mono text-xs text-slate-500">{k}</td>
                      <td className="py-1.5">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {drawer === 'settings' && (
          <SettingsDrawer
            settings={runSettings}
            onChange={setRunSettings}
            pageCount={active?.pages ?? 0}
            onClose={() => setDrawer(null)}
          />
        )}
      </main>
    </div>
  )
}

function EmptyState(props: { onUpload: () => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
      <p className="text-xl font-semibold text-slate-800">Extract tables from Khmer documents in three steps</p>
      <ol className="space-y-1 text-sm text-slate-600">
        <li>1 · Upload a bulletin PDF or scan</li>
        <li>2 · Run the extraction</li>
        <li>3 · Review the numbers, then export to Excel</li>
      </ol>
      <button className={`${primaryBtnCls} mt-2 px-5 py-2`} onClick={props.onUpload}>
        <FileUp size={ICON} aria-hidden />
        Upload documents
      </button>
    </div>
  )
}
