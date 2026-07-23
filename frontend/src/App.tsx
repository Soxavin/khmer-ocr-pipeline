import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CircleHelp, Files, FileUp, Grid3x3, Languages, Monitor, Moon, MoreHorizontal, ScanSearch, Search, Settings2, ShieldCheck, Square, StickyNote, Sun, TriangleAlert, X } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './api/client'
import type { RunSettings } from './api/types'
import { QueueRail } from './components/queue/QueueRail'
import { HeaderProgress } from './components/run/HeaderProgress'
import { RunControls } from './components/run/RunControls'
import { SettingsDrawer } from './components/run/SettingsDrawer'
import { CommandPalette, type Command } from './components/CommandPalette'
import { GridThumb, PageGrid, ViewToggle } from './components/viewer/PageGrid'
import { PageViewer } from './components/viewer/PageViewer'
// AG-Grid (~1 MB) lives entirely inside TablesPanel and is only ever rendered
// once a run has results — lazy-load it so it stays out of the initial bundle
// (the empty state and pre-run flow never pay for it).
const TablesPanel = lazy(() => import('./components/review/TablesPanel').then((m) => ({ default: m.TablesPanel })))
import { IssuesDrawer } from './components/review/IssuesDrawer'
import { useFocusTrap } from './hooks/useFocusTrap'
import { useRunStatus } from './hooks/useRunStatus'
import { encodePages, gridPages, pagesFromSettings, processedIndex } from './lib/pages'
import { countOverrides, mergeSuggestion, scanSummary } from './lib/settings'
import { configDiffers, guardedRun, isBusy } from './lib/run'
import { useT } from './i18n.tsx'
import { btnCls, chipCls, ICON, ICON_SM, iconBtnCls, kbdCls, menuCls, menuItemCls, panelMainCls, primaryBtnCls, withViewTransition } from './ui'

type ThemePref = 'light' | 'dark' | 'system'

/** Theme preference cycles light → dark → system; CSS gets the RESOLVED theme
    stamped on <html data-theme>, so index.css needs a single dark block. */
function useTheme(): [ThemePref, () => void] {
  const [pref, setPref] = useState<ThemePref>(
    () => (localStorage.getItem('theme') as ThemePref) ?? 'system',
  )
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const apply = () => {
      const resolved = pref === 'system' ? (mq.matches ? 'dark' : 'light') : pref
      document.documentElement.dataset.theme = resolved
    }
    apply()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [pref])
  const cycle = () => {
    setPref((p) => {
      const next: ThemePref = p === 'light' ? 'dark' : p === 'dark' ? 'system' : 'light'
      localStorage.setItem('theme', next)
      return next
    })
  }
  return [pref, cycle]
}

/** Adjustable viewer⇄tables split: ratio of the viewer pane, remembered per machine.
    Drag the seam, double-click to reset, arrow keys when the divider is focused. */
function useSplit(): {
  ratio: number
  dividerProps: React.HTMLAttributes<HTMLDivElement>
  dragging: boolean
} {
  const DEFAULT = 0.6
  const [ratio, setRatio] = useState(() => {
    const v = Number(localStorage.getItem('workspaceSplit'))
    return v >= 0.2 && v <= 0.8 ? v : DEFAULT
  })
  const [dragging, setDragging] = useState(false)
  const commit = (r: number) => {
    const clamped = Math.min(0.8, Math.max(0.2, r))
    setRatio(clamped)
    localStorage.setItem('workspaceSplit', String(clamped))
  }
  return {
    ratio,
    dragging,
    dividerProps: {
      role: 'separator',
      'aria-orientation': 'vertical',
      'aria-valuenow': Math.round(ratio * 100),
      'aria-valuemin': 20,
      'aria-valuemax': 80,
      tabIndex: 0,
      onDoubleClick: () => commit(DEFAULT),
      onKeyDown: (e) => {
        if (e.key === 'ArrowLeft') commit(ratio - 0.02)
        if (e.key === 'ArrowRight') commit(ratio + 0.02)
      },
      onPointerDown: (e) => {
        e.preventDefault()
        setDragging(true)
        const parent = (e.currentTarget as HTMLElement).parentElement
        if (!parent) return
        const rect = parent.getBoundingClientRect()
        const move = (ev: PointerEvent) => commit((ev.clientX - rect.left) / rect.width)
        const up = () => {
          setDragging(false)
          window.removeEventListener('pointermove', move)
          window.removeEventListener('pointerup', up)
        }
        window.addEventListener('pointermove', move)
        window.addEventListener('pointerup', up)
      },
    },
  }
}

/** Errors shown to analysts, not developers: translate transport noise. */
function friendlyError(e: unknown, unreachable: string): string {
  const msg = e instanceof Error ? e.message : String(e)
  if (/failed to fetch|networkerror|load failed/i.test(msg)) {
    return unreachable
  }
  return msg.replace(/^(TypeError|Error):\s*/, '')
}

export default function App() {
  const qc = useQueryClient()
  const { t, lang, setLang } = useT()
  const [themePref, cycleTheme] = useTheme()
  const [activeId, setActiveId] = useState<string | null>(null)
  const [pageIdx, setPageIdx] = useState(0)
  const [selectedTable, setSelectedTable] = useState<string | null>(null)
  const [flashToken, setFlashToken] = useState<{ tid: string; n: number } | null>(null)
  const [focusCell, setFocusCell] = useState<{ tid: string; row: number; col: number; n: number } | null>(null)
  // The page image and the Page-text cards are two views of one block list, so the
  // link between them lives here — the nearest common owner. `from` records which
  // side initiated, so only the OTHER side scrolls and the two never chase each
  // other. Hover highlights without moving anything; a click moves the camera.
  const [blockSel, setBlockSel] = useState<{ i: number; from: 'canvas' | 'text'; n: number } | null>(null)
  const [blockHover, setBlockHover] = useState<number | null>(null)
  const activeBlock = blockHover ?? blockSel?.i ?? null
  const pickBlock = useCallback((from: 'canvas' | 'text') => (i: number) => {
    setBlockSel((p) => ({ i, from, n: (p?.n ?? 0) + 1 }))
  }, [])
  const clearBlockLink = useCallback(() => {
    setBlockSel(null)
    setBlockHover(null)
  }, [])
  const canvasPickBlock = useMemo(() => pickBlock('canvas'), [pickBlock])
  const textPickBlock = useMemo(() => pickBlock('text'), [pickBlock])
  // Silently remember the analyst's last-used engine/settings (no presets UI).
  const [engine, setEngine] = useState(() => localStorage.getItem('engine') ?? 'surya')
  // Joining continuation tables is an EXPORT choice; extraction always stays
  // per-page so every row keeps a page image to verify against.
  const [combineExport, setCombineExport] = useState(() => localStorage.getItem('combineExport') !== 'false')
  const [batchRunning, setBatchRunning] = useState(false)
  const [drawer, setDrawer] = useState<'issues' | 'settings' | null>(null)
  const [showFind, setShowFind] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  const [showNotes, setShowNotes] = useState(false)
  const [showMore, setShowMore] = useState(false)
  const [showPalette, setShowPalette] = useState(false)
  const [issueIdx, setIssueIdx] = useState(-1)
  const [canvasView, setCanvasView] = useState<'single' | 'grid'>('single')
  const split = useSplit()
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
  // Dismissals are global triage state: the header chip, drawer count, and n/p
  // stepping all agree on the same filtered list. Per-session, reset per document.
  const [dismissedIssues, setDismissedIssues] = useState<Set<string>>(new Set())
  const allIssues = lowconf.data?.issues ?? []
  // Memoized: a fresh array identity every render was rebinding the global
  // keyboard listener (whose deps chain through jumpToIssue) on every paint.
  const issues = useMemo(
    () => allIssues.filter((it) => !dismissedIssues.has(`${it.table_id}:${it.row}:${it.col}`)),
    [allIssues, dismissedIssues],
  )
  const dismissIssue = useCallback((key: string) => {
    setDismissedIssues((prev) => new Set(prev).add(key))
    setIssueIdx(-1) // indexes renumber after a removal: drop the highlight, n restarts cleanly
  }, [])

  // Auto preprocessing suggestions: advisory values computed from the page
  // images. Applied at most once per document; the user's changes always win.
  const suggestion = useQuery({
    queryKey: ['suggest', active?.id],
    queryFn: () => api.suggest(active!.id),
    enabled: active !== null,
    staleTime: Infinity,
  })
  // toggle key → rationale line, for the "Auto" badge in the settings drawer.
  const [autoApplied, setAutoApplied] = useState<Record<string, string>>({})
  // The scan check speaks once per document: what it changed (or that all is well).
  const [settingsPulse, setSettingsPulse] = useState(false)
  // Telemetry-badge jump target: which drawer row to scroll to + pulse.
  const [highlightFlag, setHighlightFlag] = useState<{ k: string; n: number } | null>(null)
  // Doc ids are content-derived (md5), so a re-uploaded file KEEPS its old id. The
  // map remembers the doc's state when the scan check last spoke: a doc seen with
  // results that is queued again is a NEW upload generation — speak again.
  const suggestSeenRef = useRef<Map<string, 'queued' | 'run'>>(new Map())
  // Live view of settings for effects that must NOT re-run on every keystroke but
  // still need current values (the seen-doc badge filter read a stale closure before).
  const runSettingsRef = useRef(runSettings)
  runSettingsRef.current = runSettings
  useEffect(() => {
    const s = suggestion.data
    if (!active || !s) {
      setAutoApplied({})
      return
    }
    const seenAs = suggestSeenRef.current.get(active.id)
    const isFreshGeneration = seenAs === 'run' && active.status === 'queued'
    if (seenAs && !isFreshGeneration) {
      if (active.status !== 'queued') suggestSeenRef.current.set(active.id, 'run')
      // Re-activated doc: badge only toggles still holding the suggested value
      // (a user override made earlier persists, unbadged).
      setAutoApplied(Object.fromEntries(
        Object.entries(s.rationale).filter(([k]) => runSettingsRef.current[k] === s.suggested[k]),
      ))
      return
    }
    suggestSeenRef.current.set(active.id, active.status === 'queued' ? 'queued' : 'run')
    // A fresh document: report what the scan check saw, once, in passing.
    const sum = scanSummary(s.checks ?? [])
    if (sum) {
      setScanToast({
        msg: sum.active > 0 ? t('scan_toast_active', { n: sum.active }) : t('scan_toast_clean'),
        field: sum.fields[0] ?? null,
        n: Date.now(),
      })
    }
    const keys = Object.keys(s.suggested)
    setAutoApplied(keys.length ? { ...s.rationale } : {})
    if (keys.length) {
      setRunSettings((prev) => mergeSuggestion(prev, s.suggested, touchedRef.current))
      setSettingsPulse(true)
    }
    // runSettings intentionally omitted: this effect must run on doc/suggestion
    // changes only, not on every settings keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.id, suggestion.data])
  // Keys the operator changed by hand for the active document. The advisory
  // suggestion merges around these — automation never overrules a person.
  const touchedRef = useRef<Set<string>>(new Set())
  // The user touched a toggle: their choice wins — drop its Auto badge.
  const clearAuto = useCallback((k: string) => {
    touchedRef.current.add(k)
    setAutoApplied((prev) => {
      if (!(k in prev)) return prev
      const { [k]: _dropped, ...rest } = prev
      return rest
    })
  }, [])

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

  // Focus one grid cell on the current page: select its table, flash it, scroll
  // the editor to the cell AND fly the page image to the table (the focusCell
  // token also drives PageViewer's flyToken). Used by the triage-band chips.
  const focusGridCell = useCallback((tid: string, row: number, col: number) => {
    setSelectedTable(tid)
    setFlashToken((f) => ({ tid, n: (f?.n ?? 0) + 1 }))
    setFocusCell((f) => ({ tid, row, col, n: (f?.n ?? 0) + 1 }))
  }, [])

  // Clearing the last low-confidence cell is the document-level payoff:
  // the Issues chip pulses once when the count crosses to zero.
  const [issuesCleared, setIssuesCleared] = useState(false)
  const prevIssueCount = useRef(0)
  useEffect(() => {
    if (hasResults && prevIssueCount.current > 0 && issues.length === 0) setIssuesCleared(true)
    prevIssueCount.current = issues.length
  }, [issues.length, hasResults])

  // Help dialog: real focus management (trap Tab inside, restore focus on close).
  const helpRef = useRef<HTMLDivElement>(null)
  useFocusTrap(helpRef, showHelp)

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

  // Passive scan-check notice: slides in on upload, leaves on its own after 4s.
  const [scanToast, setScanToast] = useState<{ msg: string; field: string | null; n: number } | null>(null)
  useEffect(() => {
    if (scanToast === null) return
    const id = setTimeout(() => setScanToast(null), 4000)
    return () => clearTimeout(id)
  }, [scanToast])
  const [dupNotice, setDupNotice] = useState<string | null>(null)
  const upload = useMutation({
    mutationFn: api.upload,
    // Snapshot the queue BEFORE the upload: the server dedupes identical content
    // (content-hash ids), so a "new" doc whose id already existed is a re-add.
    onMutate: () => ({ before: new Set(documents.map((d) => d.id)) }),
    onSuccess: (res, _files, ctx) => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      if (res.documents.length) setActiveId(res.documents[res.documents.length - 1].id)
      setError(null)
      const dup = res.documents.find((d) => ctx?.before.has(d.id))
      setDupNotice(dup ? dup.name : null)
    },
    onError: (e) => setError(friendlyError(e, t('err_unreachable'))),
  })
  useEffect(() => {
    if (dupNotice === null) return
    const id = setTimeout(() => setDupNotice(null), 6000)
    return () => clearTimeout(id)
  }, [dupNotice])
  const removeAll = useMutation({
    mutationFn: api.clear,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      // Back to a bare workspace: no active document, no carried-over triage state.
      setActiveId(null)
      setPageIdx(0)
      setSelectedTable(null)
      setIssueIdx(-1)
      setDismissedIssues(new Set())
      setDrawer(null)
      suggestSeenRef.current.clear()
      touchedRef.current = new Set()
    },
    onError: (e) => setError(friendlyError(e, t('err_unreachable'))),
  })
  const remove = useMutation({
    mutationFn: api.remove,
    onSuccess: (_res, id) => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      // The same file re-uploaded later reuses this content-derived id: forget it
      // so the scan check speaks again for the new upload.
      suggestSeenRef.current.delete(id)
    },
  })
  // Every run is per-page: stitching is applied at export instead, so the review
  // panel can always link a row back to the page image it came from.
  // draftConfiguration: the live sidebar state, sent on the NEXT run.
  const draftConfiguration: RunSettings = { ...runSettings, ocr_engine_key: engine, stitch_pages: false }
  // The server extracts one document at a time; this gate keeps every launch
  // control in the workspace honest instead of letting the API 409 explain it.
  const pipelineBusy = isBusy(documents, batchRunning) || (status.data?.active ?? false)

  const rememberSettings = useCallback(() => {
    localStorage.setItem('engine', engine)
    localStorage.setItem('runSettings', JSON.stringify(runSettings))
    localStorage.setItem('combineExport', String(combineExport))
  }, [engine, runSettings, combineExport])

  const run = useMutation({
    // Guarded so a double-click or a stale button never races the single-run
    // server: a collision is caught here (or swallowed from a 409) rather than
    // surfacing "Another extraction is already running." as a red banner.
    mutationFn: (id: string) => guardedRun(pipelineBusy, () => api.run(id, draftConfiguration)),
    onSuccess: (outcome) => {
      if (outcome === 'blocked') return
      setError(null)
      rememberSettings()
      // A new run invalidates the results on screen: drop the caches for the old
      // ones so nothing from the previous configuration lingers into review.
      qc.removeQueries({ queryKey: ['overview', active?.id] })
      qc.removeQueries({ queryKey: ['page', active?.id] })
      qc.removeQueries({ queryKey: ['lowconf', active?.id] })
      setSelectedTable(null)
      setFlashToken(null)
      setFocusCell(null)
      setIssueIdx(-1)
      setDismissedIssues(new Set())
      qc.invalidateQueries({ queryKey: ['status'] })
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (e) => setError(friendlyError(e, t('err_unreachable'))),
  })
  const cancel = useMutation({ mutationFn: (id: string) => api.cancel(id) })

  // Switching documents resets every per-document view state in one place.
  // Declared before runAll, which both calls it and lists it as a dep.
  const selectDoc = useCallback((id: string) => {
    setActiveId(id)
    touchedRef.current = new Set()
    setPageIdx(0)
    setSelectedTable(null)
    setFlashToken(null)
    setFocusCell(null)
    clearBlockLink()
    setIssueIdx(-1)
    setDismissedIssues(new Set())
  }, [clearBlockLink])

  // "Run all": sequential client-side loop (single GPU — the server 409s overlap).
  const runAll = useCallback(async () => {
    setBatchRunning(true)
    rememberSettings()
    try {
      const targets = documents.filter((d) => d.status === 'queued' || d.status === 'error' || d.status === 'stopped')
      for (const d of targets) {
        // A doc removed while the batch was working through the queue must be
        // skipped, not run — the snapshot in `targets` is minutes old by now.
        const live = qc.getQueryData<{ documents: { id: string }[] }>(['documents'])
        if (live && !live.documents.some((x) => x.id === d.id)) continue
        // Full doc-switch reset (pageIdx, selection, triage): plain setActiveId
        // bled the previous doc's page index into the next one's preview.
        selectDoc(d.id)
        try {
          await guardedRun(false, () => api.run(d.id, draftConfiguration))
        } catch (e) {
          setError(friendlyError(e, t('err_unreachable')))
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
  }, [documents, engine, runSettings, qc, rememberSettings, selectDoc])

  // Keyboard: r runs, n/p step issues, ←/→ pages, Ctrl/Cmd-F find, Esc closes drawers.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
        e.preventDefault()
        setShowFind((s) => !s)
        return
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setShowPalette((s) => !s)
        return
      }
      if (target.closest('input,textarea,select,[contenteditable],.ag-cell')) return
      if (e.key === 'Escape') {
        setShowPalette(false)
        setDrawer(null)
        setShowFind(false)
        setShowHelp(false)
        setShowNotes(false)
        setShowMore(false)
      } else if (e.key === '?') {
        setShowHelp((h) => !h)
      } else if (e.key === 'r' && active && !pipelineBusy && !run.isPending) {
        run.mutate(active.id)
      } else if (e.key === 'v' && active && selectedTable && page.data) {
        const tbl = page.data.tables.find((x) => x.table_id === selectedTable)
        if (tbl) {
          api.review(active.id, selectedTable, !tbl.verified).then(() => {
            qc.invalidateQueries({ queryKey: ['documents'] })
            qc.invalidateQueries({ queryKey: ['page'] })
          }).catch((err) => setError(friendlyError(err, t('err_unreachable'))))
        }
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
  }, [issues.length, issueIdx, pageIdx, pageCount, jumpToIssue, active, status.data?.active, run, selectedTable, page.data, qc, t])

  // Grid page selection ⇄ run settings (single source of truth: runSettings).
  // Derived values + handlers below are memoized so React.memo on PageGrid /
  // QueueRail actually holds during the 2.5 Hz status polls of a run.
  const docPages = active?.pages ?? 0
  const selectedPages = useMemo(() => pagesFromSettings(runSettings, docPages), [runSettings, docPages])
  const togglePage = useCallback((n: number) => {
    setRunSettings((prev) => {
      const cur = pagesFromSettings(prev, docPages)
      if (cur.has(n)) {
        if (cur.size === 1) return prev // a run of zero pages is meaningless
        cur.delete(n)
      } else {
        cur.add(n)
      }
      return { ...prev, ...encodePages(cur, docPages) }
    })
  }, [docPages])
  // Post-analysis: which document pages the finished run actually covered.
  const lastRunSettings = status.data?.last_run_settings ?? null
  const processedPages = useMemo(
    () => gridPages('post-analysis', docPages, lastRunSettings),
    [docPages, lastRunSettings],
  )
  const preGridPages = useMemo(() => gridPages('pre-upload', Math.max(1, docPages), null), [docPages])
  const openPageFromGrid = useCallback((n: number) => {
    withViewTransition(() => {
      setCanvasView('single')
      setPageIdx(n)
    })
  }, [])
  // Grid image sources, stable across polls (processed_pages keeps identity via
  // structural sharing once stage 2 has run).
  const activeDocId = active?.id ?? null
  const statusProcessed = status.data?.processed_pages
  const preImageUrl = useCallback(
    (n: number) => {
      const k = processedIndex(n, statusProcessed)
      return k >= 0 && activeDocId
        ? api.pageImageUrl(activeDocId, k, 'processed')
        : api.previewImageUrl(activeDocId ?? '', n)
    },
    [activeDocId, statusProcessed],
  )
  const previewUrl = useCallback((n: number) => api.previewImageUrl(activeDocId ?? '', n), [activeDocId])
  const postImageUrl = useCallback(
    (n: number) => api.pageImageUrl(activeDocId ?? '', processedPages.indexOf(n), 'processed'),
    [activeDocId, processedPages],
  )
  const openProcessedFromGrid = useCallback(
    (n: number) => openPageFromGrid(processedPages.indexOf(n)),
    [openPageFromGrid, processedPages],
  )
  // Queue-rail handlers, stable for React.memo.
  const uploadFiles = useCallback((files: File[]) => upload.mutate(files), [upload.mutate])
  const removeDoc = useCallback(
    (id: string) => {
      remove.mutate(id)
      if (id === activeId) setActiveId(null)
    },
    [remove.mutate, activeId],
  )
  const removeAllDocs = useCallback(() => removeAll.mutate(), [removeAll.mutate])

  const runError = status.data?.run_error ?? null
  // The backend's only cancel signal is its message string; keep the match in ONE place.
  const wasCancelled = runError !== null && runError.includes('cancelled')

  // Everything the workspace can do, searchable from one place (⌘K). All rows
  // orchestrate existing actions — the palette owns no logic of its own.
  const commands = useMemo<Command[]>(() => {
    const cmds: Command[] = []
    if (active && !pipelineBusy && !run.isPending) {
      cmds.push({ id: 'run', group: t('group_actions'), label: t('run_extraction'), keywords: 're-run extract', shortcut: 'r', run: () => run.mutate(active.id) })
    }
    if (active?.status === 'done') {
      cmds.push(
        { id: 'exp-xlsx', group: t('group_actions'), label: t('cmd_export_xlsx'), keywords: 'excel export', run: () => { window.location.href = api.exportUrl(active.id, 'xlsx', combineExport) } },
        { id: 'exp-json', group: t('group_actions'), label: t('cmd_export_json'), keywords: 'export', run: () => { window.location.href = api.exportUrl(active.id, 'json', combineExport) } },
        { id: 'exp-zip', group: t('group_actions'), label: t('cmd_export_zip'), keywords: 'zip export all', run: () => { window.location.href = api.exportZipUrl(active.id, combineExport) } },
      )
    }
    cmds.push(
      { id: 'settings', group: t('group_actions'), label: t('cmd_open_settings'), keywords: 'settings preprocess engine dpi', run: () => setDrawer('settings') },
      { id: 'theme', group: t('group_actions'), label: t('cmd_theme'), keywords: 'dark light theme', run: cycleTheme },
      { id: 'lang', group: t('group_actions'), label: t('cmd_language'), keywords: 'khmer english language', run: () => setLang(lang === 'en' ? 'km' : 'en') },
    )
    if (issues.length) {
      cmds.push({ id: 'issues', group: t('group_actions'), label: t('cmd_open_issues'), keywords: 'low confidence problems', shortcut: 'n', run: () => setDrawer('issues') })
    }
    for (const e2 of meta.data?.engines ?? []) {
      if (e2.key !== engine) {
        cmds.push({ id: `eng-${e2.key}`, group: t('group_actions'), label: t('cmd_engine', { x: e2.label }), keywords: 'engine recognition ocr', run: () => setEngine(e2.key) })
      }
    }
    const flags: [string, string][] = [
      ['deskew', t('flag_deskew')],
      ['remove_stamps', t('flag_stamps')],
      ['sharpen', t('flag_sharpen')],
      ['normalise', t('flag_contrast')],
      ['normalise_table_backgrounds', t('flag_tablebg')],
      ['repair_tables', t('flag_repair')],
      ['convert_numerals', t('flag_numerals')],
    ]
    for (const [k, label] of flags) {
      const on = Boolean(runSettings[k])
      cmds.push({
        id: `flag-${k}`,
        group: t('group_actions'),
        label: t(on ? 'cmd_turn_off' : 'cmd_turn_on', { x: label }),
        keywords: 'preprocess toggle setting',
        run: () => setRunSettings((prev) => ({ ...prev, [k]: !on })),
      })
    }
    for (const d of documents) {
      if (d.id !== activeId) {
        cmds.push({ id: `doc-${d.id}`, group: t('group_documents'), label: d.name, keywords: d.status, run: () => selectDoc(d.id) })
      }
    }
    if (active) {
      for (let i = 0; i < Math.min(pageCount, 40); i++) {
        if (i !== pageIdx) {
          cmds.push({ id: `page-${i}`, group: t('group_pages'), label: t('cmd_goto_page', { n: i + 1 }), keywords: `page ${i + 1}`, run: () => setPageIdx(i) })
        }
      }
    }
    issues.slice(0, 20).forEach((_, i) => {
      cmds.push({ id: `issue-${i}`, group: t('group_issues'), label: t('cmd_goto_issue', { n: i + 1, total: issues.length }), keywords: `issue ${i + 1}`, run: () => jumpToIssue(i) })
    })
    return cmds
  }, [active, activeId, status.data?.active, run, combineExport, cycleTheme, setLang, lang, issues, meta.data?.engines, engine, runSettings, documents, pageCount, pageIdx, jumpToIssue, selectDoc, t])
  const warnings = overview.data?.warnings ?? []

  // Deliberate overrides on the Settings button — deviations on the controls the
  // drawer actually exposes, so seeded/stale non-UI fields never inflate it.
  const changedSettings = countOverrides(runSettings, defaults)

  // Staleness: current form vs the settings the visible results were made with.
  // appliedConfiguration: the frozen snapshot the visible results were made with.
  const appliedConfiguration = status.data?.last_run_settings ?? null
  const stale = hasResults && configDiffers(appliedConfiguration, draftConfiguration)
  // The stale toast shows for 10s per settings change, then gets out of the way;
  // further edits while stale bring it back (each restarts the window).
  const [staleToastHidden, setStaleToastHidden] = useState(false)
  useEffect(() => {
    setStaleToastHidden(false)
    if (!stale) return
    const id = setTimeout(() => setStaleToastHidden(true), 10_000)
    return () => clearTimeout(id)
  }, [stale, runSettings, engine])

  return (
    // h-full, not h-screen: 100vh excludes a horizontal scrollbar's height, so the
    // moment anything forces one the root exceeds the remaining space and creates a
    // VERTICAL scrollbar too. height:100% chained from html tracks the real client
    // box and has no such failure mode.
    <div className="flex h-full flex-col overflow-hidden bg-canvas text-ink">
      {/* Top bar: identity left, run controls right — the one primary action lives here. */}
      {/* overflow-hidden + min-w-0 on the clusters below: the top bar is the one
          region outside main's clipping, so a crowded right side (notes + issues +
          more + the export split button) could otherwise push the layout wider than
          the viewport and force a horizontal scrollbar. */}
      <header className="relative z-20 flex h-12 shrink-0 items-center justify-between gap-2 overflow-hidden border-b border-line-strong/60 bg-surface px-4 shadow-raised">
        {/* The run's pulse — its 250ms tick renders inside this leaf only. */}
        <HeaderProgress status={status.data} />
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-primary" aria-hidden>
            <Grid3x3 size={14} className="text-white" strokeWidth={2.25} />
          </span>
          <h1 className="truncate text-[15px] font-semibold tracking-[-0.01em] text-ink">{t('app_title')}</h1>
          <span
            className={`inline-block h-1.5 w-1.5 self-center rounded-full ${meta.data?.backend_ready ? 'bg-ok' : 'bg-line-strong'}`}
            title={meta.data?.backend_ready ? t('backend_ready') : t('backend_off')}
          />
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {/* Processing notes: observations, not errors — a quiet chip, not a banner. */}
          {hasResults && warnings.length > 0 && (
            <span className="relative">
              <button
                className={`${chipCls} bg-warn-soft text-warn-ink hover:bg-warn-soft/70`}
                onClick={() => setShowNotes((n) => !n)}
                aria-expanded={showNotes}
                title={t('notes_tip')}
              >
                <StickyNote size={12} aria-hidden />
                {t('notes', { n: warnings.length })}
              </button>
              {showNotes && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setShowNotes(false)} />
                  <div className={`${menuCls} absolute right-0 top-full z-50 mt-1 w-96 p-3`}>
                    <p className="mb-0.5 text-sm font-semibold text-ink">{t('processing_notes')}</p>
                    <p className="mb-2 text-xs text-ink-2">
                      {t('notes_intro')}
                    </p>
                    <ul className="max-h-72 space-y-2 overflow-y-auto">
                      {warnings.map((w, i) => {
                        const m = /Page (\d+)/.exec(w)
                        const target = m ? Number(m[1]) - 1 : null
                        return (
                          <li key={i} className="flex items-start gap-2 text-xs text-ink">
                            <TriangleAlert size={13} className="mt-0.5 shrink-0 text-warn" aria-hidden />
                            <span className="min-w-0">
                              {w}
                              {target !== null && target < pageCount && (
                                <button
                                  className="ml-1.5 font-medium text-primary underline underline-offset-2"
                                  onClick={() => {
                                    setPageIdx(target)
                                    setShowNotes(false)
                                  }}
                                >
                                  {t('view_page_n', { n: target + 1 })}
                                </button>
                              )}
                            </span>
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                </>
              )}
            </span>
          )}
          {hasResults && (
            <button
              className={`${chipCls} ${
                issues.length
                  ? 'bg-danger-soft font-semibold text-danger-ink hover:bg-danger-soft/70'
                  : 'bg-ok-soft text-ok-ink'
              }`}
              onClick={() => setDrawer((d) => (d === 'issues' ? null : 'issues'))}
              title={t('issues_tip')}
            >
              {issues.length ? (
                <>
                  <TriangleAlert size={12} aria-hidden />
                  {t('issues_n', { n: issues.length })}
                </>
              ) : (
                <>
                  <span
                    className={issuesCleared ? 'verify-pop inline-flex' : 'inline-flex'}
                    onAnimationEnd={() => setIssuesCleared(false)}
                  >
                    <ShieldCheck size={12} aria-hidden />
                  </span>
                  {t('no_issues')}
                </>
              )}
            </button>
          )}
          <RunControls
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
          {/* Settings is a labeled, first-class control — extraction quality is the
              analyst's main lever, not an afterthought behind an anonymous gear. */}
          <button
            className={`${btnCls} ${drawer === 'settings' ? 'border-primary/50 bg-primary-soft text-primary-strong' : ''}`}
            onClick={() => setDrawer((d) => (d === 'settings' ? null : 'settings'))}
            title={t('settings_tip')}
          >
            <span className={settingsPulse ? 'verify-pop inline-flex' : 'inline-flex'} onAnimationEnd={() => setSettingsPulse(false)}>
              <Settings2 size={ICON} aria-hidden />
            </span>
            {t('settings')}
            {changedSettings > 0 && (
              <span className="rounded-full bg-primary-soft px-1.5 text-2xs font-semibold text-primary-strong">
                {changedSettings}
              </span>
            )}
          </button>
          {/* Global preferences live behind one ⋯ menu — the run group keeps the row. */}
          <span className="mx-1 h-5 w-px bg-line" aria-hidden />
          <button
            className={iconBtnCls}
            onClick={() => setShowPalette(true)}
            aria-label={t('palette_tip')}
            title={t('palette_tip')}
          >
            <Search size={ICON} aria-hidden />
          </button>
          <span className="relative">
            <button
              className={`${iconBtnCls} ${showMore ? 'bg-primary-soft text-primary' : ''}`}
              onClick={() => setShowMore((m) => !m)}
              aria-expanded={showMore}
              aria-label={t('more_menu')}
              title={t('more_menu')}
            >
              <MoreHorizontal size={ICON} aria-hidden />
            </button>
            {showMore && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setShowMore(false)} />
                <div className={`${menuCls} absolute right-0 top-full z-50 mt-1 w-56`}>
                  <button
                    className={`${menuItemCls} flex items-center gap-2`}
                    onClick={() => {
                      setLang(lang === 'en' ? 'km' : 'en')
                      setShowMore(false)
                    }}
                  >
                    <Languages size={ICON_SM} aria-hidden />
                    {lang === 'en' ? 'ភាសាខ្មែរ' : 'English'}
                  </button>
                  <button className={`${menuItemCls} flex items-center gap-2`} onClick={cycleTheme}>
                    {themePref === 'light' ? (
                      <Sun size={ICON_SM} aria-hidden />
                    ) : themePref === 'dark' ? (
                      <Moon size={ICON_SM} aria-hidden />
                    ) : (
                      <Monitor size={ICON_SM} aria-hidden />
                    )}
                    {t(themePref === 'light' ? 'theme_tip_light' : themePref === 'dark' ? 'theme_tip_dark' : 'theme_tip_system')}
                  </button>
                  <button
                    className={`${menuItemCls} flex items-center gap-2`}
                    onClick={() => {
                      setShowMore(false)
                      setShowHelp(true)
                    }}
                  >
                    <CircleHelp size={ICON_SM} aria-hidden />
                    {t('shortcuts')}
                    <kbd className={`${kbdCls} ml-auto`}>?</kbd>
                  </button>
                </div>
              </>
            )}
          </span>
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
      {error || (runError && !wasCancelled) ? (
        <div className="border-b border-line bg-danger-soft px-4 py-1.5 text-sm text-danger-ink">
          {error ?? runError}
        </div>
      ) : wasCancelled && !status.data?.active ? (
        /* A stop the user asked for is not a failure: neutral tone, clear next step. */
        <div className="flex items-center gap-3 border-b border-line bg-rail px-4 py-1.5 text-sm text-ink">
          <Square size={11} className="shrink-0 text-ink-3" fill="currentColor" aria-hidden />
          <span>{t('stopped_msg')}</span>
        </div>
      ) : dupNotice !== null ? (
        <div className="flex items-center gap-3 border-b border-line bg-primary-soft px-4 py-1.5 text-sm text-ink">
          <Files size={14} className="shrink-0 text-primary" aria-hidden />
          <span className="min-w-0 truncate">{t('dup_doc_notice', { name: dupNotice })}</span>
          <button className={`${iconBtnCls} ml-auto h-6 w-6`} onClick={() => setDupNotice(null)} aria-label={t('ks_esc')}>
            <X size={13} aria-hidden />
          </button>
        </div>
      ) : null}

      {/* Stale-settings warning: a floating bottom-left toast instead of a banner —
          it advises without stealing a full-width row. Auto-hides after 10s; the
          wrapper stays mounted so aria-live announces arrivals politely. */}
      <div aria-live="polite" className="pointer-events-none fixed bottom-4 left-4 z-50 flex w-full max-w-[320px] flex-col gap-2">
        {scanToast && (
          <div key={scanToast.n} className="toast-in pointer-events-auto flex items-start gap-2.5 rounded-lg border border-line-strong bg-surface/95 p-3.5 shadow-modal backdrop-blur-md">
            <ScanSearch size={15} className="mt-0.5 shrink-0 text-primary" aria-hidden />
            <p className="min-w-0 text-sm leading-5 text-ink">
              {scanToast.msg}{' '}
              <button
                className="font-medium text-primary underline underline-offset-2"
                onClick={() => {
                  setDrawer('settings')
                  if (scanToast.field) setHighlightFlag((h) => ({ k: scanToast.field!, n: (h?.n ?? 0) + 1 }))
                  setScanToast(null)
                }}
              >
                {t('scan_toast_open')}
              </button>
            </p>
          </div>
        )}
        {stale && !status.data?.active && !staleToastHidden && (
          <div className="toast-in pointer-events-auto w-full rounded-lg border border-warn/25 bg-surface/95 p-3.5 shadow-modal backdrop-blur-md">
            <div className="flex items-start gap-2.5">
              <TriangleAlert size={15} className="mt-0.5 shrink-0 text-warn" aria-hidden />
              <p className="min-w-0 text-sm leading-5 text-ink">
                {t('stale_notice')}{' '}
                <button
                  className="font-medium text-primary underline underline-offset-2"
                  onClick={() => active && run.mutate(active.id)}
                >
                  {t('rerun_now')}
                </button>
              </p>
              <button
                className={`${iconBtnCls} -mr-1.5 -mt-1 h-6 w-6 shrink-0`}
                onClick={() => setStaleToastHidden(true)}
                aria-label={t('ks_esc')}
              >
                <X size={13} aria-hidden />
              </button>
            </div>
          </div>
        )}
      </div>

      <main className="flex min-h-0 flex-1 overflow-hidden p-1.5">
        <QueueRail
          documents={documents}
          activeId={active?.id ?? null}
          onSelect={selectDoc}
          onUpload={uploadFiles}
          onRemove={removeDoc}
          uploading={upload.isPending}
          onRunAll={runAll}
          onRemoveAll={removeAllDocs}
          pipelineBusy={pipelineBusy}
          batchRunning={batchRunning}
          exportAllUrl={documents.some((d) => d.status === 'done') ? api.exportAllUrl(combineExport) : null}
        />

        {/* The workbench: cards sit lifted on the canvas ground, not walled columns. */}
        {active === null ? (
          <EmptyState onUpload={() => uploadRef.current?.click()} />
        ) : !hasResults ? (
          /* Pre-run: show the actual pages, not a placeholder — the analyst needs to
             SEE the document to pick a page range before running. */
          <div className={`${panelMainCls} flex min-w-0 flex-1 flex-col overflow-hidden`}>
            <div className="flex h-10 shrink-0 items-center justify-center gap-1 whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3 text-sm">
              {canvasView === 'single' && (
                <>
                  <button className={iconBtnCls} disabled={pageIdx <= 0} onClick={() => setPageIdx(pageIdx - 1)} aria-label={t('ks_pages')}>
                    ‹
                  </button>
                  <span className="text-ink-2">{t('page_of', { a: pageIdx + 1, b: Math.max(1, active.pages) })}</span>
                  <button
                    className={iconBtnCls}
                    disabled={pageIdx >= active.pages - 1}
                    onClick={() => setPageIdx(pageIdx + 1)}
                    aria-label={t('ks_pages')}
                  >
                    ›
                  </button>
                  <span className="mx-1.5 h-4 w-px bg-line" aria-hidden />
                </>
              )}
              <ViewToggle view={canvasView} onChange={setCanvasView} />
            </div>
            {canvasView === 'grid' ? (
              <PageGrid
                pages={preGridPages}
                pageCount={Math.max(1, active.pages)}
                /* Cleaned renditions land at stage 2, so during a run the grid
                   upgrades each page the moment its processed image exists and
                   falls back to the raw preview until then. */
                imageUrl={preImageUrl}
                fallbackUrl={previewUrl}
                selected={selectedPages}
                onTogglePage={togglePage}
                onOpenPage={openPageFromGrid}
              />
            ) : (
              <div className="min-h-0 flex-1 overflow-auto bg-canvas p-4">
                {/* Same resilient lifecycle as the grid cards (skeleton → retry →
                    never the browser's broken glyph) — this was the last raw <img>
                    on the lazy-rasterized preview endpoint. */}
                <GridThumb
                  key={`${active.id}:${pageIdx}`}
                  src={api.previewImageUrl(active.id, pageIdx)}
                  alt={t('page_of', { a: pageIdx + 1, b: active.pages })}
                  className="relative mx-auto aspect-[3/4] w-full max-w-3xl shadow-overlay"
                />
              </div>
            )}
            <div className="flex h-10 shrink-0 items-center gap-2 whitespace-nowrap border-t border-line-strong/60 bg-surface px-3 text-xs text-ink-2">
              {status.data?.active
                ? status.data.stage || t('working')
                : wasCancelled
                  ? t('stopped_msg')
                  : runError
                    ? <span className="text-danger-ink">{t('failed_msg')}</span>
                    : t('preview_hint')}
            </div>
          </div>
        ) : (
          <>
            <section className={`${panelMainCls} flex min-w-[320px] basis-0 flex-col overflow-hidden`} style={{ flexGrow: split.ratio }}>
              {canvasView === 'grid' ? (
                <>
                  <div className="flex h-10 shrink-0 items-center justify-center whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3">
                    <ViewToggle view={canvasView} onChange={setCanvasView} />
                  </div>
                  {/* Post-analysis grid: ONLY the pages the run processed — excluded
                      pages never render as empty frames. Result index k maps to the
                      k-th processed document page, so cards show the processed
                      rendition and open the right result page. */}
                  <PageGrid
                    pages={processedPages}
                    pageCount={Math.max(1, active.pages)}
                    imageUrl={postImageUrl}
                    fallbackUrl={previewUrl}
                    selected={selectedPages}
                    onTogglePage={togglePage}
                    onOpenPage={openProcessedFromGrid}
                  />
                </>
              ) : pageCount > 0 && (
                <PageViewer
                  view={canvasView}
                  onViewChange={setCanvasView}
                  docId={active.id}
                  pageIdx={pageIdx}
                  pageCount={pageCount}
                  onPageChange={(i) => {
                    setPageIdx(i)
                    setSelectedTable(null)
                    setFlashToken(null)
                    clearBlockLink()
                  }}
                  page={page.data}
                  selectedTable={selectedTable}
                  flyToken={focusCell?.n ?? 0}
                  activeBlock={activeBlock}
                  blockFocus={blockSel?.from === 'text' ? blockSel : null}
                  onBlockClick={canvasPickBlock}
                  onTableClick={(tid) => {
                    setSelectedTable(tid)
                    setFlashToken((f) => ({ tid, n: (f?.n ?? 0) + 1 }))
                  }}
                />
              )}
            </section>
            {/* The divider IS the gap between sheets; hairline appears on hover/drag. */}
            <div
              {...split.dividerProps}
              className={`group relative z-10 w-1.5 shrink-0 cursor-col-resize focus-visible:outline-2 focus-visible:outline-primary ${
                split.dragging ? 'select-none' : ''
              }`}
            >
              <span
                aria-hidden
                className={`absolute inset-y-3 left-1/2 w-0.5 -translate-x-1/2 rounded-full transition-colors duration-150 group-hover:bg-primary group-focus-visible:bg-primary ${
                  split.dragging ? 'bg-primary' : 'bg-transparent'
                }`}
              />
            </div>
            <section className={`${panelMainCls} flex min-w-[360px] basis-0 flex-col overflow-hidden`} style={{ flexGrow: 1 - split.ratio }}>
              {page.isError ? (
                /* An honest failure beats an eternal skeleton: name the problem,
                   offer the retry (trust is the product). */
                <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center">
                  <TriangleAlert size={20} className="text-danger" aria-hidden />
                  <p className="max-w-sm text-sm text-ink-2">
                    {t('page_load_failed', { e: friendlyError(page.error, t('err_unreachable')) })}
                  </p>
                  <button className={btnCls} onClick={() => page.refetch()}>
                    {t('retry')}
                  </button>
                </div>
              ) : page.data ? (
                <Suspense fallback={<TablesSkeleton label={t('loading_tables')} />}>
                  <TablesPanel
                    docId={active.id}
                    pageIdx={pageIdx}
                    page={page.data}
                    selectedTable={selectedTable}
                    onSelectTable={setSelectedTable}
                    flashToken={flashToken}
                    focusCell={focusCell}
                    showFind={showFind}
                    onOpenFind={() => setShowFind(true)}
                    onCloseFind={() => setShowFind(false)}
                    onFocusCell={focusGridCell}
                    activeBlock={activeBlock}
                    blockFocus={blockSel?.from === 'canvas' ? blockSel : null}
                    onSelectBlock={textPickBlock}
                    onHoverBlock={setBlockHover}
                  />
                </Suspense>
              ) : (
                <TablesSkeleton label={t('loading_tables')} />
              )}
            </section>
          </>
        )}

        {/* Drawers arrive as elevated decks: mounted wrappers animate width so the
            sheets compress smoothly instead of jumping. */}
        {/* The WRAPPER is the card: one box owns rounding/border/bg/shadow so the
            clip and the chrome can never disagree (the old two-box split left a
            clipped-shadow fringe + corner slivers at the bottom edge). */}
        <div
          className={`flex shrink-0 overflow-hidden transition-[width,margin-left] duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] motion-reduce:transition-none ${
            drawer === 'issues' && hasResults
              ? 'ml-1.5 w-80 rounded-xl border border-line-strong bg-surface shadow-overlay'
              : 'ml-0 w-0'
          }`}
          aria-hidden={drawer !== 'issues' || !hasResults}
          inert={drawer !== 'issues' || !hasResults}
        >
          {hasResults && (
            <IssuesDrawer
              issues={issues}
              currentIdx={issueIdx}
              onJump={jumpToIssue}
              onDismiss={dismissIssue}
              onClose={() => setDrawer(null)}
            />
          )}
        </div>
        {showPalette && (
          <CommandPalette commands={commands} onClose={() => setShowPalette(false)} />
        )}
        {showHelp && (
          <div className="backdrop-enter fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setShowHelp(false)}>
            <div
              ref={helpRef}
              role="dialog"
              aria-modal="true"
              aria-label={t('shortcuts')}
              tabIndex={-1}
              // overlay-enter, like every other floating surface. It previously
              // animated ONLY through a view transition, which meant a browser
              // without that API showed no entrance at all while every other
              // overlay animated — and where it was supported it ran at the ~250ms
              // default rather than the workspace's 90ms.
              className="overlay-enter w-96 rounded-xl border border-line bg-raised p-5 shadow-modal focus:outline-none"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="mb-3 text-sm font-semibold text-ink">{t('shortcuts')}</h2>
              <table className="w-full text-sm text-ink-2">
                <tbody>
                  {/* Each shortcut is a list of combos; each combo a list of atomic
                      keys. Keys render as adjacent caps that never wrap mid-combo,
                      alternatives joined by a thin "/". Fixed key column keeps the
                      descriptions aligned regardless of combo length. */}
                  {(
                    [
                      [[['⌘', 'K']], t('ks_palette')],
                      [[['R']], t('ks_run')],
                      [[['←'], ['→']], t('ks_pages')],
                      [[['N'], ['P']], t('ks_issue')],
                      [[['V']], t('ks_verify')],
                      [[['⌘', 'F']], t('ks_find')],
                      [[['⌘', 'Z'], ['⇧', '⌘', 'Z']], t('ks_undo')],
                      [[[t('ks_rowmenu_key')]], t('ks_rowmenu')],
                      [[['Esc']], t('ks_esc')],
                      [[['?']], t('ks_overlay')],
                    ] as const
                  ).map(([combos, desc], i) => (
                    <tr key={i} className="border-b border-line last:border-0">
                      <td className="w-32 py-1.5 pr-3 align-top">
                        <span className="flex flex-wrap items-center gap-1">
                          {combos.map((keys, ci) => (
                            <span key={ci} className="flex items-center gap-1 whitespace-nowrap">
                              {ci > 0 && <span className="px-0.5 text-ink-3">/</span>}
                              {keys.map((part, ki) => (
                                <kbd key={ki} className={kbdCls}>{part}</kbd>
                              ))}
                            </span>
                          ))}
                        </span>
                      </td>
                      <td className="py-1.5 align-top">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        <div
          className={`flex shrink-0 overflow-hidden transition-[width,margin-left] duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] motion-reduce:transition-none ${
            drawer === 'settings'
              ? 'ml-1.5 w-96 rounded-xl border border-line-strong bg-surface shadow-overlay max-[1279px]:w-80'
              : 'ml-0 w-0'
          }`}
          aria-hidden={drawer !== 'settings'}
          inert={drawer !== 'settings'}
        >
          <SettingsDrawer
            settings={runSettings}
            onChange={setRunSettings}
            engines={meta.data?.engines ?? []}
            engine={engine}
            onEngineChange={setEngine}
            disabled={pipelineBusy}
            checks={suggestion.data?.checks ?? []}
            scores={suggestion.data?.scores ?? null}
            auto={autoApplied}
            onAutoOverride={clearAuto}
            highlight={highlightFlag}
            effectiveEngine={status.data?.effective_engine ?? null}
            effectiveDpi={status.data?.effective_dpi ?? null}
            pageCount={active?.pages ?? 0}
            onClose={() => setDrawer(null)}
          />
        </div>
      </main>
    </div>
  )
}

/** Content-shaped placeholder while the tables chunk loads (lazy import) or the
    page's tables are still fetching — a skeleton, not a spinner-in-content. */
function TablesSkeleton(props: { label: string }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 p-4" aria-busy="true" aria-label={props.label}>
      <span className="sr-only">{props.label}</span>
      {[0, 1].map((card) => (
        <div key={card} className="overflow-hidden rounded-lg border border-line-strong/50">
          <div className="h-8 animate-pulse bg-rail/60" />
          <div className="divide-y divide-line/70">
            {[0, 1, 2, 3].map((row) => (
              <div key={row} className="flex gap-3 p-2">
                {[0, 1, 2].map((cell) => (
                  <div key={cell} className="h-3.5 flex-1 animate-pulse rounded bg-rail/50" style={{ animationDelay: `${(row + cell) * 60}ms` }} />
                ))}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function EmptyState(props: { onUpload: () => void }) {
  const { t } = useT()
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-7">
      {/* A quiet page-with-table sketch in the product's own vocabulary. */}
      <svg width="88" height="104" viewBox="0 0 88 104" aria-hidden className="text-line-strong">
        <rect x="10" y="6" width="68" height="88" rx="4" fill="var(--color-surface)" stroke="currentColor" strokeWidth="1.5" />
        <line x1="20" y1="20" x2="56" y2="20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        <line x1="20" y1="28" x2="44" y2="28" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <rect x="20" y="40" width="48" height="40" rx="2" fill="none" stroke="var(--color-primary)" strokeWidth="1.5" opacity="0.55" />
        <line x1="20" y1="53" x2="68" y2="53" stroke="var(--color-primary)" strokeWidth="1" opacity="0.55" />
        <line x1="20" y1="66" x2="68" y2="66" stroke="var(--color-primary)" strokeWidth="1" opacity="0.55" />
        <line x1="40" y1="40" x2="40" y2="80" stroke="var(--color-primary)" strokeWidth="1" opacity="0.55" />
        <line x1="55" y1="40" x2="55" y2="80" stroke="var(--color-primary)" strokeWidth="1" opacity="0.55" />
      </svg>
      <div className="text-center">
        <p className="text-xl font-semibold tracking-[-0.015em] text-ink" style={{ textWrap: 'balance' }}>
          {t('empty_title')}
        </p>
        <p className="mt-1 text-sm text-ink-2">{t('empty_sub')}</p>
      </div>
      {/* A real 3-step sequence — the numbers carry order, not decoration. */}
      <ol className="space-y-3 text-sm text-ink-2">
        {(
          [
            [t('step_upload'), '1'],
            [t('step_run'), '2'],
            [t('step_review'), '3'],
          ] as const
        ).map(([step, n]) => (
          <li key={n} className="flex items-center gap-3">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary-soft text-xs font-semibold text-primary">
              {n}
            </span>
            {step}
          </li>
        ))}
      </ol>
      <button className={`${primaryBtnCls} h-9 px-5`} onClick={props.onUpload}>
        <FileUp size={ICON} aria-hidden />
        {t('upload_documents')}
      </button>
    </div>
  )
}
