import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, ChevronUp, Eye, EyeOff, Info, Search, SlidersHorizontal, Undo2, X } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { TableEditor } from './TableEditor'
import { PageTextPanel } from './PageTextPanel'
import { AnchoredMenu } from '../AnchoredMenu'
import { useT } from '../../i18n.tsx'
import { bandCells, type Band } from '../../lib/confidence'
import { findMatches } from '../../lib/search'
import { btnSmCls, ICON_SM, iconBtnCls, inputCls, menuItemCls } from '../../ui'

// Stepper inside the match cluster — same geometry and states as the page
// viewer's zoom segment, so the two clusters are visibly one control family.
const stepBtn =
  'inline-flex h-6 items-center px-1.5 text-ink-2 transition-colors duration-150 ' +
  'hover:bg-rail hover:text-ink disabled:opacity-40 disabled:pointer-events-none ' +
  'focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary'

// The tinted tiers, in the order the colour key lists them. Clean cells carry no
// wash, so they're not part of the tint legend.
const TINTED_BANDS: Band[] = ['check', 'skim']
const BAND_STYLE: Record<Band, { dot: string; chip: string }> = {
  check: { dot: 'bg-danger', chip: 'bg-danger-soft text-danger-ink hover:bg-danger-soft/70' },
  skim: { dot: 'bg-warn', chip: 'bg-warn-soft text-warn-ink hover:bg-warn-soft/70' },
  clean: { dot: 'bg-ok', chip: 'bg-ok-soft text-ok-ink hover:bg-ok-soft/70' },
}

function useKhmerSize(): [number, (n: number) => void] {
  const [size, setSize] = useState(() => Number(localStorage.getItem('khmerSize') ?? 14))
  useEffect(() => {
    document.documentElement.style.setProperty('--khmer-size', `${size}px`)
    localStorage.setItem('khmerSize', String(size))
  }, [size])
  return [size, setSize]
}

// Page-level preference: whether the confidence marks are shown. Persisted like
// khmerSize so a de-noised read survives page turns and reloads. Default on —
// the model's uncertainty is the whole point of the review loop.
function useShowConf(): [boolean, () => void] {
  const [on, setOn] = useState(() => localStorage.getItem('showConf') !== 'false')
  const toggle = () =>
    setOn((v) => {
      localStorage.setItem('showConf', String(!v))
      return !v
    })
  return [on, toggle]
}

export function TablesPanel(props: {
  docId: string
  pageIdx: number
  page: PageData
  selectedTable: string | null
  onSelectTable: (tableId: string | null) => void
  flashToken: { tid: string; n: number } | null
  focusCell: { tid: string; row: number; col: number; n: number } | null
  showFind: boolean
  onOpenFind: () => void
  onCloseFind: () => void
  /** Focus a grid cell (triage-band jump): selects, flashes, scrolls + flies the page. */
  onFocusCell: (tid: string, row: number, col: number) => void
  /** Page-text block link — passed straight through to PageTextPanel. */
  activeBlock?: number | null
  blockFocus?: { i: number; n: number } | null
  onSelectBlock?: (i: number) => void
  onHoverBlock?: (i: number | null) => void
  /** A drawer issue being hovered — its cell, on whichever table it belongs to. */
  hoverCell?: { tid: string; row: number; col: number } | null
}) {
  const { docId, pageIdx, page, selectedTable, onSelectTable, flashToken, focusCell, showFind, onOpenFind, onCloseFind, onFocusCell, activeBlock = null, blockFocus = null, onSelectBlock, onHoverBlock, hoverCell = null } = props
  const qc = useQueryClient()
  const { t } = useT()
  const findRef = useRef<HTMLInputElement>(null)
  const viewAnchor = useRef<HTMLSpanElement>(null)
  const [showView, setShowView] = useState(false)
  const [find, setFind] = useState('')
  const [repl, setRepl] = useState('')
  const [findMsg, setFindMsg] = useState<string | null>(null)
  useEffect(() => {
    if (showFind) findRef.current?.focus()
  }, [showFind])

  const [canUndoReplace, setCanUndoReplace] = useState(false)

  // Live search over THIS page's tables — the only cells the panel can scroll to,
  // so every counted match is reachable. (Replace-all stays document-wide and says
  // so in its own confirm; the two scopes are deliberately distinct, not conflated.)
  const matches = useMemo(() => findMatches(page.tables, find), [page.tables, find])
  const [cursorIdx, setCursorIdx] = useState(0)
  // A new query, page, or document invalidates the old position — without this the
  // counter could read "7 / 3" after the match list shrinks under it.
  useEffect(() => setCursorIdx(0), [find, docId, pageIdx])
  const stepMatch = useCallback(
    (dir: 1 | -1) => {
      if (!matches.length) return
      // Wraps both ways: cycling is the expected behaviour of a find bar, and
      // dead-ending at the last match makes the analyst re-type to start over.
      const next = (cursorIdx + dir + matches.length) % matches.length
      setCursorIdx(next)
      const m = matches[next]
      // Reuses the triage jump: selects the table, flashes it, scrolls the cell
      // into view and flies the page image to the matching region.
      onFocusCell(m.table_id, m.row, m.col)
    },
    [matches, cursorIdx, onFocusCell],
  )
  const activeMatch = matches[cursorIdx] ?? null

  // Replace-all is the only bulk mutation and it can touch pages the analyst has
  // not reviewed — so it confirms with a count first and stays undoable after.
  const doReplace = () => {
    if (!find) return
    const ok = window.confirm(t('replace_confirm', { a: find, b: repl }))
    if (!ok) return
    api
      .replace(docId, find, repl)
      .then((r) => {
        setFindMsg(r.total ? t('replaced_msg', { n: r.total, t: r.tables_changed }) : t('no_matches'))
        setCanUndoReplace(r.total > 0)
        if (r.total) {
          qc.invalidateQueries({ queryKey: ['page'] })
          qc.invalidateQueries({ queryKey: ['lowconf'] })
        }
      })
      .catch((e) => setFindMsg(t('replace_failed', { e: e instanceof Error ? e.message : String(e) })))
  }

  const undoReplace = () => {
    api
      .undoReplace(docId)
      .then(() => {
        setFindMsg(t('replace_undone'))
        setCanUndoReplace(false)
        qc.invalidateQueries({ queryKey: ['page'] })
        qc.invalidateQueries({ queryKey: ['lowconf'] })
      })
      .catch((e) => setFindMsg(t('undo_failed', { e: e instanceof Error ? e.message : String(e) })))
  }
  // Shown until the analyst dismisses it once, then gone for good on this machine.
  const [showIntro, setShowIntro] = useState(() => localStorage.getItem('reviewIntroDone') !== 'true')
  const dismissIntro = () => {
    localStorage.setItem('reviewIntroDone', 'true')
    setShowIntro(false)
  }

  const [size, setSize] = useKhmerSize()
  const [showConf, toggleConf] = useShowConf()
  const [text, setText] = useState(page.corrected_text)
  const [textSaved, setTextSaved] = useState(true)
  // Keyed on the TEXT, not the page object: any refetch (a verify flips one
  // boolean, replace-all touches grids) produces a new page identity, and an
  // object-keyed reset would silently discard the analyst's unsaved draft.
  // Structural sharing keeps corrected_text's identity until it truly changes.
  useEffect(() => {
    setText(page.corrected_text)
    setTextSaved(true)
  }, [page.corrected_text])

  // Per-page confidence bands — a passive health gauge + the colour legend for the
  // cell tints. Not a worklist: the Issues drawer (+ n/p) is the one place cells are
  // stepped and actioned, so the bands no longer cycle (§2.89 — that removed the
  // duplicate stepper and its cursor state). The counts stay objective (model
  // confidence), independent of what's been dismissed.
  const bands = useMemo(() => bandCells(page.tables), [page.tables])

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Utility header, three zones: page facts (left) · interactive triage bands
          (center) · grid utilities (right). One fixed row; overflow scrolls. */}
      <div className="flex h-10 shrink-0 items-center justify-between gap-3 whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3 text-xs">
        {/* ── Facts ── numbers first, then the page-level verify action. */}
        <span className="flex min-w-0 shrink items-center gap-4 overflow-hidden">
          <span className="shrink-0 text-ink-2">
            <strong className="text-sm font-semibold text-ink">{page.tables.length}</strong>{' '}
            {page.tables.length === 1 ? t('tables_one') : t('tables_other')}
          </span>
          <span className="hidden shrink-0 text-ink-2 sm:inline">
            <strong className="text-sm font-semibold text-ink">{page.text_blocks.length}</strong>{' '}
            {page.text_blocks.length === 1 ? t('blocks_one') : t('blocks_other')}
          </span>
          {page.tables.length > 1 && page.tables.some((t) => !t.verified) && (
            <button
              className={btnSmCls}
              title={t('verify_page_tip')}
              onClick={() => {
                const unverified = page.tables.filter((t) => !t.verified)
                Promise.all(unverified.map((t) => api.review(docId, t.table_id, true)))
                  .then(() => {
                    qc.invalidateQueries({ queryKey: ['documents'] })
                    qc.invalidateQueries({ queryKey: ['page'] })
                  })
                  .catch(() => undefined)
              }}
            >
              <Check size={12} aria-hidden />
              {t('verify_page')}
            </button>
          )}
        </span>

        {/* ── Tint legend ── the colour KEY for the cell washes (rose Check · amber
            Skim), not a count. The per-page cell worklist is the Issues drawer + n/p
            (its top-bar chip owns the number). Clean cells carry no wash, so the key
            lists only the tinted tiers. Shown only when this page has tinted cells.
            The show/hide toggle now lives in the View-options popover (§distill). */}
        {bands.check.length + bands.skim.length > 0 && (
          <span className="flex shrink-0 items-center gap-2.5" aria-label={t('conf_toggle')}>
            {TINTED_BANDS.map((b) => {
              const style = BAND_STYLE[b]
              return (
                <span key={b} className="inline-flex items-center gap-1.5 text-ink-2">
                  <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${style.dot}`} aria-hidden />
                  {t(`band_${b}` as Parameters<typeof t>[0])}
                </span>
              )
            })}
          </span>
        )}

        {/* ── Grid utilities ── Find + a View-options popover. The font-size stepper
            and the confidence-tint toggle moved into the popover so the header holds
            one primary control (Find) plus a single overflow (§distill). */}
        <span className="flex shrink-0 items-center gap-1">
          <button
            className={`${btnSmCls} ${showFind ? 'border-primary/50 bg-primary-soft text-primary-strong' : ''}`}
            onClick={showFind ? onCloseFind : onOpenFind}
            aria-pressed={showFind}
            title={t('find_tip')}
          >
            <Search size={12} aria-hidden />
            {t('find_btn')}
          </button>
          <span ref={viewAnchor} className="relative">
            <button
              className={`${btnSmCls} ${showView ? 'border-primary/50 bg-primary-soft text-primary-strong' : ''}`}
              onClick={() => setShowView((v) => !v)}
              aria-haspopup="menu"
              aria-expanded={showView}
              title={t('view_options_tip')}
            >
              <SlidersHorizontal size={12} aria-hidden />
              {t('view_options')}
            </button>
            {showView && (
              <AnchoredMenu anchorRef={viewAnchor} onClose={() => setShowView(false)} width="w-64" className="p-1.5">
                {/* Confidence tints: a reversible view, never a state change — the
                    model's signal is never erased, only hidden. */}
                <button
                  className={`${menuItemCls} flex items-center gap-2`}
                  role="switch"
                  aria-checked={showConf}
                  onClick={toggleConf}
                  title={t('conf_toggle_tip')}
                >
                  {showConf ? <Eye size={ICON_SM} aria-hidden /> : <EyeOff size={ICON_SM} aria-hidden />}
                  <span className="min-w-0 flex-1 truncate">{t('conf_toggle')}</span>
                  <span className={`text-2xs font-semibold ${showConf ? 'text-primary-strong' : 'text-ink-3'}`}>
                    {showConf ? t('toggle_on') : t('toggle_off')}
                  </span>
                </button>
                <div className="mt-0.5 flex items-center justify-between gap-2 px-2 py-1.5" title={t('size_tip')}>
                  <span className="text-sm text-ink-2">{t('text_size')}</span>
                  <span className="flex items-center gap-1">
                    <button className={btnSmCls} onClick={() => setSize(Math.max(11, size - 1))} aria-label={t('size_smaller')}>
                      A−
                    </button>
                    <span className="w-9 text-center tabular-nums text-ink-2">{size}px</span>
                    <button className={btnSmCls} onClick={() => setSize(Math.min(24, size + 1))} aria-label={t('size_larger')}>
                      A+
                    </button>
                  </span>
                </div>
              </AnchoredMenu>
            )}
          </span>
        </span>
      </div>

      {showFind && (
        <div className="flex flex-wrap items-center gap-2 border-b border-line bg-rail px-3 py-1.5 text-sm">
          <input ref={findRef} className={`${inputCls} khmer-content w-36`}
                 placeholder={t('find_ph')} value={find}
                 onChange={(e) => setFind(e.target.value)}
                 // Enter steps through matches. It used to fire replace-all — a
                 // document-wide mutation on Enter-while-typing, now that this is a
                 // live search field, is the wrong reflex to arm.
                 onKeyDown={(e) => {
                   if (e.key !== 'Enter') return
                   e.preventDefault()
                   stepMatch(e.shiftKey ? -1 : 1)
                 }} />
          {/* Counter + steppers share one bordered track — the same cluster the
              page viewer's zoom control uses, so the two read as one vocabulary. */}
          {find.trim() !== '' && (
            <span className="flex shrink-0 items-center overflow-hidden rounded-md border border-line-strong"
                  role="group" aria-label={t('find_matches_aria')}>
              <span
                className={`inline-flex h-6 min-w-[3.25rem] items-center justify-center px-1.5 text-xs font-medium tabular-nums ${
                  matches.length ? 'text-ink-2' : 'text-ink-3'
                }`}
                aria-live="polite"
              >
                {matches.length ? `${cursorIdx + 1} / ${matches.length}` : t('find_none')}
              </span>
              <span className="h-4 w-px self-center bg-line" aria-hidden />
              <button className={stepBtn} disabled={!matches.length} onClick={() => stepMatch(-1)}
                      aria-label={t('find_prev')} title={t('find_prev')}>
                <ChevronUp size={ICON_SM} aria-hidden />
              </button>
              <button className={stepBtn} disabled={!matches.length} onClick={() => stepMatch(1)}
                      aria-label={t('find_next')} title={t('find_next')}>
                <ChevronDown size={ICON_SM} aria-hidden />
              </button>
            </span>
          )}
          <input className={`${inputCls} khmer-content w-36`}
                 placeholder={t('replace_ph')} value={repl}
                 onChange={(e) => setRepl(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && doReplace()} />
          <button className={btnSmCls} disabled={!find} onClick={doReplace}
                  title={t('replace_all_tip')}>
            {t('replace_all_btn')}
          </button>
          {canUndoReplace && (
            <button className={btnSmCls} onClick={undoReplace} title={t('undo_replace_tip')}>
              <Undo2 size={13} aria-hidden />
              {t('undo_replace')}
            </button>
          )}
          {findMsg && <span className="text-xs text-ink-2">{findMsg}</span>}
          <button className={`${iconBtnCls} ml-auto`} onClick={onCloseFind} aria-label={t('close_find')}>
            <X size={14} aria-hidden />
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
        {page.tables.length === 0 && (
          <p className="py-6 text-center text-sm text-ink-2">{t('no_tables')}</p>
        )}
        {/* First-run guidance: names the loop in place, once, then never again.
            No tour, no coach marks — the analyst is already mid-task. */}
        {showIntro && page.tables.length > 0 && (
          <div className="flex items-start gap-2 rounded-lg border border-primary/20 bg-primary-soft px-3 py-2 text-xs text-ink">
            <Info size={14} className="mt-0.5 shrink-0 text-primary" aria-hidden />
            <p className="min-w-0">
              <span className="font-medium">{t('intro_title')}</span> {t('intro_body')}
              <span className="font-medium">{t('intro_verify')}</span>
              {t('intro_tail')}
            </p>
            <button className={`${iconBtnCls} shrink-0`} onClick={dismissIntro} aria-label={t('dismiss_tip')}>
              <X size={13} aria-hidden />
            </button>
          </div>
        )}
        {page.tables.map((t, i) => (
          /* Cards settle onto the sheet with a short stagger when the page changes
             (keyed remount fires the animation; base state stays visible). */
          <div
            key={`${docId}:${pageIdx}:${t.table_id}`}
            className="sheet-in"
            style={{ '--sheet-delay': `${Math.min(i, 6) * 25}ms` } as React.CSSProperties}
          >
            <TableEditor
              docId={docId}
              table={t}
              focused={selectedTable === t.table_id}
              onFocus={() => onSelectTable(t.table_id)}
              flash={flashToken?.tid === t.table_id ? flashToken.n : 0}
              focusCell={focusCell?.tid === t.table_id ? focusCell : null}
              findQuery={find}
              activeMatch={activeMatch?.table_id === t.table_id ? activeMatch : null}
              showConf={showConf}
              hoverCell={hoverCell?.tid === t.table_id ? hoverCell : null}
            />
          </div>
        ))}

        {/* raw_ocr_text also gates this panel: a trial engine (e.g. qwen_ardb)
            can leave corrected_text/text empty but still have raw output worth
            showing — see PageTextPanel's empty-state fallback. */}
        {(page.corrected_text || text || page.raw_ocr_text) && (
          <PageTextPanel
            docId={docId}
            pageIdx={pageIdx}
            page={page}
            text={text}
            onTextChange={(v) => {
              setText(v)
              setTextSaved(false)
            }}
            saved={textSaved}
            onSaved={() => setTextSaved(true)}
            activeBlock={activeBlock}
            blockFocus={blockFocus}
            onSelectBlock={onSelectBlock}
            onHoverBlock={onHoverBlock}
          />
        )}
      </div>
    </div>
  )
}
