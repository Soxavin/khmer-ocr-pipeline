import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, Info, Search, Undo2, X } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { TableEditor } from './TableEditor'
import { PageTextPanel } from './PageTextPanel'
import { useT } from '../../i18n.tsx'
import { bandCells, nextInBand, type Band } from '../../lib/confidence'
import { btnSmCls, chipCls, iconBtnCls, inputCls } from '../../ui'

// Triage bands, most-urgent first, with their semantic chip + dot styling.
const BAND_ORDER: Band[] = ['check', 'skim', 'clean']
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
}) {
  const { docId, pageIdx, page, selectedTable, onSelectTable, flashToken, focusCell, showFind, onOpenFind, onCloseFind, onFocusCell } = props
  const qc = useQueryClient()
  const { t } = useT()
  const findRef = useRef<HTMLInputElement>(null)
  const [find, setFind] = useState('')
  const [repl, setRepl] = useState('')
  const [findMsg, setFindMsg] = useState<string | null>(null)
  useEffect(() => {
    if (showFind) findRef.current?.focus()
  }, [showFind])

  const [canUndoReplace, setCanUndoReplace] = useState(false)

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

  // Grid-cell confidence bands + a per-band jump cursor (§2.70). Clicking a chip
  // cycles focus to the next cell in that band; the counter shows the position.
  const bands = useMemo(() => bandCells(page.tables), [page.tables])
  const [cursor, setCursor] = useState<Record<Band, number>>({ check: -1, skim: -1, clean: -1 })
  // Band contents differ per page/doc — reset the cursors so they never index
  // out of range on a page turn.
  useEffect(() => setCursor({ check: -1, skim: -1, clean: -1 }), [docId, pageIdx])
  const jumpBand = (b: Band) => {
    const list = bands[b]
    if (!list.length) return
    const next = nextInBand(list.length, Math.min(cursor[b], list.length - 1))
    setCursor((c) => ({ ...c, [b]: next }))
    const cell = list[next]
    onFocusCell(cell.table_id, cell.row, cell.col)
  }

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

        {/* ── Triage ── clickable band chips; click cycles to the next cell in the
            band and flies the page image. Colour is redundant: each chip shows the
            band word + count (+ live position while cycling) and an aria-label. */}
        <span className="flex shrink-0 items-center gap-1.5">
          {BAND_ORDER.map((b) => {
            const total = bands[b].length
            if (total === 0) return null
            const style = BAND_STYLE[b]
            const active = cursor[b] >= 0
            return (
              <button
                key={b}
                className={`${chipCls} ${style.chip}`}
                onClick={() => jumpBand(b)}
                aria-label={t(`triage_aria_${b}` as Parameters<typeof t>[0], { n: total })}
                title={t('triage_tip')}
              >
                <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${style.dot}`} aria-hidden />
                {active ? t('band_progress', { band: t(`band_${b}` as Parameters<typeof t>[0]), i: cursor[b] + 1, n: total })
                        : <>{total} {t(`band_${b}` as Parameters<typeof t>[0])}</>}
              </button>
            )
          })}
        </span>

        {/* ── Grid utilities ── Find + content size. */}
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
          <span className="mx-0.5 h-4 w-px bg-line" aria-hidden />
          <span className="flex items-center gap-1" title={t('size_tip')}>
            <button className={btnSmCls} onClick={() => setSize(Math.max(11, size - 1))} aria-label={t('size_smaller')}>
              A−
            </button>
            <span className="tabular-nums text-ink-2">{size}px</span>
            <button className={btnSmCls} onClick={() => setSize(Math.min(24, size + 1))} aria-label={t('size_larger')}>
              A+
            </button>
          </span>
        </span>
      </div>

      {showFind && (
        <div className="flex flex-wrap items-center gap-2 border-b border-line bg-rail px-3 py-1.5 text-sm">
          <input ref={findRef} className={`${inputCls} khmer-content w-36`}
                 placeholder={t('find_ph')} value={find}
                 onChange={(e) => setFind(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && doReplace()} />
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
            />
          </div>
        ))}

        {(page.corrected_text || text) && (
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
          />
        )}
      </div>
    </div>
  )
}
