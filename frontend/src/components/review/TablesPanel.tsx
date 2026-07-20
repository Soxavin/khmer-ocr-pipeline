import { useEffect, useRef, useState } from 'react'
import { Check, Info, Undo2, X } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { TableEditor } from './TableEditor'
import { useT } from '../../i18n.tsx'
import { btnSmCls, iconBtnCls, inputCls } from '../../ui'

const CONF_LOW = 0.5 // text-block bucket (model_config.py CONFIDENCE_LOW)

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
  onCloseFind: () => void
}) {
  const { docId, pageIdx, page, selectedTable, onSelectTable, flashToken, focusCell, showFind, onCloseFind } = props
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
  useEffect(() => {
    setText(page.corrected_text)
    setTextSaved(true)
  }, [page])

  const lowConfBlocks = page.text_blocks.filter((b) => (b.confidence ?? 0) < CONF_LOW).length

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Per-page quality banner + content size control. Numbers carry the weight;
          labels recede — the banner should scan as facts, not a gray sentence. */}
      {/* Utility header: facts anchored left, type-size controls right, one fixed row.
          Overflow scrolls sideways — the header never grows a second line. */}
      <div className="flex h-10 shrink-0 items-center justify-between gap-3 whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3 text-xs">
        <span className="flex min-w-0 items-center gap-4 overflow-hidden">
        {/* Facts read as figures first: numbers a step larger and heavier than labels. */}
        <span className="shrink-0 text-ink-2">
          <strong className="text-sm font-semibold text-ink">{page.tables.length}</strong>{' '}
          {page.tables.length === 1 ? t('tables_one') : t('tables_other')}
        </span>
        <span className="shrink-0 text-ink-2">
          <strong className="text-sm font-semibold text-ink">{page.text_blocks.length}</strong>{' '}
          {page.text_blocks.length === 1 ? t('blocks_one') : t('blocks_other')}
        </span>
        {lowConfBlocks > 0 && (
          <span className="font-semibold text-danger-ink">{t('n_lowconf', { n: lowConfBlocks })}</span>
        )}
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
        {/* Legend lives in the header as micro swatches — only when this page has tinted cells. */}
        {page.tables.some((t) => t.confidence.some((row) => row.some((v) => v !== null && v < 0.95))) && (
          <span className="hidden shrink-0 items-center gap-3 text-ink-2 md:flex">
            <span className="flex shrink-0 items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px] border border-danger/25" style={{ background: 'var(--cell-low-bg)' }} aria-hidden />
              <span className="whitespace-nowrap">
                <span className="underline decoration-danger-ink decoration-dotted underline-offset-2">{t('legend_check_pct')}</span> {t('legend_check')}
              </span>
            </span>
            <span className="flex shrink-0 items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px] border border-warn/25" style={{ background: 'var(--cell-mid-bg)' }} aria-hidden />
              <span className="whitespace-nowrap">{t('legend_skim')}</span>
            </span>
          </span>
        )}
        </span>
        <span className="flex shrink-0 items-center gap-1" title={t('size_tip')}>
          <button className={btnSmCls} onClick={() => setSize(Math.max(11, size - 1))} aria-label={t('size_smaller')}>
            A−
          </button>
          <span className="text-ink-2">{size}px</span>
          <button className={btnSmCls} onClick={() => setSize(Math.min(24, size + 1))} aria-label={t('size_larger')}>
            A+
          </button>
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
          <details className="rounded-lg border border-line bg-surface p-2 shadow-raised">
            <summary className="cursor-pointer text-sm font-medium text-ink-2">
              {t('page_text')}{textSaved ? '' : t('unsaved_suffix')}
            </summary>
            <textarea
              className="khmer-content mt-2 w-full rounded-md border border-line-strong p-2 text-ink"
              rows={8}
              value={text}
              onChange={(e) => {
                setText(e.target.value)
                setTextSaved(false)
              }}
            />
            <button
              className={`${btnSmCls} mt-1`}
              disabled={textSaved}
              onClick={() => api.putPageText(docId, pageIdx, text).then(() => setTextSaved(true))}
            >
              {t('save_text')}
            </button>
          </details>
        )}
      </div>
    </div>
  )
}
