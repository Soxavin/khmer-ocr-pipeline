import { useEffect, useRef, useState } from 'react'
import { Check, Info, Undo2, X } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { TableEditor } from './TableEditor'
import { btnSmCls, iconBtnCls } from '../../ui'

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
    const ok = window.confirm(
      `Replace “${find}” with “${repl}” in every table of this document — including pages you have not reviewed?\n\nYou can undo this immediately afterwards.`,
    )
    if (!ok) return
    api
      .replace(docId, find, repl)
      .then((r) => {
        setFindMsg(r.total ? `Replaced ${r.total} occurrence(s) in ${r.tables_changed} table(s).` : 'No matches found.')
        setCanUndoReplace(r.total > 0)
        if (r.total) {
          qc.invalidateQueries({ queryKey: ['page'] })
          qc.invalidateQueries({ queryKey: ['lowconf'] })
        }
      })
      .catch((e) => setFindMsg(`Replace failed: ${e instanceof Error ? e.message : String(e)}`))
  }

  const undoReplace = () => {
    api
      .undoReplace(docId)
      .then(() => {
        setFindMsg('Replace undone.')
        setCanUndoReplace(false)
        qc.invalidateQueries({ queryKey: ['page'] })
        qc.invalidateQueries({ queryKey: ['lowconf'] })
      })
      .catch((e) => setFindMsg(`Undo failed: ${e instanceof Error ? e.message : String(e)}`))
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
      <div className="flex items-center gap-4 border-b border-slate-200 bg-white px-3 py-1.5 text-xs">
        <span className="text-slate-500">
          <strong className="font-semibold text-slate-800">{page.tables.length}</strong>{' '}
          table{page.tables.length === 1 ? '' : 's'}
        </span>
        <span className="text-slate-500">
          <strong className="font-semibold text-slate-800">{page.text_blocks.length}</strong>{' '}
          text block{page.text_blocks.length === 1 ? '' : 's'}
        </span>
        {lowConfBlocks > 0 && (
          <span className="font-semibold text-conf-low">{lowConfBlocks} low-confidence</span>
        )}
        {page.tables.length > 1 && page.tables.some((t) => !t.verified) && (
          <button
            className={btnSmCls}
            title="Mark every table on this page as reviewed"
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
            Verify page
          </button>
        )}
        <span className="ml-auto flex items-center gap-1" title="Content text size (Khmer legibility)">
          <button className={btnSmCls} onClick={() => setSize(Math.max(11, size - 1))} aria-label="Smaller content text">
            A−
          </button>
          <span className="text-slate-600">{size}px</span>
          <button className={btnSmCls} onClick={() => setSize(Math.min(24, size + 1))} aria-label="Larger content text">
            A+
          </button>
        </span>
      </div>

      {/* Legend on its own quiet line — only when this page actually has tinted cells. */}
      {page.tables.some((t) => t.confidence.some((row) => row.some((v) => v !== null && v < 0.95))) && (
        <div className="flex items-center gap-4 border-b border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 shrink-0 rounded-sm border border-red-200" style={{ background: '#fdecec' }} aria-hidden />
            <span className="whitespace-nowrap">
              <span className="underline decoration-red-700 decoration-dotted underline-offset-2">below 80%</span> — check
            </span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 shrink-0 rounded-sm border border-amber-200" style={{ background: '#fffcf0' }} aria-hidden />
            <span className="whitespace-nowrap">80–95% — skim</span>
          </span>
        </div>
      )}

      {showFind && (
        <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-blue-50/50 px-3 py-1.5 text-sm">
          <input ref={findRef} className="khmer-content w-36 rounded border border-slate-300 px-2 py-0.5"
                 placeholder="Find…" value={find}
                 onChange={(e) => setFind(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && doReplace()} />
          <input className="khmer-content w-36 rounded border border-slate-300 px-2 py-0.5"
                 placeholder="Replace with…" value={repl}
                 onChange={(e) => setRepl(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && doReplace()} />
          <button className={btnSmCls} disabled={!find} onClick={doReplace}
                  title="Replace in every table of this document">
            Replace in all tables
          </button>
          {canUndoReplace && (
            <button className={btnSmCls} onClick={undoReplace} title="Restore the tables to before the replace">
              <Undo2 size={13} aria-hidden />
              Undo replace
            </button>
          )}
          {findMsg && <span className="text-xs text-slate-600">{findMsg}</span>}
          <button className={`${iconBtnCls} ml-auto`} onClick={onCloseFind} aria-label="Close find and replace">
            <X size={14} aria-hidden />
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-3">
        {page.tables.length === 0 && (
          <p className="py-6 text-center text-sm text-slate-500">No tables on this page.</p>
        )}
        {/* First-run guidance: names the loop in place, once, then never again.
            No tour, no coach marks — the analyst is already mid-task. */}
        {showIntro && page.tables.length > 0 && (
          <div className="flex items-start gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-slate-700">
            <Info size={14} className="mt-0.5 shrink-0 text-primary" aria-hidden />
            <p className="min-w-0">
              <span className="font-medium">Checking a table:</span> tinted cells are the ones the recogniser was
              unsure of — compare each against the page image on the left, correct it here, then mark the table{' '}
              <span className="font-medium">Verify</span>. Export stays available at any point; it tells you how much
              is still unverified.
            </p>
            <button className={`${iconBtnCls} shrink-0`} onClick={dismissIntro} aria-label="Dismiss this tip">
              <X size={13} aria-hidden />
            </button>
          </div>
        )}
        {page.tables.map((t) => (
          <TableEditor
            key={`${docId}:${pageIdx}:${t.table_id}`}
            docId={docId}
            table={t}
            focused={selectedTable === t.table_id}
            onFocus={() => onSelectTable(t.table_id)}
            flash={flashToken?.tid === t.table_id ? flashToken.n : 0}
            focusCell={focusCell?.tid === t.table_id ? focusCell : null}
          />
        ))}

        {(page.corrected_text || text) && (
          <details className="rounded-md border border-slate-200 bg-white p-2">
            <summary className="cursor-pointer text-sm font-medium text-slate-600">
              Page text{textSaved ? '' : ' — unsaved'}
            </summary>
            <textarea
              className="khmer-content mt-2 w-full rounded border border-slate-200 p-2 text-slate-700"
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
              Save text
            </button>
          </details>
        )}
      </div>
    </div>
  )
}
