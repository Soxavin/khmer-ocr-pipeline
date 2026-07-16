import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AgGridReact } from 'ag-grid-react'
import {
  AllCommunityModule,
  ModuleRegistry,
  themeQuartz,
  type CellContextMenuEvent,
  type ColDef,
  type GridApi,
} from 'ag-grid-community'
import { Check, Diff, Download, ListPlus, Redo2, RotateCcw, Undo2 } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/client'
import type { PageTable } from '../../api/types'
import { btnSmCls } from '../../ui'

ModuleRegistry.registerModules([AllCommunityModule])

const theme = themeQuartz.withParams({
  accentColor: '#1565c0',
  headerFontSize: 12,
  cellHorizontalPadding: 8,
  wrapperBorderRadius: 6,
})

// Calibrated per-cell buckets (model_config.py §2.33) — same palette as P1.
const CELL_CONF_LOW = 0.8
const CELL_CONF_MID = 0.95

type Row = { __r: number; [col: `c${number}`]: string }

function toRows(grid: string[][]): Row[] {
  return grid.map((row, r) => {
    const o: Row = { __r: r }
    row.forEach((v, c) => (o[`c${c}`] = v))
    return o
  })
}

type Menu = { x: number; y: number; rowIndex: number }

export function TableEditor(props: {
  docId: string
  table: PageTable
  focused: boolean
  onFocus: () => void
  flash: number // increments on every image-side click; 0 = never flashed
  focusCell?: { row: number; col: number; n: number } | null // triage jump target
}) {
  const { docId, table, focused, onFocus, flash, focusCell } = props
  const qc = useQueryClient()
  const gridApi = useRef<GridApi | null>(null)
  const [verified, setVerified] = useState(table.verified)
  const [grid, setGrid] = useState<string[][]>(table.grid)
  const [history, setHistory] = useState<string[][][]>([])
  const [redo, setRedo] = useState<string[][][]>([])
  const [diffOn, setDiffOn] = useState(false)
  const [saveState, setSaveState] = useState<'saved' | 'saving' | 'error'>('saved')
  const [menu, setMenu] = useState<Menu | null>(null)
  const [edited, setEdited] = useState(table.edited)
  const wrapRef = useRef<HTMLDivElement>(null)

  // New server data (page change, re-run) replaces local state.
  useEffect(() => {
    setGrid(table.grid)
    setHistory([])
    setRedo([])
    setEdited(table.edited)
    setVerified(table.verified)
  }, [table])

  // Triage jump: scroll the exact cell into view and focus it.
  useEffect(() => {
    if (focusCell && focusCell.n > 0 && gridApi.current) {
      gridApi.current.ensureIndexVisible(focusCell.row, 'middle')
      gridApi.current.ensureColumnVisible(`c${focusCell.col}`)
      gridApi.current.setFocusedCell(focusCell.row, `c${focusCell.col}`)
    }
  }, [focusCell])

  useEffect(() => {
    if (flash > 0 && wrapRef.current) {
      const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      wrapRef.current.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'nearest' })
      wrapRef.current.classList.remove('table-flash')
      void wrapRef.current.offsetWidth // restart the animation
      wrapRef.current.classList.add('table-flash')
    }
  }, [flash])

  const save = useCallback(
    (next: string[][]) => {
      setSaveState('saving')
      api
        .putTable(docId, table.table_id, next)
        .then(() => {
          setSaveState('saved')
          qc.invalidateQueries({ queryKey: ['lowconf'] })
        })
        .catch(() => setSaveState('error'))
      setEdited(true)
    },
    [docId, table.table_id, qc],
  )

  const commit = useCallback(
    (next: string[][]) => {
      setHistory((h) => [...h.slice(-49), grid])
      setRedo([])
      setGrid(next)
      save(next)
    },
    [grid, save],
  )

  const undoEdit = useCallback(() => {
    setHistory((h) => {
      if (!h.length) return h
      const prev = h[h.length - 1]
      setRedo((r) => [...r, grid])
      setGrid(prev)
      save(prev)
      return h.slice(0, -1)
    })
  }, [grid, save])

  const redoEdit = useCallback(() => {
    setRedo((r) => {
      if (!r.length) return r
      const next = r[r.length - 1]
      setHistory((h) => [...h, grid])
      setGrid(next)
      save(next)
      return r.slice(0, -1)
    })
  }, [grid, save])

  const reset = useCallback(() => {
    api.resetTable(docId, table.table_id).catch(() => setSaveState('error'))
    setGrid(table.original_grid)
    setHistory([])
    setRedo([])
    setEdited(false)
    setDiffOn(false)
  }, [docId, table])

  const nCols = grid[0]?.length ?? 0

  const insertRow = (at: number) => {
    const next = [...grid.slice(0, at), Array.from({ length: nCols }, () => ''), ...grid.slice(at)]
    commit(next)
  }
  const deleteRow = (at: number) => {
    if (grid.length <= 1) return
    commit(grid.filter((_, r) => r !== at))
  }

  const columnDefs = useMemo<ColDef[]>(
    () =>
      Array.from({ length: nCols }, (_, c) => ({
        field: `c${c}`,
        headerName: String(c + 1),
        editable: true,
        minWidth: 60,
        flex: 1,
        cellClassRules: {
          'cell-diff': (p) => {
            if (!diffOn) return false
            const r = (p.data as Row).__r
            return (table.original_grid[r]?.[c] ?? '') !== (p.value ?? '')
          },
          // Empty cells are intentional table structure, not OCR errors — never tint them.
          'cell-conf-low': (p) => {
            if (diffOn || !String(p.value ?? '').trim()) return false
            const r = (p.data as Row).__r
            const v = table.confidence[r]?.[c]
            return v !== null && v !== undefined && v < CELL_CONF_LOW
          },
          'cell-conf-mid': (p) => {
            if (diffOn || !String(p.value ?? '').trim()) return false
            const r = (p.data as Row).__r
            const v = table.confidence[r]?.[c]
            return v !== null && v !== undefined && v >= CELL_CONF_LOW && v < CELL_CONF_MID
          },
        },
      })),
    [nCols, diffOn, table],
  )

  const rows = useMemo(() => toRows(grid), [grid])

  return (
    <section
      ref={wrapRef}
      className={`rounded-md ${focused ? 'ring-1 ring-primary/40' : ''}`}
      onFocusCapture={onFocus}
      onClickCapture={onFocus}
      onKeyDown={(e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
          e.preventDefault()
          if (e.shiftKey) redoEdit()
          else undoEdit()
        }
      }}
    >
      <div className="mb-1 flex min-h-7 items-center gap-2 text-xs">
        <span className="min-w-0 max-w-56 truncate font-semibold text-slate-700" title={table.table_id}>
          {table.table_id}
        </span>
        <button
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium ${
            verified
              ? 'bg-green-100 text-green-800'
              : 'border border-slate-300 text-slate-600 hover:bg-slate-50'
          }`}
          title={verified ? 'Marked verified — click to unmark' : 'Mark this table as reviewed'}
          onClick={() => {
            const next = !verified
            setVerified(next)
            setSaveState('saved')
            api.review(docId, table.table_id, next).then(() => {
              qc.invalidateQueries({ queryKey: ['documents'] })
              qc.invalidateQueries({ queryKey: ['page'] })
            }).catch(() => {
              // Never let the export's trust count lie: revert AND say so.
              setVerified(!next)
              setSaveState('error')
            })
          }}
        >
          <Check size={12} aria-hidden />
          {verified ? 'Verified' : 'Verify'}
        </button>
        {edited && <span className="rounded-full bg-blue-100 px-2 py-0.5 font-medium text-blue-800">Edited</span>}
        {saveState === 'saving' && <span className="text-slate-500">saving…</span>}
        {saveState === 'error' && (
          <span className="font-medium text-red-700">Not saved to the server — try that again</span>
        )}
        {/* Contextual toolbar: only on the focused table (residency: contextual tier). */}
        {focused && (
          <span className="ml-auto flex items-center gap-1">
            <button className={btnSmCls} onClick={undoEdit} disabled={!history.length}
                    aria-label="Undo" title="Undo (Cmd-Z)">
              <Undo2 size={13} aria-hidden />
            </button>
            <button className={btnSmCls} onClick={redoEdit} disabled={!redo.length}
                    aria-label="Redo" title="Redo (Shift-Cmd-Z)">
              <Redo2 size={13} aria-hidden />
            </button>
            <button className={`${btnSmCls} ${diffOn ? 'border-primary text-primary' : ''}`}
                    onClick={() => setDiffOn((d) => !d)} disabled={!edited}
                    title="Highlight cells that differ from the OCR result">
              <Diff size={13} aria-hidden />
              Diff
            </button>
            <button className={btnSmCls} onClick={() => insertRow(grid.length)} title="Add a row at the end">
              <ListPlus size={13} aria-hidden />
              Row
            </button>
            <a className={btnSmCls} href={api.exportCsvUrl(docId, table.table_id)} download
               title="Download just this table as CSV">
              <Download size={13} aria-hidden />
              CSV
            </a>
            <button className={btnSmCls} onClick={reset} disabled={!edited}
                    title="Discard all edits to this table">
              <RotateCcw size={13} aria-hidden />
              Reset
            </button>
          </span>
        )}
      </div>

      <div className="ag-theme-workspace" onContextMenu={(e) => e.preventDefault()}>
        <AgGridReact
          theme={theme}
          columnDefs={columnDefs}
          rowData={rows}
          getRowId={(p) => String((p.data as Row).__r)}
          domLayout="autoHeight"
          rowHeight={38}
          headerHeight={26}
          stopEditingWhenCellsLoseFocus
          onGridReady={(e) => (gridApi.current = e.api)}
          onCellValueChanged={(e) => {
            const next = grid.map((row) => [...row])
            next[(e.data as Row).__r][Number(String(e.colDef.field).slice(1))] = e.newValue ?? ''
            commit(next)
          }}
          onCellContextMenu={(e: CellContextMenuEvent) => {
            const ev = e.event as MouseEvent
            ev.preventDefault()
            setMenu({ x: ev.clientX, y: ev.clientY, rowIndex: (e.data as Row).__r })
          }}
        />
      </div>

      {menu && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setMenu(null)} onContextMenu={(e) => { e.preventDefault(); setMenu(null) }} />
          <div className="fixed z-50 w-44 rounded-md border border-slate-200 bg-white py-1 text-sm shadow-lg"
               style={{ left: menu.x, top: menu.y }}>
            <button className="block w-full px-3 py-1 text-left hover:bg-slate-50"
                    onClick={() => { insertRow(menu.rowIndex); setMenu(null) }}>Insert row above</button>
            <button className="block w-full px-3 py-1 text-left hover:bg-slate-50"
                    onClick={() => { insertRow(menu.rowIndex + 1); setMenu(null) }}>Insert row below</button>
            <button className="block w-full px-3 py-1 text-left text-red-600 hover:bg-red-50 disabled:opacity-40"
                    disabled={grid.length <= 1}
                    onClick={() => { deleteRow(menu.rowIndex); setMenu(null) }}>Delete row</button>
          </div>
        </>
      )}
    </section>
  )
}
