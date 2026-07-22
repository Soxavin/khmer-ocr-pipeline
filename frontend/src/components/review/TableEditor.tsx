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
import { useT } from '../../i18n.tsx'
import { btnSmCls, menuCls, menuItemCls } from '../../ui'

ModuleRegistry.registerModules([AllCommunityModule])

// AG Grid tuned to the workspace tokens: rail-toned headers, hairline borders,
// primary selection — so the grid reads as part of the app, not a widget.
// CSS-variable references so the grid follows the token theme (incl. dark mode).
const theme = themeQuartz.withParams({
  accentColor: 'var(--color-primary)',
  backgroundColor: 'var(--color-surface)',
  foregroundColor: 'var(--color-ink)',
  headerFontSize: 12,
  headerFontFamily: "'Inter Variable', system-ui, sans-serif",
  headerBackgroundColor: 'var(--color-rail)',
  headerTextColor: 'var(--color-ink-2)',
  borderColor: 'var(--color-line)',
  rowHoverColor: 'color-mix(in oklab, var(--color-primary) 5%, transparent)',
  selectedRowBackgroundColor: 'var(--color-primary-soft)',
  cellHorizontalPadding: 8,
  wrapperBorderRadius: 8,
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
  const { t } = useT()
  const gridApi = useRef<GridApi | null>(null)
  const [verified, setVerified] = useState(table.verified)
  const [grid, setGrid] = useState<string[][]>(table.grid)
  const [history, setHistory] = useState<string[][][]>([])
  const [redo, setRedo] = useState<string[][][]>([])
  const [diffOn, setDiffOn] = useState(false)
  const [saveState, setSaveState] = useState<'saved' | 'saving' | 'error'>('saved')
  const [menu, setMenu] = useState<Menu | null>(null)
  const [edited, setEdited] = useState(table.edited)
  // Marking a table verified is the payoff of the review loop — one earned pulse.
  const [justVerified, setJustVerified] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  // New server data replaces local state — but ONLY when it genuinely differs
  // from what we already hold. Every page refetch (a verify flips one boolean)
  // hands down a new table identity, and after an edit→verify the refetched grid
  // even has new CONTENT (the server echoes the saved edit back); an unconditional
  // reset therefore destroyed the undo stack at the exact moment the analyst
  // committed trust. Confirmation is not new data.
  useEffect(() => {
    if (JSON.stringify(grid) === JSON.stringify(table.grid)) return
    setGrid(table.grid)
    setHistory([])
    setRedo([])
    // `grid` read from the render that delivered the new table.grid — current by
    // construction; adding it to deps would re-run the reset on every local edit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [table.grid])
  // Verified/edited are server facts with no local history: sync unconditionally.
  useEffect(() => setVerified(table.verified), [table.verified])
  useEffect(() => setEdited(table.edited), [table.edited])

  // Triage jump: scroll the exact cell into view and focus it.
  useEffect(() => {
    if (focusCell && focusCell.n > 0 && gridApi.current) {
      const go = () => {
        gridApi.current?.ensureIndexVisible(focusCell.row, 'middle')
        gridApi.current?.ensureColumnVisible(`c${focusCell.col}`)
        gridApi.current?.setFocusedCell(focusCell.row, `c${focusCell.col}`)
      }
      go()
      // autoHeight rows settle a frame later; re-center so wrap growth can't
      // shift the just-focused cell out of view.
      requestAnimationFrame(() => setTimeout(go, 50))
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
        // Long Khmer values wrap and the row grows — nothing hides behind a
        // truncation; reading never requires clicking into the cell.
        wrapText: true,
        autoHeight: true,
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
      // Depth + border encode focus: the table under review lifts and sharpens,
      // the rest sit flat — hierarchy through commitment, not decoration.
      className={`rounded-lg border bg-surface p-2 transition-[border-color,box-shadow] duration-150 ${
        focused ? 'border-primary/50 shadow-overlay' : 'border-line'
      }`}
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
      <div className="mb-1.5 flex min-h-7 items-center gap-2 overflow-x-auto whitespace-nowrap text-xs">
        <span
          className={`min-w-0 max-w-56 truncate font-semibold ${focused ? 'text-primary-strong' : 'text-ink-2'}`}
          title={table.table_id}
        >
          {table.table_id}
        </span>
        <button
          className={`inline-flex h-6 items-center gap-1 rounded-full px-2 font-medium transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-primary ${
            verified
              ? 'bg-ok-soft text-ok-ink'
              : 'border border-line-strong text-ink-2 hover:bg-rail hover:text-ink'
          }`}
          title={verified ? t('verify_tip_on') : t('verify_tip_off')}
          onClick={() => {
            const next = !verified
            setVerified(next)
            if (next) setJustVerified(true)
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
          <span
            className={justVerified ? 'verify-pop inline-flex' : 'inline-flex'}
            onAnimationEnd={() => setJustVerified(false)}
          >
            <Check size={12} aria-hidden />
          </span>
          {verified ? t('verified') : t('verify')}
        </button>
        {edited && <span className="rounded-full bg-primary-soft px-2 py-0.5 font-medium text-primary-strong">{t('edited')}</span>}
        {/* Fixed-width slot so the toolbar doesn't jump as save state flickers. */}
        {saveState === 'saving' && <span className="w-14 text-ink-3">{t('saving')}</span>}
        {saveState === 'error' && (
          <span className="font-medium text-danger-ink">{t('not_saved')}</span>
        )}
        {/* Contextual toolbar: only on the focused table (residency: contextual tier). */}
        {focused && (
          <span className="ml-auto flex shrink-0 items-center gap-1 pl-2">
            <button className={btnSmCls} onClick={undoEdit} disabled={!history.length}
                    aria-label={t('undo')} title={t('undo_tip')}>
              <Undo2 size={13} aria-hidden />
            </button>
            <button className={btnSmCls} onClick={redoEdit} disabled={!redo.length}
                    aria-label={t('redo')} title={t('redo_tip')}>
              <Redo2 size={13} aria-hidden />
            </button>
            <button className={`${btnSmCls} ${diffOn ? 'border-primary text-primary' : ''}`}
                    onClick={() => setDiffOn((d) => !d)} disabled={!edited}
                    title={t('diff_tip')}>
              <Diff size={13} aria-hidden />
              {t('diff')}
            </button>
            <button className={btnSmCls} onClick={() => insertRow(grid.length)} title={t('row_tip')}>
              <ListPlus size={13} aria-hidden />
              {t('row')}
            </button>
            <a className={btnSmCls} href={api.exportCsvUrl(docId, table.table_id)} download
               title={t('csv_tip')}>
              <Download size={13} aria-hidden />
              CSV
            </a>
            <button className={btnSmCls} onClick={reset} disabled={!edited}
                    title={t('reset_tip')}>
              <RotateCcw size={13} aria-hidden />
              {t('reset')}
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
          <div className={`${menuCls} fixed z-50 w-44`} style={{ left: menu.x, top: menu.y }}>
            <button className={menuItemCls}
                    onClick={() => { insertRow(menu.rowIndex); setMenu(null) }}>{t('insert_above')}</button>
            <button className={menuItemCls}
                    onClick={() => { insertRow(menu.rowIndex + 1); setMenu(null) }}>{t('insert_below')}</button>
            <button className={`${menuItemCls} text-danger-ink hover:bg-danger-soft hover:text-danger-ink disabled:opacity-40`}
                    disabled={grid.length <= 1}
                    onClick={() => { deleteRow(menu.rowIndex); setMenu(null) }}>{t('delete_row')}</button>
          </div>
        </>
      )}
    </section>
  )
}
