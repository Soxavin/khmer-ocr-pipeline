import { useEffect, useRef, useState } from 'react'
import { ChevronDown, Download, FileUp, Play, RotateCw, ShieldCheck, Square } from 'lucide-react'
import { api } from '../../api/client'
import type { EngineInfo, RunStatus } from '../../api/types'
import { ICON, btnCls, primaryBtnCls, selectCls } from '../../ui'

// Stage labels emitted by webapp/runner.py, in pipeline order.
const STAGES = [
  ['Reading the document…', 'Read'],
  ['Cleaning the pages…', 'Clean'],
  ['Finding text & tables…', 'OCR'],
  ['Tidying the text…', 'Tidy'],
  ['Preparing your files…', 'Export'],
] as const

function stageIndex(stage: string): number {
  return STAGES.findIndex(([label]) => label === stage)
}

function fmtEta(seconds: number): string {
  if (seconds < 60) return `~${Math.max(1, Math.round(seconds))}s left`
  return `~${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s left`
}

export function RunControls(props: {
  engines: EngineInfo[]
  engine: string
  onEngineChange: (key: string) => void
  status: RunStatus | undefined
  docSelected: boolean
  onUploadClick: () => void
  onRun: () => void
  onStop: () => void
  exportUrl: string | null
  docId: string | null
  unverifiedTables: number
  openIssues: number
  combineExport: boolean
  onCombineChange: (v: boolean) => void
  multiPage: boolean
}) {
  const { engines, engine, onEngineChange, status, docSelected, onUploadClick, onRun, onStop, exportUrl, docId, unverifiedTables, openIssues, combineExport, onCombineChange, multiPage } = props
  const active = status?.active ?? false
  const [stopping, setStopping] = useState(false)
  const [exportMenu, setExportMenu] = useState(false)

  // Client-side ETA: from OCR page progress (the dominant stage by far).
  const runStart = useRef<number | null>(null)
  useEffect(() => {
    if (active && runStart.current === null) runStart.current = Date.now()
    if (!active) {
      runStart.current = null
      setStopping(false)
    }
  }, [active])
  let eta: string | null = null
  if (active && status && status.fraction > 0.05 && runStart.current !== null) {
    const elapsed = (Date.now() - runStart.current) / 1000
    eta = fmtEta((elapsed * (1 - status.fraction)) / status.fraction)
  }

  // The ONE morphing primary action: Upload → Run → Export.
  let primary: { label: string; icon: typeof Play; onClick?: () => void; href?: string; disabled?: boolean }
  if (!docSelected) {
    primary = { label: 'Upload documents', icon: FileUp, onClick: onUploadClick }
  } else if (active) {
    primary = { label: 'Extracting…', icon: Play, disabled: true }
  } else if (status?.has_results && exportUrl) {
    primary = { label: 'Export results', icon: Download, href: exportUrl }
  } else if (status?.run_error) {
    primary = { label: 'Retry extraction', icon: RotateCw, onClick: onRun }
  } else {
    primary = { label: 'Run extraction', icon: Play, onClick: onRun }
  }
  const PrimaryIcon = primary.icon

  const idx = status ? stageIndex(status.stage) : -1

  return (
    <div className="flex items-center gap-3">
      {docSelected && !active && (
        <select
          className={`${selectCls} max-w-44 truncate`}
          value={engine}
          onChange={(e) => onEngineChange(e.target.value)}
          aria-label="Recognition engine"
          title={engines.find((e2) => e2.key === engine)?.guidance ?? ''}
        >
          {engines.map((e2) => (
            <option key={e2.key} value={e2.key}>
              {e2.label}
            </option>
          ))}
        </select>
      )}

      {active && status && (
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1" aria-label="Pipeline stages">
            {STAGES.map(([label, short], i) => (
              <span
                key={label}
                className={`rounded-full px-2 py-0.5 text-xs ${
                  i < idx
                    ? 'bg-green-100 font-medium text-green-800'
                    : i === idx
                      ? 'bg-blue-100 font-semibold text-blue-800'
                      : 'bg-slate-100 text-slate-500'
                }`}
              >
                {short}
              </span>
            ))}
          </div>
          {status.total > 0 && (
            <span className="text-xs font-medium text-slate-600">
              page {status.page}/{status.total}
              {eta ? ` · ${eta}` : ''}
            </span>
          )}
          <button
            className={`${btnCls} border-red-300 text-red-700 hover:bg-red-50`}
            onClick={() => {
              setStopping(true)
              onStop()
            }}
            disabled={stopping}
            title="Stop the extraction (finishes the current page, then cancels)"
          >
            <Square size={12} fill="currentColor" aria-hidden />
            {stopping ? 'Stopping…' : 'Stop'}
          </button>
        </div>
      )}

      {docSelected && !active && status?.has_results && (
        <button className={btnCls} onClick={onRun} title="Run again with the selected engine">
          <RotateCw size={ICON} aria-hidden />
          Re-run
        </button>
      )}

      {primary.href ? (
        <span className="relative flex items-stretch">
          <a
            className={`${primaryBtnCls} rounded-r-none`}
            href={primary.href}
            download
            title={
              unverifiedTables > 0 || openIssues > 0
                ? `Export is always allowed — but ${[
                    unverifiedTables > 0 ? `${unverifiedTables} table${unverifiedTables === 1 ? '' : 's'} not yet verified` : '',
                    openIssues > 0 ? `${openIssues} low-confidence cell${openIssues === 1 ? '' : 's'} unreviewed` : '',
                  ].filter(Boolean).join(' and ')}.`
                : 'All tables verified — export away.'
            }
          >
            <PrimaryIcon size={ICON} aria-hidden />
            {primary.label}
            {/* The export button carries the trust state: never identical when work remains. */}
            {unverifiedTables > 0 ? (
              <span className="rounded-full bg-white/25 px-1.5 py-0.5 text-xs font-medium">
                {unverifiedTables} unverified
              </span>
            ) : (
              <ShieldCheck size={13} aria-hidden className="opacity-90" />
            )}
          </a>
          <button
            className="inline-flex items-center rounded-md rounded-l-none border-l border-white/30 bg-primary px-1.5 text-white hover:bg-primary-strong focus-visible:outline-2 focus-visible:outline-primary"
            onClick={() => setExportMenu((m) => !m)}
            aria-label="Other export formats"
            title="Other formats"
          >
            <ChevronDown size={ICON} aria-hidden />
          </button>
          {exportMenu && docId && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setExportMenu(false)} />
              <div className="absolute right-0 top-full z-50 mt-1 w-64 rounded-md border border-slate-200 bg-white py-1 text-sm shadow-lg">
                {/* Joining continuation tables lives HERE — it is a shape-of-the-export
                    decision, so it never has to compromise page-linked review. */}
                {multiPage && (
                  <div className="border-b border-slate-100 px-3 pb-2 pt-1">
                    <p className="mb-1 text-xs font-medium text-slate-600">Tables that continue across pages</p>
                    {(
                      [
                        [true, 'Join into one table', 'One sheet, header once — paste straight into Excel.'],
                        [false, 'Keep one table per page', 'Matches the pages exactly.'],
                      ] as const
                    ).map(([val, label, hint]) => (
                      <label key={String(val)} className="flex cursor-pointer gap-2 py-0.5">
                        <input
                          type="radio"
                          className="mt-0.5"
                          name="combine-export"
                          checked={combineExport === val}
                          onChange={() => onCombineChange(val)}
                        />
                        <span className="text-slate-700">
                          {label}
                          <span className="block text-xs text-slate-500">{hint}</span>
                        </span>
                      </label>
                    ))}
                  </div>
                )}
                {(
                  [
                    ['xlsx', 'Excel (.xlsx)'],
                    ['json', 'JSON'],
                    ['txt', 'Text report'],
                  ] as const
                ).map(([fmt, label]) => (
                  <a
                    key={fmt}
                    className="block px-3 py-1 text-slate-700 hover:bg-slate-50"
                    href={api.exportUrl(docId, fmt, combineExport)}
                    download
                    onClick={() => setExportMenu(false)}
                  >
                    {label}
                  </a>
                ))}
              </div>
            </>
          )}
        </span>
      ) : (
        <button className={primaryBtnCls} onClick={primary.onClick} disabled={primary.disabled}>
          <PrimaryIcon size={ICON} aria-hidden />
          {primary.label}
        </button>
      )}
    </div>
  )
}
