import { ShieldCheck, X } from 'lucide-react'
import type { Issue } from '../../api/types'
import { iconBtnCls } from '../../ui'

/** Triage drawer: low-confidence cells, worst first. Click (or n/p) jumps to the cell. */
export function IssuesDrawer(props: {
  issues: Issue[]
  currentIdx: number
  onJump: (idx: number) => void
  onClose: () => void
}) {
  const { issues, currentIdx, onJump, onClose } = props
  return (
    <div className="flex w-80 shrink-0 flex-col border-l border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
        <span className="text-sm font-semibold text-slate-800">Issues ({issues.length})</span>
        <span className="text-xs text-slate-500">n / p to step</span>
        <button className={iconBtnCls} onClick={onClose} aria-label="Close issues panel">
          <X size={14} aria-hidden />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {issues.length === 0 && (
          <p className="flex items-center justify-center gap-2 p-4 text-sm text-green-800">
            <ShieldCheck size={16} aria-hidden />
            All cells are above the confidence threshold.
          </p>
        )}
        {issues.map((it, i) => (
          <button
            key={`${it.table_id}:${it.row}:${it.col}`}
            className={`block w-full border-b border-slate-100 px-3 py-2 text-left text-sm hover:bg-slate-50 ${
              i === currentIdx ? 'bg-blue-50' : ''
            }`}
            onClick={() => onJump(i)}
          >
            <div className="flex items-center justify-between">
              <span className="khmer-content truncate font-medium text-slate-700">
                {it.text || <em className="text-slate-400">empty cell</em>}
              </span>
              <span className={`ml-2 shrink-0 text-xs ${it.conf < 0.5 ? 'text-conf-low' : 'text-conf-mid'}`}>
                {(it.conf * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-0.5 text-xs text-slate-500">
              {it.page !== null ? `page ${it.page + 1} · ` : ''}row {it.row + 1}, col {it.col + 1}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
