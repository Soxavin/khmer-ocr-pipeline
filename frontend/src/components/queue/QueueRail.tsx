import { useRef, useState } from 'react'
import { Download, Play, X } from 'lucide-react'
import type { DocSummary } from '../../api/types'
import { btnSmCls } from '../../ui'

const STATUS_STYLE: Record<string, string> = {
  queued: 'bg-slate-100 text-slate-600',
  running: 'bg-blue-100 text-blue-800',
  done: 'bg-green-100 text-green-800',
  error: 'bg-red-100 text-red-700',
}

export function QueueRail(props: {
  documents: DocSummary[]
  activeId: string | null
  onSelect: (id: string) => void
  onUpload: (files: File[]) => void
  onRemove: (id: string) => void
  uploading: boolean
  onRunAll: () => void
  batchRunning: boolean
  exportAllUrl: string | null
}) {
  const { documents, activeId, onSelect, onUpload, onRemove, uploading, onRunAll, batchRunning, exportAllUrl } = props
  const pending = documents.filter((d) => d.status === 'queued' || d.status === 'error').length
  const unverifiedAcrossDocs = documents
    .filter((d) => d.status === 'done')
    .reduce((n, d) => n + Math.max(0, d.total_tables - d.reviewed_tables), 0)
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-200 p-3">
        <div
          className={`cursor-pointer rounded-md border-2 border-dashed p-4 text-center text-sm transition-colors ${
            dragOver ? 'border-primary bg-blue-50 text-primary' : 'border-slate-300 text-slate-600 hover:border-primary hover:text-primary'
          }`}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            onUpload(Array.from(e.dataTransfer.files))
          }}
        >
          {uploading ? 'Uploading…' : (
            <>
              <span className="font-semibold">Add documents</span>
              <br />
              <span className="text-xs text-slate-500">drop PDFs/images or click</span>
            </>
          )}
        </div>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) onUpload(Array.from(e.target.files))
            e.target.value = ''
          }}
        />
      </div>

      {documents.length > 1 && (
        <div className="flex gap-2 border-b border-slate-200 p-2 text-xs">
          {pending > 0 && (
            <button
              className={`${btnSmCls} flex-1 justify-center py-1`}
              onClick={onRunAll}
              disabled={batchRunning}
              title="Run every unprocessed document, one after another"
            >
              <Play size={12} aria-hidden />
              {batchRunning ? 'Running all…' : `Run all (${pending})`}
            </button>
          )}
          {exportAllUrl && (
            <a
              className={`${btnSmCls} flex-1 justify-center py-1`}
              href={exportAllUrl}
              download
              title={
                unverifiedAcrossDocs > 0
                  ? `One zip with every finished document. ${unverifiedAcrossDocs} table${unverifiedAcrossDocs === 1 ? '' : 's'} across the queue ${unverifiedAcrossDocs === 1 ? 'is' : 'are'} not verified yet.`
                  : "One zip with every finished document's results — all tables verified."
              }
            >
              <Download size={12} aria-hidden />
              Export all
              {/* The batch export carries trust state too — it is where unchecked work most easily ships. */}
              {unverifiedAcrossDocs > 0 && (
                <span className="rounded-full bg-amber-100 px-1 text-[10px] font-medium text-amber-900">
                  {unverifiedAcrossDocs}
                </span>
              )}
            </a>
          )}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {documents.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-slate-400">
            No documents yet.
            <br />
            Add a bulletin PDF to begin.
          </p>
        )}
        {documents.map((d) => (
          <div
            key={d.id}
            className={`group mb-1 cursor-pointer rounded-md border p-2 text-sm transition-colors ${
              d.id === activeId ? 'border-primary bg-blue-50' : 'border-transparent hover:bg-slate-50'
            }`}
            onClick={() => onSelect(d.id)}
          >
            <div className="flex items-center justify-between gap-1">
              <span className="truncate font-medium text-slate-700" title={d.name}>
                {d.name}
              </span>
              <button
                className="hidden shrink-0 rounded p-0.5 text-slate-500 hover:bg-red-50 hover:text-red-600 group-hover:block"
                aria-label={`Remove ${d.name}`}
                title="Remove document"
                onClick={(e) => {
                  e.stopPropagation()
                  // Removal discards results AND edits — irreversible, so confirm.
                  if (window.confirm(`Remove “${d.name}”? Its results and any edits will be discarded.`)) {
                    onRemove(d.id)
                  }
                }}
              >
                <X size={14} aria-hidden />
              </button>
            </div>
            <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
              <span>
                {d.pages} page{d.pages === 1 ? '' : 's'} · {d.size_kb} KB
              </span>
              <span className={`rounded-full px-2 py-0.5 ${STATUS_STYLE[d.status] ?? STATUS_STYLE.queued}`}>{d.status}</span>
            </div>
            {d.status === 'done' && d.total_tables > 0 && (
              <div className={`mt-0.5 text-xs font-medium ${d.reviewed_tables === d.total_tables ? 'text-green-700' : 'text-slate-500'}`}>
                {d.reviewed_tables}/{d.total_tables} tables verified
              </div>
            )}
          </div>
        ))}
      </div>
    </aside>
  )
}
