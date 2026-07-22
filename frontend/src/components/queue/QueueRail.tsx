import { memo, useRef, useState } from 'react'
import { Download, PanelLeft, PanelLeftClose, Play, Plus, Trash2, X } from 'lucide-react'
import type { DocSummary } from '../../api/types'
import { ConfirmPopover } from '../ConfirmPopover'
import { useT, type Key } from '../../i18n.tsx'
import { btnCls, btnSmCls, ICON, ICON_SM, iconBtnCls } from '../../ui'

/** Which destructive action is awaiting confirmation, and where to anchor it. */
type PendingConfirm =
  | { kind: 'all'; anchor: { x: number; y: number } }
  | { kind: 'doc'; id: string; name: string; anchor: { x: number; y: number } }

const anchorOf = (e: React.MouseEvent): { x: number; y: number } => {
  const r = e.currentTarget.getBoundingClientRect()
  return { x: r.right, y: r.bottom }
}

// Linear-style status: a 6px dot + neutral text, not a colored pill.
const STATUS_DOT: Record<string, string> = {
  queued: 'bg-line-strong',
  running: 'bg-primary',
  done: 'bg-ok',
  error: 'bg-danger',
  stopped: 'bg-ink-3', // user-requested stop: neutral slate, never failure-red
}
const STATUS_KEY: Record<string, Key> = {
  queued: 'status_queued',
  running: 'status_running',
  done: 'status_done',
  error: 'status_error',
  stopped: 'status_stopped',
}

// Memoized: the run-status poll re-renders App ~2.5x/s for the whole run;
// the rail's props are all stable (App memoizes its handlers), so the queue
// list must not repaint on every tick.
export const QueueRail = memo(function QueueRail(props: {
  documents: DocSummary[]
  activeId: string | null
  onSelect: (id: string) => void
  onUpload: (files: File[]) => void
  onRemove: (id: string) => void
  uploading: boolean
  onRunAll: () => void
  onRemoveAll: () => void
  batchRunning: boolean
  /** A run is in flight somewhere in the workspace: no second one may start. */
  pipelineBusy: boolean
  exportAllUrl: string | null
}) {
  const { documents, activeId, onSelect, onUpload, onRemove, onRemoveAll, uploading, onRunAll, batchRunning, pipelineBusy, exportAllUrl } = props
  const pending = documents.filter((d) => d.status === 'queued' || d.status === 'error').length
  const unverifiedAcrossDocs = documents
    .filter((d) => d.status === 'done')
    .reduce((n, d) => n + Math.max(0, d.total_tables - d.reviewed_tables), 0)
  const { t } = useT()
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  // The queue is management chrome; the page image + tables are the work. The rail
  // folds to a slim strip so the review zones get the width (remembered per machine).
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('railCollapsed') === 'true')
  // ONE confirmation guard for both destructive actions (clear-all, remove-doc):
  // the shared popover names the consequence and never fires on the first click.
  const [confirm, setConfirm] = useState<PendingConfirm | null>(null)
  const toggleCollapsed = () => {
    setCollapsed((c) => {
      localStorage.setItem('railCollapsed', String(!c))
      return !c
    })
  }

  return (
    <aside
      className={`relative mr-1.5 flex shrink-0 flex-col overflow-hidden rounded-xl border border-line-strong/60 bg-surface shadow-sm transition-[width] duration-150 ${
        collapsed ? 'w-11' : 'w-64'
      }`}
      // The whole rail is the drop target — no permanent dashed box needed.
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        onUpload(Array.from(e.dataTransfer.files))
      }}
    >
      {dragOver && (
        <div className="pointer-events-none absolute inset-1 z-10 flex items-center justify-center rounded-lg border-2 border-dashed border-primary bg-primary-soft/80 text-sm font-medium text-primary">
          {t('drop_to_add')}
        </div>
      )}

      {collapsed ? (
        <div className="flex flex-col items-center gap-2 py-2">
          <button className={iconBtnCls} onClick={toggleCollapsed} aria-label={t('show_queue')} title={t('show_queue')}>
            <PanelLeft size={ICON} aria-hidden />
          </button>
          {documents.length > 0 && (
            <span
              className="flex h-5 w-5 items-center justify-center rounded-full bg-primary-soft text-2xs font-semibold text-primary-strong"
              title={t('queue_count_tip', { n: documents.length })}
            >
              {documents.length}
            </span>
          )}
          <button className={iconBtnCls} onClick={() => fileInput.current?.click()} aria-label={t('add_documents')} title={t('add_documents')}>
            <Plus size={ICON} aria-hidden />
          </button>
        </div>
      ) : (
      <>
        {/* Structural card header: identity left, panel controls right. */}
        <div className="flex h-10 shrink-0 items-center justify-between whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3">
          <span className="flex min-w-0 items-center gap-2 text-sm font-semibold text-ink">
            {t('group_documents')}
            {documents.length > 0 && (
              <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-primary-soft px-1 text-2xs font-semibold text-primary-strong">
                {documents.length}
              </span>
            )}
          </span>
          <span className="relative flex shrink-0 items-center gap-0.5">
            {/* Bulk clear earns its place only once the queue is genuinely a queue. */}
            {documents.length > 1 && (
              <button
                className={`${iconBtnCls} ${confirm?.kind === 'all' ? 'bg-danger-soft text-danger-ink' : 'hover:bg-danger-soft hover:text-danger-ink'}`}
                onClick={(e) => setConfirm((c) => (c?.kind === 'all' ? null : { kind: 'all', anchor: anchorOf(e) }))}
                aria-label={t('delete_all')}
                aria-expanded={confirm?.kind === 'all'}
                title={t('delete_all')}
              >
                <Trash2 size={ICON} aria-hidden />
              </button>
            )}
            <button className={iconBtnCls} onClick={toggleCollapsed} aria-label={t('hide_queue')} title={t('hide_queue')}>
              <PanelLeftClose size={ICON} aria-hidden />
            </button>
          </span>
        </div>
        <div className="p-3 pb-2">
          <button
            className={`${btnCls} w-full justify-center`}
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
          >
            <Plus size={ICON_SM} aria-hidden />
            {uploading ? t('uploading') : t('add_documents')}
          </button>
        </div>
      </>
      )}
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

      {!collapsed && documents.length > 1 && (
        <div className="flex gap-2 px-3 pb-2">
          {pending > 0 && (
            <button
              className={`${btnSmCls} flex-1 justify-center`}
              onClick={onRunAll}
              disabled={batchRunning || pipelineBusy}
              title={t('run_all_tip')}
            >
              <Play size={ICON_SM} aria-hidden />
              {batchRunning ? t('running_all') : t('run_all', { n: pending })}
            </button>
          )}
          {exportAllUrl && (
            <a
              className={`${btnSmCls} flex-1 justify-center`}
              href={exportAllUrl}
              download
              title={
                unverifiedAcrossDocs > 0
                  ? t('export_all_tip_warn', { n: unverifiedAcrossDocs })
                  : t('export_all_tip_ok')
              }
            >
              <Download size={ICON_SM} aria-hidden />
              {t('export_all')}
              {/* The batch export carries trust state too — it is where unchecked work most easily ships. */}
              {unverifiedAcrossDocs > 0 && (
                <span className="rounded-full bg-warn-soft px-1 text-2xs font-semibold text-warn-ink">
                  {unverifiedAcrossDocs}
                </span>
              )}
            </a>
          )}
        </div>
      )}

      <div className={`min-h-0 flex-1 overflow-y-auto px-2 pb-2 ${collapsed ? 'hidden' : ''}`}>
        {documents.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-ink-2">
            {t('no_docs_1')}
            <br />
            {t('no_docs_2')}
          </p>
        )}
        {documents.map((d) => {
          const selected = d.id === activeId
          return (
            <div
              key={d.id}
              role="button"
              tabIndex={0}
              aria-label={d.name}
              aria-current={selected ? 'true' : undefined}
              className={`group mb-0.5 cursor-pointer rounded-md p-2 text-sm transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary ${
                selected ? 'bg-primary-soft' : 'hover:bg-rail'
              }`}
              onClick={() => onSelect(d.id)}
              // Keyboard parity with the click: Enter/Space selects, but only when
              // focus is on the row itself — not when it bubbles up from the
              // nested remove button.
              onKeyDown={(e) => {
                if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault()
                  onSelect(d.id)
                }
              }}
            >
              <div className="flex items-center justify-between gap-1">
                <span className={`truncate font-medium ${selected ? 'text-primary-strong' : 'text-ink'}`} title={d.name}>
                  {d.name}
                </span>
                {/* Quiet at rest, but always in the tab order — revealed by hover
                    OR keyboard focus, never mouse-only. */}
                <button
                  className="shrink-0 rounded-md p-0.5 text-ink-3 opacity-0 transition-[opacity,color,background-color] duration-150 hover:bg-danger-soft hover:text-danger-ink focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-primary group-hover:opacity-100"
                  aria-label={`${t('remove_doc')}: ${d.name}`}
                  title={t('remove_doc')}
                  onClick={(e) => {
                    e.stopPropagation()
                    // Removal discards results AND edits — irreversible, so the same
                    // popover guard as clear-all (one confirmation vocabulary).
                    setConfirm({ kind: 'doc', id: d.id, name: d.name, anchor: anchorOf(e) })
                  }}
                >
                  <X size={14} aria-hidden />
                </button>
              </div>
              <div className="mt-1 flex items-center justify-between text-xs text-ink-2">
                <span>{t('pages_kb', { p: d.pages, kb: d.size_kb })}</span>
                <span className="flex items-center gap-1.5">
                  <span className={`inline-block h-1.5 w-1.5 rounded-full ${STATUS_DOT[d.status] ?? STATUS_DOT.queued}`} aria-hidden />
                  {t(STATUS_KEY[d.status] ?? 'status_queued')}
                </span>
              </div>
              {d.status === 'done' && d.total_tables > 0 && (
                <div className="mt-1.5 flex items-center gap-2">
                  {/* Verification progress as a quiet 2px bar + tabular count. */}
                  <span className="h-0.5 flex-1 overflow-hidden rounded-full bg-line" aria-hidden>
                    <span
                      className={`block h-full rounded-full transition-[width] duration-300 ${d.reviewed_tables === d.total_tables ? 'bg-ok' : 'bg-primary'}`}
                      style={{ width: `${(d.reviewed_tables / d.total_tables) * 100}%` }}
                    />
                  </span>
                  <span className={`text-2xs font-medium ${d.reviewed_tables === d.total_tables ? 'text-ok-ink' : 'text-ink-2'}`}>
                    {t('verified_count', { a: d.reviewed_tables, b: d.total_tables })}
                  </span>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Rendered at the rail root (fixed coords): outside the role="button" rows
          and immune to the scroll container's clipping. */}
      {confirm && (
        <ConfirmPopover
          title={confirm.kind === 'all' ? t('delete_all_title') : t('remove_doc')}
          body={
            confirm.kind === 'all'
              ? t('delete_all_confirm', { n: documents.length })
              : t('remove_confirm')
          }
          subject={confirm.kind === 'doc' ? confirm.name : undefined}
          actionLabel={confirm.kind === 'all' ? t('delete_all_action') : t('remove_doc')}
          cancelLabel={t('cancel')}
          anchor={confirm.anchor}
          onConfirm={() => (confirm.kind === 'all' ? onRemoveAll() : onRemove(confirm.id))}
          onClose={() => setConfirm(null)}
        />
      )}
    </aside>
  )
})
