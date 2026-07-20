import { useState } from 'react'
import { Check, ShieldCheck, X } from 'lucide-react'
import type { Issue } from '../../api/types'
import { useT, type Key } from '../../i18n.tsx'
import { iconBtnCls } from '../../ui'

// Failure-mode taxonomy (validate.py) → plain localized phrase. An analyst reads
// WHY a cell is flagged in her own language, not a cryptic Latin badge.
const REASON_KEY: Record<string, Key> = {
  numeric_mismatch: 'reason_numeric_mismatch',
  sequence_illegal: 'reason_sequence_illegal',
  digit_mixed: 'reason_digit_mixed',
  numeric_unparseable: 'reason_numeric_unparseable',
  structure_ragged: 'reason_structure_ragged',
  low_conf: 'reason_low_conf',
}

const issueKey = (it: Issue) => `${it.table_id}:${it.row}:${it.col}`

/** Triage drawer: flagged cells (failure-mode classified), worst first. Click (or n/p)
    jumps to the cell; Dismiss removes the row from the GLOBAL triage list (header chip,
    count, stepping all agree) — the underlying flag clears for real when the cell is fixed. */
export function IssuesDrawer(props: {
  issues: Issue[]
  currentIdx: number
  onJump: (idx: number) => void
  onDismiss: (key: string) => void
  onClose: () => void
}) {
  const { issues, currentIdx, onJump, onDismiss, onClose } = props
  const { t } = useT()
  // Exit animation only lives here; the removal itself is App state (onDismiss).
  const [leaving, setLeaving] = useState<Set<string>>(new Set())
  const dismiss = (k: string) => {
    if (leaving.has(k)) return
    setLeaving((s) => new Set(s).add(k))
    window.setTimeout(() => {
      onDismiss(k)
      setLeaving((s) => {
        const next = new Set(s)
        next.delete(k)
        return next
      })
    }, 150)
  }
  const visible = issues
  return (
    <div className="flex h-full w-80 shrink-0 flex-col bg-surface">
      <div className="flex h-10 shrink-0 items-center justify-between whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3">
        <span className="text-sm font-semibold text-ink">{t('issues_n', { n: visible.length })}</span>
        <span className="text-xs text-ink-3">{t('np_step')}</span>
        <button className={iconBtnCls} onClick={onClose} aria-label={t('close_issues')}>
          <X size={14} aria-hidden />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {visible.length === 0 && (
          <p className="flex items-center justify-center gap-2 p-4 text-sm text-ok-ink">
            <ShieldCheck size={16} aria-hidden />
            {t('all_confident')}
          </p>
        )}
        {issues.map((it, i) => {
          const k = issueKey(it)
          // Two honest columns: content left, verdicts (conf% + Dismiss) right.
          // No absolute positioning — nothing can overlap in either language.
          return (
            <div
              key={k}
              className={`group flex items-start gap-2 border-b border-line px-3 py-2 transition-colors duration-150 hover:bg-rail ${
                i === currentIdx ? 'bg-primary-soft' : ''
              } ${leaving.has(k) ? 'issue-out' : ''}`}
            >
              <button
                className="min-w-0 flex-1 text-left text-sm focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary"
                onClick={() => onJump(i)}
              >
                <span className="khmer-content block truncate font-medium text-ink">
                  {it.text || <em className="text-ink-2">{t('empty_cell')}</em>}
                </span>
                {/* The problem, stated plainly. Severity dot: danger = validator finding,
                    warn = low recognition confidence. */}
                <span className="mt-0.5 flex items-center gap-1.5 text-xs text-ink-2">
                  <span
                    className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${
                      it.reason === 'low_conf' ? 'bg-warn' : 'bg-danger'
                    }`}
                    aria-hidden
                  />
                  <span className="min-w-0 truncate">
                    {(it.reasons?.length ? it.reasons : [it.reason])
                      .map((r) => (REASON_KEY[r] ? t(REASON_KEY[r]) : r))
                      .join(' · ')}
                  </span>
                </span>
                <span className="mt-0.5 block text-xs text-ink-3">
                  {it.page !== null ? `${t('issue_page', { p: it.page + 1 })} · ` : ''}{t('issue_loc', { r: it.row + 1, c: it.col + 1 })}
                </span>
              </button>
              <span className="flex shrink-0 flex-col items-end gap-1">
                {it.conf !== null && (
                  <span className={`text-xs font-medium ${it.conf < 0.5 ? 'text-danger-ink' : 'text-warn-ink'}`}>
                    {(it.conf * 100).toFixed(0)}%
                  </span>
                )}
                {/* Low-profile action badge: appears on hover/focus, scales on press. */}
                <button
                  className="hidden items-center gap-1 rounded-full border border-line-strong bg-surface px-2 py-0.5 text-2xs font-medium text-ink-2 transition-[color,background-color,border-color,transform] duration-150 hover:border-ok/40 hover:bg-ok-soft hover:text-ok-ink focus-visible:flex focus-visible:outline-2 focus-visible:outline-primary active:scale-95 group-hover:flex"
                  title={t('dismiss_issue')}
                  onClick={() => dismiss(k)}
                >
                  <Check size={11} aria-hidden />
                  {t('dismiss_badge')}
                </button>
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
