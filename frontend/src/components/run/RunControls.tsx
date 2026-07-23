import { useEffect, useState } from 'react'
import { ChevronDown, Download, FileUp, Play, RotateCw, ShieldCheck, Square } from 'lucide-react'
import { api } from '../../api/client'
import type { RunStatus } from '../../api/types'
import { useT, type Key } from '../../i18n.tsx'
import { ICON, btnCls, dangerBtnCls, menuCls, menuItemCls, primaryBtnCls } from '../../ui'

// Server stage labels (webapp/runner.py emits English), mapped to display keys.
const STAGES: [string, Key][] = [
  ['Reading the document…', 'stage_read'],
  ['Cleaning the pages…', 'stage_clean'],
  ['Finding text & tables…', 'stage_ocr'],
  ['Tidying the text…', 'stage_tidy'],
  ['Preparing your files…', 'stage_export'],
]

function stageIndex(stage: string): number {
  return STAGES.findIndex(([label]) => label === stage)
}

// Sub-stages the OCR engine reports inside a single page (webapp/state.py Progress.step).
const SUB_STEPS: Record<string, Key> = {
  layout: 'step_layout',
  text: 'step_text',
  tables: 'step_tables',
}

export function RunControls(props: {
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
  const { status, docSelected, onUploadClick, onRun, onStop, exportUrl, docId, unverifiedTables, openIssues, combineExport, onCombineChange, multiPage } = props
  const { t } = useT()
  const active = status?.active ?? false
  const [stopping, setStopping] = useState(false)
  const [exportMenu, setExportMenu] = useState(false)

  // Reset the Stop debounce when the run actually ends.
  useEffect(() => {
    if (!active) setStopping(false)
  }, [active])

  // The ONE morphing primary action: Upload → Run → Export.
  let primary: { label: string; icon: typeof Play; onClick?: () => void; href?: string; disabled?: boolean }
  if (!docSelected) {
    primary = { label: t('upload_documents'), icon: FileUp, onClick: onUploadClick }
  } else if (active) {
    primary = { label: t('extracting'), icon: Play, disabled: true }
  } else if (status?.has_results && exportUrl) {
    primary = { label: t('export_results'), icon: Download, href: exportUrl }
  } else if (status?.run_error) {
    primary = { label: t('retry_extraction'), icon: RotateCw, onClick: onRun }
  } else {
    primary = { label: t('run_extraction'), icon: Play, onClick: onRun }
  }
  const PrimaryIcon = primary.icon

  const idx = status ? stageIndex(status.stage) : -1

  return (
    <div className="flex items-center gap-3">
      {active && status && (
        <div className="flex items-center gap-3">
          {/* Stage + page facts only — the PROGRESS itself lives as a 2px line on the
              header's bottom edge (no ETA guesses, no second bar). */}
          <div className="flex items-baseline gap-3 whitespace-nowrap text-xs" aria-live="polite">
            <span className="font-medium text-ink">
              {idx >= 0 ? t(STAGES[idx][1]) : status.stage || t('working')}
            </span>
            {/* The OCR stage runs for minutes: its sub-step keeps the line moving
                so a working pipeline never reads as a frozen one. */}
            {SUB_STEPS[status.step] && (
              <span className="text-ink-2">{t(SUB_STEPS[status.step])}</span>
            )}
            {status.total > 0 && (
              <span className="text-ink-2">{t('page_of', { a: status.page, b: status.total })}</span>
            )}
          </div>
          <button
            className={dangerBtnCls}
            onClick={() => {
              setStopping(true)
              onStop()
            }}
            disabled={stopping}
            title={t('stop_tip')}
          >
            <Square size={10} fill="currentColor" aria-hidden />
            {stopping ? t('stopping') : t('stop')}
          </button>
        </div>
      )}

      {docSelected && !active && status?.has_results && (
        <button className={btnCls} onClick={onRun} title={t('rerun_tip')}>
          <RotateCw size={ICON} aria-hidden />
          {t('rerun')}
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
                ? t('export_warn_prefix') + [
                    unverifiedTables > 0 ? t(unverifiedTables === 1 ? 'warn_tables_one' : 'warn_tables_other', { n: unverifiedTables }) : '',
                    openIssues > 0 ? t(openIssues === 1 ? 'warn_cells_one' : 'warn_cells_other', { n: openIssues }) : '',
                  ].filter(Boolean).join(t('and')) + '.'
                : t('export_ok_tip')
            }
          >
            <PrimaryIcon size={ICON} aria-hidden />
            {primary.label}
            {/* The export button carries the trust state: never identical when work remains. */}
            {unverifiedTables > 0 ? (
              <span className="rounded-full bg-white/25 px-1.5 py-0.5 text-xs font-medium">
                {t('n_unverified', { n: unverifiedTables })}
              </span>
            ) : (
              <ShieldCheck size={13} aria-hidden className="opacity-90" />
            )}
          </a>
          {/* The trigger and the panel deliberately share NO viewTransitionName.
              They used to: the button carried 'export-menu' while closed and the
              panel carried it while open, so the View Transitions API was asked to
              morph a 28px chevron into a 256px panel — and stretched the arrow on
              the way. The panel already rises via .overlay-enter in menuCls, which
              is the entrance every other floating surface uses, so the morph was
              never carrying the animation, only distorting the icon. */}
          <button
            className="inline-flex items-center rounded-md rounded-l-none border-l border-primary-strong bg-primary px-1.5 text-white transition-colors duration-150 hover:bg-primary-strong focus-visible:outline-2 focus-visible:outline-primary"
            onClick={() => setExportMenu((m) => !m)}
            // The other three menu triggers already carry aria-expanded; this one
            // was the outlier. (None carries aria-haspopup — adding it here alone
            // would just trade one inconsistency for another.)
            aria-expanded={exportMenu}
            aria-label={t('other_formats_aria')}
            title={t('other_formats')}
          >
            <ChevronDown size={ICON} aria-hidden />
          </button>
          {exportMenu && docId && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setExportMenu(false)} />
              <div className={`${menuCls} absolute right-0 top-full z-50 mt-1 w-64`}>
                {/* Joining continuation tables lives HERE — it is a shape-of-the-export
                    decision, so it never has to compromise page-linked review. */}
                {multiPage && (
                  <div className="border-b border-line px-3 pb-2 pt-1">
                    <p className="mb-1 text-xs font-medium text-ink-2">{t('combine_title')}</p>
                    {(
                      [
                        [true, t('combine_join'), t('combine_join_hint')],
                        [false, t('combine_keep'), t('combine_keep_hint')],
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
                        <span className="text-ink">
                          {label}
                          <span className="block text-xs text-ink-2">{hint}</span>
                        </span>
                      </label>
                    ))}
                  </div>
                )}
                {(
                  [
                    ['xlsx', t('fmt_xlsx')],
                    ['json', t('fmt_json')],
                    ['txt', t('fmt_txt')],
                  ] as const
                ).map(([fmt, label]) => (
                  <a
                    key={fmt}
                    className={menuItemCls}
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
          {/* keyed span: the morphing action crossfades its label instead of snapping */}
          <span key={primary.label} className="label-fade">{primary.label}</span>
        </button>
      )}
    </div>
  )
}
