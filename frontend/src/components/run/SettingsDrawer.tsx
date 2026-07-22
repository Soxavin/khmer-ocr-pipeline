import { useEffect, useRef, useState } from 'react'
import { Check, Eraser, FileOutput, Files, ScanSearch, Sparkles, X } from 'lucide-react'
import type { EngineInfo, RunSettings, SuggestCheck, Suggestion } from '../../api/types'
import { useT, type Key } from '../../i18n.tsx'
import { SegmentedToggle } from '../viewer/PageGrid'
import { scanWordingKey } from '../../lib/scan'
import { autoBadge } from '../../lib/settings'
import { iconBtnCls, inputCls } from '../../ui'

const PREPROCESS_FLAGS: [string, Key, Key][] = [
  ['deskew', 'flag_deskew', 'hint_deskew'],
  ['remove_stamps', 'flag_stamps', 'hint_stamps'],
  ['sharpen', 'flag_sharpen', 'hint_sharpen'],
  ['normalise', 'flag_contrast', 'hint_contrast'],
  ['normalise_table_backgrounds', 'flag_tablebg', 'hint_tablebg'],
]
// NOTE: joining tables across pages is deliberately NOT here — it is an export
// choice, not an extraction one. Extraction always keeps per-page tables so the
// review panel can link every row to the page image it came from.
const OUTPUT_FLAGS: [string, Key, Key][] = [
  ['repair_tables', 'flag_repair', 'hint_repair'],
  ['convert_numerals', 'flag_numerals', 'hint_numerals'],
]

/** A real switch, not a checkbox: the drawer's clearest "designed" signal.
    36×20 track, 150ms knob travel, token colors, reduced-motion covered globally. */
function Switch(props: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  const { checked, onChange, label, disabled = false } = props
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative h-5 w-9 shrink-0 rounded-full transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-50 ${
        checked ? 'bg-primary' : 'bg-line-strong'
      }`}
    >
      <span
        aria-hidden
        className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow-raised transition-[left] duration-100 ${
          checked ? 'left-[18px]' : 'left-0.5'
        }`}
      />
    </button>
  )
}

/** What an 'Auto' option actually resolved to for this document.
    'Auto' without an outcome is a black box: the analyst cannot tell 200 DPI from
    300, or which recognizer read the page. Rendered only once a run has decided —
    a badge that guessed would be worse than no badge. */
function ResolvedBadge(props: { text: string; title: string }) {
  return (
    <span
      className="ml-1.5 inline-block rounded bg-primary-soft px-1.5 py-0.5 text-2xs font-semibold text-primary-strong"
      title={props.title}
    >
      {props.text}
    </span>
  )
}

function SectionTitle(props: { icon: typeof Files; label: string }) {
  const Icon = props.icon
  return (
    <h3 className="mb-2.5 flex items-center gap-1.5 text-[15px] font-semibold text-ink">
      <Icon size={13} className="text-ink-3" aria-hidden />
      {props.label}
    </h3>
  )
}

/** Advanced settings (summoned tier) — most analysts never open this; defaults do the work. */
export function SettingsDrawer(props: {
  settings: RunSettings
  onChange: (s: RunSettings) => void
  engines: EngineInfo[]
  engine: string
  onEngineChange: (key: string) => void
  /** Scan-check assessment for the active document (empty until it loads). */
  checks?: SuggestCheck[]
  /** Raw scan scores backing the checks — pick the phrasing tier per finding. */
  scores?: Suggestion['scores'] | null
  /** Auto-suggested toggle key → rationale line (badge shown while present). */
  auto?: Record<string, string>
  /** The user changed a toggle: its Auto badge no longer applies. */
  onAutoOverride?: (k: string) => void
  /** Telemetry-bar jump target: scroll to + pulse this flag's row (n re-triggers). */
  highlight?: { k: string; n: number } | null
  pageCount: number
  /** What the last run's 'Auto' choices resolved to for the active document —
      the engine key the router used, and the concrete render DPI. */
  effectiveEngine?: string | null
  effectiveDpi?: number | null
  /** A run is in flight: its parameters are frozen until it finishes. */
  disabled?: boolean
  onClose: () => void
}) {
  const { settings, onChange, engines, engine, onEngineChange, checks = [], scores = null, auto = {}, onAutoOverride, highlight = null, pageCount, effectiveEngine = null, effectiveDpi = null, disabled = false, onClose } = props
  const { t } = useT()
  const rowRefs = useRef(new Map<string, HTMLDivElement>())
  const [pulsing, setPulsing] = useState<string | null>(null)
  useEffect(() => {
    if (!highlight) return
    const el = rowRefs.current.get(highlight.k)
    if (!el) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    el.scrollIntoView({ block: 'center', behavior: reduced ? 'auto' : 'smooth' })
    setPulsing(highlight.k)
    const id = setTimeout(() => setPulsing(null), 1600)
    return () => clearTimeout(id)
  }, [highlight])
  const set = (k: string, v: unknown) => onChange({ ...settings, [k]: v })
  const bool = (k: string) => Boolean(settings[k])
  const scope = String(settings.page_scope ?? 'all')
  // The router reports an engine KEY; show the same label the card carries.
  const resolvedEngineLabel =
    effectiveEngine && effectiveEngine !== 'auto'
      ? (engines.find((e2) => e2.key === effectiveEngine)?.label ?? effectiveEngine)
      : null
  const dpiIsAuto = String(settings.dpi ?? 'auto') === 'auto'

  return (
    <div className="flex h-full min-h-0 w-96 shrink-0 flex-col max-[1279px]:w-80">
      <div className="flex h-10 shrink-0 items-center justify-between whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3">
        <span className="flex min-w-0 items-baseline gap-2 overflow-hidden">
          <span className="text-sm font-semibold text-ink">{t('extraction_settings')}</span>
          <span className="truncate text-xs text-ink-2">{t('settings_subtitle')}</span>
        </span>
        <button className={iconBtnCls} onClick={onClose} aria-label={t('close_settings')}>
          <X size={14} aria-hidden />
        </button>
      </div>
      {/* Solid, continuous scroll surface: every section carries its own explicit
          spacing block (no parent-selector magic), last one pads the bottom radius. */}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto bg-surface px-4 pb-8 pt-5 text-sm">
        {/* The engine is a run-setup decision, not an every-minute control. */}
        <section className="mt-5 border-t border-line-strong/30 pt-5 first:mt-0 first:border-0 first:pt-0">
          <SectionTitle icon={Sparkles} label={t('engine_section')} />
          {/* Selection deck: each engine is a bounded option card; the chosen one
              carries the ring + tint, unchosen cards stay quiet. */}
          <div className="space-y-1.5" role="radiogroup" aria-label={t('engine_section')}>
            {engines.map((e2) => {
              const selected = e2.key === engine
              return (
                <button
                  key={e2.key}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  disabled={disabled}
                  onClick={() => onEngineChange(e2.key)}
                  className={`group flex w-full items-start gap-2.5 rounded-lg border p-2.5 text-left transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-primary ${
                    selected
                      ? 'border-primary/60 bg-primary-soft shadow-raised'
                      : 'border-line-strong/30 bg-rail/20 hover:border-line-strong hover:bg-rail'
                  }`}
                >
                  <span
                    aria-hidden
                    className={`mt-[3px] h-3 w-3 shrink-0 rounded-full border transition-colors duration-150 ${
                      selected ? 'border-4 border-primary bg-surface' : 'border-line-strong bg-surface group-hover:border-ink-3'
                    }`}
                  />
                  <span className="min-w-0">
                    <span className={`block text-sm leading-5 ${selected ? 'font-semibold text-primary-strong' : 'font-medium text-ink'}`}>
                      {e2.label}
                      {/* Only the Auto card, only once the router has ruled. */}
                      {selected && e2.key === 'auto' && resolvedEngineLabel && (
                        <ResolvedBadge
                          text={t('auto_resolved_engine', { v: resolvedEngineLabel })}
                          title={t('auto_resolved_engine_tip', { v: resolvedEngineLabel })}
                        />
                      )}
                    </span>
                    {e2.guidance && <span className="mt-0.5 block text-xs leading-4 text-ink-2">{e2.guidance}</span>}
                  </span>
                </button>
              )
            })}
          </div>
        </section>

        <section className="mt-5 border-t border-line-strong/30 pt-5 first:mt-0 first:border-0 first:pt-0">
          <SectionTitle icon={Files} label={t('pages')} />
          {/* Stacked rows with block labels — no jagged side-by-side alignment. */}
          <div className="mb-3">
            <span className="mb-1 block text-xs font-medium text-ink-2">
              {t('dpi')}
              {dpiIsAuto && effectiveDpi && (
                <ResolvedBadge
                  text={t('auto_resolved_dpi', { n: effectiveDpi })}
                  title={t('auto_resolved_dpi_tip', { n: effectiveDpi })}
                />
              )}
            </span>
            {/* The shared segment control; 'Auto' leads — it reads the document's
                density and picks 200 or 300. Values ride as strings, stored as
                'auto' | number to match the API contract. */}
            <SegmentedToggle
              value={String(settings.dpi ?? 'auto')}
              onChange={(v) => set('dpi', v === 'auto' ? 'auto' : Number(v))}
              label={t('dpi')}
              disabled={disabled}
              options={[
                ['auto', t('dpi_auto'), t('dpi_auto_tip')],
                ['150', '150'],
                ['200', '200'],
                ['300', '300'],
              ] as const}
            />
          </div>
          <span className="mb-1 block text-xs font-medium text-ink-2">{t('pages')}</span>
          <div className="flex items-center gap-2">
            <select className={`${inputCls} min-w-0 flex-1 pr-6`} disabled={disabled}
                    value={scope} onChange={(e) => set('page_scope', e.target.value)}>
              <option value="all">{t('all_pages')}</option>
              <option value="single">{t('single_page')}</option>
              <option value="range">{t('page_range')}</option>
              {/* Appears only while the grid overview drives a disjoint selection;
                  choosing any other option exits list mode normally. */}
              {scope === 'list' && (
                <option value="list">
                  {t('scope_list_option', { n: ((settings.page_list as number[] | undefined) ?? []).length })}
                </option>
              )}
            </select>
            {scope === 'single' && (
              <input type="number" disabled={disabled} min={1} max={Math.max(1, pageCount)} className={`${inputCls} w-16 px-1`}
                     value={Number(settings.page_num ?? 1)} onChange={(e) => set('page_num', Number(e.target.value))} />
            )}
            {scope === 'range' && (
              <>
                <input type="number" disabled={disabled} min={1} className={`${inputCls} w-14 px-1`}
                       aria-label={t('first_page')}
                       value={Number(settings.page_start ?? 1)} onChange={(e) => set('page_start', Number(e.target.value))} />
                <span>–</span>
                <input type="number" disabled={disabled} min={1} className={`${inputCls} w-14 px-1`}
                       aria-label={t('last_page')}
                       value={Number(settings.page_end ?? 5)} onChange={(e) => set('page_end', Number(e.target.value))} />
              </>
            )}
          </div>
          {scope === 'range' && Number(settings.page_end ?? 5) < Number(settings.page_start ?? 1) && (
            <p className="mt-1 text-xs font-medium text-danger-ink">{t('range_error')}</p>
          )}
          {scope === 'single' && pageCount > 0 && Number(settings.page_num ?? 1) > pageCount && (
            <p className="mt-1 text-xs font-medium text-danger-ink">{t('single_error', { n: pageCount })}</p>
          )}
        </section>

        <section className="mt-5 border-t border-line-strong/30 pt-5 first:mt-0 first:border-0 first:pt-0">
          <SectionTitle icon={Eraser} label={t('page_cleanup')} />
          {/* What the scan check found — the permanent record of "what was done". */}
          {checks.length > 0 && (
            <div className="mb-1.5 rounded-md border border-line-strong/30 bg-rail/20 p-2">
              <p className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-ink">
                <ScanSearch size={12} className="text-primary" aria-hidden />
                {t('scan_check_title')}
              </p>
              {/* Items in the SAME order as the switches below, so each finding sits
                  directly above the toggle it explains. */}
              <ul className="space-y-0.5">
                {[...checks]
                  .sort((a, b) =>
                    PREPROCESS_FLAGS.findIndex(([k]) => k === a.field) -
                    PREPROCESS_FLAGS.findIndex(([k]) => k === b.field))
                  .map((c) => (
                  <li key={c.field} className="flex items-start gap-1.5 text-xs text-ink-2" title={c.detail}>
                    {c.active ? (
                      <Check size={12} className="mt-0.5 shrink-0 text-ok" aria-hidden />
                    ) : (
                      /* Neutral finding: a quiet dot, same optical slot as the check. */
                      <span className="mx-[3px] mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-ink-3/50" aria-hidden />
                    )}
                    <span className="min-w-0">{scores ? t(scanWordingKey(c, scores)) : c.detail}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="space-y-1.5">
            {PREPROCESS_FLAGS.map(([k, labelKey, hintKey]) => (
              <div
                key={k}
                ref={(el) => { if (el) rowRefs.current.set(k, el); else rowRefs.current.delete(k) }}
                className={`flex items-start justify-between gap-3 rounded-md border p-2 transition-[box-shadow,border-color] duration-150 ${
                  pulsing === k ? 'border-primary ring-2 ring-primary/40' : 'border-line-strong/30'
                } bg-rail/20`}
              >
                <span className="min-w-0">
                  <span className="text-sm font-semibold text-ink">
                    {t(labelKey)}
                    {/* Present only where the scan check made the call — and it says
                        which way, so a step it disabled stays auditable without ever
                        looking like it is running. */}
                    {(() => {
                      const badge = autoBadge(bool(k), k in auto)
                      if (badge === null) return null
                      return (
                        <span
                          className={`ml-1.5 rounded px-1.5 py-0.5 text-2xs font-semibold ${
                            badge === 'applied' ? 'bg-ok-soft text-ok-ink' : 'bg-rail text-ink-2'
                          }`}
                        >
                          {t(badge === 'applied' ? 'auto_applied' : 'auto_off')}
                        </span>
                      )
                    })()}
                  </span>
                  <span className="mt-1 block text-xs leading-4 text-ink-2">{t(hintKey)}</span>
                </span>
                <Switch
                  checked={bool(k)}
                  disabled={disabled}
                  onChange={(v) => {
                    set(k, v)
                    onAutoOverride?.(k)
                  }}
                  label={t(labelKey)}
                />
              </div>
            ))}
          </div>
        </section>

        <section className="mt-5 border-t border-line-strong/30 pt-5 first:mt-0 first:border-0 first:pt-0">
          <SectionTitle icon={ScanSearch} label={t('ai_correction')} />
          <div className="flex items-start justify-between gap-3 rounded-md border border-line-strong/30 bg-rail/20 p-2">
            <span className="min-w-0 text-sm font-semibold text-ink">{t('ai_enable')}</span>
            <Switch checked={bool('enable_qwen')} disabled={disabled} onChange={(v) => set('enable_qwen', v)} label={t('ai_correction')} />
          </div>
          {bool('enable_qwen') && (
            <label className="mt-2 flex items-center justify-between">
              <span className="text-ink-2">{t('anomaly')}</span>
              <input type="number" disabled={disabled} step={0.05} min={0} max={1} className={`${inputCls} w-20 px-1`}
                     value={Number(settings.anomaly_threshold ?? 0.15)}
                     onChange={(e) => set('anomaly_threshold', Number(e.target.value))} />
            </label>
          )}
        </section>

        {/* Export settings close the drawer: the last decisions before files leave. */}
        <section className="mt-5 border-t border-line-strong/30 pt-5 first:mt-0 first:border-0 first:pt-0">
          <SectionTitle icon={FileOutput} label={t('output')} />
          <div className="space-y-1.5">
            {OUTPUT_FLAGS.map(([k, labelKey, hintKey]) => (
              <div key={k} className="flex items-start justify-between gap-3 rounded-md border border-line-strong/30 bg-rail/20 p-2">
                <span className="min-w-0">
                  <span className="text-sm font-semibold text-ink">{t(labelKey)}</span>
                  <span className="mt-1 block text-xs leading-4 text-ink-2">{t(hintKey)}</span>
                </span>
                <Switch checked={bool(k)} disabled={disabled} onChange={(v) => set(k, v)} label={t(labelKey)} />
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs text-ink-2">{t('join_note')}</p>
        </section>
      </div>
    </div>
  )
}
