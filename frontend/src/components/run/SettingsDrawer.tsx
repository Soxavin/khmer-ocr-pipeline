import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import { Check, Cloud, Eraser, FileOutput, Files, FlaskConical, Laptop, ScanSearch, Sparkles, TriangleAlert, X } from 'lucide-react'
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
// Group-header labels for the engine picker, keyed by the API's `group` field. An
// unknown group falls back to its raw name so a future group still renders a header.
const ENGINE_GROUP_LABELS: Record<string, Key> = {
  local: 'engine_group_local',
  cloud: 'engine_group_cloud',
}
// Group-header micro-icons, keyed by `group`. Falls back to no icon for an unknown group.
const ENGINE_GROUP_ICONS: Record<string, typeof Laptop> = {
  local: Laptop,
  cloud: Cloud,
}
// Short tab labels (the switcher is compact); the full label rides as the tooltip + SR name.
const ENGINE_GROUP_LABELS_SHORT: Record<string, Key> = {
  local: 'engine_group_local_short',
  cloud: 'engine_group_cloud_short',
}

// NOTE: joining tables across pages is deliberately NOT here — it is an export
// choice, not an extraction one. Extraction always keeps per-page tables so the
// review panel can link every row to the page image it came from.
const OUTPUT_FLAGS: [string, Key, Key][] = [
  ['repair_tables', 'flag_repair', 'hint_repair'],
  ['convert_numerals', 'flag_numerals', 'hint_numerals'],
]

/** A real switch, not a checkbox: the drawer's clearest "designed" signal.
    36×20 track, 150ms knob travel, token colors, reduced-motion covered globally. */
function Switch(props: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean; id?: string }) {
  const { checked, onChange, label, disabled = false, id } = props
  return (
    <button
      type="button"
      role="switch"
      id={id}
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      // stopPropagation: when a SettingRow wraps this switch and makes the whole row
      // clickable, the row's own onClick must not also fire and double-toggle.
      onClick={(e) => { e.stopPropagation(); onChange(!checked) }}
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

/** One toggle row inside a SettingList — the shared vocabulary for the Preprocessing,
    AI-correction, and Output flags. The whole row is clickable (not just the switch);
    stopPropagation on the Switch keeps that from double-toggling. `rowRef` lands on the
    outer node so Preprocessing can register it for telemetry scroll-to + pulse. */
function SettingRow(props: {
  id: string
  label: string
  hint?: string
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
  badge?: ReactNode
  pulsing?: boolean
  rowRef?: (el: HTMLDivElement | null) => void
}) {
  const { id, label, hint, checked, onChange, disabled = false, badge, pulsing = false, rowRef } = props
  return (
    <div
      ref={rowRef}
      onClick={() => { if (!disabled) onChange(!checked) }}
      className={`flex cursor-pointer items-start justify-between gap-3 px-3 py-2 transition-colors duration-150 ${
        pulsing ? 'bg-primary-soft' : ''
      } ${disabled ? 'cursor-not-allowed' : ''}`}
    >
      <span className="min-w-0">
        <span className="text-sm font-semibold text-ink">
          {label}
          {badge}
        </span>
        {hint && <span className="mt-1 block text-xs leading-4 text-ink-2">{hint}</span>}
      </span>
      <Switch id={id} checked={checked} disabled={disabled} onChange={onChange} label={label} />
    </div>
  )
}

/** The single-perimeter divided container the rows live in — matches the engine card
    list, so the whole drawer speaks one "list" language instead of stacked boxes. */
function SettingList(props: { children: ReactNode }) {
  return (
    <div className="divide-y divide-line-strong/40 overflow-hidden rounded-lg border border-line-strong/60">
      {props.children}
    </div>
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
    <h3 className="mb-2.5 flex items-center gap-1.5 text-title font-semibold text-ink">
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
  /** Labs mode reveals the custom ARDB fine-tunes (experimental engines). */
  labsMode?: boolean
  onLabsModeChange?: (v: boolean) => void
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
  const { settings, onChange, engines, engine, onEngineChange, labsMode = false, onLabsModeChange, checks = [], scores = null, auto = {}, onAutoOverride, highlight = null, pageCount, effectiveEngine = null, effectiveDpi = null, disabled = false, onClose } = props
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
  // Engine label/guidance are localized via i18n when an entry exists (see i18n.tsx);
  // otherwise fall back to the backend English on EngineInfo (e.g. a future engine key).
  const localizedEngines = new Set(['auto', 'surya', 'surya_kiri', 'surya_kiri_vlm', 'gemini'])
  const engineLabel = (e2: EngineInfo) =>
    localizedEngines.has(e2.key) ? t(`engine_label_${e2.key}` as Parameters<typeof t>[0]) : e2.label
  const engineGuidance = (e2: EngineInfo) =>
    localizedEngines.has(e2.key) ? t(`engine_guidance_${e2.key}` as Parameters<typeof t>[0]) : e2.guidance
  // The router reports an engine KEY; show the same (localized) label the card carries.
  const resolvedEngine = engines.find((e2) => e2.key === effectiveEngine)
  const resolvedEngineLabel =
    effectiveEngine && effectiveEngine !== 'auto'
      ? (resolvedEngine ? engineLabel(resolvedEngine) : effectiveEngine)
      : null
  const dpiIsAuto = String(settings.dpi ?? 'auto') === 'auto'
  // Bucket engines by `group`, preserving first-encounter order (local before cloud,
  // as the API returns them). Driven entirely by the data — no hardcoded engine keys.
  const engineGroups: [string, EngineInfo[]][] = []
  for (const e2 of engines) {
    const bucket = engineGroups.find(([name]) => name === e2.group)
    if (bucket) bucket[1].push(e2)
    else engineGroups.push([e2.group, [e2]])
  }
  // The picker is tabbed by group: open on the selected engine's tab, but treat the
  // active tab as a pure VIEW filter — switching tabs never changes the selection.
  const selectedGroup = engines.find((e2) => e2.key === engine)?.group
  const [activeGroup, setActiveGroup] = useState<string>(selectedGroup ?? engineGroups[0]?.[0] ?? 'local')
  // Follow the selected engine ONLY when it actually changes (e.g. selected from
  // outside, or a persisted cloud engine on load) — an unrelated re-render must never
  // yank the tab back while the user is exploring the other group.
  const prevEngineRef = useRef(engine)
  useEffect(() => {
    if (engine !== prevEngineRef.current) {
      prevEngineRef.current = engine
      const g = engines.find((e2) => e2.key === engine)?.group
      if (g) setActiveGroup(g)
    }
  }, [engine, engines])
  // Roving-focus refs for the tablist: ArrowLeft/Right/Home/End move the active tab.
  const tabRefs = useRef(new Map<string, HTMLButtonElement>())
  const onTabKeyDown = (e: ReactKeyboardEvent) => {
    const names = engineGroups.map(([n]) => n)
    const cur = names.indexOf(activeGroup)
    let next = -1
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (cur + 1) % names.length
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (cur - 1 + names.length) % names.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = names.length - 1
    if (next < 0) return
    e.preventDefault()
    setActiveGroup(names[next])
    tabRefs.current.get(names[next])?.focus()
  }
  // The Labs toggle only exists once the API actually returns a fine-tune to reveal;
  // until the backend flags one, the whole feature stays invisible (no dead control).
  const hasExperimental = engines.some((e2) => e2.experimental)

  // One engine option card. Shared by the production list and the Labs subsection so
  // both render identically; only the grouping around them differs.
  const engineCard = (e2: EngineInfo) => {
    const selected = e2.key === engine
    const isCloud = e2.group === 'cloud'
    // 'Recommended.' rides in the backend guidance for the auto engine; lift it into a
    // crisp badge (detected off the canonical English). The localized caption already
    // omits it (`engine_guidance_auto`), so it is never said twice.
    const isRecommended = e2.key === 'auto' && / Recommended\.?$/.test(e2.guidance)
    const caption = engineGuidance(e2)
    return (
      <button
        key={e2.key}
        type="button"
        role="radio"
        aria-checked={selected}
        disabled={disabled}
        onClick={() => onEngineChange(e2.key)}
        className={`group flex w-full items-start gap-2 px-3 py-2 text-left transition-colors duration-100 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary ${
          selected ? 'bg-primary-soft' : 'hover:bg-rail/20'
        }`}
      >
        <span
          aria-hidden
          className={`mt-[5px] h-2.5 w-2.5 shrink-0 rounded-full border transition-colors duration-100 ${
            selected ? 'border-[3px] border-primary bg-surface' : 'border-line-strong bg-surface group-hover:border-ink-3'
          }`}
        />
        <span className="min-w-0 flex-1">
          <span className="flex items-baseline justify-between gap-2">
            <span className={`block text-sm leading-5 ${selected ? 'font-semibold text-primary-strong' : 'font-medium text-ink'}`}>
              {engineLabel(e2)}
              {/* The engine KEY as a muted technical id: analysts scan by the descriptive
                  name, but the precise id (surya_kiri, …) makes support/troubleshooting
                  unambiguous. It's an identifier, so it is never localized. */}
              <span className="ml-1.5 font-mono text-2xs font-normal text-ink-3">{e2.key}</span>
              {/* Only the Auto card, only once the router has ruled. */}
              {selected && e2.key === 'auto' && resolvedEngineLabel && (
                <ResolvedBadge
                  text={t('auto_resolved_engine', { v: resolvedEngineLabel })}
                  title={t('auto_resolved_engine_tip', { v: resolvedEngineLabel })}
                />
              )}
            </span>
            {isRecommended && (
              <span className="shrink-0 rounded bg-primary-soft px-1.5 py-0 text-2xs font-semibold text-primary-strong">
                {t('engine_recommended')}
              </span>
            )}
          </span>
          {/* Cloud guidance is a privacy caution — same integrated line as local
              captions, distinguished only by warn color + a small icon. */}
          {caption && (
            <span className={`mt-0.5 flex items-start gap-1 text-xs leading-4 ${isCloud ? 'text-warn-ink' : 'text-ink-2'}`}>
              {isCloud && <TriangleAlert size={11} className="mt-px shrink-0" aria-hidden />}
              <span className="min-w-0">{caption}</span>
            </span>
          )}
        </span>
      </button>
    )
  }

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
          {/* Labs gate — light row (no box), shown only when the API returns a fine-tune.
              Off → production-clean engine list; on → the Experimental subsection appears. */}
          {hasExperimental && onLabsModeChange && (
            <div className="mb-2.5 flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-xs text-ink-2" title={t('labs_mode_tip')}>
                <FlaskConical size={12} className="text-ink-3" aria-hidden />
                {t('labs_mode')}
              </span>
              <Switch checked={labsMode} disabled={disabled} onChange={onLabsModeChange} label={t('labs_mode')} />
            </div>
          )}
          {/* Group SWITCHER — the house segmented control (§DESIGN Inputs): a bordered
              joined row, active option takes primary-soft fill + primary-strong text,
              matching the DPI toggle right below. Tab semantics + roving keys are kept
              on top of that look; data-driven from `engineGroups`. */}
          <div
            role="tablist"
            aria-label={t('engine_section')}
            onKeyDown={onTabKeyDown}
            className="mb-3 flex w-full overflow-hidden rounded-md border border-line-strong"
          >
            {engineGroups.map(([groupName], gi) => {
              const GroupIcon = ENGINE_GROUP_ICONS[groupName]
              const active = groupName === activeGroup
              const fullLabel = ENGINE_GROUP_LABELS[groupName] ? t(ENGINE_GROUP_LABELS[groupName]) : groupName
              const shortLabel = ENGINE_GROUP_LABELS_SHORT[groupName] ? t(ENGINE_GROUP_LABELS_SHORT[groupName]) : fullLabel
              // The selected engine lives on this tab but it isn't showing: a marker dot
              // (plus an SR-only note) keeps "which engine is active" from hiding.
              const holdsHiddenSelection = !active && selectedGroup === groupName
              return (
                <button
                  key={groupName}
                  ref={(el) => { if (el) tabRefs.current.set(groupName, el); else tabRefs.current.delete(groupName) }}
                  type="button"
                  role="tab"
                  id={`engine-tab-${groupName}`}
                  aria-controls={`engine-panel-${groupName}`}
                  aria-selected={active}
                  tabIndex={active ? 0 : -1}
                  title={fullLabel}
                  aria-label={holdsHiddenSelection
                    ? `${fullLabel} (${t('contains_selected_engine')})`
                    : fullLabel}
                  onClick={() => setActiveGroup(groupName)}
                  className={`flex flex-1 items-center justify-center gap-1.5 h-7 px-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary ${
                    gi > 0 ? 'border-l border-line-strong' : ''
                  } ${active ? 'bg-primary-soft text-primary-strong' : 'bg-surface text-ink-2 hover:bg-rail'}`}
                >
                  {GroupIcon && <GroupIcon size={12} aria-hidden />}
                  {shortLabel}
                  {holdsHiddenSelection && <span aria-hidden className="ml-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />}
                </button>
              )
            })}
          </div>
          {/* Only the active group's cards render. ONE outer perimeter + hairline
              dividers; a SINGLE radiogroup wraps the cards. When Labs is on, the local
              group splits into production + an "Experimental" subsection (fine-tunes are
              local; the subheader stays inside the radiogroup so arrows span both). */}
          <div
            role="tabpanel"
            id={`engine-panel-${activeGroup}`}
            aria-labelledby={`engine-tab-${activeGroup}`}
          >
            <div
              className="divide-y divide-line-strong/40 overflow-hidden rounded-lg border border-line-strong/60"
              role="radiogroup"
              aria-label={t('engine_section')}
            >
              {(() => {
                const activeEngines = engineGroups.find(([n]) => n === activeGroup)?.[1] ?? []
                const production = activeEngines.filter((e2) => !e2.experimental)
                const experimental = activeEngines.filter((e2) => e2.experimental)
                return (
                  <>
                    {production.map(engineCard)}
                    {labsMode && experimental.length > 0 && (
                      <>
                        <p className="flex items-center gap-1.5 bg-rail/30 px-3 py-1.5 text-2xs font-semibold uppercase tracking-wide text-ink-3">
                          <FlaskConical size={12} aria-hidden />
                          {t('engine_group_experimental')}
                        </p>
                        {experimental.map(engineCard)}
                      </>
                    )}
                  </>
                )
              })()}
            </div>
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
          {/* What the scan check found — the permanent record of "what was done".
              Zones-not-borders: a light rail ground, no heavy border. */}
          {checks.length > 0 && (
            <div className="mb-1.5 rounded-md bg-rail/30 p-2">
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
          <SettingList>
            {PREPROCESS_FLAGS.map(([k, labelKey, hintKey]) => {
              // Present only where the scan check made the call — and it says which way,
              // so a step it disabled stays auditable without ever looking like it is running.
              const badge = autoBadge(bool(k), k in auto)
              return (
                <SettingRow
                  key={k}
                  id={`preprocess-${k}`}
                  rowRef={(el) => { if (el) rowRefs.current.set(k, el); else rowRefs.current.delete(k) }}
                  pulsing={pulsing === k}
                  label={t(labelKey)}
                  hint={t(hintKey)}
                  checked={bool(k)}
                  disabled={disabled}
                  onChange={(v) => { set(k, v); onAutoOverride?.(k) }}
                  badge={badge && (
                    <span
                      className={`ml-1.5 rounded px-1.5 py-0.5 text-2xs font-semibold ${
                        badge === 'applied' ? 'bg-ok-soft text-ok-ink' : 'bg-rail text-ink-2'
                      }`}
                    >
                      {t(badge === 'applied' ? 'auto_applied' : 'auto_off')}
                    </span>
                  )}
                />
              )
            })}
          </SettingList>
        </section>

        {/* Export settings close the drawer: the last decisions before files leave. */}
        <section className="mt-5 border-t border-line-strong/30 pt-5 first:mt-0 first:border-0 first:pt-0">
          <SectionTitle icon={FileOutput} label={t('output')} />
          <SettingList>
            {OUTPUT_FLAGS.map(([k, labelKey, hintKey]) => (
              <SettingRow
                key={k}
                id={`output-${k}`}
                label={t(labelKey)}
                hint={t(hintKey)}
                checked={bool(k)}
                disabled={disabled}
                onChange={(v) => set(k, v)}
              />
            ))}
          </SettingList>
          <p className="mt-2 text-xs text-ink-2">{t('join_note')}</p>
        </section>
      </div>
    </div>
  )
}
