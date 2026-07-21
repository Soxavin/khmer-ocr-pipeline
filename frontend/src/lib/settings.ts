import type { RunSettings, SuggestCheck } from '../api/types'

/** What badge a preprocessing row shows. The badge answers ONE question — did the
    scan check decide this row? — because the switch already answers "is it on".
    `applied` — the scan check switched this step on.
    `auto-off` — the scan check switched it off (neutral: it states the step is off
      while keeping the decision auditable; every suggestion this backend emits is
      a turn-off, so erasing it would hide all of automation's work).
    `null` — the operator's own choice, or an untouched default: no badge. */
export type AutoBadge = 'applied' | 'auto-off' | null

export function autoBadge(on: boolean, isAuto: boolean): AutoBadge {
  if (!isAuto) return null
  return on ? 'applied' : 'auto-off'
}

/** Fold an advisory suggestion into the current settings WITHOUT overruling the
    operator: any key they changed by hand keeps its value, even if the scan
    check disagrees. Automation advises; the person decides. */
export function mergeSuggestion(
  prev: RunSettings,
  suggested: Record<string, boolean>,
  touched: Set<string>,
): RunSettings {
  const next = { ...prev }
  for (const [k, v] of Object.entries(suggested)) {
    if (!touched.has(k)) next[k] = v
  }
  return next
}

// The settings the drawer actually exposes as distinct controls. The badge counts
// deviations over THESE only, so seeded or stale non-UI fields (show_layout,
// overlay_mode, tables_only, stitch_pages, ocr_engine_key) never inflate it. Page
// sub-fields (page_num/start/end/list) are omitted so one page-scope change counts
// once, not several times.
const OVERRIDE_KEYS = [
  'dpi', 'page_scope',
  'remove_stamps', 'sharpen', 'normalise', 'deskew', 'normalise_table_backgrounds',
  'enable_qwen', 'anomaly_threshold',
  'repair_tables', 'convert_numerals',
] as const

/** How many user-facing controls deviate from their defaults — the number on the
    Settings badge. 0 (badge hidden) when the configuration is untouched. */
export function countOverrides(settings: RunSettings, defaults: RunSettings | undefined): number {
  if (!defaults) return 0
  return OVERRIDE_KEYS.filter(
    (k) => k in defaults && k in settings && JSON.stringify(settings[k]) !== JSON.stringify(defaults[k]),
  ).length
}

export type ScanSummary = { total: number; active: number; fields: string[] }

/** One-line digest of a document's scan check, for the post-upload notice.
    Null when there is nothing to report yet. */
export function scanSummary(checks: SuggestCheck[]): ScanSummary | null {
  if (checks.length === 0) return null
  const fields = checks.filter((c) => c.active).map((c) => c.field)
  return { total: checks.length, active: fields.length, fields }
}
