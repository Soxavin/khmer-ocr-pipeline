import { X } from 'lucide-react'
import type { RunSettings } from '../../api/types'
import { iconBtnCls, selectCls } from '../../ui'

const PREPROCESS_FLAGS: [string, string][] = [
  ['deskew', 'Deskew'],
  ['remove_stamps', 'Remove stamps'],
  ['sharpen', 'Sharpen'],
  ['normalise', 'Enhance contrast'],
  ['normalise_table_backgrounds', 'Normalise table backgrounds'],
]
// NOTE: joining tables across pages is deliberately NOT here — it is an export
// choice, not an extraction one. Extraction always keeps per-page tables so the
// review panel can link every row to the page image it came from.
const OUTPUT_FLAGS: [string, string, string][] = [
  ['repair_tables', 'Repair table structure', 'Fills ragged rows the recogniser left uneven.'],
  ['convert_numerals', 'Convert Khmer numerals to Arabic', 'Writes ១២៣ as 123 in exports.'],
]

const PREPROCESS_HINTS: Record<string, string> = {
  deskew: 'Straightens a scan that was fed in crooked.',
  remove_stamps: 'Erases signature stamps that sit over the numbers.',
  sharpen: 'Crispens soft or faxed scans.',
  normalise: 'Evens out faded or unevenly lit pages.',
  normalise_table_backgrounds: 'Flattens coloured table shading so text reads cleanly.',
}

/** Advanced settings (summoned tier) — most analysts never open this; defaults do the work. */
export function SettingsDrawer(props: {
  settings: RunSettings
  onChange: (s: RunSettings) => void
  pageCount: number
  onClose: () => void
}) {
  const { settings, onChange, pageCount, onClose } = props
  const set = (k: string, v: unknown) => onChange({ ...settings, [k]: v })
  const bool = (k: string) => Boolean(settings[k])
  const scope = String(settings.page_scope ?? 'all')

  return (
    <div className="flex w-80 shrink-0 flex-col border-l border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
        <span className="text-sm font-semibold text-slate-800">Extraction settings</span>
        <button className={iconBtnCls} onClick={onClose} aria-label="Close settings">
          <X size={14} aria-hidden />
        </button>
      </div>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3 text-sm">
        <label className="flex items-center justify-between">
          <span>Scan quality (DPI)</span>
          <select className={selectCls}
                  value={Number(settings.dpi ?? 200)} onChange={(e) => set('dpi', Number(e.target.value))}>
            {[150, 200, 300].map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>

        <fieldset>
          <legend className="mb-1 font-medium text-slate-600">Pages</legend>
          <div className="flex items-center gap-2">
            <select className={selectCls}
                    value={scope} onChange={(e) => set('page_scope', e.target.value)}>
              <option value="all">All pages</option>
              <option value="single">Single page</option>
              <option value="range">Page range</option>
            </select>
            {scope === 'single' && (
              <input type="number" min={1} max={Math.max(1, pageCount)} className="w-16 rounded border border-slate-300 px-1 py-0.5"
                     value={Number(settings.page_num ?? 1)} onChange={(e) => set('page_num', Number(e.target.value))} />
            )}
            {scope === 'range' && (
              <>
                <input type="number" min={1} className="w-14 rounded border border-slate-300 px-1 py-0.5"
                       aria-label="First page"
                       value={Number(settings.page_start ?? 1)} onChange={(e) => set('page_start', Number(e.target.value))} />
                <span>–</span>
                <input type="number" min={1} className="w-14 rounded border border-slate-300 px-1 py-0.5"
                       aria-label="Last page"
                       value={Number(settings.page_end ?? 5)} onChange={(e) => set('page_end', Number(e.target.value))} />
              </>
            )}
          </div>
          {scope === 'range' && Number(settings.page_end ?? 5) < Number(settings.page_start ?? 1) && (
            <p className="mt-1 text-xs font-medium text-red-700">The last page is before the first — fix the range before running.</p>
          )}
          {scope === 'single' && pageCount > 0 && Number(settings.page_num ?? 1) > pageCount && (
            <p className="mt-1 text-xs font-medium text-red-700">This document has only {pageCount} page{pageCount === 1 ? '' : 's'}.</p>
          )}
        </fieldset>

        {/* Help lives where the decision is: one plain line under each control,
            so a colleague inheriting this never has to guess or hover. */}
        <fieldset>
          <legend className="mb-1 font-medium text-slate-600">Page cleanup</legend>
          {PREPROCESS_FLAGS.map(([k, label]) => (
            <label key={k} className="flex gap-2 py-1">
              <input type="checkbox" className="mt-0.5" checked={bool(k)} onChange={(e) => set(k, e.target.checked)} />
              <span>
                {label}
                <span className="block text-xs text-slate-500">{PREPROCESS_HINTS[k]}</span>
              </span>
            </label>
          ))}
        </fieldset>

        <fieldset>
          <legend className="mb-1 font-medium text-slate-600">Output</legend>
          {OUTPUT_FLAGS.map(([k, label, hint]) => (
            <label key={k} className="flex gap-2 py-1">
              <input type="checkbox" className="mt-0.5" checked={bool(k)} onChange={(e) => set(k, e.target.checked)} />
              <span>
                {label}
                <span className="block text-xs text-slate-500">{hint}</span>
              </span>
            </label>
          ))}
          <p className="mt-1 text-xs text-slate-500">
            Joining tables that continue across pages is chosen when you export, not here — so review always
            stays linked to the page each row came from.
          </p>
        </fieldset>

        <fieldset>
          <legend className="mb-1 font-medium text-slate-600">AI text correction</legend>
          <label className="flex items-center gap-2 py-0.5">
            <input type="checkbox" checked={bool('enable_qwen')} onChange={(e) => set('enable_qwen', e.target.checked)} />
            Enable (slower; uses the local correction model)
          </label>
          {bool('enable_qwen') && (
            <label className="flex items-center justify-between py-0.5">
              <span>Anomaly threshold</span>
              <input type="number" step={0.05} min={0} max={1} className="w-20 rounded border border-slate-300 px-1 py-0.5"
                     value={Number(settings.anomaly_threshold ?? 0.15)}
                     onChange={(e) => set('anomaly_threshold', Number(e.target.value))} />
            </label>
          )}
        </fieldset>

        <p className="text-xs text-slate-500">
          Changes apply to the next run. If results were made with different settings, a
          “settings changed” notice appears until you re-run.
        </p>
      </div>
    </div>
  )
}
