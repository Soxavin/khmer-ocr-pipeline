// Shapes returned by webapp/api.py — keep in lockstep with the handlers.

export type EngineInfo = { key: string; label: string; guidance: string }

export type Meta = {
  engines: EngineInfo[]
  defaults: Record<string, unknown>
  setting_fields: string[]
  backend_ready: boolean
}

export type DocStatus = 'queued' | 'running' | 'done' | 'error' | 'stopped'

export type DocSummary = {
  id: string
  name: string
  pages: number
  size_kb: number
  status: DocStatus
  total_tables: number
  reviewed_tables: number
}

export type RunStatus = {
  active: boolean
  stage: string
  /** Sub-stage within the current stage ("layout"/"text"/"tables"); "" if unknown. */
  step: string
  page: number
  total: number
  fraction: number
  has_results: boolean
  run_error: string | null
  last_run_settings: Record<string, unknown> | null
}

export type PageTable = {
  table_id: string
  grid: string[][]
  original_grid: string[][]
  confidence: (number | null)[][]
  edited: boolean
  verified: boolean
}

export type Issue = {
  page: number | null
  table_id: string
  row: number
  col: number
  conf: number | null
  text: string
  reason: string
  reasons: string[]
}

export type TextBlock = { bbox: number[]; confidence?: number | null; label?: string }

export type PageData = {
  corrected_text: string
  tables: PageTable[]
  text_blocks: TextBlock[]
  table_bboxes: (number[] | null)[]
  table_bbox_index: Record<string, number[]>
  qwen_used: boolean
}

export type Overview = {
  pages: number
  total_tables: number
  warnings: string[]
  stitched: boolean
  stage_times: Record<string, number>
}

export type RunSettings = Record<string, unknown>

// GET /documents/{id}/suggest — advisory preprocessing suggestions. `suggested`
// holds only toggles deviating from the defaults (usually empty); `rationale`
// mirrors its keys with one plain-English sentence each.
export type SuggestCheck = {
  field: string
  active: boolean // "this cleanup is useful for THIS document"
  reason: string // stable key, localized by the frontend
  detail: string // measured evidence (English, tooltip/fallback)
}

export type Suggestion = {
  scores: { laplacian_var: number; contrast_std: number; skew_deg: number; stamp_ink_ratio: number }
  suggested: Record<string, boolean>
  rationale: Record<string, string>
  checks: SuggestCheck[]
}
