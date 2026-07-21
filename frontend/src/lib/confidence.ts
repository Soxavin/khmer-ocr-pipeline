import type { PageTable } from '../api/types'

// One confidence vocabulary for the whole workspace (Page Viewer boxes AND grid
// cells). Bands: Check <80% · Skim 80–95% · Clean >95%. Kept here so the two
// surfaces can never drift apart.
export const CHECK_MAX = 0.8 // below → Check
export const SKIM_MAX = 0.95 // [0.8, 0.95) → Skim; ≥0.95 → Clean

export type Band = 'check' | 'skim' | 'clean'

export function confBand(v: number | null | undefined): Band | null {
  if (v === null || v === undefined) return null
  if (v < CHECK_MAX) return 'check'
  if (v < SKIM_MAX) return 'skim'
  return 'clean'
}

export type BandCounts = { check: number; skim: number; clean: number }

/** Pool a flat list of confidences into the three bands; nulls are unranked. */
export function bucketCounts(values: (number | null | undefined)[]): BandCounts {
  const counts: BandCounts = { check: 0, skim: 0, clean: 0 }
  for (const v of values) {
    const b = confBand(v)
    if (b) counts[b] += 1
  }
  return counts
}

export type BandCell = { table_id: string; row: number; col: number }
export type BandCells = { check: BandCell[]; skim: BandCell[]; clean: BandCell[] }

/** Per-band cell lists in table→row→col order — the ordered target list the
    triage "jump to next" cursor walks. */
export function bandCells(tables: PageTable[]): BandCells {
  const out: BandCells = { check: [], skim: [], clean: [] }
  for (const t of tables) {
    t.confidence.forEach((rowConf, row) =>
      rowConf.forEach((v, col) => {
        const b = confBand(v)
        if (b) out[b].push({ table_id: t.table_id, row, col })
      }),
    )
  }
  return out
}

/** Wrapping advance of a band cursor; -1 for an empty band. */
export function nextInBand(length: number, cursor: number): number {
  if (length === 0) return -1
  return (cursor + 1) % length
}
