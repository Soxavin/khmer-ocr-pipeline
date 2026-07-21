import { describe, expect, it } from 'vitest'
import { bandCells, bucketCounts, confBand, nextInBand } from './confidence'
import type { PageTable } from '../api/types'

describe('confBand (Check <80 · Skim 80–95 · Clean >95)', () => {
  it('buckets by the exact band cutoffs', () => {
    expect(confBand(0.5)).toBe('check')
    expect(confBand(0.799)).toBe('check')
    expect(confBand(0.8)).toBe('skim') // 80% is the start of Skim
    expect(confBand(0.94)).toBe('skim')
    expect(confBand(0.95)).toBe('clean') // 95% is the start of Clean
    expect(confBand(1)).toBe('clean')
  })
  it('treats a missing confidence as unranked (null)', () => {
    expect(confBand(null)).toBeNull()
    expect(confBand(undefined)).toBeNull()
  })
})

describe('bucketCounts', () => {
  it('pools a flat list into the three bands, ignoring nulls', () => {
    expect(bucketCounts([0.4, 0.85, 0.99, null, 0.7, 0.96])).toEqual({ check: 2, skim: 1, clean: 2 })
  })
  it('is all-zero for an empty list', () => {
    expect(bucketCounts([])).toEqual({ check: 0, skim: 0, clean: 0 })
  })
})

const table = (id: string, conf: (number | null)[][]): PageTable => ({
  table_id: id,
  grid: conf.map((r) => r.map(() => '')),
  original_grid: [],
  confidence: conf,
  edited: false,
  verified: false,
})

describe('bandCells (row-major cell lists per band, for the jump cursor)', () => {
  const tables = [
    table('t1', [[0.4, 0.9], [0.99, null]]),
    table('t2', [[0.6]]),
  ]
  it('collects cells per band in table→row→col order', () => {
    const b = bandCells(tables)
    expect(b.check).toEqual([
      { table_id: 't1', row: 0, col: 0 },
      { table_id: 't2', row: 0, col: 0 },
    ])
    expect(b.skim).toEqual([{ table_id: 't1', row: 0, col: 1 }])
    expect(b.clean).toEqual([{ table_id: 't1', row: 1, col: 0 }])
  })
  it('skips null (unranked) cells', () => {
    const b = bandCells([table('t', [[null, null]])])
    expect(b.check).toEqual([])
    expect(b.skim).toEqual([])
    expect(b.clean).toEqual([])
  })
})

describe('nextInBand (wrapping cursor advance)', () => {
  it('advances then wraps to 0', () => {
    expect(nextInBand(3, -1)).toBe(0)
    expect(nextInBand(3, 0)).toBe(1)
    expect(nextInBand(3, 2)).toBe(0)
  })
  it('returns -1 for an empty band', () => {
    expect(nextInBand(0, -1)).toBe(-1)
  })
})
