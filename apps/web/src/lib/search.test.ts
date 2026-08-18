import { describe, expect, it } from 'vitest'
import { findMatches, matchIndexAt } from './search'
import type { PageTable } from '../api/types'

const table = (id: string, grid: string[][]): PageTable => ({
  table_id: id,
  grid,
  original_grid: grid,
  confidence: grid.map((r) => r.map(() => 1)),
  edited: false,
  verified: false,
})

describe('findMatches', () => {
  it('is empty for an empty or whitespace-only query — not "everything matches"', () => {
    const t = [table('t1', [['a', 'b']])]
    expect(findMatches(t, '')).toEqual([])
    expect(findMatches(t, '   ')).toEqual([])
  })

  it('orders matches table, then row, then column', () => {
    const tables = [
      table('t1', [['x', 'a'], ['x', 'x']]),
      table('t2', [['a', 'x']]),
    ]
    expect(findMatches(tables, 'x')).toEqual([
      { table_id: 't1', row: 0, col: 0 },
      { table_id: 't1', row: 1, col: 0 },
      { table_id: 't1', row: 1, col: 1 },
      { table_id: 't2', row: 0, col: 1 },
    ])
  })

  it('matches substrings, case-insensitively', () => {
    const t = [table('t1', [['Total Revenue', 'total']])]
    expect(findMatches(t, 'TOTAL')).toHaveLength(2)
    expect(findMatches(t, 'reven')).toHaveLength(1)
  })

  it('matches Khmer as a plain substring', () => {
    const t = [table('t1', [['ខ្ញុំ', 'x']])]
    expect(findMatches(t, 'ខ្ញុំ')).toHaveLength(1)
  })

  it('DOCUMENTS A LIMIT: a non-canonical Khmer mark order does not match', () => {
    // Not aspiration — the current, honest behaviour. Khmer combining marks all
    // have combining class 0, so NFC never reorders them; these two strings render
    // identically but stay distinct. Stored cell text is normalized by the Python
    // `khmer_normalize`, which has no TS twin, so a query typed in another order
    // misses. Flip this assertion when that normalizer is ported.
    const canonical = 'ខ្ញុំ' // ខ ្ញ ុ ំ
    const swapped = 'ខ្ញំុ' // ខ ្ញ ំ ុ — same word, marks transposed
    expect(canonical).not.toBe(swapped)
    expect(findMatches([table('t1', [[canonical]])], swapped)).toHaveLength(0)
  })

  it('skips empty cells rather than counting them as matches', () => {
    expect(findMatches([table('t1', [['', 'a']])], '')).toEqual([])
  })

  it('tolerates ragged rows without throwing', () => {
    const t = [table('t1', [['a'], ['a', 'a']])]
    expect(findMatches(t, 'a')).toHaveLength(3)
  })
})

describe('matchIndexAt', () => {
  const ms = [
    { table_id: 't1', row: 0, col: 0 },
    { table_id: 't2', row: 1, col: 2 },
  ]

  it('locates a cell in the match list so the grid can mark the active one', () => {
    expect(matchIndexAt(ms, 't2', 1, 2)).toBe(1)
  })

  it('returns -1 when the cell is not a match', () => {
    expect(matchIndexAt(ms, 't1', 5, 5)).toBe(-1)
  })
})
