import { describe, expect, it } from 'vitest'
import { encodePages, gridPages, pagesFromSettings, processedIndex } from './pages'

describe('gridPages (post-analysis filtering)', () => {
  it('pre-upload mode shows every document page', () => {
    expect(gridPages('pre-upload', 5, null)).toEqual([0, 1, 2, 3, 4])
  })

  it('post-analysis drops unselected indices completely (range run)', () => {
    const lastRun = { page_scope: 'range', page_start: 2, page_end: 4 }
    expect(gridPages('post-analysis', 6, lastRun)).toEqual([1, 2, 3])
  })

  it('post-analysis drops unselected indices completely (disjoint list run)', () => {
    const lastRun = { page_scope: 'list', page_list: [1, 3, 6] }
    expect(gridPages('post-analysis', 6, lastRun)).toEqual([0, 2, 5])
  })

  it('post-analysis single-page run keeps exactly one entry', () => {
    expect(gridPages('post-analysis', 6, { page_scope: 'single', page_num: 4 })).toEqual([3])
  })

  it('post-analysis without recorded settings falls back to all pages', () => {
    expect(gridPages('post-analysis', 3, null)).toEqual([0, 1, 2])
  })
})

describe('pagesFromSettings never invents a page the document does not have', () => {
  it('a range starting past the end clamps to the last real page', () => {
    // The bug: start=2, end=min(5,1)=1, then Math.max(1, end-start) FORCED one
    // entry — page index 2 of a one-page document. The grid then rendered a card
    // for a page that does not exist.
    const s = { page_scope: 'range', page_start: 3, page_end: 5 }
    expect(Array.from(pagesFromSettings(s, 1))).toEqual([0])
  })

  it('a range running past the end keeps only the pages that exist', () => {
    expect(Array.from(pagesFromSettings({ page_scope: 'range', page_start: 2, page_end: 99 }, 3)))
      .toEqual([1, 2])
  })

  it('an inverted range still yields one real page, never a phantom', () => {
    expect(Array.from(pagesFromSettings({ page_scope: 'range', page_start: 4, page_end: 2 }, 6)))
      .toEqual([3])
  })

  it('a single page past the end clamps instead of going negative', () => {
    expect(Array.from(pagesFromSettings({ page_scope: 'single', page_num: 9 }, 3))).toEqual([2])
  })

  it('a document with no pages yields nothing at all (never index -1)', () => {
    expect(Array.from(pagesFromSettings({ page_scope: 'single', page_num: 1 }, 0))).toEqual([])
    expect(Array.from(pagesFromSettings({ page_scope: 'range', page_start: 1, page_end: 3 }, 0)))
      .toEqual([])
  })
})

describe('pagesFromSettings ⇄ encodePages round trip', () => {
  it('contiguity uses strict numeric sort (pages 9,10,11 collapse to a range)', () => {
    expect(encodePages(new Set([8, 9, 10]), 20)).toEqual({ page_scope: 'range', page_start: 9, page_end: 11 })
  })

  it('disjoint selection encodes as a list scope', () => {
    expect(encodePages(new Set([0, 2]), 5)).toEqual({ page_scope: 'list', page_list: [1, 3] })
  })

  it('full selection collapses to all', () => {
    expect(encodePages(new Set([0, 1, 2]), 3)).toEqual({ page_scope: 'all' })
  })

  it('list scope decodes back to the same set', () => {
    const s = pagesFromSettings({ page_scope: 'list', page_list: [1, 3] }, 5)
    expect(Array.from(s).sort((a, b) => a - b)).toEqual([0, 2])
  })
})

describe('processedIndex — mid-run rendition mapping', () => {
  it('returns -1 before stage 2 has produced anything', () => {
    expect(processedIndex(0, undefined)).toBe(-1)
    expect(processedIndex(0, [])).toBe(-1)
  })

  it('maps a document page to its POSITION, not its number, for a page-scoped run', () => {
    // Pages 3 and 7 were selected: result 0 is page 3, result 1 is page 7.
    expect(processedIndex(3, [3, 7])).toBe(0)
    expect(processedIndex(7, [3, 7])).toBe(1)
    expect(processedIndex(5, [3, 7])).toBe(-1)
  })

  it('is identity for a whole-document run', () => {
    expect(processedIndex(2, [0, 1, 2, 3])).toBe(2)
  })
})
