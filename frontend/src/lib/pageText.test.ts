import { describe, expect, it } from 'vitest'
import { pairSegments, replaceSegment, segmentRange, splitSegments } from './pageText'
import type { TextBlock } from '../api/types'

const block = (over: Partial<TextBlock> = {}): TextBlock => ({ bbox: [0, 0, 1, 1], ...over })

describe('splitSegments', () => {
  it('splits on the blank line the backend joins blocks with', () => {
    expect(splitSegments('one\n\ntwo\n\nthree')).toEqual(['one', 'two', 'three'])
  })

  it('treats empty text as no segments, not one empty segment', () => {
    expect(splitSegments('')).toEqual([])
    expect(splitSegments('   ')).toEqual([])
  })

  it('keeps single newlines inside a segment — only a BLANK line separates blocks', () => {
    expect(splitSegments('line one\nline two\n\nnext')).toEqual(['line one\nline two', 'next'])
  })
})

describe('segmentRange', () => {
  const text = 'alpha\n\nbeta\n\ngamma'

  it('gives exact offsets for the first segment', () => {
    const { start, end } = segmentRange(text, 0)
    expect(text.slice(start, end)).toBe('alpha')
    expect(start).toBe(0)
  })

  it('accounts for every separator when locating a middle segment', () => {
    const { start, end } = segmentRange(text, 1)
    expect(text.slice(start, end)).toBe('beta')
    expect(start).toBe(7) // 'alpha' (5) + '\n\n' (2)
  })

  it('reaches the very end of the last segment', () => {
    const { start, end } = segmentRange(text, 2)
    expect(text.slice(start, end)).toBe('gamma')
    expect(end).toBe(text.length)
  })

  it('is a zero-width range at 0 for an out-of-bounds index, never NaN', () => {
    expect(segmentRange(text, 99)).toEqual({ start: 0, end: 0 })
    expect(segmentRange(text, -1)).toEqual({ start: 0, end: 0 })
  })

  it('handles Khmer text, where offsets are code units not glyphs', () => {
    const km = 'ខ្ញុំ\n\nលេខ'
    const { start, end } = segmentRange(km, 1)
    expect(km.slice(start, end)).toBe('លេខ')
  })
})

describe('replaceSegment', () => {
  const text = 'alpha\n\nbeta\n\ngamma'

  it('swaps one segment and leaves the rest byte-identical', () => {
    expect(replaceSegment(text, 1, 'BETA')).toBe('alpha\n\nBETA\n\ngamma')
  })

  it('round-trips: replacing a segment with itself is a no-op', () => {
    for (let i = 0; i < 3; i++) {
      expect(replaceSegment(text, i, splitSegments(text)[i])).toBe(text)
    }
  })

  it('leaves the text untouched for an out-of-range index', () => {
    expect(replaceSegment(text, 9, 'x')).toBe(text)
  })

  it('keeps an edit that itself contains a blank line addressable as one segment', () => {
    // The analyst pasted a paragraph break inside one block. The write is honest —
    // it goes in verbatim — but the result now has MORE segments than blocks, which
    // is exactly the misalignment pairSegments is required to detect.
    const out = replaceSegment(text, 0, 'a\n\nb')
    expect(out).toBe('a\n\nb\n\nbeta\n\ngamma')
    expect(splitSegments(out)).toHaveLength(4)
  })
})

describe('pairSegments', () => {
  const entry = (text: string, index: number, confidence?: number) => ({
    block: block({ text, confidence }),
    index,
  })

  it('pairs one-to-one and reports aligned when the counts match', () => {
    const r = pairSegments(['a', 'b'], [entry('a', 0, 0.5), entry('b', 1)])
    expect(r.aligned).toBe(true)
    expect(r.rows.map((x) => x.block?.confidence)).toEqual([0.5, undefined])
  })

  it('keeps the SOURCE index separate from the segment index', () => {
    // The bug this exists to prevent: reading order reverses these two blocks, so
    // card 1 is segment 0 but source block 2. Linking by the segment index would
    // halo the wrong region on the scan while the list looked perfectly right.
    const r = pairSegments(['first', 'second'], [entry('first', 2), entry('second', 0)])
    expect(r.rows.map((x) => x.segIndex)).toEqual([0, 1])
    expect(r.rows.map((x) => x.srcIndex)).toEqual([2, 0])
  })

  it('reports NOT aligned and withholds every block when counts differ', () => {
    // Withholding all metadata is deliberate: once the counts diverge there is no
    // way to know WHERE the drift began, so pairing even the leading segments would
    // be a guess. A wrong confidence badge on the wrong text is worse than none.
    const r = pairSegments(['a', 'b', 'c'], [entry('a', 0), entry('b', 1)])
    expect(r.aligned).toBe(false)
    expect(r.rows).toHaveLength(3)
    expect(r.rows.every((x) => x.block === null && x.srcIndex === null)).toBe(true)
  })

  it('still carries the segment index when unaligned, so editing keeps working', () => {
    const r = pairSegments(['a', 'b', 'c'], [entry('a', 0)])
    expect(r.rows.map((x) => x.segIndex)).toEqual([0, 1, 2])
  })

  it('is aligned-and-empty for a page with no text at all', () => {
    const r = pairSegments([], [])
    expect(r.aligned).toBe(true)
    expect(r.rows).toEqual([])
  })
})
