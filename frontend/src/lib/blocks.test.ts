import { describe, expect, it } from 'vitest'
import { blockLabel, mergedText, orderedBlocks } from './blocks'
import type { TextBlock } from '../api/types'

const b = (over: Partial<TextBlock> = {}): TextBlock => ({ bbox: [0, 0, 1, 1], ...over })

describe('orderedBlocks', () => {
  it('is empty when the page has no blocks', () => {
    expect(orderedBlocks(undefined)).toEqual([])
    expect(orderedBlocks([])).toEqual([])
  })

  it('sorts by reading order, not array order', () => {
    const out = orderedBlocks([
      b({ text: 'second', reading_order: 2 }),
      b({ text: 'first', reading_order: 1 }),
    ])
    expect(out.map((x) => x.text)).toEqual(['first', 'second'])
  })

  it('drops blocks with no text — empty layout regions are not reading material', () => {
    const out = orderedBlocks([b({ text: 'kept' }), b({ text: '   ' }), b({})])
    expect(out).toHaveLength(1)
  })

  it('falls back to array position when reading_order is absent, keeping order stable', () => {
    const out = orderedBlocks([b({ text: 'a' }), b({ text: 'b' }), b({ text: 'c' })])
    expect(out.map((x) => x.text)).toEqual(['a', 'b', 'c'])
  })
})

describe('blockLabel', () => {
  it('prefers region_label, then label, then the fallback', () => {
    expect(blockLabel(b({ region_label: 'Header', label: 'Text' }), 'Block')).toBe('Header')
    expect(blockLabel(b({ label: 'Text' }), 'Block')).toBe('Text')
    expect(blockLabel(b({}), 'Block')).toBe('Block')
    expect(blockLabel(b({ region_label: '  ' }), 'Block')).toBe('Block')
  })
})

describe('mergedText', () => {
  it('joins with a blank line, matching how the backend builds ocr_text', () => {
    expect(mergedText([b({ text: 'one' }), b({ text: 'two' })])).toBe('one\n\ntwo')
  })

  it('skips empties so the copy output has no ragged gaps', () => {
    expect(mergedText([b({ text: 'one' }), b({ text: '' }), b({ text: 'two' })])).toBe('one\n\ntwo')
  })
})
