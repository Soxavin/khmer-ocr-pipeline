import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { PageTextPanel } from './PageTextPanel'
import type { PageData, TextBlock } from '../../api/types'

vi.mock('../../api/client', () => ({ api: { putPageText: vi.fn(() => Promise.resolve({ ok: true })) } }))

const block = (over: Partial<TextBlock>): TextBlock => ({ bbox: [0, 0, 10, 10], ...over })

// Reading order REVERSES the array and the middle block is dropped for having no
// text, so card position and source index deliberately disagree. `text` is the
// CORRECTED string — two segments, matching the two surviving blocks, so the panel
// is in its aligned state.
const page: PageData = {
  corrected_text: 'first block\n\nsecond block',
  tables: [],
  text_blocks: [
    block({ text: 'second', reading_order: 2, confidence: 0.4, region_label: 'Text' }),
    block({ text: '' }),
    block({ text: 'first', reading_order: 1, confidence: 0.99, region_label: 'SectionHeader' }),
  ],
  table_bboxes: [],
  table_bbox_index: {},
  qwen_used: false,
}
const TEXT = 'first block\n\nsecond block'

function renderPanel(over: Record<string, unknown> = {}) {
  const onSelectBlock = vi.fn()
  const onHoverBlock = vi.fn()
  const onTextChange = vi.fn()
  const view = render(
    <PageTextPanel
      docId="d1"
      pageIdx={0}
      page={page}
      text={TEXT}
      onTextChange={onTextChange}
      saved
      onSaved={vi.fn()}
      onSelectBlock={onSelectBlock}
      onHoverBlock={onHoverBlock}
      {...over}
    />,
  )
  return { onSelectBlock, onHoverBlock, onTextChange, view }
}

describe('PageTextPanel shows the corrected text, not raw OCR', () => {
  it('renders the corrected segments — the same string Raw shows', () => {
    renderPanel()
    expect(screen.getByText('first block')).toBeTruthy()
    expect(screen.getByText('second block')).toBeTruthy()
    // The raw OCR strings must NOT appear: they are what the correction pass replaced.
    expect(screen.queryByText('first')).toBeNull()
    expect(screen.queryByText('second')).toBeNull()
  })

  it('attaches each block\'s confidence to its segment when counts align', () => {
    renderPanel()
    expect(screen.getByText('99%')).toBeTruthy()
    expect(screen.getByText('40%')).toBeTruthy()
  })
})

describe('PageTextPanel block linking', () => {
  it('reports the SOURCE index on click, not the segment or card position', () => {
    const { onSelectBlock } = renderPanel()
    // Card 1 is segment 0, but its block is text_blocks[2] — reading order reverses
    // them. Passing the segment index here would halo the wrong region on the scan.
    fireEvent.click(screen.getByText(/^1 ·/))
    expect(onSelectBlock).toHaveBeenCalledWith(2)
    fireEvent.click(screen.getByText(/^2 ·/))
    expect(onSelectBlock).toHaveBeenCalledWith(0)
  })

  it('reports hover enter and leave so the halo can follow the pointer', () => {
    const { onHoverBlock } = renderPanel()
    const card = screen.getByText('first block').closest('li')!
    fireEvent.mouseEnter(card)
    expect(onHoverBlock).toHaveBeenLastCalledWith(2)
    fireEvent.mouseLeave(card)
    expect(onHoverBlock).toHaveBeenLastCalledWith(null)
  })

  it('a canvas pick forces Blocks view — Raw has no card to scroll to', () => {
    const { view } = renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'Raw' }))
    expect(screen.queryByText('first block')).toBeNull()

    view.rerender(
      <PageTextPanel
        docId="d1"
        pageIdx={0}
        page={page}
        text={TEXT}
        onTextChange={vi.fn()}
        saved
        onSaved={vi.fn()}
        blockFocus={{ i: 2, n: 1 }}
      />,
    )
    expect(screen.getByText('first block')).toBeTruthy()
  })
})

describe('PageTextPanel inline editing', () => {
  it('an inline edit rewrites only its own segment, leaving the rest byte-identical', () => {
    const { onTextChange } = renderPanel()
    fireEvent.click(screen.getAllByRole('button', { name: 'Edit this block' })[0])
    const ta = screen.getByRole('textbox') as HTMLTextAreaElement
    fireEvent.change(ta, { target: { value: 'EDITED' } })
    expect(onTextChange).toHaveBeenCalledWith('EDITED\n\nsecond block')
  })

  it('editing the SECOND card does not touch the first', () => {
    const { onTextChange } = renderPanel()
    fireEvent.click(screen.getAllByRole('button', { name: 'Edit this block' })[1])
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'X' } })
    expect(onTextChange).toHaveBeenCalledWith('first block\n\nX')
  })

  it('Edit in Raw switches view and selects exactly that segment', () => {
    renderPanel()
    fireEvent.click(screen.getAllByRole('button', { name: 'Edit in Raw' })[1])
    const ta = screen.getByRole('textbox') as HTMLTextAreaElement
    expect(ta.value).toBe(TEXT)
    // 'second block' starts after 'first block' + the blank-line separator.
    expect(TEXT.slice(ta.selectionStart, ta.selectionEnd)).toBe('second block')
  })
})

describe('PageTextPanel confidence filter', () => {
  it('filters the cards to one band and back', () => {
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: /1 need checking/i }))
    expect(screen.queryByText('first block')).toBeNull() // 99% is Clean
    expect(screen.getByText('second block')).toBeTruthy() // 40% is Check
    fireEvent.click(screen.getByRole('button', { name: /1 need checking/i }))
    expect(screen.getByText('first block')).toBeTruthy()
  })

  it('a filtered card still edits its own segment, not the one at its visible position', () => {
    // The regression this guards: with only the second block visible it sits at
    // list position 0, so any code using the loop index would rewrite segment 0.
    const { onTextChange } = renderPanel()
    fireEvent.click(screen.getByRole('button', { name: /1 need checking/i }))
    fireEvent.click(screen.getByRole('button', { name: 'Edit this block' }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Z' } })
    expect(onTextChange).toHaveBeenCalledWith('first block\n\nZ')
  })
})

describe('PageTextPanel when segments and blocks disagree', () => {
  // One segment, two blocks: the pairing is refused rather than guessed.
  const mismatched = { ...page, text_blocks: page.text_blocks }

  it('still shows the text, but withholds confidence it cannot vouch for', () => {
    render(
      <PageTextPanel
        docId="d1"
        pageIdx={0}
        page={mismatched}
        text="only one segment here"
        onTextChange={vi.fn()}
        saved
        onSaved={vi.fn()}
      />,
    )
    expect(screen.getByText('only one segment here')).toBeTruthy()
    expect(screen.queryByText('99%')).toBeNull()
    expect(screen.getByText(/could not be matched/i)).toBeTruthy()
  })
})
