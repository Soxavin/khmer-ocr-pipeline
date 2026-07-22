import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { PageTextPanel } from './PageTextPanel'
import type { PageData, TextBlock } from '../../api/types'

vi.mock('../../api/client', () => ({ api: { putPageText: vi.fn(() => Promise.resolve({ ok: true })) } }))

const block = (over: Partial<TextBlock>): TextBlock => ({ bbox: [0, 0, 10, 10], ...over })

// Reading order REVERSES the array and the middle block is dropped for having no
// text — so card position and source index deliberately disagree here.
const page: PageData = {
  corrected_text: 'server text',
  tables: [],
  text_blocks: [
    block({ text: 'second', reading_order: 2 }),
    block({ text: '' }),
    block({ text: 'first', reading_order: 1 }),
  ],
  table_bboxes: [],
  table_bbox_index: {},
  qwen_used: false,
}

function renderPanel(over: Record<string, unknown> = {}) {
  const onSelectBlock = vi.fn()
  const onHoverBlock = vi.fn()
  const view = render(
    <PageTextPanel
      docId="d1"
      pageIdx={0}
      page={page}
      text="server text"
      onTextChange={vi.fn()}
      saved
      onSaved={vi.fn()}
      onSelectBlock={onSelectBlock}
      onHoverBlock={onHoverBlock}
      {...over}
    />,
  )
  return { onSelectBlock, onHoverBlock, view }
}

describe('PageTextPanel block linking', () => {
  it('reports the SOURCE index on click, not the card position', () => {
    const { onSelectBlock } = renderPanel()
    // Card 1 is 'first', which lives at text_blocks[2].
    fireEvent.click(screen.getByText(/^1 ·/))
    expect(onSelectBlock).toHaveBeenCalledWith(2)
    fireEvent.click(screen.getByText(/^2 ·/))
    expect(onSelectBlock).toHaveBeenCalledWith(0)
  })

  it('reports hover enter and leave so the halo can follow the pointer', () => {
    const { onHoverBlock } = renderPanel()
    const card = screen.getByText('first').closest('li')!
    fireEvent.mouseEnter(card)
    expect(onHoverBlock).toHaveBeenLastCalledWith(2)
    fireEvent.mouseLeave(card)
    expect(onHoverBlock).toHaveBeenLastCalledWith(null)
  })

  it('marks the active card pressed, keyed on the source index', () => {
    renderPanel({ activeBlock: 0 })
    expect(screen.getByText(/^2 ·/)).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText(/^1 ·/)).toHaveAttribute('aria-pressed', 'false')
  })

  it('a canvas pick forces Blocks view — Raw has no card to scroll to', () => {
    const { view } = renderPanel()
    fireEvent.click(screen.getByText('Raw'))
    expect(screen.queryByText('first')).toBeNull()

    view.rerender(
      <PageTextPanel
        docId="d1"
        pageIdx={0}
        page={page}
        text="server text"
        onTextChange={vi.fn()}
        saved
        onSaved={vi.fn()}
        blockFocus={{ i: 2, n: 1 }}
      />,
    )
    expect(screen.getByText('first')).toBeTruthy()
  })
})
