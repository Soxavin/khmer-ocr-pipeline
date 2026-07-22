import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TablesPanel } from './TablesPanel'
import type { PageData, PageTable } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    putPageText: vi.fn(() => Promise.resolve({ ok: true })),
    replace: vi.fn(),
    undoReplace: vi.fn(),
    review: vi.fn(),
    putTable: vi.fn(),
    resetTable: vi.fn(),
    exportCsvUrl: () => '#',
  },
}))

// AG-Grid renders a canvas-like DOM that jsdom cannot lay out; the find bar's
// counter and stepping are panel-level concerns, so the grid itself is stubbed.
vi.mock('./TableEditor', () => ({ TableEditor: () => null }))

const table = (id: string, grid: string[][]): PageTable => ({
  table_id: id,
  grid,
  original_grid: grid,
  confidence: grid.map((r) => r.map(() => 1)),
  edited: false,
  verified: false,
})

const page: PageData = {
  corrected_text: '',
  // 3 matches for "total": two in t1, one in t2 — so ordering across tables matters.
  tables: [
    table('t1', [['Total', 'x'], ['y', 'subtotal']]),
    table('t2', [['TOTAL', 'z']]),
  ],
  text_blocks: [],
  table_bboxes: [],
  table_bbox_index: {},
  qwen_used: false,
}

function renderPanel() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const onFocusCell = vi.fn()
  render(
    <QueryClientProvider client={qc}>
      <TablesPanel
        docId="d1"
        pageIdx={0}
        page={page}
        selectedTable={null}
        onSelectTable={vi.fn()}
        flashToken={null}
        focusCell={null}
        showFind
        onOpenFind={vi.fn()}
        onCloseFind={vi.fn()}
        onFocusCell={onFocusCell}
      />
    </QueryClientProvider>,
  )
  const input = screen.getByPlaceholderText('Find…')
  return { input, onFocusCell }
}

describe('Find bar match counter and stepping', () => {
  it('shows no counter until something is typed', () => {
    renderPanel()
    expect(screen.queryByText(/\d+ \/ \d+/)).toBeNull()
  })

  it('counts every match on the page, case-insensitively across tables', () => {
    const { input } = renderPanel()
    fireEvent.change(input, { target: { value: 'total' } })
    expect(screen.getByText('1 / 3')).toBeTruthy()
  })

  it('steps forward in reading order and jumps to the matched cell', () => {
    const { input, onFocusCell } = renderPanel()
    fireEvent.change(input, { target: { value: 'total' } })
    fireEvent.click(screen.getByRole('button', { name: 'Next match' }))
    expect(screen.getByText('2 / 3')).toBeTruthy()
    // Second match is t1's "subtotal" at row 1, col 1 — not t2's, which comes last.
    expect(onFocusCell).toHaveBeenLastCalledWith('t1', 1, 1)
  })

  it('wraps around rather than dead-ending at the last match', () => {
    const { input, onFocusCell } = renderPanel()
    fireEvent.change(input, { target: { value: 'total' } })
    const next = screen.getByRole('button', { name: 'Next match' })
    fireEvent.click(next)
    fireEvent.click(next)
    expect(screen.getByText('3 / 3')).toBeTruthy()
    fireEvent.click(next)
    expect(screen.getByText('1 / 3')).toBeTruthy()
    expect(onFocusCell).toHaveBeenLastCalledWith('t1', 0, 0)
  })

  it('steps backwards from the first match to the last', () => {
    const { input } = renderPanel()
    fireEvent.change(input, { target: { value: 'total' } })
    fireEvent.click(screen.getByRole('button', { name: 'Previous match' }))
    expect(screen.getByText('3 / 3')).toBeTruthy()
  })

  it('reports no matches calmly and disables the steppers', () => {
    const { input } = renderPanel()
    fireEvent.change(input, { target: { value: 'zzzz' } })
    expect(screen.getByText('none')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Next match' })).toBeDisabled()
  })

  it('resets the cursor when the query changes, so the counter cannot exceed the total', () => {
    const { input } = renderPanel()
    fireEvent.change(input, { target: { value: 'total' } })
    fireEvent.click(screen.getByRole('button', { name: 'Next match' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next match' }))
    expect(screen.getByText('3 / 3')).toBeTruthy()
    // Narrower query with a single match: a stale cursor would read "3 / 1".
    fireEvent.change(input, { target: { value: 'subtotal' } })
    expect(screen.getByText('1 / 1')).toBeTruthy()
  })

  it('Enter steps to the next match instead of firing the document-wide replace', () => {
    const { input, onFocusCell } = renderPanel()
    fireEvent.change(input, { target: { value: 'total' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(screen.getByText('2 / 3')).toBeTruthy()
    expect(onFocusCell).toHaveBeenCalled()
  })

  it('Shift+Enter steps backwards', () => {
    const { input } = renderPanel()
    fireEvent.change(input, { target: { value: 'total' } })
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
    expect(screen.getByText('3 / 3')).toBeTruthy()
  })
})
