import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TableEditor, suppressNpWhenFocused } from './TableEditor'
import type { PageTable } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    putTable: vi.fn(() => Promise.resolve({ ok: true })),
    resetTable: vi.fn(() => Promise.resolve({ ok: true })),
    review: vi.fn(() => Promise.resolve({ ok: true })),
    exportCsvUrl: () => '#',
  },
}))

const table = (over: Partial<PageTable> = {}): PageTable => ({
  table_id: 'p1_t1',
  grid: [['a']],
  original_grid: [['a']],
  confidence: [[0.9]],
  edited: false,
  verified: false,
  ...over,
})

function renderEditor(t: PageTable) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const props = { docId: 'd1', table: t, focused: true, onFocus: vi.fn(), flash: 0 }
  const utils = render(
    <QueryClientProvider client={qc}>
      <TableEditor {...props} />
    </QueryClientProvider>,
  )
  const rerenderTable = (next: PageTable) =>
    utils.rerender(
      <QueryClientProvider client={qc}>
        <TableEditor {...props} table={next} />
      </QueryClientProvider>,
    )
  return { ...utils, rerenderTable }
}

describe('TableEditor undo-history survival', () => {
  it('keeps the undo stack when a refetch returns the grid we already have (edit → verify)', () => {
    const { rerenderTable } = renderEditor(table())
    // Commit an edit through the toolbar (adds a row → history gains an entry).
    fireEvent.click(screen.getByRole('button', { name: /^row$/i }))
    const undo = screen.getByRole('button', { name: /^undo$/i })
    expect(undo).toBeEnabled()

    // Verify elsewhere → page refetch: NEW table object, grid CONTENT equal to the
    // saved local state (the server echoed our edit back), verified flipped.
    rerenderTable(table({ grid: [['a'], ['']], edited: true, verified: true }))
    expect(screen.getByRole('button', { name: /^undo$/i })).toBeEnabled()
  })

  it('still adopts a genuinely different server grid and clears stale history', () => {
    const { rerenderTable } = renderEditor(table())
    fireEvent.click(screen.getByRole('button', { name: /^row$/i }))
    expect(screen.getByRole('button', { name: /^undo$/i })).toBeEnabled()

    // A re-run produced different content: local history no longer applies.
    rerenderTable(table({ grid: [['completely new']] }))
    expect(screen.getByRole('button', { name: /^undo$/i })).toBeDisabled()
  })

  it('syncs the verified pill from a refetch without touching the grid state', () => {
    const { rerenderTable } = renderEditor(table())
    fireEvent.click(screen.getByRole('button', { name: /^row$/i }))
    rerenderTable(table({ grid: [['a'], ['']], edited: true, verified: true }))
    // The pill reflects the server's verified state after the refetch.
    expect(screen.getByRole('button', { name: /verified/i })).toBeInTheDocument()
  })
})

// The Raw/Edited view toggle is icon-only, so its accessible name ("Original")
// is the only thing naming it, and the Eye/EyeOff swap is the only visible signal
// of which view is showing. A non-pristine table (grid ≠ original) is required —
// with nothing to compare against, the toggle is deliberately disabled.
describe('TableEditor raw/edited view toggle', () => {
  const view = () => screen.getByRole('button', { name: /^raw ocr$/i })
  const edited = () => table({ edited: true, grid: [['b']], original_grid: [['a']] })

  it('announces its state through aria-pressed and flips on click', () => {
    renderEditor(edited())
    expect(view()).toHaveAttribute('aria-pressed', 'false')

    fireEvent.click(view())
    expect(view()).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(view())
    expect(view()).toHaveAttribute('aria-pressed', 'false')
  })

  it('is icon-only: reachable by accessible name, with no visible label text', () => {
    renderEditor(edited())
    expect(view()).toBeInTheDocument()
    expect(view().textContent).toBe('')
  })

  it('is disabled while the table is pristine (grid equals the OCR original)', () => {
    renderEditor(table({ edited: false }))
    expect(view()).toBeDisabled()
  })

  it('shows EyeOff in the edited view and Eye when viewing raw', () => {
    renderEditor(edited())
    expect(view().querySelector('.lucide-eye-off')).not.toBeNull()
    expect(view().querySelector('.lucide-eye')).toBeNull()

    fireEvent.click(view())
    // .lucide-eye must match Eye exactly, not the EyeOff prefix.
    expect(view().querySelector('svg.lucide-eye')).not.toBeNull()
    expect(view().querySelector('.lucide-eye-off')).toBeNull()
  })
})

// Undoing every edit back to the OCR original must retire the "Edited" badge and
// disable Reset on its own — the content is identical, so a lingering badge or an
// enabled Reset would misrepresent the table.
describe('TableEditor pristine derivation', () => {
  it('hides the Edited badge when grid matches original despite edited=true', () => {
    renderEditor(table({ edited: true, grid: [['a']], original_grid: [['a']] }))
    expect(screen.queryByText(/^edited$/i)).toBeNull()
  })

  it('shows the Edited badge when the grid genuinely differs', () => {
    renderEditor(table({ edited: true, grid: [['b']], original_grid: [['a']] }))
    expect(screen.getByText(/^edited$/i)).toBeInTheDocument()
  })

  it('disables Reset when pristine, enables it once the grid differs', () => {
    const { rerenderTable } = renderEditor(table({ edited: true, grid: [['a']], original_grid: [['a']] }))
    expect(screen.getByRole('button', { name: /reset/i })).toBeDisabled()
    rerenderTable(table({ edited: true, grid: [['b']], original_grid: [['a']] }))
    expect(screen.getByRole('button', { name: /reset/i })).toBeEnabled()
  })
})

// The n/p issue-stepping fix rests entirely on this predicate telling AG Grid to
// stand down for n/p on a focused cell — but keep its edit behaviour everywhere
// else. A regression here silently reopens the "typing n into the cell" bug.
describe('suppressNpWhenFocused', () => {
  it('suppresses n and p only when the cell is NOT editing', () => {
    expect(suppressNpWhenFocused({ editing: false, event: { key: 'n' } })).toBe(true)
    expect(suppressNpWhenFocused({ editing: false, event: { key: 'p' } })).toBe(true)
  })

  it('lets n and p through while the cell IS editing, so words can be typed', () => {
    expect(suppressNpWhenFocused({ editing: true, event: { key: 'n' } })).toBe(false)
    expect(suppressNpWhenFocused({ editing: true, event: { key: 'p' } })).toBe(false)
  })

  it('never suppresses any other key, editing or not', () => {
    for (const editing of [true, false]) {
      for (const key of ['a', 'z', 'Enter', ' ', 'ArrowDown', 'N', 'P']) {
        expect(suppressNpWhenFocused({ editing, event: { key } })).toBe(false)
      }
    }
  })
})
