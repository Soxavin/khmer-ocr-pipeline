import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TableEditor } from './TableEditor'
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
