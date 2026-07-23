import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TablesPanel } from './TablesPanel'
import type { PageData } from '../../api/types'

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

// A page with NO tables: the draft-loss scenario needs only the text panel, and
// zero tables keeps AG-Grid out of the render entirely.
const pageWith = (over: Partial<PageData> = {}): PageData => ({
  corrected_text: 'server text',
  tables: [],
  text_blocks: [],
  table_bboxes: [],
  table_bbox_index: {},
  qwen_used: false,
  ...over,
})

function renderPanel(page: PageData) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const props = {
    docId: 'd1',
    pageIdx: 0,
    page,
    selectedTable: null,
    onSelectTable: vi.fn(),
    flashToken: null,
    focusCell: null,
    showFind: false,
    onOpenFind: vi.fn(),
    onCloseFind: vi.fn(),
    onFocusCell: vi.fn(),
  }
  const utils = render(
    <QueryClientProvider client={qc}>
      <TablesPanel {...props} />
    </QueryClientProvider>,
  )
  const rerenderPage = (next: PageData) =>
    utils.rerender(
      <QueryClientProvider client={qc}>
        <TablesPanel {...props} page={next} />
      </QueryClientProvider>,
    )
  return { ...utils, rerenderPage }
}

function openRawTextarea(container: HTMLElement): HTMLTextAreaElement {
  const details = container.querySelector('details')!
  details.open = true
  // Exact: the block cards now also carry an "Edit in Raw" action, so /raw/i is
  // ambiguous. This is the view toggle.
  fireEvent.click(screen.getByRole('button', { name: 'Raw' }))
  return container.querySelector('textarea')!
}

describe('TablesPanel page-text draft safety', () => {
  it('keeps an unsaved draft when the page refetches with unchanged text (e.g. a verify)', () => {
    const { container, rerenderPage } = renderPanel(pageWith())
    const ta = openRawTextarea(container)
    fireEvent.change(ta, { target: { value: 'my unsaved draft' } })
    expect(ta.value).toBe('my unsaved draft')

    // A verify elsewhere refetches the page: NEW object identity, SAME text.
    rerenderPage(pageWith())
    expect((container.querySelector('textarea') as HTMLTextAreaElement).value).toBe(
      'my unsaved draft',
    )
  })

  it('still adopts genuinely new server text (a re-run changed the page)', () => {
    const { container, rerenderPage } = renderPanel(pageWith())
    const ta = openRawTextarea(container)
    fireEvent.change(ta, { target: { value: 'draft' } })

    rerenderPage(pageWith({ corrected_text: 'newly extracted text' }))
    expect((container.querySelector('textarea') as HTMLTextAreaElement).value).toBe(
      'newly extracted text',
    )
  })
})
