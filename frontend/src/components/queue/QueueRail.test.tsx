import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueueRail } from './QueueRail'
import type { DocSummary } from '../../api/types'

const doc = (id: string, status: DocSummary['status'] = 'queued'): DocSummary => ({
  id,
  name: `${id}.pdf`,
  pages: 2,
  size_kb: 100,
  status,
  total_tables: 0,
  reviewed_tables: 0,
})

function renderRail(over: Partial<Parameters<typeof QueueRail>[0]> = {}) {
  const props = {
    documents: [doc('a'), doc('b')],
    activeId: 'a',
    onSelect: vi.fn(),
    onUpload: vi.fn(),
    onRemove: vi.fn(),
    onRemoveAll: vi.fn(),
    uploading: false,
    onRunAll: vi.fn(),
    batchRunning: false,
    pipelineBusy: false,
    exportAllUrl: null,
    ...over,
  }
  // The i18n context default already resolves English, so no provider is needed.
  render(<QueueRail {...props} />)
  return props
}

describe('QueueRail clear-all guard', () => {
  it('clicking the trash icon opens a confirmation without clearing anything', async () => {
    const user = userEvent.setup()
    const props = renderRail()

    await user.click(screen.getByRole('button', { name: /delete all/i }))

    expect(screen.getByRole('dialog')).toBeInTheDocument()
    // The guard is the whole point: nothing may be removed before confirming.
    expect(props.onRemoveAll).not.toHaveBeenCalled()
    expect(screen.getByText('a.pdf')).toBeInTheDocument()
    expect(screen.getByText('b.pdf')).toBeInTheDocument()
  })

  it('confirming in the popover performs the clear exactly once', async () => {
    const user = userEvent.setup()
    const props = renderRail()

    await user.click(screen.getByRole('button', { name: /delete all/i }))
    await user.click(screen.getByRole('button', { name: /^remove all$/i }))

    expect(props.onRemoveAll).toHaveBeenCalledTimes(1)
  })

  it('cancelling closes the popover and leaves the queue intact', async () => {
    const user = userEvent.setup()
    const props = renderRail()

    await user.click(screen.getByRole('button', { name: /delete all/i }))
    await user.click(screen.getByRole('button', { name: /cancel/i }))

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(props.onRemoveAll).not.toHaveBeenCalled()
  })

  it('offers no clear-all control for a single document', () => {
    renderRail({ documents: [doc('a')] })
    expect(screen.queryByRole('button', { name: /delete all/i })).not.toBeInTheDocument()
  })

  it('blocks Run all while the pipeline is busy elsewhere', () => {
    renderRail({ pipelineBusy: true })
    expect(screen.getByRole('button', { name: /run all/i })).toBeDisabled()
  })

  it('selects a document by keyboard — the row is a real control', async () => {
    const user = userEvent.setup()
    const props = renderRail()
    // The row exposes a button role and takes focus…
    const row = screen.getByRole('button', { name: 'a.pdf' })
    row.focus()
    await user.keyboard('{Enter}')
    expect(props.onSelect).toHaveBeenCalledWith('a')
  })

  it('per-document remove uses the same popover guard, not window.confirm', async () => {
    const user = userEvent.setup()
    const props = renderRail()

    await user.click(screen.getByRole('button', { name: /remove.*a\.pdf/i }))
    // A designed dialog appears; nothing is removed until confirmed.
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(props.onRemove).not.toHaveBeenCalled()

    // Exact match: the trigger's own label is "Remove document: a.pdf".
    await user.click(screen.getByRole('button', { name: /^remove document$/i }))
    expect(props.onRemove).toHaveBeenCalledWith('a')
    expect(props.onRemove).toHaveBeenCalledTimes(1)
  })

  it('keeps the remove control reachable (not display:none at rest)', () => {
    renderRail()
    // The per-document remove button must be in the tree (focusable), not
    // conditionally mounted only on hover.
    expect(screen.getByRole('button', { name: /remove.*a\.pdf/i })).toBeInTheDocument()
  })
})
