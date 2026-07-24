import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { IssuesDrawer } from './IssuesDrawer'
import type { Issue } from '../../api/types'

const issue = (over: Partial<Issue> = {}): Issue => ({
  page: 0,
  table_id: 'p1_t1',
  row: 0,
  col: 0,
  conf: 0.4,
  text: 'x',
  reason: 'low_conf',
  reasons: ['low_conf'],
  ...over,
})

function renderDrawer(issues: Issue[]) {
  const onDismissAll = vi.fn()
  const onHoverIssue = vi.fn()
  render(
    <IssuesDrawer
      issues={issues}
      currentIdx={-1}
      onJump={vi.fn()}
      onDismiss={vi.fn()}
      onDismissAll={onDismissAll}
      onHoverIssue={onHoverIssue}
      onClose={vi.fn()}
    />,
  )
  return { onDismissAll, onHoverIssue }
}

describe('IssuesDrawer dismiss-all', () => {
  const dismissAll = () => screen.getByRole('button', { name: /dismiss all/i })

  it('fires onDismissAll when the header button is clicked', () => {
    const { onDismissAll } = renderDrawer([issue({ col: 0 }), issue({ col: 1 })])
    fireEvent.click(dismissAll())
    expect(onDismissAll).toHaveBeenCalledTimes(1)
  })

  // The confirm dialog lives in App, not here — but an always-present button on an
  // empty list would offer to dismiss nothing, so the drawer hides it itself.
  it('is absent when there are no issues to dismiss', () => {
    renderDrawer([])
    expect(screen.queryByRole('button', { name: /dismiss all/i })).toBeNull()
  })
})

describe('IssuesDrawer hover preview', () => {
  it('reports the hovered row index on enter and null on leave', () => {
    const { onHoverIssue } = renderDrawer([issue({ col: 0 }), issue({ col: 1 })])
    // The jump button carries the cell text; its row is the hover target's parent.
    const rows = screen.getAllByText('x').map((el) => el.closest('div.group') as HTMLElement)
    fireEvent.mouseEnter(rows[1])
    expect(onHoverIssue).toHaveBeenLastCalledWith(1)
    fireEvent.mouseLeave(rows[1])
    expect(onHoverIssue).toHaveBeenLastCalledWith(null)
  })
})
