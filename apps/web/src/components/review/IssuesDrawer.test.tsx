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
  const dismissAll = () => screen.getAllByRole('button', { name: /dismiss all/i })[0]

  // Bulk dismiss is guarded in-app now (ConfirmPopover): the trigger opens the
  // confirmation and does NOT fire onDismissAll on its own; only the popover's
  // action does. This is the "no native window.confirm" contract.
  it('confirms via popover before firing onDismissAll', () => {
    const { onDismissAll } = renderDrawer([issue({ col: 0 }), issue({ col: 1 })])
    fireEvent.click(dismissAll())
    expect(onDismissAll).not.toHaveBeenCalled()
    // Trigger + popover action share the label; the popover's action is the last.
    const buttons = screen.getAllByRole('button', { name: /dismiss all/i })
    fireEvent.click(buttons[buttons.length - 1])
    expect(onDismissAll).toHaveBeenCalledTimes(1)
  })

  it('does not fire onDismissAll if the confirmation is cancelled', () => {
    const { onDismissAll } = renderDrawer([issue({ col: 0 }), issue({ col: 1 })])
    fireEvent.click(dismissAll())
    fireEvent.click(screen.getByRole('button', { name: /cancel/i }))
    expect(onDismissAll).not.toHaveBeenCalled()
  })

  // An always-present button on an empty list would offer to dismiss nothing, so
  // the drawer hides it itself.
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
