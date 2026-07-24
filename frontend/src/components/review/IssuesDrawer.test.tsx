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
  render(
    <IssuesDrawer
      issues={issues}
      currentIdx={-1}
      onJump={vi.fn()}
      onDismiss={vi.fn()}
      onDismissAll={onDismissAll}
      onClose={vi.fn()}
    />,
  )
  return { onDismissAll }
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
