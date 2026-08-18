import { describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { PageGrid } from './PageGrid'

function renderGrid(over: Partial<Parameters<typeof PageGrid>[0]> = {}) {
  const props = {
    pages: [0, 1, 2],
    pageCount: 3,
    imageUrl: (n: number) => `/proc/${n}`,
    selected: new Set<number>(),
    onTogglePage: vi.fn(),
    onOpenPage: vi.fn(),
    ...over,
  }
  render(<PageGrid {...props} />)
  return props
}

describe('PageGrid thumbnails', () => {
  it('renders one thumbnail per page at its primary src', () => {
    renderGrid()
    const imgs = screen.getAllByRole('img')
    expect(imgs).toHaveLength(3)
    expect(imgs[2]).toHaveAttribute('src', '/proc/2')
  })

  it('swaps to the fallback rendition on the first error before any retry', () => {
    renderGrid({ fallbackUrl: (n) => `/preview/${n}` })
    const img = screen.getAllByRole('img')[2]
    fireEvent.error(img)
    expect(img).toHaveAttribute('src', '/preview/2')
  })

  it('retries with a cache-busting param when there is no fallback left to try', () => {
    vi.useFakeTimers()
    try {
      renderGrid()
      const img = screen.getAllByRole('img')[0]
      fireEvent.error(img)
      act(() => vi.runOnlyPendingTimers())
      expect(img.getAttribute('src')).toMatch(/\/proc\/0\?retry=1/)
    } finally {
      vi.useRealTimers()
    }
  })

  it('drops the <img> for a calm skeleton once retries are exhausted (never a broken glyph)', () => {
    vi.useFakeTimers()
    try {
      renderGrid()
      const altP0 = 'page 1/3'
      // 4 retries on page 0's thumbnail, each firing another error, then it gives up.
      for (let i = 0; i < 5; i++) {
        const img = screen.queryByAltText(altP0)
        if (!img) break
        fireEvent.error(img)
        act(() => vi.runOnlyPendingTimers())
      }
      // The exhausted thumbnail is gone (skeleton, no <img>); the others stay.
      expect(screen.queryByAltText(altP0)).toBeNull()
      expect(screen.queryAllByRole('img')).toHaveLength(2)
    } finally {
      vi.useRealTimers()
    }
  })
})
