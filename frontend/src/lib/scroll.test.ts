import { describe, expect, it } from 'vitest'
import { scrollTopFor } from './scroll'

// The container shows [viewScrollTop, viewScrollTop + viewHeight); the element sits
// at [elemTop, elemTop + elemHeight) in the same content-coordinate space.
describe('scrollTopFor', () => {
  it('does not move when the element is already fully visible', () => {
    // view [100, 300); element [150, 200) — inside it.
    expect(scrollTopFor({ elemTop: 150, elemHeight: 50, viewHeight: 200, viewScrollTop: 100, margin: 8 })).toBe(100)
  })

  it('scrolls down just enough to reveal an element below the fold, plus margin', () => {
    // view [0, 200); element [300, 340). Bottom-align: 340 - 200 + 8 = 148.
    expect(scrollTopFor({ elemTop: 300, elemHeight: 40, viewHeight: 200, viewScrollTop: 0, margin: 8 })).toBe(148)
  })

  it('scrolls up to reveal an element above the current view, minus margin', () => {
    // view [500, 700); element [420, 460). Top-align: 420 - 8 = 412.
    expect(scrollTopFor({ elemTop: 420, elemHeight: 40, viewHeight: 200, viewScrollTop: 500, margin: 8 })).toBe(412)
  })

  it('never returns a negative scrollTop', () => {
    // Element at the very top; margin would push below zero.
    expect(scrollTopFor({ elemTop: 2, elemHeight: 40, viewHeight: 200, viewScrollTop: 100, margin: 8 })).toBe(0)
  })

  it('prefers showing the TOP of an element taller than the viewport', () => {
    // element [100, 500), view height 200: top-align at 100 - 8, not bottom-align
    // (which would hide the start of a long block).
    expect(scrollTopFor({ elemTop: 100, elemHeight: 400, viewHeight: 200, viewScrollTop: 0, margin: 8 })).toBe(92)
  })
})
