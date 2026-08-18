/** Scroll a descendant into view by moving ONLY its scroll container — never the
    window.

    The trap this exists to avoid: `Element.scrollIntoView()` and `.focus()` scroll
    every scroll ancestor up to and including the document, and — critically —
    `overflow:hidden` does not stop them. `overflow:hidden` blocks *user* scrolling,
    not scripted scrolling, so a card revealed with `scrollIntoView` yanks the whole
    page even inside a viewport-locked layout. These helpers write a container's
    `scrollTop` directly and touch nothing above it. */

/** The `scrollTop` that brings `[elemTop, elemTop+elemHeight)` inside the viewport
    `[viewScrollTop, viewScrollTop+viewHeight)`, with `margin` of breathing room.
    Returns the current `viewScrollTop` unchanged when the element already fits
    (nearest-style — no gratuitous motion). Pure, so it is unit-testable without a
    layout engine. */
export function scrollTopFor(p: {
  elemTop: number
  elemHeight: number
  viewHeight: number
  viewScrollTop: number
  margin: number
}): number {
  const { elemTop, elemHeight, viewHeight, viewScrollTop, margin } = p
  const viewBottom = viewScrollTop + viewHeight
  const elemBottom = elemTop + elemHeight
  let top = viewScrollTop
  // Above the fold, OR taller than the viewport: align the element's top (showing
  // the start of a long block beats bottom-aligning and hiding it).
  if (elemTop < viewScrollTop || elemHeight >= viewHeight) {
    top = elemTop - margin
  } else if (elemBottom > viewBottom) {
    top = elemBottom - viewHeight + margin
  }
  return Math.max(0, top)
}

/** The nearest ancestor that actually scrolls, stopping BEFORE body/documentElement
    so the window is never a candidate. Null if none — then there is nothing to move
    and nothing should. */
function scrollParent(el: HTMLElement): HTMLElement | null {
  let node = el.parentElement
  while (node && node !== document.body && node !== document.documentElement) {
    const oy = getComputedStyle(node).overflowY
    if ((oy === 'auto' || oy === 'scroll') && node.scrollHeight > node.clientHeight) {
      return node
    }
    node = node.parentElement
  }
  return null
}

/** Reveal `el` within its own scroll container. No-op when it has no scrollable
    ancestor below the document. Honours reduced motion. */
export function scrollIntoViewWithin(el: HTMLElement, opts: { margin?: number } = {}): void {
  const container = scrollParent(el)
  if (!container) return
  const margin = opts.margin ?? 8
  // offsetTop is relative to the nearest positioned ancestor, which is not
  // necessarily the container — measure through rects instead, in the container's
  // current scroll frame.
  const elemTop = el.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop
  const top = scrollTopFor({
    elemTop,
    elemHeight: el.offsetHeight,
    viewHeight: container.clientHeight,
    viewScrollTop: container.scrollTop,
    margin,
  })
  const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches
  container.scrollTo({ top, behavior: smooth ? 'smooth' : 'auto' })
}
