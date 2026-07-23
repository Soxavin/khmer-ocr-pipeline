import { useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { menuCls } from '../ui'

/** A dropdown anchored under its trigger, positioned `fixed` rather than `absolute`.
 *
 *  This is not a style preference. Every top-bar menu sits inside three nested
 *  `overflow: hidden` boxes — the `<header>`, the app shell below it, and `#root`
 *  in index.css — all of which are the §2.82 guards that keep a stray wide element
 *  from forcing a scrollbar on the document. An `absolute` menu hangs below a 48px
 *  header and is therefore clipped to nothing: the panel never paints, while its
 *  `fixed` scrim still arms, so the trigger appears dead and swallows the next click.
 *
 *  A fixed element's containing block is the viewport, which is outside all three
 *  clippers, so it escapes without any guard being weakened. We keep it in the React
 *  tree (no portal) so focus, state, and outside-click behave exactly as before. */
export function AnchoredMenu(props: {
  /** The element the menu hangs from — its bottom-right corner is the anchor. */
  anchorRef: RefObject<HTMLElement | null>
  onClose: () => void
  /** Tailwind width class, e.g. `w-64`. */
  width: string
  className?: string
  children: ReactNode
}) {
  const { anchorRef, onClose, width, className = '', children } = props
  const [pos, setPos] = useState<{ top: number; right: number } | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  // Measured in a layout effect so the menu paints already positioned; a state
  // update in a passive effect would show it at 0,0 for one frame first.
  useLayoutEffect(() => {
    const place = () => {
      const el = anchorRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      setPos({ top: r.bottom + 4, right: Math.max(8, window.innerWidth - r.right) })
    }
    place()
    window.addEventListener('resize', place)
    return () => window.removeEventListener('resize', place)
  }, [anchorRef])

  useLayoutEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        ref={menuRef}
        className={`${menuCls} ${width} ${className} fixed z-50`}
        // Hidden until measured: rendering at the default 0,0 and snapping into
        // place reads as a flicker on every open.
        style={
          pos
            ? { top: pos.top, right: pos.right, maxHeight: `calc(100vh - ${pos.top + 8}px)` }
            : { top: 0, right: 0, visibility: 'hidden' }
        }
      >
        {children}
      </div>
    </>
  )
}
