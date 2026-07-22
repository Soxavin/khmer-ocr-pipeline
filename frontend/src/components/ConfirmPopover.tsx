import { useEffect, useRef } from 'react'
import { btnSmCls, dangerBtnCls } from '../ui'

/** The workspace's one destructive-action guard: an anchored popover that names
    the consequence, never fires on the first click, and closes on Escape or an
    outside press. Fixed-positioned at the trigger, so it works from scrollable
    lists (an absolute popover would clip inside overflow containers) and never
    nests inside a `role="button"` row. Replaces ad-hoc `window.confirm`s so the
    product keeps one confirmation vocabulary. */
export function ConfirmPopover(props: {
  title: string
  body: string
  actionLabel: string
  cancelLabel: string
  /** Viewport point to anchor at — pass the trigger's bounding rect corner. */
  anchor: { x: number; y: number }
  onConfirm: () => void
  onClose: () => void
}) {
  const { title, body, actionLabel, cancelLabel, anchor, onConfirm, onClose } = props
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    const onDown = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose()
    }
    window.addEventListener('keydown', onKey)
    // Deferred: the opening click itself must not immediately close the popover.
    const id = setTimeout(() => window.addEventListener('pointerdown', onDown), 0)
    // Keyboard users land on the SAFE choice first.
    ref.current?.querySelector<HTMLButtonElement>('[data-cancel]')?.focus()
    return () => {
      clearTimeout(id)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown)
    }
  }, [onClose])

  const W = 240
  const left = Math.max(8, Math.min(anchor.x - W, window.innerWidth - W - 8))
  const top = Math.min(anchor.y + 4, window.innerHeight - 140)

  return (
    <div
      role="dialog"
      aria-label={title}
      ref={ref}
      className="overlay-enter fixed z-50 w-60 rounded-lg border border-line-strong bg-raised p-3 text-left shadow-modal"
      style={{ left, top }}
    >
      <p className="text-sm font-semibold text-ink">{title}</p>
      <p className="mt-1 text-xs leading-4 text-ink-2">{body}</p>
      <div className="mt-3 flex justify-end gap-2">
        <button data-cancel className={btnSmCls} onClick={onClose}>
          {cancelLabel}
        </button>
        <button
          className={dangerBtnCls}
          onClick={() => {
            onClose()
            onConfirm()
          }}
        >
          {actionLabel}
        </button>
      </div>
    </div>
  )
}
