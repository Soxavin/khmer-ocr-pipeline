import { useEffect, useRef } from 'react'
import { btnCls, dangerBtnCls } from '../ui'

/** The workspace's one destructive-action guard: an anchored popover that names
    the consequence, never fires on the first click, and closes on Escape or an
    outside press. Fixed-positioned at the trigger, so it works from scrollable
    lists (an absolute popover would clip inside overflow containers) and never
    nests inside a `role="button"` row. Replaces ad-hoc `window.confirm`s so the
    product keeps one confirmation vocabulary. */
export function ConfirmPopover(props: {
  title: string
  body: string
  /** The subject of the action (a filename), shown in its own bounded slot above
      the consequence. Interpolating it into `body` instead would let one long
      unbroken name blow out the card — and truncating `body` would cut the
      sentence, not the name. Omitted for actions with no single subject. */
  subject?: string
  actionLabel: string
  cancelLabel: string
  /** Viewport point to anchor at — pass the trigger's bounding rect corner. */
  anchor: { x: number; y: number }
  onConfirm: () => void
  onClose: () => void
}) {
  const { title, body, subject, actionLabel, cancelLabel, anchor, onConfirm, onClose } = props
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
  // Bottom reserve covers the tallest the card can get: title + a two-line subject
  // + body + buttons. Under-reserving pushed the actions off-screen for long names.
  const top = Math.min(anchor.y + 4, window.innerHeight - (subject ? 200 : 140))

  return (
    <div
      role="dialog"
      aria-label={title}
      ref={ref}
      className="overlay-enter fixed z-50 w-60 rounded-lg border border-line-strong bg-raised p-3 text-left shadow-modal"
      style={{ left, top }}
    >
      <p className="text-sm font-semibold text-ink">{title}</p>
      {/* break-all, not truncate: a filename has no spaces to wrap at, so only
          mid-token breaking bounds the width. line-clamp caps the height, and
          `title` keeps the full name reachable on hover. */}
      {subject && (
        <p className="mt-1 line-clamp-2 break-all text-xs font-medium leading-4 text-ink" title={subject}>
          {subject}
        </p>
      )}
      <p className="mt-1 text-xs leading-4 text-ink-2">{body}</p>
      <div className="mt-3 flex items-center justify-end gap-2">
        {/* btnCls (h-7), not btnSmCls (h-6) — the danger button is h-7, and two
            confirm actions at different heights read as a layout accident. */}
        <button data-cancel className={btnCls} onClick={onClose}>
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
