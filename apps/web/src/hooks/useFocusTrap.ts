import { useEffect, type RefObject } from 'react'

/** Real modal focus management: focus the container on open, keep Tab cycling
    inside it, and restore focus to the opener on close. One implementation for
    every dialog (help modal, ⌘K palette) — an `aria-modal` without a trap is a
    promise the keyboard can't keep. */
export function useFocusTrap(ref: RefObject<HTMLElement | null>, active: boolean) {
  useEffect(() => {
    if (!active) return
    const prev = document.activeElement as HTMLElement | null
    // Focus the container only when nothing inside it took focus already
    // (the palette autofocuses its input; don't steal it back).
    if (!ref.current?.contains(document.activeElement)) ref.current?.focus()
    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !ref.current) return
      const focusables = ref.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      )
      if (!focusables.length) {
        e.preventDefault()
        return
      }
      const first = focusables[0]
      const last = focusables[focusables.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', trap)
    return () => {
      window.removeEventListener('keydown', trap)
      prev?.focus()
    }
  }, [ref, active])
}
