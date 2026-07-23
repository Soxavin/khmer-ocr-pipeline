// One control vocabulary for the whole workspace. Every select, button, chip,
// input, and menu composes from these; per-instance styling drift is the enemy.

const focus = 'focus-visible:outline-2 focus-visible:outline-primary'
const trans = 'transition-[color,background-color,border-color,box-shadow,transform] duration-150 ease-out'
const press = 'active:scale-[0.98]'

// --- NEW: Spatial Depth & Layering Primitives ---
// Gives Claude the vocabulary to create layered panels instead of flat walls
// (panelCanvasCls removed: it carried min-h-screen, was exported, and was used
// nowhere. A dead viewport-height token is a loaded gun next to a layout that
// depends on nothing exceeding the viewport.)
export const panelMainCls = `bg-surface border border-line-strong/60 rounded-xl shadow-sm ${trans}` // Standard crisp panel
export const panelFloatingCls = `bg-raised border border-line-strong shadow-overlay rounded-xl backdrop-blur-md ${trans}` // Elevated contextual panel

// --- Controls Framework ---
export const selectCls =
  `h-7 rounded-md border border-line-strong bg-surface px-2 text-sm text-ink ${trans} ` +
  `hover:border-ink-3 hover:bg-rail/30 ${focus}`

// Buttons are fixed-height controls: a wrapped label breaks out of that height and
// spills past the border. Set at the token, not per instance, so the next narrow
// container cannot rediscover the bug — the answer to a long label is shorter copy.
const nowrap = 'whitespace-nowrap'

export const btnCls =
  `inline-flex h-7 items-center gap-1.5 rounded-md border border-line-strong bg-surface px-2.5 text-sm ` +
  `font-medium text-ink-2 ${nowrap} ${trans} ${press} hover:bg-rail hover:text-ink hover:border-ink-3 ` +
  `disabled:opacity-40 disabled:pointer-events-none ${focus}`

export const btnSmCls =
  `inline-flex h-6 items-center gap-1 rounded-md border border-line-strong bg-surface px-1.5 text-xs ` +
  `font-medium text-ink-2 ${nowrap} ${trans} ${press} hover:bg-rail hover:text-ink ` +
  `disabled:opacity-40 disabled:pointer-events-none ${focus}`

// Modernized primary button with subtle depth instead of a flat fill
export const primaryBtnCls =
  `inline-flex h-8 items-center gap-1.5 rounded-md bg-primary px-4 text-sm font-semibold text-white ` +
  `${nowrap} shadow-md shadow-primary/10 ${trans} ${press} hover:bg-primary-strong hover:shadow-lg hover:shadow-primary/15 ` +
  `disabled:cursor-default disabled:opacity-50 ${focus}`

export const iconBtnCls =
  `inline-flex h-7 w-7 items-center justify-center rounded-md text-ink-3 ${trans} ${press} ` +
  `hover:bg-rail hover:text-ink ${focus}`

export const dangerBtnCls =
  `inline-flex h-7 items-center gap-1.5 rounded-md border border-danger/40 bg-surface px-2.5 text-sm ` +
  `font-medium text-danger-ink ${nowrap} ${trans} ${press} hover:bg-danger-soft hover:border-danger ` +
  `disabled:opacity-40 disabled:pointer-events-none ${focus}`

export const chipCls =
  `inline-flex h-6 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium ${trans} ${focus}`

export const inputCls =
  `h-7 rounded-md border border-line-strong bg-surface px-2 text-sm text-ink ` +
  `placeholder:text-ink-3 ${trans} focus:border-primary ${focus}`

// Floating menus/popovers + their rows.
// (zoom-rise lives in the overlay-in keyframe — tailwindcss-animate isn't installed,
// so animate-in/* classes would be silent no-ops here.)
export const menuCls =
  'overlay-enter rounded-lg border border-line bg-raised py-1 text-sm shadow-overlay'
export const menuItemCls =
  `block w-full px-3 py-1.5 text-left text-ink-2 ${trans} hover:bg-rail hover:text-ink ${focus}`

export const kbdCls =
  'inline-flex h-[18px] min-w-[18px] items-center justify-center rounded border border-line-strong ' +
  'bg-surface px-1 font-mono text-2xs font-medium text-ink-2 shadow-raised'

export const ICON = 15
export const ICON_SM = 12

import { flushSync } from 'react-dom'
export function withViewTransition(update: () => void) {
  const d = document as Document & { startViewTransition?: (cb: () => void) => unknown }
  if (d.startViewTransition && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    d.startViewTransition(() => flushSync(update))
  } else {
    update()
  }
}