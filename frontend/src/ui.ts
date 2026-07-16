// One control vocabulary for the whole workspace. Every select, button, and
// icon-button composes from these; per-instance styling drift is the enemy.

export const selectCls =
  'rounded-md border border-slate-300 bg-white px-2 py-1 text-sm text-slate-700 ' +
  'focus-visible:outline-2 focus-visible:outline-primary'

export const btnCls =
  'inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-sm ' +
  'text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none ' +
  'focus-visible:outline-2 focus-visible:outline-primary'

export const btnSmCls =
  'inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs ' +
  'text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none ' +
  'focus-visible:outline-2 focus-visible:outline-primary'

export const primaryBtnCls =
  'inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-1.5 text-sm font-semibold text-white ' +
  'shadow-sm hover:bg-primary-strong disabled:cursor-default disabled:opacity-50 ' +
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary'

export const iconBtnCls =
  'inline-flex items-center justify-center rounded-md p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700 ' +
  'focus-visible:outline-2 focus-visible:outline-primary'

// Standard icon sizing: 15px chrome icons, stroke 2 (lucide default).
export const ICON = 15
