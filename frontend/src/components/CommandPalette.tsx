import { useEffect, useMemo, useRef, useState } from 'react'
import { Search } from 'lucide-react'
import { useFocusTrap } from '../hooks/useFocusTrap'
import { useT } from '../i18n.tsx'
import { kbdCls, panelFloatingCls } from '../ui'

export type Command = {
  id: string
  group: string
  label: string
  /** Extra hidden search text (e.g. English label alongside km UI). */
  keywords?: string
  /** Real keyboard shortcut, shown as a kbd chip. */
  shortcut?: string
  run: () => void
}

/** Subsequence fuzzy score: all query chars must appear in order; earlier,
    denser matches win. Returns null on no match, higher = better. */
function fuzzy(query: string, text: string): number | null {
  const q = query.toLowerCase()
  const s = text.toLowerCase()
  let score = 0
  let last = -1
  for (const ch of q) {
    const i = s.indexOf(ch, last + 1)
    if (i === -1) return null
    score += last >= 0 && i === last + 1 ? 3 : 1 // consecutive runs beat scattered hits
    if (i === 0) score += 2
    last = i
  }
  return score - s.length * 0.01 // prefer shorter targets on ties
}

/** ⌘K palette: one fuzzy search over every action the workspace already has. */
export function CommandPalette(props: { commands: Command[]; onClose: () => void }) {
  const { commands, onClose } = props
  const { t } = useT()
  const [query, setQuery] = useState('')
  const [sel, setSel] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)
  // aria-modal promises the keyboard stays inside; the trap keeps it (Tab was
  // escaping into the background workspace before).
  useFocusTrap(dialogRef, true)

  useEffect(() => inputRef.current?.focus(), [])

  const matches = useMemo(() => {
    if (!query.trim()) return commands.slice(0, 12)
    return commands
      .map((c) => ({ c, s: fuzzy(query.trim(), `${c.label} ${c.keywords ?? ''}`) }))
      .filter((m): m is { c: Command; s: number } => m.s !== null)
      .sort((a, b) => b.s - a.s)
      .slice(0, 12)
      .map((m) => m.c)
  }, [commands, query])

  const clamped = Math.min(sel, Math.max(0, matches.length - 1))
  const pick = (c: Command) => {
    onClose()
    c.run()
  }

  // Keep the selected row in view while arrowing.
  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-idx="${clamped}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [clamped])

  let lastGroup = ''
  return (
    <div
      className="backdrop-enter fixed inset-0 z-50 flex items-start justify-center bg-black/30 pt-[15vh]"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={t('ks_palette')}
        tabIndex={-1}
        className={`${panelFloatingCls} overlay-enter flex max-h-[55vh] w-[480px] max-w-[calc(100vw-2rem)] flex-col overflow-hidden`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-line px-3">
          <Search size={15} className="shrink-0 text-ink-3" aria-hidden />
          <input
            ref={inputRef}
            role="combobox"
            aria-expanded
            aria-controls="palette-list"
            aria-activedescendant={matches[clamped] ? `palette-${matches[clamped].id}` : undefined}
            className="h-11 min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-ink-3"
            placeholder={t('palette_placeholder')}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setSel(0)
            }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setSel((v) => Math.min(v + 1, matches.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setSel((v) => Math.max(v - 1, 0))
              } else if (e.key === 'Enter' && matches[clamped]) {
                e.preventDefault()
                pick(matches[clamped])
              } else if (e.key === 'Escape') {
                onClose()
              }
            }}
          />
          <kbd className={kbdCls}>esc</kbd>
        </div>
        <div ref={listRef} id="palette-list" role="listbox" className="min-h-0 flex-1 overflow-y-auto py-1">
          {matches.length === 0 && (
            <p className="px-4 py-6 text-center text-sm text-ink-3">{t('palette_empty')}</p>
          )}
          {matches.map((c, i) => {
            const header = c.group !== lastGroup ? c.group : null
            lastGroup = c.group
            return (
              <div key={c.id}>
                {header && (
                  <p className="px-3 pb-0.5 pt-2 text-2xs font-semibold text-ink-2">{header}</p>
                )}
                <button
                  id={`palette-${c.id}`}
                  data-idx={i}
                  role="option"
                  aria-selected={i === clamped}
                  className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm ${
                    i === clamped ? 'bg-primary-soft text-primary-strong' : 'text-ink-2'
                  }`}
                  onMouseMove={() => setSel(i)}
                  onClick={() => pick(c)}
                >
                  <span className="min-w-0 flex-1 truncate">{c.label}</span>
                  {c.shortcut && <kbd className={kbdCls}>{c.shortcut}</kbd>}
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
