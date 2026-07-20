import { useT } from '../../i18n.tsx'

/** [Single] [Grid] segmented control — same pattern as the Cleaned⇄Original segment. */
export function ViewToggle(props: { view: 'single' | 'grid'; onChange: (v: 'single' | 'grid') => void }) {
  const { view, onChange } = props
  const { t } = useT()
  return (
    <span className="flex shrink-0 overflow-hidden rounded-md border border-line-strong" role="group" aria-label={t('grid_tip')}>
      {(
        [
          ['single', t('view_single')],
          ['grid', t('view_grid')],
        ] as const
      ).map(([val, label]) => (
        <button
          key={val}
          className={`h-6 px-2 text-xs font-medium transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary ${
            view === val ? 'bg-primary-soft text-primary-strong' : 'bg-surface text-ink-2 hover:bg-rail'
          }`}
          aria-pressed={view === val}
          onClick={() => onChange(val)}
        >
          {label}
        </button>
      ))}
    </span>
  )
}

/** Grid overview: one card per rendered page. Click a card to open it in single
    view; the corner checkbox includes/excludes the page from the next run's scope.
    The checkbox and the card are separate tab stops — Space on the checkbox
    toggles selection, Enter on the card opens the page, never crosswise.
    `pages` is the explicit 0-based document-page list to render: pre-upload it is
    every page; post-analysis it is only the pages the run processed. */
export function PageGrid(props: {
  pages: number[]
  pageCount: number
  imageUrl: (n: number) => string
  selected: Set<number>
  onTogglePage: (n: number) => void
  onOpenPage: (n: number) => void
}) {
  const { pages, pageCount, imageUrl, selected, onTogglePage, onOpenPage } = props
  const { t } = useT()
  return (
    <div className="grid h-full grid-cols-2 gap-4 overflow-y-auto bg-rail/10 p-4 xl:grid-cols-3" style={{ gridAutoRows: 'min-content' }}>
      {pages.map((n) => (
        <div key={n} className="group relative">
          <button
            className={`block w-full overflow-hidden rounded-lg border bg-surface text-left shadow-raised transition-[border-color,box-shadow] duration-150 focus-visible:outline-2 focus-visible:outline-primary ${
              selected.has(n) ? 'border-primary/60' : 'border-line-strong/60 hover:border-line-strong'
            } hover:shadow-overlay`}
            onClick={() => onOpenPage(n)}
            title={t('grid_tip')}
          >
            <img
              src={imageUrl(n)}
              alt={t('page_of', { a: n + 1, b: pageCount })}
              loading="lazy"
              draggable={false}
              className="aspect-[3/4] w-full object-contain"
            />
          </button>
          {/* Page number chip — bottom center, always readable over the thumbnail. */}
          <span className="pointer-events-none absolute bottom-1.5 left-1/2 -translate-x-1/2 rounded-full border border-line bg-surface/90 px-2 text-2xs font-semibold text-ink">
            {n + 1}
          </span>
          {/* Include-in-run checkbox: its own island; keyboard toggling (Space) must
              never bubble into the card's open action. */}
          <label
            className="absolute left-1.5 top-1.5 flex h-6 w-6 cursor-pointer items-center justify-center rounded-md border border-line-strong bg-surface/80 backdrop-blur-sm"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => e.stopPropagation()}
            title={t('grid_include_page', { n: n + 1 })}
          >
            <input
              type="checkbox"
              className="h-4 w-4 accent-[var(--color-primary)]"
              checked={selected.has(n)}
              onChange={() => onTogglePage(n)}
              aria-label={t('grid_include_page', { n: n + 1 })}
            />
          </label>
        </div>
      ))}
    </div>
  )
}
