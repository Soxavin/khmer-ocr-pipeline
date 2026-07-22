import { useEffect, useRef, useState } from 'react'
import { useT } from '../../i18n.tsx'

const MAX_THUMB_RETRIES = 4

/** A single grid thumbnail that never shows the browser's broken-image glyph.
    Page images are rasterized lazily server-side (see api_preview_image): the first
    burst of card requests during an active extraction can lose the race and error.
    So we hold a calm skeleton while loading, retry with backoff (the cached ingest
    lands within a beat or two), optionally fall back to the raw preview rendition,
    and — critically — reset the whole lifecycle whenever `src` changes so a stale
    thumbnail never bleeds across a view toggle or a document switch. */
function GridThumb(props: { src: string; fallbackSrc?: string; alt: string }) {
  const { src, fallbackSrc, alt } = props
  const [status, setStatus] = useState<'load' | 'ok' | 'fail'>('load')
  const [current, setCurrent] = useState(src)
  const attempt = useRef(0)
  const usedFallback = useRef(false)
  const timer = useRef<number | undefined>(undefined)

  // New src → new rendition finished, or the active document changed: start over.
  useEffect(() => {
    attempt.current = 0
    usedFallback.current = false
    setStatus('load')
    setCurrent(src)
    return () => window.clearTimeout(timer.current)
  }, [src])

  function onError() {
    // 1) One shot at the fallback rendition (post-analysis grid → raw preview).
    if (fallbackSrc && !usedFallback.current) {
      usedFallback.current = true
      setCurrent(fallbackSrc)
      return
    }
    // 2) Back off and retry with a cache-busting param so the browser refetches
    //    rather than replaying its cached error once the lazy ingest is ready.
    if (attempt.current < MAX_THUMB_RETRIES) {
      attempt.current += 1
      const base = usedFallback.current && fallbackSrc ? fallbackSrc : src
      const n = attempt.current
      timer.current = window.setTimeout(() => {
        setCurrent(`${base}${base.includes('?') ? '&' : '?'}retry=${n}`)
      }, 250 * n)
      return
    }
    // 3) Out of retries: keep the skeleton, never the broken '?'.
    setStatus('fail')
  }

  return (
    <div className="relative aspect-[3/4] w-full">
      {status !== 'ok' && (
        <div className={`absolute inset-0 ${status === 'fail' ? 'bg-rail/40' : 'animate-pulse bg-rail/60'}`} aria-hidden />
      )}
      {status !== 'fail' && (
        <img
          src={current}
          alt={alt}
          loading="lazy"
          draggable={false}
          onLoad={() => setStatus('ok')}
          onError={onError}
          className={`aspect-[3/4] w-full object-contain transition-opacity duration-200 ${status === 'ok' ? 'opacity-100' : 'opacity-0'}`}
        />
      )}
    </div>
  )
}

/** The workspace's one segmented control: a small set of exclusive choices shown
    inline (Single⇄Grid, Cleaned⇄Original, Blocks⇄Raw). Generic over the value so
    every segment in the app shares one set of states and one focus treatment —
    a second hand-rolled copy is how vocabularies drift apart. */
export function SegmentedToggle<T extends string>(props: {
  value: T
  options: readonly (readonly [T, string])[]
  onChange: (v: T) => void
  label: string
}) {
  const { value, options, onChange, label } = props
  return (
    <span className="flex shrink-0 overflow-hidden rounded-md border border-line-strong" role="group" aria-label={label}>
      {options.map(([val, text]) => (
        <button
          key={val}
          className={`h-6 px-2 text-xs font-medium transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary ${
            value === val ? 'bg-primary-soft text-primary-strong' : 'bg-surface text-ink-2 hover:bg-rail'
          }`}
          aria-pressed={value === val}
          onClick={() => onChange(val)}
        >
          {text}
        </button>
      ))}
    </span>
  )
}

/** [Single] [Grid] — the canvas's view segment. */
export function ViewToggle(props: { view: 'single' | 'grid'; onChange: (v: 'single' | 'grid') => void }) {
  const { t } = useT()
  return (
    <SegmentedToggle
      value={props.view}
      onChange={props.onChange}
      label={t('grid_tip')}
      options={[
        ['single', t('view_single')],
        ['grid', t('view_grid')],
      ] as const}
    />
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
  /** Optional secondary rendition tried once before retrying (post-analysis grid
      falls back to the raw preview when a processed page isn't ready yet). */
  fallbackUrl?: (n: number) => string
  selected: Set<number>
  onTogglePage: (n: number) => void
  onOpenPage: (n: number) => void
}) {
  const { pages, pageCount, imageUrl, fallbackUrl, selected, onTogglePage, onOpenPage } = props
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
            <GridThumb
              src={imageUrl(n)}
              fallbackSrc={fallbackUrl?.(n)}
              alt={t('page_of', { a: n + 1, b: pageCount })}
            />
          </button>
          {/* Page number chip — bottom center, always readable over the thumbnail.
              Clear of the top-left checkbox island; nowrap so it never clips. */}
          <span className="pointer-events-none absolute bottom-1.5 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full border border-line bg-surface/90 px-2 text-2xs font-semibold leading-tight text-ink">
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
