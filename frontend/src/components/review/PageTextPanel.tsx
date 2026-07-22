import { useEffect, useMemo, useRef, useState } from 'react'
import { Copy } from 'lucide-react'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { useT } from '../../i18n.tsx'
import { SegmentedToggle } from '../viewer/PageGrid'
import { blockLabel, mergedText, orderedBlockEntries } from '../../lib/blocks'
import { confBand, type Band } from '../../lib/confidence'
import { btnSmCls, chipCls, iconBtnCls } from '../../ui'

// Same three-band vocabulary as the tables and the page overlay — one confidence
// language across the whole workspace, never a second scale for one panel.
const BAND_STYLE: Record<Band, string> = {
  check: 'bg-danger-soft text-danger-ink',
  skim: 'bg-warn-soft text-warn-ink',
  clean: 'bg-ok-soft text-ok-ink',
}

/** Page text outside the tables.
 *
 *  Two views over the same content. **Blocks** is the audit view: one card per
 *  detected region in reading order, carrying its type and confidence, so an
 *  analyst can walk the page against the image and see exactly where the
 *  recogniser was unsure. **Raw** is the edit-and-export view.
 *
 *  Blocks are deliberately READ-ONLY: they hold the raw OCR text, while the
 *  textarea edits `corrected_text` (post-correction). Those are different strings,
 *  so an editable block list would silently overwrite the correction pass. Editing
 *  stays in exactly one place until that is resolved deliberately. */
export function PageTextPanel(props: {
  docId: string
  pageIdx: number
  page: PageData
  text: string
  onTextChange: (v: string) => void
  saved: boolean
  onSaved: () => void
  /** Source index of the block currently linked to the page canvas, or null. */
  activeBlock?: number | null
  /** A selection that came from the CANVAS: open this panel and scroll its card
      into view. `n` increments per request so re-picking the same box still moves. */
  blockFocus?: { i: number; n: number } | null
  /** A card was clicked — link it to the page image. */
  onSelectBlock?: (i: number) => void
  /** A card was hovered (null on leave) — highlight without moving the camera. */
  onHoverBlock?: (i: number | null) => void
}) {
  const { docId, pageIdx, page, text, onTextChange, saved, onSaved, activeBlock = null, blockFocus = null, onSelectBlock, onHoverBlock } = props
  const { t } = useT()
  const [mode, setMode] = useState<'blocks' | 'raw'>('blocks')
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')
  // Controlled so a canvas pick can open the panel; the analyst's own toggling
  // still works because every path writes this one piece of state.
  const [open, setOpen] = useState(true)
  const cardRefs = useRef(new Map<number, HTMLLIElement>())

  const blocks = useMemo(() => orderedBlockEntries(page.text_blocks), [page.text_blocks])

  // A box was clicked on the page: surface the matching card. Blocks view is the
  // only one that HAS cards, so the panel switches to it rather than opening onto
  // a textarea that cannot answer the question.
  const lastFocus = useRef(0)
  useEffect(() => {
    if (!blockFocus || blockFocus.n === lastFocus.current) return
    lastFocus.current = blockFocus.n
    setOpen(true)
    setMode('blocks')
  }, [blockFocus])

  // Separate pass: the card only exists to scroll to once the panel is open and
  // in Blocks mode, which the effect above may have just changed.
  useEffect(() => {
    if (!blockFocus || !open || mode !== 'blocks') return
    const el = cardRefs.current.get(blockFocus.i)
    if (!el) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    el.scrollIntoView({ block: 'center', behavior: reduced ? 'auto' : 'smooth' })
  }, [blockFocus, open, mode])

  async function copyAll() {
    const payload = mode === 'raw' ? text : mergedText(blocks.map((e) => e.block))
    let ok = false
    // navigator.clipboard is absent on non-secure origins (a LAN demo over
    // http://<ip>:8600) and can reject even where present — fall back to the
    // selection-based path, and never fail silently either way.
    try {
      await navigator.clipboard.writeText(payload)
      ok = true
    } catch {
      const ta = document.createElement('textarea')
      ta.value = payload
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      try {
        ok = document.execCommand('copy')
      } catch {
        ok = false
      }
      ta.remove()
    }
    setCopyState(ok ? 'copied' : 'failed')
    window.setTimeout(() => setCopyState('idle'), 1500)
  }

  return (
    <details
      className="rounded-lg border border-line bg-surface p-2 shadow-raised"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary className="cursor-pointer text-sm font-medium text-ink-2">
        {t('page_text')}
        {blocks.length > 0 && (
          <span className="ml-1.5 text-xs font-normal text-ink-3">
            {t('text_blocks_count', { n: blocks.length })}
          </span>
        )}
        {saved ? '' : t('unsaved_suffix')}
      </summary>

      <div className="mt-2 flex items-center justify-between gap-2">
        <SegmentedToggle
          value={mode}
          onChange={setMode}
          label={t('view_text_mode')}
          options={[
            ['blocks', t('view_structured')],
            ['raw', t('view_raw')],
          ] as const}
        />
        <button className={iconBtnCls} onClick={copyAll} title={t('copy_all')} aria-label={t('copy_all')}>
          <Copy size={14} aria-hidden />
        </button>
      </div>
      {/* Failure is visible text, not sr-only — a silent broken Copy is a lie. */}
      <span aria-live="polite" className={copyState === 'failed' ? 'text-2xs text-danger-ink' : 'sr-only'}>
        {copyState === 'copied' ? t('copied') : copyState === 'failed' ? t('copy_failed') : ''}
      </span>

      {mode === 'blocks' ? (
        blocks.length === 0 ? (
          <p className="mt-2 text-xs text-ink-3">{t('no_text_blocks')}</p>
        ) : (
          <>
            <ol className="mt-2 space-y-1.5">
              {blocks.map(({ block: b, index }, i) => {
                const band = confBand(b.confidence)
                const active = activeBlock === index
                const linkable = Boolean(onSelectBlock)
                return (
                  <li
                    key={index}
                    ref={(el) => {
                      if (el) cardRefs.current.set(index, el)
                      else cardRefs.current.delete(index)
                    }}
                    // The active card carries the same primary ring the page halo
                    // does — one highlight language across the two panes.
                    className={`rounded-md border p-2 transition-all duration-150 ease-out ${
                      active
                        ? 'border-primary bg-primary-soft/40 ring-2 ring-primary/40'
                        : 'border-line-strong/30 bg-rail/20 hover:border-line-strong/60'
                    }`}
                    onMouseEnter={() => onHoverBlock?.(index)}
                    onMouseLeave={() => onHoverBlock?.(null)}
                  >
                    <div className="flex items-center justify-between gap-2 text-2xs">
                      {/* The number is the card's position in reading order; the
                          identity it links by is the source index, kept off-screen. */}
                      {linkable ? (
                        <button
                          type="button"
                          className="min-w-0 truncate rounded font-semibold uppercase tracking-wide text-ink-3 transition-colors duration-150 ease-out hover:text-primary-strong focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                          onClick={() => onSelectBlock?.(index)}
                          onFocus={() => onHoverBlock?.(index)}
                          onBlur={() => onHoverBlock?.(null)}
                          title={t('block_focus')}
                          aria-pressed={active}
                        >
                          {i + 1} · {blockLabel(b, t('block_untitled'))}
                        </button>
                      ) : (
                        <span className="min-w-0 truncate font-semibold uppercase tracking-wide text-ink-3">
                          {i + 1} · {blockLabel(b, t('block_untitled'))}
                        </span>
                      )}
                      {band && (
                        // Colour is never the only signal: the percentage is visible
                        // text and the band is named for screen readers.
                        <span
                          className={`${chipCls} shrink-0 ${BAND_STYLE[band]}`}
                          aria-label={`${t(`band_${band}` as const)} — ${Math.round((b.confidence ?? 0) * 100)}%`}
                        >
                          {Math.round((b.confidence ?? 0) * 100)}%
                        </span>
                      )}
                    </div>
                    <p className="khmer-content mt-1 whitespace-pre-wrap break-words text-ink">{b.text}</p>
                  </li>
                )
              })}
            </ol>
            <p className="mt-1.5 text-2xs text-ink-3">{t('blocks_readonly')}</p>
          </>
        )
      ) : (
        <>
          <textarea
            className="khmer-content mt-2 w-full rounded-md border border-line-strong p-2 text-ink"
            rows={8}
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
          />
          <button
            className={`${btnSmCls} mt-1`}
            disabled={saved}
            onClick={() => api.putPageText(docId, pageIdx, text).then(onSaved)}
          >
            {t('save_text')}
          </button>
        </>
      )}
    </details>
  )
}
