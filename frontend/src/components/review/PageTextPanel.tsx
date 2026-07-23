import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, Copy, PenLine, TextCursorInput } from 'lucide-react'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { useT } from '../../i18n.tsx'
import { SegmentedToggle } from '../viewer/PageGrid'
import { orderedBlockEntries } from '../../lib/blocks'
import { blockLabel } from '../../lib/blocks'
import { pairSegments, replaceSegment, segmentRange, splitSegments } from '../../lib/pageText'
import { copyText } from '../../lib/clipboard'
import { scrollIntoViewWithin } from '../../lib/scroll'
import { confBand, type Band } from '../../lib/confidence'
import { btnSmCls, chipCls, iconBtnCls } from '../../ui'

// Same three-band vocabulary as the tables and the page overlay — one confidence
// language across the whole workspace, never a second scale for one panel.
const BAND_STYLE: Record<Band, string> = {
  check: 'bg-danger-soft text-danger-ink',
  skim: 'bg-warn-soft text-warn-ink',
  clean: 'bg-ok-soft text-ok-ink',
}
const BANDS: Band[] = ['check', 'skim', 'clean']

/** Page text outside the tables — ONE string, two presentations.
 *
 *  Both views now render `corrected_text`: **Blocks** splits it into its per-block
 *  segments (the backend builds it as `"\n\n".join(block_texts)`, so the split is
 *  exact) and decorates each with its region label and confidence; **Raw** is the
 *  same string undivided. Editing either writes back to the same text.
 *
 *  This replaced an earlier split where the cards showed `text_blocks[].text` — the
 *  RAW OCR — while the textarea showed the corrected string. The two silently
 *  disagreed wherever the correction pass had done its job, and no block edit could
 *  be saved because the API persists only the whole page text.
 *
 *  Block METADATA (label, confidence, canvas link) is attached only when the segment
 *  and block counts match; see `pairSegments` for why a partial pairing is refused. */
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
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  const [editing, setEditing] = useState<number | null>(null)
  const [filter, setFilter] = useState<Band | null>(null)
  // Controlled so a canvas pick can open the panel; the analyst's own toggling
  // still works because every path writes this one piece of state.
  const [open, setOpen] = useState(true)
  const cardRefs = useRef(new Map<number, HTMLLIElement>())
  const rawRef = useRef<HTMLTextAreaElement>(null)
  // Set when Edit-in-Raw switches modes; consumed once the textarea has mounted.
  const pendingSelect = useRef<number | null>(null)

  const blocks = useMemo(() => orderedBlockEntries(page.text_blocks), [page.text_blocks])
  const segments = useMemo(() => splitSegments(text), [text])
  const { rows, aligned } = useMemo(() => pairSegments(segments, blocks), [segments, blocks])
  // Filtering is only meaningful where blocks are paired — with no pairing there
  // are no confidences to filter by.
  const counts = useMemo(() => {
    const c: Record<Band, number> = { check: 0, skim: 0, clean: 0 }
    for (const r of rows) {
      const b = confBand(r.block?.confidence)
      if (b) c[b] += 1
    }
    return c
  }, [rows])
  const visible = useMemo(
    () => (filter === null ? rows : rows.filter((r) => confBand(r.block?.confidence) === filter)),
    [rows, filter],
  )

  // A box was clicked on the page: surface the matching card. Blocks view is the
  // only one that HAS cards, so the panel switches to it rather than opening onto
  // a textarea that cannot answer the question.
  const lastFocus = useRef(0)
  useEffect(() => {
    if (!blockFocus || blockFocus.n === lastFocus.current) return
    lastFocus.current = blockFocus.n
    setOpen(true)
    setMode('blocks')
    // A filter hiding the target card would make the jump land on nothing.
    setFilter(null)
  }, [blockFocus])

  // Separate pass: the card only exists to scroll to once the panel is open and
  // in Blocks mode, which the effect above may have just changed.
  useEffect(() => {
    if (!blockFocus || !open || mode !== 'blocks') return
    const el = cardRefs.current.get(blockFocus.i)
    if (!el) return
    // scrollIntoViewWithin, not el.scrollIntoView: the latter scrolls the document
    // too (even under overflow:hidden), which jumped the whole page on a box click.
    scrollIntoViewWithin(el)
  }, [blockFocus, open, mode])

  // Edit in Raw: select that segment once the textarea exists. Runs after the mode
  // switch has painted, which is why the target rides in a ref rather than being
  // selected inline at click time.
  useEffect(() => {
    if (mode !== 'raw' || pendingSelect.current === null) return
    const i = pendingSelect.current
    pendingSelect.current = null
    const ta = rawRef.current
    if (!ta) return
    const { start, end } = segmentRange(text, i)
    // preventScroll: a textarea .focus() otherwise scrolls ancestors (incl. the
    // document) to reveal the caret; we position the selection ourselves below.
    ta.focus({ preventScroll: true })
    ta.setSelectionRange(start, end)
    // Put the selection on screen: scroll proportionally to where it sits in the
    // text, since a textarea offers no scrollIntoView for a character range.
    const ratio = text.length > 0 ? start / text.length : 0
    ta.scrollTop = Math.max(0, ratio * ta.scrollHeight - ta.clientHeight / 2)
  }, [mode, text])

  function flashCopy(ok: boolean, idx: number | null) {
    setCopyState(ok ? 'copied' : 'failed')
    setCopiedIdx(ok ? idx : null)
    window.setTimeout(() => {
      setCopyState('idle')
      setCopiedIdx(null)
    }, 1500)
  }

  async function copyAll() {
    flashCopy(await copyText(text), null)
  }

  function editInRaw(index: number) {
    pendingSelect.current = index
    setEditing(null)
    setMode('raw')
  }

  const save = () => api.putPageText(docId, pageIdx, text).then(onSaved)

  return (
    <details
      className="rounded-lg border border-line bg-surface p-2 shadow-raised"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary className="cursor-pointer text-sm font-medium text-ink-2">
        {t('page_text')}
        {rows.length > 0 && (
          <span className="ml-1.5 text-xs font-normal text-ink-3">
            {t('text_blocks_count', { n: rows.length })}
          </span>
        )}
        {saved ? '' : t('unsaved_suffix')}
      </summary>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <SegmentedToggle
          value={mode}
          onChange={setMode}
          label={t('view_text_mode')}
          options={[
            ['blocks', t('view_structured')],
            ['raw', t('view_raw')],
          ] as const}
        />
        <span className="flex items-center gap-1">
          {/* Save lives outside the Raw branch: edits now originate in BOTH views,
              so a save button reachable from only one of them would strand them. */}
          <button className={btnSmCls} disabled={saved} onClick={save}>
            {t('save_text')}
          </button>
          <button className={iconBtnCls} onClick={copyAll} title={t('copy_all')} aria-label={t('copy_all')}>
            <Copy size={14} aria-hidden />
          </button>
        </span>
      </div>
      {/* Failure is visible text, not sr-only — a silent broken Copy is a lie. */}
      <span aria-live="polite" className={copyState === 'failed' ? 'text-2xs text-danger-ink' : 'sr-only'}>
        {copyState === 'copied' ? t('copied') : copyState === 'failed' ? t('copy_failed') : ''}
      </span>

      {mode === 'blocks' ? (
        rows.length === 0 ? (
          <p className="mt-2 text-xs text-ink-3">{t('no_text_blocks')}</p>
        ) : (
          <>
            {/* Triage pills — same bands, same words as the tables and the overlay.
                Disabled rather than hidden at zero so the row never reflows. */}
            {aligned && (
              <div className="mt-2 flex flex-wrap items-center gap-1">
                <button
                  className={`${chipCls} ${filter === null ? 'bg-primary-soft text-primary-strong' : 'bg-rail text-ink-2 hover:bg-rail/70'}`}
                  onClick={() => setFilter(null)}
                  aria-pressed={filter === null}
                >
                  {t('filter_all', { n: rows.length })}
                </button>
                {BANDS.map((b) => (
                  <button
                    key={b}
                    className={`${chipCls} ${filter === b ? BAND_STYLE[b] + ' ring-1 ring-current' : 'bg-rail text-ink-2 hover:bg-rail/70'}`}
                    onClick={() => setFilter(filter === b ? null : b)}
                    disabled={counts[b] === 0}
                    aria-pressed={filter === b}
                    aria-label={t(`band_aria_${b}` as const, { n: counts[b] })}
                  >
                    {t(`band_${b}` as const)} {counts[b]}
                  </button>
                ))}
              </div>
            )}
            {/* The one honest note when the pairing could not be trusted. */}
            {!aligned && <p className="mt-2 text-2xs text-warn-ink">{t('blocks_unmatched')}</p>}

            <ol className="mt-2 space-y-1.5">
              {visible.map((row, i) => {
                // segIndex addresses the TEXT (edit/copy/raw-range); srcIndex
                // addresses the page canvas. Never interchangeable — see SegmentRow.
                const { segIndex, srcIndex, segment, block } = row
                const band = confBand(block?.confidence)
                const active = srcIndex !== null && activeBlock === srcIndex
                const linkable = Boolean(onSelectBlock) && block !== null && srcIndex !== null
                const isEditing = editing === segIndex
                return (
                  <li
                    key={segIndex}
                    // Keyed by SOURCE index: blockFocus.i arrives from the canvas.
                    ref={(el) => {
                      if (srcIndex === null) return
                      if (el) cardRefs.current.set(srcIndex, el)
                      else cardRefs.current.delete(srcIndex)
                    }}
                    // The active card carries the same primary ring the page halo
                    // does — one highlight language across the two panes.
                    className={`rounded-md border p-2 transition-all duration-150 ease-out ${
                      active
                        ? 'border-primary bg-primary-soft/40 ring-2 ring-primary/40'
                        : 'border-line-strong/30 bg-rail/20 hover:border-line-strong/60'
                    }`}
                    onMouseEnter={() => srcIndex !== null && onHoverBlock?.(srcIndex)}
                    onMouseLeave={() => onHoverBlock?.(null)}
                  >
                    <div className="flex items-center justify-between gap-2 text-2xs">
                      {/* The number is the card's position in the CURRENT list; the
                          identity it links and edits by is the segment index. */}
                      {linkable ? (
                        <button
                          type="button"
                          className="min-w-0 truncate rounded font-semibold uppercase tracking-wide text-ink-3 transition-colors duration-150 ease-out hover:text-primary-strong focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                          onClick={() => onSelectBlock?.(srcIndex!)}
                          onFocus={() => onHoverBlock?.(srcIndex!)}
                          onBlur={() => onHoverBlock?.(null)}
                          title={t('block_focus')}
                          aria-pressed={active}
                        >
                          {i + 1} · {blockLabel(block!, t('block_untitled'))}
                        </button>
                      ) : (
                        <span className="min-w-0 truncate font-semibold uppercase tracking-wide text-ink-3">
                          {i + 1}
                          {block ? ` · ${blockLabel(block, t('block_untitled'))}` : ''}
                        </span>
                      )}
                      <span className="flex shrink-0 items-center gap-1">
                        {band && (
                          // Colour is never the only signal: the percentage is
                          // visible text and the band is named for screen readers.
                          <span
                            className={`${chipCls} ${BAND_STYLE[band]}`}
                            aria-label={`${t(`band_${band}` as const)} — ${Math.round((block?.confidence ?? 0) * 100)}%`}
                          >
                            {Math.round((block?.confidence ?? 0) * 100)}%
                          </span>
                        )}
                        <button
                          className={iconBtnCls}
                          onClick={() => setEditing(isEditing ? null : segIndex)}
                          aria-pressed={isEditing}
                          title={isEditing ? t('block_edit_done') : t('block_edit')}
                          aria-label={isEditing ? t('block_edit_done') : t('block_edit')}
                        >
                          {isEditing ? <Check size={13} aria-hidden /> : <PenLine size={13} aria-hidden />}
                        </button>
                        <button
                          className={iconBtnCls}
                          onClick={() => editInRaw(segIndex)}
                          title={t('block_edit_raw')}
                          aria-label={t('block_edit_raw')}
                        >
                          <TextCursorInput size={13} aria-hidden />
                        </button>
                        <button
                          className={iconBtnCls}
                          onClick={async () => flashCopy(await copyText(segment), segIndex)}
                          title={t('block_copy')}
                          aria-label={t('block_copy')}
                        >
                          <Copy size={13} aria-hidden />
                        </button>
                      </span>
                    </div>
                    {isEditing ? (
                      <textarea
                        className="khmer-content mt-1 w-full rounded-md border border-primary/60 bg-surface p-1.5 text-ink"
                        rows={Math.min(8, segment.split('\n').length + 1)}
                        value={segment}
                        autoFocus
                        onChange={(e) => onTextChange(replaceSegment(text, segIndex, e.target.value))}
                      />
                    ) : (
                      <p className="khmer-content mt-1 whitespace-pre-wrap break-words text-ink">{segment}</p>
                    )}
                    {copiedIdx === segIndex && (
                      <span className="mt-1 block text-2xs font-medium text-ok-ink">{t('copied')}</span>
                    )}
                  </li>
                )
              })}
            </ol>
            {visible.length === 0 && (
              <p className="mt-2 text-xs text-ink-3">{t('filter_empty')}</p>
            )}
          </>
        )
      ) : (
        <textarea
          ref={rawRef}
          className="khmer-content mt-2 w-full rounded-md border border-line-strong p-2 text-ink"
          rows={8}
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
        />
      )}
    </details>
  )
}
