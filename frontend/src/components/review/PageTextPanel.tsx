import { useMemo, useState } from 'react'
import { Copy } from 'lucide-react'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { useT } from '../../i18n.tsx'
import { SegmentedToggle } from '../viewer/PageGrid'
import { blockLabel, mergedText, orderedBlocks } from '../../lib/blocks'
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
}) {
  const { docId, pageIdx, page, text, onTextChange, saved, onSaved } = props
  const { t } = useT()
  const [mode, setMode] = useState<'blocks' | 'raw'>('blocks')
  const [copied, setCopied] = useState(false)

  const blocks = useMemo(() => orderedBlocks(page.text_blocks), [page.text_blocks])

  async function copyAll() {
    await navigator.clipboard.writeText(mode === 'raw' ? text : mergedText(blocks))
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <details className="rounded-lg border border-line bg-surface p-2 shadow-raised">
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
      <span aria-live="polite" className="sr-only">{copied ? t('copied') : ''}</span>

      {mode === 'blocks' ? (
        blocks.length === 0 ? (
          <p className="mt-2 text-xs text-ink-3">{t('no_text_blocks')}</p>
        ) : (
          <>
            <ol className="mt-2 space-y-1.5">
              {blocks.map((b, i) => {
                const band = confBand(b.confidence)
                return (
                  <li
                    key={i}
                    className="rounded-md border border-line-strong/30 bg-rail/20 p-2 transition-colors duration-150 hover:border-line-strong/60"
                  >
                    <div className="flex items-center justify-between gap-2 text-2xs">
                      <span className="min-w-0 truncate font-semibold uppercase tracking-wide text-ink-3">
                        {i + 1} · {blockLabel(b, t('block_untitled'))}
                      </span>
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
