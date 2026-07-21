import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Maximize2, ZoomIn } from 'lucide-react'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { useT } from '../../i18n.tsx'
import { btnSmCls, ICON, ICON_SM, iconBtnCls, selectCls } from '../../ui'
import { ViewToggle } from './PageGrid'

// Text-block confidence buckets (model_config.py: CONFIDENCE_LOW/MID).
const CONF_LOW = 0.5
const CONF_MID = 0.8
// These are SVG stroke colors drawn OVER the page photograph, so they are
// deliberately theme-independent fixed hex (a dark-mode swap would make the
// overlay fight the scan, not the UI). They intentionally equal the
// --color-conf-* / webapp components.py values so the page overlay and the grid
// tints read as one system; keep the two in sync by hand if the tokens change.
const PALETTE = { high: '#16a34a', mid: '#f59e0b', low: '#dc2626' }
// Region-type colors, ported 1:1 from webapp/components.py LABEL_COLORS — same
// rationale: fixed hex over the photo, matched to the Python side on purpose.
const LABEL_COLORS: Record<string, string> = {
  Text: '#4A90D9',
  Table: '#E74C3C',
  TableOfContents: '#E67E22',
  Picture: '#27AE60',
  Figure: '#27AE60',
  Caption: '#8E44AD',
}

type OverlayMode = 'confidence' | 'regions' | 'none'

function confColor(v: number | null | undefined): string {
  const c = v ?? 0
  return c < CONF_LOW ? PALETTE.low : c < CONF_MID ? PALETTE.mid : PALETTE.high
}

export function PageViewer(props: {
  docId: string
  pageIdx: number
  pageCount: number
  onPageChange: (idx: number) => void
  page: PageData | undefined
  selectedTable: string | null
  onTableClick: (tableId: string) => void
  flyToken?: number // increments on triage jumps: fly the camera to the evidence
  /** Canvas view toggle (single ⇄ grid) shown in the strip when provided. */
  view?: 'single' | 'grid'
  onViewChange?: (v: 'single' | 'grid') => void
}) {
  const { docId, pageIdx, pageCount, onPageChange, page, selectedTable, onTableClick, flyToken = 0, view: canvasView, onViewChange } = props
  const { t } = useT()
  const [variant, setVariant] = useState<'processed' | 'original'>('processed')
  const [overlay, setOverlay] = useState<OverlayMode>('confidence')
  const [loupeOn, setLoupeOn] = useState(false)
  const [lensPos, setLensPos] = useState<{ x: number; y: number } | null>(null)
  const LENS = 180 // lens diameter (px)
  const LENS_MAG = 3 // magnification relative to the scan's natural pixels
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 })
  const containerRef = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number } | null>(null)

  const fit = useCallback(
    (nat?: { w: number; h: number }) => {
      const n = nat ?? natural
      const el = containerRef.current
      if (!n || !el) return
      const pad = 16
      const scale = Math.min((el.clientWidth - pad) / n.w, (el.clientHeight - pad) / n.h)
      setView({
        scale,
        tx: (el.clientWidth - n.w * scale) / 2,
        ty: (el.clientHeight - n.h * scale) / 2,
      })
    },
    [natural],
  )

  // New page: image reloads; refit once its size is known.
  useEffect(() => {
    setNatural(null)
  }, [docId, pageIdx, variant])

  // Wheel zoom toward the cursor (non-passive: we must preventDefault page scroll).
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      setView((v) => {
        const factor = Math.exp(-e.deltaY * 0.0015)
        const scale = Math.min(8, Math.max(0.05, v.scale * factor))
        const k = scale / v.scale
        return { scale, tx: mx - k * (mx - v.tx), ty: my - k * (my - v.ty) }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  const bboxIndex = page?.table_bbox_index ?? {}
  const selectedBbox = selectedTable ? bboxIndex[selectedTable] : undefined

  // Zoom-to-evidence: on a triage jump, fly the camera to the flagged table and
  // spotlight it for a beat. The tool navigates the proof; the analyst just reads.
  const flyAnim = useRef<number | null>(null)
  const lastFlown = useRef(0)
  const [spotlight, setSpotlight] = useState<{ bbox: number[]; n: number } | null>(null)
  useEffect(() => {
    // Retriggers when page data arrives late (cross-page jumps load async).
    if (flyToken === 0 || flyToken === lastFlown.current) return
    const el = containerRef.current
    if (!el || !natural || !selectedBbox) return
    lastFlown.current = flyToken
    const [x0, y0, x1, y1] = selectedBbox
    const bw = Math.max(1, x1 - x0)
    const bh = Math.max(1, y1 - y0)
    // Frame the region at 90% of the canvas, capped at 3× so text stays crisp.
    const scale = Math.min((el.clientWidth * 0.9) / bw, (el.clientHeight * 0.9) / bh, 3)
    const target = {
      scale,
      tx: el.clientWidth / 2 - ((x0 + x1) / 2) * scale,
      ty: el.clientHeight / 2 - ((y0 + y1) / 2) * scale,
    }
    const done = () => setSpotlight({ bbox: selectedBbox, n: flyToken })
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setView(target)
      done()
      return
    }
    if (flyAnim.current !== null) cancelAnimationFrame(flyAnim.current)
    const from = { ...view }
    const t0 = performance.now()
    const DUR = 380
    const easeOutExpo = (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t))
    const step = (now: number) => {
      const k = easeOutExpo(Math.min(1, (now - t0) / DUR))
      setView({
        scale: from.scale + (target.scale - from.scale) * k,
        tx: from.tx + (target.tx - from.tx) * k,
        ty: from.ty + (target.ty - from.ty) * k,
      })
      if (k < 1) flyAnim.current = requestAnimationFrame(step)
      else {
        flyAnim.current = null
        done()
      }
    }
    flyAnim.current = requestAnimationFrame(step)
    return () => {
      if (flyAnim.current !== null) cancelAnimationFrame(flyAnim.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flyToken, selectedBbox, natural])

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Top strip carries reading position ONLY — view controls float on the
          canvas they act on (control placement architecture, §2.45). */}
      <div className="flex h-10 shrink-0 items-center justify-center gap-1 overflow-x-auto whitespace-nowrap border-b border-line-strong/50 bg-rail/30 px-3 text-sm">
        <button
          className={iconBtnCls}
          onClick={() => onPageChange(pageIdx - 1)}
          disabled={pageIdx <= 0}
          aria-label={t('prev_page')}
        >
          <ChevronLeft size={ICON} aria-hidden />
        </button>
        <select
          className={selectCls}
          value={pageIdx}
          onChange={(e) => onPageChange(Number(e.target.value))}
          aria-label={t('pages')}
        >
          {Array.from({ length: pageCount }, (_, i) => (
            <option key={i} value={i}>
              {t('page_n_of', { i: i + 1, n: pageCount })}
            </option>
          ))}
        </select>
        <button
          className={iconBtnCls}
          onClick={() => onPageChange(pageIdx + 1)}
          disabled={pageIdx >= pageCount - 1}
          aria-label={t('next_page')}
        >
          <ChevronRight size={ICON} aria-hidden />
        </button>
        {canvasView && onViewChange && (
          <>
            <span className="mx-1.5 h-4 w-px shrink-0 bg-line" aria-hidden />
            <ViewToggle view={canvasView} onChange={onViewChange} />
          </>
        )}
      </div>

      <div
        ref={containerRef}
        // The canvas is focusable so pan/zoom have a keyboard path (mouse drag +
        // wheel are enhancements, not the only way in): arrows pan, +/− zoom
        // toward centre, 0 refits.
        tabIndex={0}
        role="group"
        aria-label={t('canvas_aria')}
        className="relative min-h-0 flex-1 cursor-grab overflow-hidden bg-canvas focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary active:cursor-grabbing"
        onKeyDown={(e) => {
          const PAN = 60
          if (e.key === 'ArrowLeft') setView((v) => ({ ...v, tx: v.tx + PAN }))
          else if (e.key === 'ArrowRight') setView((v) => ({ ...v, tx: v.tx - PAN }))
          else if (e.key === 'ArrowUp') setView((v) => ({ ...v, ty: v.ty + PAN }))
          else if (e.key === 'ArrowDown') setView((v) => ({ ...v, ty: v.ty - PAN }))
          else if (e.key === '0') fit()
          else if (e.key === '+' || e.key === '=' || e.key === '-') {
            // Zoom toward the canvas centre, mirroring the wheel-zoom math.
            const el = containerRef.current
            if (!el) return
            const cx = el.clientWidth / 2
            const cy = el.clientHeight / 2
            setView((v) => {
              const scale = Math.min(8, Math.max(0.05, v.scale * (e.key === '-' ? 1 / 1.2 : 1.2)))
              const k = scale / v.scale
              return { scale, tx: cx - k * (cx - v.tx), ty: cy - k * (cy - v.ty) }
            })
          } else return
          e.preventDefault()
        }}
        onMouseDown={(e) => {
          drag.current = { x: e.clientX - view.tx, y: e.clientY - view.ty }
        }}
        onMouseMove={(e) => {
          if (drag.current) {
            setLensPos(null) // panning and inspecting are different hand modes
            setView((v) => ({ ...v, tx: e.clientX - drag.current!.x, ty: e.clientY - drag.current!.y }))
            return
          }
          if (loupeOn) {
            const rect = e.currentTarget.getBoundingClientRect()
            setLensPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
          }
        }}
        onMouseUp={() => (drag.current = null)}
        onMouseLeave={() => {
          drag.current = null
          setLensPos(null)
        }}
      >
        {/* Loupe: a lens showing the SAME image at 3× natural pixels around the
            cursor — glyph-level inspection without disturbing the page view. */}
        {loupeOn && lensPos && natural && (
          <div
            className="pointer-events-none absolute z-20 overflow-hidden rounded-full border-2 border-line-strong bg-surface shadow-overlay"
            style={{ width: LENS, height: LENS, left: lensPos.x - LENS / 2, top: lensPos.y - LENS / 2 }}
            aria-hidden
          >
            <img
              src={api.pageImageUrl(docId, pageIdx, variant)}
              alt=""
              draggable={false}
              className="max-w-none select-none"
              style={{
                width: natural.w,
                height: natural.h,
                transformOrigin: '0 0',
                transform: (() => {
                  // Image coords under the cursor, via the current pan/zoom…
                  const ix = (lensPos.x - view.tx) / view.scale
                  const iy = (lensPos.y - view.ty) / view.scale
                  // …centered in the lens at fixed 3× natural magnification.
                  return `translate(${LENS / 2 - ix * LENS_MAG}px, ${LENS / 2 - iy * LENS_MAG}px) scale(${LENS_MAG})`
                })(),
              }}
            />
          </div>
        )}
        <div
          style={{
            transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
            transformOrigin: '0 0',
            width: natural?.w,
            height: natural?.h,
          }}
          className="relative"
        >
          <img
            src={api.pageImageUrl(docId, pageIdx, variant)}
            alt={`Page ${pageIdx + 1} (${variant})`}
            draggable={false}
            // Pages fade in once sized — page turns read as a transition, not a pop.
            className={`block max-w-none select-none shadow-overlay transition-opacity duration-200 ${natural ? 'opacity-100' : 'opacity-0'}`}
            onLoad={(e) => {
              const img = e.currentTarget
              const n = { w: img.naturalWidth, h: img.naturalHeight }
              setNatural(n)
              fit(n)
            }}
          />
          {natural && page && overlay !== 'none' && (
            <svg
              className="absolute inset-0"
              width="100%"
              height="100%"
              viewBox={`0 0 ${natural.w} ${natural.h}`}
            >
              {page.text_blocks.map((b, i) =>
                b.bbox && b.bbox.length >= 4 ? (
                  <rect
                    key={`t${i}`}
                    x={b.bbox[0]}
                    y={b.bbox[1]}
                    width={b.bbox[2] - b.bbox[0]}
                    height={b.bbox[3] - b.bbox[1]}
                    fill="none"
                    stroke={overlay === 'confidence' ? confColor(b.confidence) : (LABEL_COLORS[b.label ?? ''] ?? '#95A5A6')}
                    strokeWidth={2 / view.scale}
                    // Confidence must not be color-only: flagged blocks are dashed too.
                    strokeDasharray={
                      overlay === 'confidence' && (b.confidence ?? 0) < CONF_MID
                        ? `${6 / view.scale} ${4 / view.scale}`
                        : undefined
                    }
                  />
                ) : null,
              )}
              {Object.entries(bboxIndex).map(([tid, bbox]) => (
                <rect
                  key={tid}
                  x={bbox[0]}
                  y={bbox[1]}
                  width={bbox[2] - bbox[0]}
                  height={bbox[3] - bbox[1]}
                  fill="transparent"
                  stroke={LABEL_COLORS.Table}
                  strokeWidth={2 / view.scale}
                  className="region-rect cursor-pointer"
                  role="button"
                  tabIndex={0}
                  aria-label={t('open_table', { tid })}
                  onClick={() => onTableClick(tid)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onTableClick(tid)
                    }
                  }}
                >
                  <title>{t('open_table_tip', { tid })}</title>
                </rect>
              ))}
              {selectedBbox && (
                <rect
                  x={selectedBbox[0]}
                  y={selectedBbox[1]}
                  width={selectedBbox[2] - selectedBbox[0]}
                  height={selectedBbox[3] - selectedBbox[1]}
                  fill="var(--color-primary)"
                  fillOpacity={0.15}
                  stroke="var(--color-primary)"
                  strokeWidth={4 / view.scale}
                  pointerEvents="none"
                />
              )}
              {/* Spotlight: dim everything around the evidence for a beat after landing. */}
              {spotlight && (
                <g key={spotlight.n} className="spotlight-fade" pointerEvents="none" fill="#0b1526">
                  <rect x={0} y={0} width={natural.w} height={Math.max(0, spotlight.bbox[1])} />
                  <rect x={0} y={spotlight.bbox[3]} width={natural.w} height={Math.max(0, natural.h - spotlight.bbox[3])} />
                  <rect x={0} y={spotlight.bbox[1]} width={Math.max(0, spotlight.bbox[0])} height={spotlight.bbox[3] - spotlight.bbox[1]} />
                  <rect x={spotlight.bbox[2]} y={spotlight.bbox[1]} width={Math.max(0, natural.w - spotlight.bbox[2])} height={spotlight.bbox[3] - spotlight.bbox[1]} />
                </g>
              )}
            </svg>
          )}
        </div>
      </div>
      {/* Docked footer: a fluid rail — the controls cluster and the legend sit at
          opposite ends and wrap gracefully while the split divider is dragged. */}
      <div className="flex min-h-10 w-full shrink-0 flex-wrap items-center justify-between gap-x-3 gap-y-1 border-t border-line-strong/60 bg-surface px-3 py-1">
        <span className="flex flex-wrap items-center gap-1 whitespace-nowrap">
        <button className={btnSmCls} onClick={() => fit()} title={t('fit_tip')}>
          <Maximize2 size={ICON_SM} aria-hidden />
          {t('fit')}
        </button>
        <button className={btnSmCls} onClick={() => setView((v) => ({ ...v, scale: 1 }))} title={t('actual_size')}>
          100%
        </button>
        <span className="mx-0.5 h-4 w-px shrink-0 bg-line" aria-hidden />
        {/* Segmented control: which rendition of the page is shown. */}
        <span className="flex shrink-0 overflow-hidden rounded-md border border-line-strong" role="group" aria-label={t('rendition_aria')}>
          {(
            [
              ['processed', t('variant_processed')],
              ['original', t('variant_original')],
            ] as const
          ).map(([val, label]) => (
            <button
              key={val}
              className={`h-6 px-2 text-xs font-medium transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary ${
                variant === val ? 'bg-primary-soft text-primary-strong' : 'bg-surface text-ink-2 hover:bg-rail'
              }`}
              aria-pressed={variant === val}
              title={val === 'processed' ? t('variant_processed_tip') : t('variant_original_tip')}
              onClick={() => setVariant(val)}
            >
              {label}
            </button>
          ))}
        </span>
        <select
          className={`${selectCls} h-6 shrink-0 text-xs`}
          value={overlay}
          onChange={(e) => setOverlay(e.target.value as OverlayMode)}
          aria-label={t('overlay_aria')}
          title={t('overlay_tip')}
        >
          <option value="confidence">{t('overlay_conf')}</option>
          <option value="regions">{t('overlay_regions')}</option>
          <option value="none">{t('overlay_none')}</option>
        </select>
        <span className="mx-0.5 h-4 w-px shrink-0 bg-line" aria-hidden />
        {/* Loupe is icon-only: the glass is the label, the tooltip carries the words. */}
        <button
          className={`${iconBtnCls} shrink-0 ${loupeOn ? 'bg-primary-soft text-primary-strong' : ''}`}
          onClick={() => setLoupeOn((l) => !l)}
          aria-pressed={loupeOn}
          aria-label={t('loupe')}
          title={t('loupe_tip')}
        >
          <ZoomIn size={ICON_SM} aria-hidden />
        </button>
        </span>
        {/* Dot-only legend: tooltips carry the meaning; visually silent. */}
        {overlay === 'confidence' && (
          <span className="flex shrink-0 items-center gap-1.5">
            <span className="inline-block h-2.5 w-3.5 rounded-[2px] border-2" style={{ borderColor: PALETTE.high }} title={t('legend_high')} />
            <span className="inline-block h-2.5 w-3.5 rounded-[2px] border-2 border-dashed" style={{ borderColor: PALETTE.mid }} title={t('legend_mid')} />
            <span className="inline-block h-2.5 w-3.5 rounded-[2px] border-2 border-dashed" style={{ borderColor: PALETTE.low }} title={t('legend_low')} />
          </span>
        )}
      </div>
    </div>
  )
}
