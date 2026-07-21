import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Crosshair, Maximize2, Minus, Plus, ZoomIn } from 'lucide-react'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { useT } from '../../i18n.tsx'
import { bucketCounts, confBand, type Band } from '../../lib/confidence'
import { ICON, ICON_SM, iconBtnCls, selectCls } from '../../ui'
import { ViewToggle } from './PageGrid'

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

// Segmented button inside the zoom cluster (shares the group's border).
const zoomBtn =
  'inline-flex h-6 items-center px-1.5 text-xs text-ink-2 transition-colors duration-150 ' +
  'hover:bg-rail hover:text-ink disabled:opacity-40 disabled:pointer-events-none ' +
  'focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary'

// One vocabulary with the grid + count cluster (lib/confidence): Check / Skim /
// Clean. A block with no confidence is unranked → drawn calm (green), never a
// false red alarm.
const BAND_COLOR: Record<Band, string> = { check: PALETTE.low, skim: PALETTE.mid, clean: PALETTE.high }
function confColor(v: number | null | undefined): string {
  return BAND_COLOR[confBand(v) ?? 'clean']
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
  const LENS = 200 // lens diameter (px) — roomy enough to read a coeng in context
  const LENS_MAG = 2 // 2× is the sweet spot: doubles a 14px glyph, keeps ~6 in view
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 })
  // Live mirror of view so callbacks (flyTo, keyboard/stepper zoom) read the
  // current camera without re-binding on every pan.
  const viewRef = useRef(view)
  viewRef.current = view
  const containerRef = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number } | null>(null)

  // Zoom toward the canvas centre, clamped — shared by the stepper and keyboard.
  const zoomBy = useCallback((factor: number) => {
    const el = containerRef.current
    if (!el) return
    const cx = el.clientWidth / 2
    const cy = el.clientHeight / 2
    setView((v) => {
      const scale = Math.min(8, Math.max(0.05, v.scale * factor))
      const k = scale / v.scale
      return { scale, tx: cx - k * (cx - v.tx), ty: cy - k * (cy - v.ty) }
    })
  }, [])

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
  const pct = Math.round(view.scale * 100)
  // Text-block confidence counts for the overlay legend (same bands as the grid).
  const confCounts = useMemo(
    () => bucketCounts((page?.text_blocks ?? []).map((b) => b.confidence ?? null)),
    [page],
  )
  const BANDS: Band[] = ['check', 'skim', 'clean']

  // Zoom-to-evidence: fly the camera to a table's bbox and spotlight it for a
  // beat. Called on triage jumps (flyToken) AND from the Focus Table button.
  const flyAnim = useRef<number | null>(null)
  const lastFlown = useRef(0)
  const flySeq = useRef(0)
  const [spotlight, setSpotlight] = useState<{ bbox: number[]; n: number } | null>(null)
  const flyTo = useCallback((bbox: number[]) => {
    const el = containerRef.current
    if (!el || !natural) return
    const [x0, y0, x1, y1] = bbox
    const bw = Math.max(1, x1 - x0)
    const bh = Math.max(1, y1 - y0)
    // Frame the region at 90% of the canvas, capped at 3× so text stays crisp.
    const scale = Math.min((el.clientWidth * 0.9) / bw, (el.clientHeight * 0.9) / bh, 3)
    const target = {
      scale,
      tx: el.clientWidth / 2 - ((x0 + x1) / 2) * scale,
      ty: el.clientHeight / 2 - ((y0 + y1) / 2) * scale,
    }
    const n = ++flySeq.current
    const done = () => setSpotlight({ bbox, n })
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setView(target)
      done()
      return
    }
    if (flyAnim.current !== null) cancelAnimationFrame(flyAnim.current)
    const from = { ...viewRef.current }
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
  }, [natural])
  useEffect(() => {
    // Triage jump: retriggers when page data arrives late (cross-page jumps load async).
    if (flyToken === 0 || flyToken === lastFlown.current) return
    if (!containerRef.current || !natural || !selectedBbox) return
    lastFlown.current = flyToken
    flyTo(selectedBbox)
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
                    // Confidence must not be color-only: anything not Clean is dashed too.
                    strokeDasharray={
                      overlay === 'confidence' && confBand(b.confidence) !== null && confBand(b.confidence) !== 'clean'
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
        {/* One zoom cluster: fit · step − 100% + · focus the active table. */}
        <span className="flex shrink-0 items-center overflow-hidden rounded-md border border-line-strong" role="group" aria-label={t('zoom_group')}>
          <button className={zoomBtn} onClick={() => fit()} aria-label={t('fit')} title={t('fit_tip')}>
            <Maximize2 size={ICON_SM} aria-hidden />
          </button>
          <span className="h-4 w-px self-center bg-line" aria-hidden />
          <button className={zoomBtn} onClick={() => zoomBy(1 / 1.2)} aria-label={t('zoom_out')} title={t('zoom_out')}>
            <Minus size={ICON_SM} aria-hidden />
          </button>
          <button
            className={`${zoomBtn} min-w-[3rem] justify-center font-medium tabular-nums`}
            onClick={() => setView((v) => ({ ...v, scale: 1 }))}
            title={t('zoom_reset')}
          >
            {pct}%
          </button>
          <button className={zoomBtn} onClick={() => zoomBy(1.2)} aria-label={t('zoom_in')} title={t('zoom_in')}>
            <Plus size={ICON_SM} aria-hidden />
          </button>
          <span className="h-4 w-px self-center bg-line" aria-hidden />
          <button
            className={zoomBtn}
            onClick={() => selectedBbox && flyTo(selectedBbox)}
            disabled={!selectedBbox}
            aria-label={t('focus_table')}
            title={selectedBbox ? t('focus_table') : t('focus_table_none')}
          >
            <Crosshair size={ICON_SM} aria-hidden />
          </button>
        </span>
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
        {/* Confidence counts, joined to the overlay selector — a legend that also
            carries data. Colour is redundant: every item shows its count AND band
            word, with an aria-label spelling it out (the Honest-Trust Rule). */}
        {overlay === 'confidence' && (
          <span className="flex shrink-0 items-center gap-2.5 text-2xs">
            {BANDS.map((b) => (
              <span key={b} className="inline-flex items-center gap-1 text-ink-2" aria-label={t(`band_aria_${b}` as Parameters<typeof t>[0], { n: confCounts[b] })}>
                <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: BAND_COLOR[b] }} aria-hidden />
                <span className="font-semibold text-ink">{confCounts[b]}</span>
                <span aria-hidden>{t(`band_${b}` as Parameters<typeof t>[0])}</span>
              </span>
            ))}
          </span>
        )}
      </div>
    </div>
  )
}
