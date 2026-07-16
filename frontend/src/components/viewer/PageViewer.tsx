import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { api } from '../../api/client'
import type { PageData } from '../../api/types'
import { btnSmCls, iconBtnCls, selectCls } from '../../ui'

// Text-block confidence buckets (model_config.py: CONFIDENCE_LOW/MID).
const CONF_LOW = 0.5
const CONF_MID = 0.8
const PALETTE = { high: '#16a34a', mid: '#f59e0b', low: '#dc2626' }
// Region-type colors, ported from webapp/components.py LABEL_COLORS.
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
}) {
  const { docId, pageIdx, pageCount, onPageChange, page, selectedTable, onTableClick } = props
  const [variant, setVariant] = useState<'processed' | 'original'>('processed')
  const [overlay, setOverlay] = useState<OverlayMode>('confidence')
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

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-white px-3 py-1.5 text-sm">
        <div className="flex items-center gap-1">
          <button
            className={iconBtnCls}
            onClick={() => onPageChange(pageIdx - 1)}
            disabled={pageIdx <= 0}
            aria-label="Previous page"
          >
            <ChevronLeft size={16} aria-hidden />
          </button>
          <select
            className={selectCls}
            value={pageIdx}
            onChange={(e) => onPageChange(Number(e.target.value))}
            aria-label="Page"
          >
            {Array.from({ length: pageCount }, (_, i) => (
              <option key={i} value={i}>
                Page {i + 1} / {pageCount}
              </option>
            ))}
          </select>
          <button
            className={iconBtnCls}
            onClick={() => onPageChange(pageIdx + 1)}
            disabled={pageIdx >= pageCount - 1}
            aria-label="Next page"
          >
            <ChevronRight size={16} aria-hidden />
          </button>
        </div>
        <div className="flex items-center gap-1 text-xs">
          <button className={btnSmCls} onClick={() => fit()}>
            Fit
          </button>
          <button className={btnSmCls} onClick={() => setView((v) => ({ ...v, scale: 1 }))}>
            100%
          </button>
          <button
            className={`${btnSmCls} ${variant === 'original' ? 'border-primary text-primary' : ''}`}
            onClick={() => setVariant((v) => (v === 'processed' ? 'original' : 'processed'))}
            title="Toggle between the cleaned page and the original scan"
          >
            {variant === 'processed' ? 'Processed' : 'Original'}
          </button>
          <select
            className={selectCls}
            value={overlay}
            onChange={(e) => setOverlay(e.target.value as OverlayMode)}
            aria-label="Overlay mode"
            title="What the colored boxes mean"
          >
            <option value="confidence">Confidence boxes</option>
            <option value="regions">Region types</option>
            <option value="none">No boxes</option>
          </select>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative min-h-0 flex-1 cursor-grab overflow-hidden bg-slate-200 active:cursor-grabbing"
        onMouseDown={(e) => {
          drag.current = { x: e.clientX - view.tx, y: e.clientY - view.ty }
        }}
        onMouseMove={(e) => {
          if (!drag.current) return
          setView((v) => ({ ...v, tx: e.clientX - drag.current!.x, ty: e.clientY - drag.current!.y }))
        }}
        onMouseUp={() => (drag.current = null)}
        onMouseLeave={() => (drag.current = null)}
      >
        {/* Legend floats on the canvas corner — out of the toolbar's flow. */}
        {overlay === 'confidence' && (
          <span className="absolute bottom-2 left-2 z-10 flex items-center gap-2.5 rounded-md border border-slate-200 bg-white/90 px-2 py-1 text-xs text-slate-600 shadow-sm">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-3.5 border-2" style={{ borderColor: PALETTE.high }} aria-hidden />
              ≥80% solid
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-3.5 border-2 border-dashed" style={{ borderColor: PALETTE.mid }} aria-hidden />
              50–80%
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-3.5 border-2 border-dashed" style={{ borderColor: PALETTE.low }} aria-hidden />
              &lt;50% dashed
            </span>
          </span>
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
            className="block max-w-none select-none shadow"
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
                  aria-label={`Open table ${tid}`}
                  onClick={() => onTableClick(tid)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onTableClick(tid)
                    }
                  }}
                >
                  <title>{`${tid} — click to open this table`}</title>
                </rect>
              ))}
              {selectedBbox && (
                <rect
                  x={selectedBbox[0]}
                  y={selectedBbox[1]}
                  width={selectedBbox[2] - selectedBbox[0]}
                  height={selectedBbox[3] - selectedBbox[1]}
                  fill="#2563eb"
                  fillOpacity={0.15}
                  stroke="#2563eb"
                  strokeWidth={4 / view.scale}
                  pointerEvents="none"
                />
              )}
            </svg>
          )}
        </div>
      </div>
    </div>
  )
}
