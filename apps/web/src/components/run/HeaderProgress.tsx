import { useEffect, useState } from 'react'
import type { RunStatus } from '../../api/types'

/** Header progress line: milestone ranges per pipeline stage — ingest 0–10,
    preprocess 10–30, OCR 30–80 (riding the REAL page fraction), post 80–95,
    export 95–99. The value approaches each ceiling asymptotically, so the line
    always creeps and never jumps backward; run end completes to 100 then clears. */
function useSmoothProgress(active: boolean, stage: string, fraction: number): number {
  const [val, setVal] = useState(0)
  useEffect(() => {
    if (!active) {
      setVal((v) => (v > 0 ? 100 : 0))
      const id = setTimeout(() => setVal(0), 700)
      return () => clearTimeout(id)
    }
    const range: [number, number] = stage.startsWith('Reading')
      ? [2, 10]
      : stage.startsWith('Cleaning')
        ? [10, 30]
        : stage.startsWith('Finding')
          ? [30, 80]
          : stage.startsWith('Tidying')
            ? [80, 95]
            : stage.startsWith('Preparing')
              ? [95, 99]
              : [1, 8]
    const target = stage.startsWith('Finding') && fraction > 0
      ? range[0] + (range[1] - range[0]) * Math.min(1, fraction)
      : range[1]
    const id = setInterval(() => {
      setVal((v) => Math.max(v, v + (target - v) * 0.08))
    }, 250)
    return () => clearInterval(id)
  }, [active, stage, fraction])
  return val
}

/** The run's pulse: a 2px line along the header's bottom edge. Lives in its own
    component so the 250 ms animation tick re-renders THIS leaf, not the whole
    App tree (which was churning the queue rail and every grid thumbnail ~7×/s
    for the length of an extraction). */
export function HeaderProgress(props: { status: RunStatus | undefined }) {
  const { status } = props
  const val = useSmoothProgress(status?.active ?? false, status?.stage ?? '', status?.fraction ?? 0)
  if (val <= 0) return null
  return (
    <span
      aria-hidden
      className="absolute bottom-0 left-0 h-[2px] bg-primary transition-[width] duration-500 motion-reduce:transition-none"
      style={{ width: `${val}%` }}
    />
  )
}
