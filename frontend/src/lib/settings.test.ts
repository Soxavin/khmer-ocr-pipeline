import { describe, expect, it } from 'vitest'
import { autoBadge, countOverrides, mergeSuggestion, scanSummary } from './settings'
import type { SuggestCheck } from '../api/types'

describe('autoBadge (the badge means "automation touched this row")', () => {
  it('a step the scan check switched ON reads as applied', () => {
    expect(autoBadge(true, true)).toBe('applied')
  })

  it('a step the scan check switched OFF keeps its provenance, without claiming to run', () => {
    expect(autoBadge(false, true)).toBe('auto-off')
  })

  it('the operator’s own choices carry no badge — the switch already says so', () => {
    expect(autoBadge(true, false)).toBe(null)
    expect(autoBadge(false, false)).toBe(null)
  })
})

describe('mergeSuggestion (operator overrides automated vision)', () => {
  it('applies suggestions onto untouched settings', () => {
    expect(mergeSuggestion({ sharpen: true, normalise: true }, { sharpen: false }, new Set())).toEqual({
      sharpen: false,
      normalise: true,
    })
  })

  it('never overwrites a setting the operator touched', () => {
    const merged = mergeSuggestion({ sharpen: true }, { sharpen: false }, new Set(['sharpen']))
    expect(merged.sharpen).toBe(true)
  })

  it('leaves untouched keys of the same suggestion applied', () => {
    const merged = mergeSuggestion(
      { sharpen: true, normalise: true },
      { sharpen: false, normalise: false },
      new Set(['sharpen']),
    )
    expect(merged).toEqual({ sharpen: true, normalise: false })
  })
})

describe('scanSummary (post-upload notification)', () => {
  const check = (field: string, active: boolean): SuggestCheck => ({ field, active, reason: '', detail: '' })

  it('counts the cleanups the scan found useful', () => {
    const s = scanSummary([check('deskew', true), check('sharpen', false), check('normalise', false)])
    expect(s).toEqual({ total: 3, active: 1, fields: ['deskew'] })
  })

  it('reports a clean document when nothing needs cleanup', () => {
    const s = scanSummary([check('deskew', false), check('sharpen', false)])
    expect(s).toEqual({ total: 2, active: 0, fields: [] })
  })

  it('is null-safe for a document with no checks yet', () => {
    expect(scanSummary([])).toBeNull()
  })
})

describe('countOverrides (Settings badge — deliberate overrides only)', () => {
  const defaults = {
    dpi: 'auto', page_scope: 'all', remove_stamps: true, sharpen: true,
    normalise: true, deskew: true, normalise_table_backgrounds: true,
    enable_qwen: false, anomaly_threshold: 0.15, repair_tables: false,
    convert_numerals: false,
    // Non-UI fields that must never inflate the badge:
    show_layout: true, overlay_mode: 'Region type', tables_only: false, stitch_pages: true,
  }

  it('is 0 when nothing deviates from defaults', () => {
    expect(countOverrides({ ...defaults }, defaults)).toBe(0)
  })

  it('counts each changed user-facing control', () => {
    expect(countOverrides({ ...defaults, sharpen: false, dpi: 300 }, defaults)).toBe(2)
  })

  it('ignores non-UI fields even when they differ (stale/seeded values)', () => {
    expect(countOverrides({ ...defaults, show_layout: false, stitch_pages: false, tables_only: true }, defaults)).toBe(0)
  })

  it('the auto DPI default reads as unchanged', () => {
    expect(countOverrides({ ...defaults, dpi: 'auto' }, defaults)).toBe(0)
  })

  it('is null-safe with no defaults yet', () => {
    expect(countOverrides({ sharpen: false }, undefined)).toBe(0)
  })
})
