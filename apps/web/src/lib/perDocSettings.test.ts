import { beforeEach, describe, expect, it } from 'vitest'
import { perDocKey, pruneToKnownFields, readPerDocSettings, resolveInitialSettings, writePerDocSettings } from './perDocSettings'

beforeEach(() => localStorage.clear())

describe('perDocKey / read+write round-trip', () => {
  it('keys by document id, mirroring the dismissed:${id} pattern', () => {
    expect(perDocKey('doc-1')).toBe('settings:doc-1')
  })

  it('round-trips a written draft', () => {
    const draft = { runSettings: { sharpen: true }, engine: 'gemini', combineExport: false }
    writePerDocSettings('doc-1', draft)
    expect(readPerDocSettings('doc-1')).toEqual(draft)
  })

  it('is null for a document that was never touched', () => {
    expect(readPerDocSettings('doc-never-seen')).toBeNull()
  })

  it('is null-safe against a corrupt entry', () => {
    localStorage.setItem('settings:doc-1', '{not json')
    expect(readPerDocSettings('doc-1')).toBeNull()
  })
})

describe('pruneToKnownFields', () => {
  it('drops keys the backend no longer knows', () => {
    const defaults = { sharpen: true, deskew: true }
    expect(pruneToKnownFields({ sharpen: false, removed_flag: true }, defaults)).toEqual({ sharpen: false })
  })
})

describe('resolveInitialSettings (three-tier restore)', () => {
  const defaults = { sharpen: true, deskew: false, dpi: 'auto' }
  const globalPrefs = { runSettings: { sharpen: false }, engine: 'surya', combineExport: true }

  it('tier 1: a per-document draft wins outright, ignoring tiers 2 and 3', () => {
    const perDoc = { runSettings: { sharpen: true, deskew: true }, engine: 'gemini', combineExport: false }
    const lastRun = { sharpen: false, ocr_engine_key: 'surya_kiri' }
    const out = resolveInitialSettings(defaults, perDoc, lastRun, globalPrefs)
    expect(out).toEqual({
      runSettings: { sharpen: true, deskew: true, dpi: 'auto' },
      engine: 'gemini',
      combineExport: false,
    })
  })

  it('tier 2: no draft, but last_run_settings present — wins over global prefs', () => {
    const lastRun = { sharpen: false, deskew: true, ocr_engine_key: 'surya_kiri' }
    const out = resolveInitialSettings(defaults, null, lastRun, globalPrefs)
    expect(out).toEqual({
      runSettings: { sharpen: false, deskew: true, dpi: 'auto' },
      engine: 'surya_kiri',
      // combineExport is never part of last_run_settings — falls through to the global pref.
      combineExport: true,
    })
  })

  it('tier 2 falls back to the global engine when last_run_settings has none', () => {
    const out = resolveInitialSettings(defaults, null, { sharpen: false }, globalPrefs)
    expect(out.engine).toBe('surya')
  })

  it('tier 3: neither a draft nor a last run — falls to global last-used preferences', () => {
    const out = resolveInitialSettings(defaults, null, null, globalPrefs)
    expect(out).toEqual({
      runSettings: { sharpen: false, deskew: false, dpi: 'auto' },
      engine: 'surya',
      combineExport: true,
    })
  })

  it('every tier fills in defaults for fields the source blob never mentioned', () => {
    const perDoc = { runSettings: {}, engine: 'gemini', combineExport: true }
    const out = resolveInitialSettings(defaults, perDoc, null, globalPrefs)
    expect(out.runSettings).toEqual(defaults)
  })
})
