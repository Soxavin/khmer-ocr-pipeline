import type { RunSettings } from '../api/types'

/** The subset of run configuration that belongs to one document: which engine ran
    it, how it was preprocessed, and whether its export joins continuation tables.
    `labsMode` is deliberately NOT here — it only gates which engines are visible
    in the picker, never rides in the run payload, and flickering the engine list
    between documents would be pure noise. */
export type PerDocDraft = { runSettings: RunSettings; engine: string; combineExport: boolean }

export function perDocKey(id: string): string {
  return `settings:${id}`
}

/** This document's own last-edited draft, if the operator has touched it in this
    browser. Mirrors the `dismissed:${id}` read/write pattern used for per-document
    triage state (App.tsx). */
export function readPerDocSettings(id: string): PerDocDraft | null {
  try {
    const raw = localStorage.getItem(perDocKey(id))
    return raw ? (JSON.parse(raw) as PerDocDraft) : null
  } catch {
    return null
  }
}

export function writePerDocSettings(id: string, draft: PerDocDraft): void {
  try {
    localStorage.setItem(perDocKey(id), JSON.stringify(draft))
  } catch { /* storage full/blocked: draft still holds for the session */ }
}

/** Drop persisted keys the backend no longer knows (e.g. a setting removed in a
    later version): the run POST validates against the current field set and 400s
    on any extra key, so a stale blob would otherwise block every run. */
export function pruneToKnownFields(settings: RunSettings, defaults: RunSettings): RunSettings {
  return Object.fromEntries(Object.entries(settings).filter(([k]) => k in defaults)) as RunSettings
}

/** Resolve which settings a document should open with, in three tiers:

    1. Its own draft (`settings:${id}`) — the operator has already configured this
       document in this browser; always wins outright.
    2. The backend's `last_run_settings` for this document — what actually produced
       the results on screen, restored when there's no local draft (new browser,
       cleared storage).
    3. The operator's global last-used preferences — a sensible starting point for
       a document neither configured nor run yet.

    Each tier's runSettings are pruned/overlaid onto `defaults` so a stale or
    partial blob never leaves a required field missing. */
export function resolveInitialSettings(
  defaults: RunSettings,
  perDocDraft: PerDocDraft | null,
  lastRunSettings: RunSettings | null,
  globalPrefs: PerDocDraft,
): PerDocDraft {
  if (perDocDraft) {
    return {
      runSettings: { ...defaults, ...pruneToKnownFields(perDocDraft.runSettings, defaults) },
      engine: perDocDraft.engine,
      combineExport: perDocDraft.combineExport,
    }
  }
  if (lastRunSettings) {
    const lastEngine = lastRunSettings.ocr_engine_key
    return {
      runSettings: { ...defaults, ...pruneToKnownFields(lastRunSettings, defaults) },
      // combineExport is an export-time choice, never part of a run's recorded
      // settings, so tier 2 has no opinion on it — fall through to the global pref.
      engine: typeof lastEngine === 'string' ? lastEngine : globalPrefs.engine,
      combineExport: globalPrefs.combineExport,
    }
  }
  return {
    runSettings: { ...defaults, ...pruneToKnownFields(globalPrefs.runSettings, defaults) },
    engine: globalPrefs.engine,
    combineExport: globalPrefs.combineExport,
  }
}
