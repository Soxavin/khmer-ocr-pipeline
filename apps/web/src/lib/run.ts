import type { DocSummary, RunSettings } from '../api/types'

/** True while the extraction pipeline is occupied anywhere in the workspace.
    The server runs one document at a time, so this gates every launch control —
    not just the active document's. `batchRunning` covers the window between
    dispatching a batch and the first document flipping to `running`. */
export function isBusy(documents: DocSummary[], batchRunning: boolean): boolean {
  return batchRunning || documents.some((d) => d.status === 'running')
}

/** Start a run only if the pipeline is free, treating the server's 409 as a
    benign collision rather than an error worth showing an analyst. Real failures
    still propagate. */
export async function guardedRun(
  busy: boolean,
  start: () => Promise<unknown>,
): Promise<'started' | 'blocked'> {
  if (busy) return 'blocked'
  try {
    await start()
    return 'started'
  } catch (e) {
    if ((e as { status?: number }).status === 409) return 'blocked'
    throw e
  }
}

/** Deep value comparison over the keys the applied snapshot actually recorded.
    Draft-only keys are ignored: the run never had an opinion about them. */
export function configDiffers(applied: RunSettings | null, draft: RunSettings): boolean {
  if (applied === null) return false
  return Object.keys(applied).some(
    (k) => k in draft && JSON.stringify(draft[k]) !== JSON.stringify(applied[k]),
  )
}
