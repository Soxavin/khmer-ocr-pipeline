import { describe, expect, it, vi } from 'vitest'
import { configDiffers, guardedRun, isBusy } from './run'
import type { DocSummary } from '../api/types'

const doc = (id: string, status: DocSummary['status']): DocSummary => ({
  id,
  name: id,
  pages: 1,
  size_kb: 1,
  status,
  total_tables: 0,
  reviewed_tables: 0,
})

describe('isBusy (workspace-wide pipeline gate)', () => {
  it('is busy while any document is running, not just the active one', () => {
    expect(isBusy([doc('a', 'done'), doc('b', 'running')], false)).toBe(true)
  })

  it('is busy during a batch run even before a document flips to running', () => {
    expect(isBusy([doc('a', 'queued')], true)).toBe(true)
  })

  it('is idle when nothing is running', () => {
    expect(isBusy([doc('a', 'done'), doc('b', 'queued')], false)).toBe(false)
  })
})

describe('guardedRun (concurrency collisions never reach the server)', () => {
  it('starts the run when the pipeline is idle', async () => {
    const start = vi.fn().mockResolvedValue(undefined)
    expect(await guardedRun(false, start)).toBe('started')
    expect(start).toHaveBeenCalledTimes(1)
  })

  it('blocks a second run while one is in flight, without calling the server', async () => {
    const start = vi.fn().mockResolvedValue(undefined)
    expect(await guardedRun(true, start)).toBe('blocked')
    expect(start).not.toHaveBeenCalled()
  })

  it('swallows a server 409 as a benign collision rather than surfacing an error', async () => {
    const start = vi.fn().mockRejectedValue(Object.assign(new Error('Another extraction is already running.'), { status: 409 }))
    expect(await guardedRun(false, start)).toBe('blocked')
  })

  it('still propagates real failures', async () => {
    const start = vi.fn().mockRejectedValue(Object.assign(new Error('boom'), { status: 500 }))
    await expect(guardedRun(false, start)).rejects.toThrow('boom')
  })
})

describe('configDiffers (applied vs draft)', () => {
  it('is false when the draft matches the applied snapshot', () => {
    expect(configDiffers({ dpi: 200, deskew: true }, { dpi: 200, deskew: true })).toBe(false)
  })

  it('is true when a tracked value is mutated', () => {
    expect(configDiffers({ dpi: 200 }, { dpi: 300 })).toBe(true)
  })

  it('ignores draft keys the applied snapshot never recorded', () => {
    expect(configDiffers({ dpi: 200 }, { dpi: 200, unrelated: 'x' })).toBe(false)
  })

  it('compares deeply, not by reference', () => {
    expect(configDiffers({ page_list: [1, 2] }, { page_list: [1, 2] })).toBe(false)
    expect(configDiffers({ page_list: [1, 2] }, { page_list: [1, 3] })).toBe(true)
  })

  it('is false with no applied snapshot — nothing has been run to drift from', () => {
    expect(configDiffers(null, { dpi: 300 })).toBe(false)
  })
})
