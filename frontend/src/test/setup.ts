import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach } from 'vitest'

// Node 22 exposes a stub global `localStorage` (no getItem) that shadows the one
// jsdom provides, so components reading persisted UI prefs crash. Install a real
// in-memory Storage and reset it between tests.
function memoryStorage(): Storage {
  let data: Record<string, string> = {}
  return {
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = String(v)
    },
    removeItem: (k) => {
      delete data[k]
    },
    clear: () => {
      data = {}
    },
    key: (i) => Object.keys(data)[i] ?? null,
    get length() {
      return Object.keys(data).length
    },
  } as Storage
}

beforeEach(() => {
  Object.defineProperty(globalThis, 'localStorage', { value: memoryStorage(), configurable: true, writable: true })
})
afterEach(cleanup)
