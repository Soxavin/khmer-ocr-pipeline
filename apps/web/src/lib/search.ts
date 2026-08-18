import type { PageTable } from '../api/types'

/** One cell that matches the current search, addressed the way the rest of the
    workspace addresses cells (table id + row + column). */
export type Match = { table_id: string; row: number; col: number }

/** Fold a string to its comparison form: NFC-normalized and lower-cased.

    NFC earns its place on the Latin side of these documents (accented characters
    and full-width forms), and lower-casing handles Latin headers and codes.

    KNOWN LIMIT — it does NOT help Khmer. Khmer combining marks all carry combining
    class 0, so canonical normalization never reorders them: `ខ្ញុំ` and the same
    syllable typed with the vowel and sign swapped stay distinct strings through
    NFC. The pipeline normalizes stored cell text with `utils/khmer_normalize.py`
    (cluster-aware mark reordering), but that logic has no TypeScript twin, so a
    query typed in a non-canonical mark order can still miss. Closing this means
    porting that normalizer and sharing its test vectors — deliberately not done
    inline here. */
function fold(s: string): string {
  return s.normalize('NFC').toLowerCase()
}

/** Does one cell value match the query? THE single matching rule — the counter and
    the grid's highlight both call it, so what is counted and what is ringed can
    never drift apart. An empty or whitespace-only query matches nothing. */
export function cellMatches(value: string, query: string): boolean {
  const q = fold(query.trim())
  return q.length > 0 && Boolean(value) && fold(value).includes(q)
}

/** Every cell on the page containing `query`, in a stable, human reading order:
    table by table, then row, then column — so stepping through matches walks the
    page the way the analyst reads it, and the order never shifts between renders.
    An empty or whitespace-only query matches nothing (not everything). */
export function findMatches(tables: PageTable[], query: string): Match[] {
  if (!query.trim()) return []
  const out: Match[] = []
  for (const t of tables) {
    t.grid.forEach((row, r) => {
      row.forEach((cell, c) => {
        if (cellMatches(cell, query)) out.push({ table_id: t.table_id, row: r, col: c })
      })
    })
  }
  return out
}

/** Position of a given cell within `matches`, or -1. Lets a grid cell ask "am I
    the active match?" without every cell re-scanning the list itself. */
export function matchIndexAt(matches: Match[], tableId: string, row: number, col: number): number {
  return matches.findIndex((m) => m.table_id === tableId && m.row === row && m.col === col)
}
