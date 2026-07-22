import type { TextBlock } from '../api/types'

/** Blocks that carry text, in the document's own reading order.

    Surya already sorts by `reading_order`, but the API contract does not promise it
    and a missing field would otherwise scatter the list — so ordering is enforced
    here, where it can be tested. Blocks with no text are layout regions the reader
    has no use for (an empty figure box), so they are dropped. */
export function orderedBlocks(blocks: TextBlock[] | undefined): TextBlock[] {
  if (!blocks) return []
  return blocks
    .filter((b) => (b.text ?? '').trim().length > 0)
    .map((b, i) => ({ b, i }))
    .sort((x, y) => (x.b.reading_order ?? x.i) - (y.b.reading_order ?? y.i))
    .map(({ b }) => b)
}

/** Human label for a block's region type, falling back to a generic word when
    layout detection did not name it. */
export function blockLabel(b: TextBlock, fallback: string): string {
  const raw = (b.region_label ?? b.label ?? '').trim()
  return raw.length > 0 ? raw : fallback
}

/** The merged plain-text rendering of a page's blocks — the same blank-line join
    the backend uses to build `ocr_text`, so Raw view and the blocks agree. */
export function mergedText(blocks: TextBlock[]): string {
  return blocks.map((b) => (b.text ?? '').trim()).filter(Boolean).join('\n\n')
}
