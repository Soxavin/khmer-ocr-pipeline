import type { TextBlock } from '../api/types'

/** Blocks that carry text, in the document's own reading order.

    Surya already sorts by `reading_order`, but the API contract does not promise it
    and a missing field would otherwise scatter the list — so ordering is enforced
    here, where it can be tested. Blocks with no text are layout regions the reader
    has no use for (an empty figure box), so they are dropped. */
export function orderedBlocks(blocks: TextBlock[] | undefined): TextBlock[] {
  return orderedBlockEntries(blocks).map((e) => e.block)
}

/** The same reading-order list, each block paired with its index in the ORIGINAL
    `text_blocks` array.

    That index is the block's identity across the workspace: the page overlay draws
    one rect per source block, so linking a text card to its box needs the source
    position, not the card's position — filtering and reading-order sorting make
    the two disagree. */
export function orderedBlockEntries(
  blocks: TextBlock[] | undefined,
): { block: TextBlock; index: number }[] {
  if (!blocks) return []
  return blocks
    .map((block, index) => ({ block, index }))
    .filter((e) => (e.block.text ?? '').trim().length > 0)
    .sort((x, y) => (x.block.reading_order ?? x.index) - (y.block.reading_order ?? y.index))
}

/** Human label for a block's region type, falling back to a generic word when
    layout detection did not name it. */
export function blockLabel(b: TextBlock, fallback: string): string {
  const raw = (b.region_label ?? b.label ?? '').trim()
  return raw.length > 0 ? raw : fallback
}

