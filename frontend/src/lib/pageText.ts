import type { TextBlock } from '../api/types'

/** The separator the backend joins text blocks with when it builds `corrected_text`
    (`"\n\n".join(...)` in postprocess.py). Splitting on it is what lets the Blocks
    view and the Raw textarea be two presentations of ONE string rather than two
    different strings — which is what they used to be. */
const SEP = '\n\n'

/** The page's corrected text as its per-block segments. Whitespace-only text has no
    segments (not one empty segment) so an empty page renders as empty, not as a
    single blank card. */
export function splitSegments(text: string): string[] {
  if (!text.trim()) return []
  return text.split(SEP)
}

/** Character offsets of segment `i` within `text`, for `setSelectionRange`.

    Computed by walking the preceding segments and their separators rather than by
    searching for the segment's content: identical segments are common on these
    documents (repeated column headers, "0,00 ដុល្លារ"), and a search would select
    the first match instead of the one the analyst clicked. Out-of-range indices
    collapse to a zero-width range so a caller can never pass NaN to the DOM. */
export function segmentRange(text: string, i: number): { start: number; end: number } {
  const segments = splitSegments(text)
  if (i < 0 || i >= segments.length) return { start: 0, end: 0 }
  let start = 0
  for (let k = 0; k < i; k++) start += segments[k].length + SEP.length
  return { start, end: start + segments[i].length }
}

/** `text` with segment `i` replaced. Out-of-range returns the text unchanged. */
export function replaceSegment(text: string, i: number, next: string): string {
  const segments = splitSegments(text)
  if (i < 0 || i >= segments.length) return text
  segments[i] = next
  return segments.join(SEP)
}

/** A rendered card. It carries TWO indices, which are not interchangeable:

    - `segIndex` — position in the corrected text. Editing, copying and
      Edit-in-Raw's character range all address this.
    - `srcIndex` — position in `page.text_blocks`. The page canvas draws one rect
      per source block and looks its bbox up by this, so the highlight link
      addresses this.

    They differ whenever reading order is not array order, or an empty block was
    filtered out — both routine. Using one where the other belongs highlights the
    wrong region on the scan while looking perfectly correct in the list. */
export type SegmentRow = {
  segIndex: number
  srcIndex: number | null
  segment: string
  block: TextBlock | null
}

/** Pair corrected-text segments with the layout blocks that produced them.

    The backend builds `corrected_text` block-wise and in order, so when the counts
    match the pairing is exact and each segment can carry its block's confidence,
    region label, and canvas link.

    When they DON'T match, every block is withheld. The counts diverge for real
    reasons — a block whose corrected text came out empty is dropped by the backend's
    `if t` filter, a block containing its own blank line splits into two segments, and
    any analyst edit that adds or removes a blank line shifts everything after it — and
    none of those tell us *where* the drift began. Pairing the leading segments anyway
    would attach a confidence score to text it does not describe, which is a worse
    failure than showing no score: the whole point of the badge is trust. The text
    itself is always authoritative, so editing keeps working either way. */
export function pairSegments(
  segments: string[],
  entries: { block: TextBlock; index: number }[],
): { rows: SegmentRow[]; aligned: boolean } {
  const aligned = segments.length === entries.length
  return {
    aligned,
    rows: segments.map((segment, segIndex) => ({
      segIndex,
      segment,
      srcIndex: aligned ? entries[segIndex].index : null,
      block: aligned ? entries[segIndex].block : null,
    })),
  }
}
