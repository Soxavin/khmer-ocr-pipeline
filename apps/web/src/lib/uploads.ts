/**
 * Accepted upload file types, in one place.
 *
 * This string was duplicated in App.tsx and QueueRail.tsx, so adding a format
 * meant remembering both. Keep in step with `ACCEPTED_SUFFIXES` in
 * apps/api/uploads.py, which drives the NiceGUI fallback's upload widget and the
 * PDF-vs-image branch in `probe_pages` — the two lists are the same set on
 * either side of the API boundary.
 */
export const ACCEPTED_SUFFIXES = [
  '.pdf',
  '.png',
  '.jpg',
  '.jpeg',
  '.tif',
  '.tiff',
] as const;

/** Value for an `<input type="file">` `accept` attribute. */
export const ACCEPT_ATTR = ACCEPTED_SUFFIXES.join(',');
