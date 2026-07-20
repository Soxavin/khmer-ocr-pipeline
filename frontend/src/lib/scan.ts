import type { Key } from '../i18n.tsx'
import type { SuggestCheck, Suggestion } from '../api/types'

// Severity cut points for the scan-check phrasing tiers. The base thresholds
// (is a cleanup useful at all) live in preprocess.py; these only split the
// "useful" band into a gentle suggestion vs a firm one.
const SEVERE_TILT_DEG = 2.5 // ≥ this: "will help" instead of "might help"
const HEAVY_STAMP_RATIO = 0.05 // ≥ 5% ink coverage: removal is recommended
const SEVERE_CONTRAST_STD = 40 // below: "will help"; 40–60 is merely close to the 60 threshold

/** Scan-check phrase key for one finding, tiered by the measured scores so the
    wording matches how bad the scan actually is. */
export function scanWordingKey(check: SuggestCheck, scores: Suggestion['scores']): Key {
  switch (check.field) {
    case 'deskew':
      if (!check.active) return 'check_straight'
      return scores.skew_deg >= SEVERE_TILT_DEG ? 'check_tilted_major' : 'check_tilted_minor'
    case 'remove_stamps':
      if (!check.active) return 'check_no_stamps'
      return scores.stamp_ink_ratio >= HEAVY_STAMP_RATIO ? 'check_stamps_major' : 'check_stamps_minor'
    case 'sharpen':
      return check.active ? 'check_soft_scan' : 'check_already_sharp'
    case 'normalise':
      if (!check.active) return 'check_good_contrast'
      return scores.contrast_std < SEVERE_CONTRAST_STD ? 'check_contrast_major' : 'check_contrast_minor'
    default:
      return 'check_table_shading_default'
  }
}
