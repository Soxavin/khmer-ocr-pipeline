import { describe, expect, it } from 'vitest'
import { scanWordingKey } from './scan'
import type { SuggestCheck, Suggestion } from '../api/types'

const scores = (over: Partial<Suggestion['scores']>): Suggestion['scores'] => ({
  laplacian_var: 800,
  contrast_std: 80,
  skew_deg: 0.1,
  stamp_ink_ratio: 0,
  ...over,
})
const check = (field: string, active: boolean, reason: string): SuggestCheck => ({
  field,
  active,
  reason,
  detail: '',
})

describe('scanWordingKey (tiered by numeric metadata)', () => {
  it('minor tilt (<2.5°) says deskewing might help', () => {
    expect(scanWordingKey(check('deskew', true, 'tilted'), scores({ skew_deg: 1.2 }))).toBe('check_tilted_minor')
  })
  it('severe tilt (≥2.5°) says deskewing will help', () => {
    expect(scanWordingKey(check('deskew', true, 'tilted'), scores({ skew_deg: 4.0 }))).toBe('check_tilted_major')
  })
  it('straight pages keep the neutral wording', () => {
    expect(scanWordingKey(check('deskew', false, 'straight'), scores({}))).toBe('check_straight')
  })

  it('minor stamp ink (<5%) says removal might help', () => {
    expect(scanWordingKey(check('remove_stamps', true, 'stamps_found'), scores({ stamp_ink_ratio: 0.01 }))).toBe('check_stamps_minor')
  })
  it('heavy stamp ink (≥5%) says removal is recommended', () => {
    expect(scanWordingKey(check('remove_stamps', true, 'stamps_found'), scores({ stamp_ink_ratio: 0.08 }))).toBe('check_stamps_major')
  })

  it('optimal sharpness reports sharpening turned off', () => {
    expect(scanWordingKey(check('sharpen', false, 'already_sharp'), scores({}))).toBe('check_already_sharp')
  })
  it('soft scan keeps the sharpening suggestion', () => {
    expect(scanWordingKey(check('sharpen', true, 'soft_scan'), scores({ laplacian_var: 100 }))).toBe('check_soft_scan')
  })

  it('contrast close to threshold says enhancement might help', () => {
    expect(scanWordingKey(check('normalise', true, 'faded'), scores({ contrast_std: 50 }))).toBe('check_contrast_minor')
  })
  it('severe low contrast says enhancement will help', () => {
    expect(scanWordingKey(check('normalise', true, 'faded'), scores({ contrast_std: 20 }))).toBe('check_contrast_major')
  })

  it('table shading always reports the automatic flatten', () => {
    expect(scanWordingKey(check('normalise_table_backgrounds', true, 'table_shading_default'), scores({}))).toBe('check_table_shading_default')
  })
})
