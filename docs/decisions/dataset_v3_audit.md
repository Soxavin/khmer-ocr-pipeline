# Audit — `Soxavin/ardb-sft-v3` (ARDB unified page-understanding SFT)

**Date:** 2026-08-04 · **Scope:** diagnosis only — nothing was fixed, regenerated, or re-uploaded.
**Subject:** local `eval/datasets/ardb_unified_sft_v3/` + `ardb_unified_sft_v3_hf/` (the upload source;
parquet verified byte-identical in its `text` column to the raw `pairs.jsonl`).

**Why this matters:** the `test` split's numbers are intended to be **cited as a result** in the
report/presentation, and Gemma Run 4 is pending on this dataset. Defects that would be cosmetic in a
scratch dataset are report-blocking here.

---

## Verdict

| # | Finding | Severity | Live in shipped v3? |
|---|---|---|---|
| 1 | Data rows emitted as `<th>` on every continuation page | **Report-blocking** | Yes — 81/81 pages, incl. **9 of 14 test rows** |
| 2 | Adjacent-day train/test leakage | **Report-blocking** | Yes — 6 of 7 dated held-out docs |
| 3 | Test split's era mix is inverted vs train | **Report-blocking** | Yes — test 57% Era A, train 19% |
| 4 | Literal `\n` (2 chars) in every target | **Training-quality** | Yes — 128/128 rows |
| 5 | Same commodity spelled 2 ways across eras | ~~Training-quality~~ → **not a defect; metric fixed** | GT faithful both sides (see §5) |
| 6 | Split fix has an undated-document blind spot | **Review note** (fix not yet applied) | n/a |
| 7 | 3 documents silently produced no rows | **Minor** | n/a |

**Clean:** schema (0 malformed targets, 0 wrong key sets, 0 missing images), eval-GT reservation
(0 frozen/anchor docs leaked into training), duplicates (0 identical targets, 0 identical images),
sequence length (max target 4,035 chars vs `max_length = 6144` — no truncation), letterhead line
order (0 mis-ordered), COCO↔SFT split consistency (0 mismatches).

**Inventory:** 128 rows / 47 docs — train 100/37, validation 14/5, test 14/5.
Labels: Table 128, Picture 128, Page-Furniture 256, Section-Header 47, Text 38.

---

## 1. Every continuation page emits a data row as `<th>` — **report-blocking**

`grid_to_html` hardcodes row 0 as the header row:

```python
# src/khmer_pipeline/datagen/harvest_table_gt.py:149
rows = [fmt(grid[0], "th", allow_span=False)]
rows += [fmt(row, "td", allow_span=True) for row in grid[1:]]
```

Page 0 genuinely starts with a header. **Pages 1 and 2 do not** — their first row is a commodity with
real prices, so it ships as `<th>`.

| measure | value |
|---|---|
| continuation pages (`page > 0`) | 81 |
| …emitting a `<th>` row | **81 (100%)** |
| …where that row holds real numeric price data | **81 (100%)** |
| affected rows by split | train 63, validation 9, **test 9** |
| share of all 3,177 `<tr>` rows | 2.55% |

Verbatim example (`doc_012` p1, validation):
`['២២', 'ពងមាន់ស្រែ', '៛/គ្រាប់', '800', '800', '0.00%']` — wrapped in `<th>`, not `<td>`.

**Why it is report-blocking rather than cosmetic:** 9 of the 14 test rows (64%) have a corrupted
target. Any per-cell or exact-match metric computed on this test split is measured against a label
that is wrong in a structurally-meaningful way, and the model is being taught "the first row of any
table is a header" — which is exactly the structural judgement the layout task is supposed to learn.

*Checked and NOT a problem:* section rows landing at position 0 and losing their `colspan` — 0 cases.

## 2. Adjacent-day leakage — **report-blocking** (fix exists, unapplied)

`assign_splits` is date-blind. ARDB publishes daily, and consecutive issues differ only in prices.

**In the shipped dataset: 6 of 7 dated held-out documents sit exactly 1 day from a training document.**

| split | doc date | distance to nearest train doc |
|---|---|---|
| validation | 2026-06-16 | **1** |
| validation | 2026-06-22 | **1** |
| validation | 2026-07-07 | **1** |
| validation | 2026-07-09 | **1** |
| test | 2026-06-12 | **1** |
| test | 2026-06-29 | **1** |
| validation | 2022-10-14 | 18 |

Target-text similarity confirms the practical impact — each held-out page against its most similar
training page:

| split | mean | min | max |
|---|---|---|---|
| validation | 0.893 | 0.766 | 0.966 |
| test | **0.886** | 0.807 | 0.940 |

A test target is on average **~89% identical** to something already in training. Part of that floor is
structural (all ARDB pages share ~73 identical commodity names), so the numeric-only similarity is the
sharper figure — it still reaches 0.83 on adjacent-day pairs (`doc_038` p2), meaning most of the
*prices* were seen too.

**Status:** the parallel session's `assign_splits_by_date_cluster` fix is written and passing tests but
**uncommitted, and neither COCO v3 nor SFT v3 has been rebuilt** — confirmed empirically: the packaged
`ardb_layout_coco_v3_hf` still shows 6 adjacent-day leaks and a 45/5/5 doc split (the pre-fix ratio).

**Ordering dependency (important):** `build_ardb_unified_sft.load_page_regions` reads `row["split"]`
straight out of the packaged COCO folder rather than recomputing it. Verified: 0 split mismatches
between COCO v3 and SFT v3 — the SFT dataset inherits COCO's splits verbatim. **So rebuilding the SFT
dataset alone cannot fix the leak. COCO v3 must be rebuilt first, then the SFT rebuilt from it.**

## 3. The test split's era mix is inverted relative to train — **report-blocking**

| split | Era A (`retail_only`) | Era B (`wholesale_retail`) |
|---|---|---|
| train | 19 rows (19%) / 8 docs | 81 rows / 29 docs |
| validation | 2 rows (14%) / 1 doc | 12 rows / 4 docs |
| **test** | **8 rows (57%) / 3 docs** | 6 rows / 2 docs |

Test is **57% Era A** while training is only **19% Era A**. Era A is the older 6-column
`retail_only` layout — structurally different, and the era the pipeline has historically handled worst.
A single cited test number therefore measures a document mix the model was barely trained on, and is
not comparable to validation (14% Era A). This is a consequence of random 5-document splitting on a
55-document corpus, and it is not fixed by the date-clustering change.

## 4. A literal `\n` sits in every training target — **training-quality**

```python
# src/khmer_pipeline/datagen/build_ardb_unified_sft.py:63
_FOOTER_TEXT = (
    "តម្មៃយកពីប្រភព៖\n"
    "១ \\nផ្សារមួយចំនួនក្នុងរាជធានីភ្នំពេញ\n"   # <-- \\n = backslash + n, not a newline
    ...
)
```

The constant contains **2 real newlines and 1 literal two-character `\n`**. Because `_FOOTER_TEXT` is
embedded in the JSON target of every page, **all 128 rows (100%)** carry it. The model is being trained
to emit a literal backslash-n as visible text.

All 38 live `Text` regions equal the constant exactly (0 deviations) — as expected, since it *is* the
constant. That is the point: this is a single-character fix at the source that propagates to 100% of
the dataset.

## 5. The same commodity is spelled two ways — **RESOLVED 2026-08-05: not a defect**

> **Update.** Filed below as a label defect on the assumption that one of the two verified GTs had a
> stray space. The user checked the rendered page: **both are faithful** — the source PDF itself
> renders a wide kerning gap inside the parentheses on some labels and not others. So the GT is
> correct on both sides and there was nothing to fix in the data.
>
> The real problem was the **metric**: `_cell` compared with `==` after `_norm`, which collapses
> whitespace *runs* and strips *ends* but preserves an interior space — so a typographic artifact
> flipped a whole cell to "wrong". Fixed by folding **bracket-hugging spaces only** in the
> exact-match metrics (`cell_accuracy`, `khmer_cell_accuracy`, `cell_content_recall`) via
> `_fold_spaces`, mirroring the space-dropping `_fold_numeric` already applied to split numbers.
>
> **Scope was chosen by measurement, not intuition:** folding *every* space would have touched
> **161 of 3,467 GT cells (4.64%)**, silently crediting word-segmentation differences never verified
> to be artifacts; the bracket-only rule touches **4 cells (0.12%)** — exactly the confirmed
> phenomenon. `table_cer` is deliberately left alone: a proportional metric already charges a lone
> space ~1/len(text).
>
> Net effect: absolute `cell_accuracy` was **understated** on p1 of both frozen eval docs. Engine
> *rankings* were never affected — every engine took the same penalty.

*Original finding, retained for the record:*

The two era templates are separately user-verified GT files, and they disagree on two items — a stray
space before the closing parenthesis:

| Era B template (`wholesale_retail`) | Era A template (`retail_only`) | edit distance |
|---|---|---|
| `មាន់ស្រែ (សាច់ )` | `មាន់ស្រែ (សាច់)` | 1 |
| `សាច់គោ (សាច់សម្ល )` | `សាច់គោ (សាច់សម្ល)` | 1 |

Reach: **35 rows** carry the spaced variant, **12 rows** the unspaced one — for the same commodity.
One of the two verified GT files has a stray space, and it propagates to every document of that era.

**This needs your call, not mine** — both strings are quoted verbatim from files you verified; I am not
judging which is correct.

## 6. Review note for the parallel session — the split fix's undated blind spot

The fix is **correct on what it can see**, verified independently:

- adjacent-day leaks: **6 → 0** on dated documents
- clusters straddling splits: **0** (transitivity holds)
- window choice validated: at 3 days the corpus chains into a single **24-document** supercluster,
  confirming the 1-day choice. *Minor correction:* the doc-level ratio is **49/3/3 at every window
  tested (1/2/3/5 days)** — window size does not drive the ratio; the old-vs-new change (45/5/5 → 49/3/3)
  does.

**The gap: `_pdf_date` returns `None` for 15 of 55 documents (27%).** Those become singleton clusters —
"never merged, since we have no evidence they're near-duplicates" — but the evidence *is* in the
filename, just unparsed. Two failure modes:

1. Khmer-digit / word-form filenames (Era A/A2), e.g. `..._ថ្ងៃទី_០៤_០៥_មករា_២០២៣.pdf`
2. A 4-digit year in the otherwise-standard form: `-06.08.2025.pdf` — `_FILENAME_DATE_RE` wants
   `(\d{2})\.(\d{2})\.(\d{2})`. Also affects `01.12.2025` / `10.11.2025` (those two recover via the
   header fallback; `06.08.2025` does not).

The header-date fallback fails for all 15 — consistent with the known `find_tables()` fragmentation on
Era A layouts, i.e. **the fallback fails on exactly the era where the filename parser also fails.**

Concrete consequence — a genuine adjacent-day pair that clustering cannot see:

```
…ថ្ងៃទី04_មីនា_2025.pdf   (2025-03-04)   undated → singleton
…ថ្ងៃទី05_មីនា_2025.pdf   (2025-03-05)   undated → singleton
```

- **19 of 20 seeds** place at least one undated document in a held-out split.
- That specific pair **straddles train/held-out in 12 of 50 seeds (24%)**.
- `seed=0` happens to be safe — **by luck, not by construction.**

Suggested (yours to implement): extend `_pdf_date` to parse Khmer-digit/word filenames and 4-digit
years, and consider treating undated docs as a single quarantine cluster rather than singletons, so an
unparseable date fails safe.

## 7. Silent data loss — minor

55 corpus PDFs → 5 intentionally excluded (frozen eval + template anchors, reservation verified intact:
**0 violations**) → 47 shipped → **3 produced no rows at all**, with no `unaligned/` directory written:

```
…ថ្ងៃទី២១_២២មីនា២០២២_With-Logo.pdf
…ថ្ងៃទី២៧_២៨-មីនា-២០២២_With-Logo.pdf
…ថ្ងៃទី05_ធ្នូ_2024.pdf
```

13 `doc_id` slots are absent overall (`doc_003`, `doc_011`, `doc_014`, `doc_015`, `doc_019`, `doc_021`,
`doc_032`, `doc_039`, `doc_055`–`doc_059`). `doc_014` (March 2022) is confirmed **absent from all three
splits**, so the parallel session's `_FOOTER_TEXT` mismatch finding does not affect live data.

---

## Assumptions I could NOT verify

**Commodity-name drift remains unverified — and is not verifiable from the PDF text layer.**

This was the largest open question, and the honest answer is that the intended method cannot work:

- item-name cells in targets: **3,061** — of which **0** pass `is_trusted_cell_text`
- unit cells: 3,061 — of which 3,014 are comparable (they start with `៛`)
- unit comparisons run: 3,014 → **176 mismatches**

But those 176 are **text-layer glyph-scrambling, not drift**. Aggregated, they collapse to 5 distinct
pairs, e.g. 127× template `'៛/គ្រាប់'` vs text-layer `'៛/រោែ ់'`. The template value is the coherent one.

Two consequences:

1. **The template-substitution assumption cannot be validated against the text layer at all.** Only a
   visual check against rendered pixels can confirm the item list didn't drift across 2022–2026 —
   which needs you, not me. The parallel session's single spot-check (March 2022, matching through row
   27 with 2 wording variants) remains the only evidence, and it is one document out of 47.
2. **Side finding:** `is_trusted_cell_text` accepted scrambled values like `'៛/រោែ ់'` because it trusts
   any `៛`-prefixed string passing `khmer_order_valid`. That is over-permissive. It does **not** corrupt
   v3 (units here come from the template, not the text layer) — but `harvest_table_gt.py` uses the same
   predicate to decide which text-layer cells are safe as *recognition* GT, where it would.

Also unverified: whether `_FOOTER_TEXT`/`_LETTERHEAD_TEXT` are character-correct on Era A pages, beyond
the parallel session's 4/4 visual spot-check (2023/2024/2025). Note 7 live pages have vertically
**overlapping** Page-Furniture boxes, where the y-order zip in `build_region_texts` is fragile — it
happens to be correct on all 128 pages today (0 mis-ordered), but a count-only guard would not catch a
swap.

---

## Recommended order of remediation

Not applied — for your approval.

1. **Fix `grid_to_html`** so only page 0 gets a `<th>` header row (pass `has_header` through, mirroring
   `build_page`). One-line class of change; largest correctness win; add a regression test on a
   continuation page.
2. **Fix the literal `\n`** in `_FOOTER_TEXT` (delete one backslash). Trivial, affects 100% of rows.
3. **Decide the two spellings** (finding 5) and make both templates agree.
4. **Rebuild in the right order:** COCO v3 (with the split fix) → then SFT v3 from it. Rebuilding SFT
   alone will not remove the leak.
5. **Extend `_pdf_date`** before that rebuild, or 27% of the corpus stays unclustered.
6. **Settle the reporting question** (below) before Run 4, since it determines whether any of this
   blocks the report.

## The reporting question this audit raises

Even after every fix above, the leak-free split is **49/3/3 documents** — about **9 test pages**. One
quarantined or mislabelled page moves a cited metric by >10%, and finding 3 shows the small split also
produces a badly skewed era mix.

On a 55-document corpus, "leak-free" and "statistically meaningful" may not both be reachable. The
defensible framing is likely:

- **headline result** → the frozen `eval/datasets/real` GT via the pipeline benchmark (independent of
  this dataset, and already the project's evaluation of record);
- **this test split** → a training-health signal, reported with its size and era skew stated.

That is a framing decision, not a code change — but it is cheaper to settle now than after Run 4
produces a number that would have to be walked back.

---

## Reproducing

Scripts (read-only; scratchpad
`/private/tmp/claude-501/-Users-vin-Internship/678bed41-1760-48a5-bad5-cb499fef1db9/scratchpad/`):

| script | produces |
|---|---|
| `a1_inventory.py` | inventory, schema integrity, label counts |
| `a1b_parquet_hf.py` | parquet↔jsonl equality, content hashes, build mtimes |
| `a2_th_defect.py` | finding 1 |
| `a5_leakage.py` | finding 2 (shipped-dataset leak + similarity) |
| `a5b_verify_fix.py` | finding 6 (fix verification, transitivity, window sweep) |
| `a5c_undated_gap.py` | finding 6 (undated blind spot, seed sensitivity) |
| `a46_boiler_ready.py` | findings 3, 4, 7 + readiness sweep |
| `a3_drift.py` | drift attempt → "could not verify" |
| `a3b_template_compare.py` | finding 5 |
| `a7_final.py` | finding 5 reach, COCO↔SFT consistency, pre/post-fix detection |

Run with `uv run python <script>` from the repo root (they need `fitz`/`datasets` from the project env).

**Isolation:** `git status --porcelain` was byte-identical before and after this audit; no
parallel-session file was modified. Note `pseudo_label_layout.py` changed *underneath* this audit
(another session editing it live) — the findings in §6 were re-verified against the current file and
still hold: 55 docs, 15 undated, 49/3/3, 19/20 seeds risky.
