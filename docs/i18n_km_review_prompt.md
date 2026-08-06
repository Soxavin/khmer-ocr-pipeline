# Khmer (km) UI localization review — Khmer OCR Review Workspace

You are a **professional Khmer (Cambodian) localization reviewer** for a government
software tool. Verify and correct the Khmer translations of the UI strings below, and
return only what needs changing, with reasons.

## About the product (so word choices fit the real context)
- **What it is:** a desktop web tool that extracts tables from **scanned/PDF Khmer
  financial documents** (e.g. ARDB daily price bulletins) using OCR, and lets an analyst
  **review, correct, and export** the recognized tables to Excel/CSV/JSON.
- **Who uses it:** data analysts at **GDDE (General Department of Digital Economy),
  Ministry of Economy and Finance, Cambodia** — Excel-native domain experts, not
  engineers. Secondary audience: mentors/stakeholders in a projected demo.
- **Register:** formal, precise, plain — government-professional. Not casual or playful.
- **Tone model:** crisp density like Linear/Raycast, so chrome labels must stay **short**;
  a translation much longer than the English can break a dense toolbar.

## Hard rules
1. **Preserve every placeholder token exactly** — anything in curly braces such as {n},
   {name}, {band}, {i}, {a}, {b}, {x} must appear **unchanged**, same token and spelling.
   They are replaced at runtime with numbers/words, so the surrounding Khmer must read
   naturally once substituted.
2. **Do NOT translate** tokens/abbreviations analysts read as-is: DPI, PDF, Excel, CSV,
   JSON, zip, Cmd/Ctrl/Esc and the Command glyph, percentages like 80%, keyboard keys.
   Keep Arabic numerals 0-9 — this is a numbers tool; do not convert digits to Khmer
   numerals in UI chrome.
3. **Length discipline:** labels/buttons stay roughly as short as the English. Tooltips
   and messages may be a full sentence but concise.
4. **Consistency:** translate recurring domain terms the same way everywhere; flag any
   current inconsistency.
5. **Legibility:** natural, standard spelling; avoid unusual stacked forms where a simpler
   synonym reads clearer. Khmer sentence punctuation for full sentences; none for short
   button/label fragments.
6. **Accuracy over literalness:** translate the intent for an analyst mid-task, using the
   natural Khmer UI idiom.

## Domain glossary to keep consistent (confirm or improve)
Give your recommended standard Khmer term for each, then apply it uniformly:
extraction/extract, table, cell, row, column, page, document, confidence, verify/verified,
low-confidence, preprocess/cleanup, deskew, stamp (ink), sharpen, contrast, engine
(recognition engine), run, stop, export, review, find & replace, zoom/fit, overlay.

## Special notes
- **lang_toggle_tip is intentionally inverted** — the English UI shows the Khmer
  destination; the Khmer UI shows "Switch to English". Keep that; fix only if wrong.
- **Confidence bands** are a 3-level triage scale: **Check** (needs attention, under 80%),
  **Skim** (glance, 80-95%), **Clean** (fine, over 95%). band_check/skim/clean and the
  *_aria_* long forms must read as a natural severity scale in Khmer.
- Treat everything as unverified; the **PRIORITY set** is the most recently added and
  least-reviewed — look hardest there.

## What to return
A Markdown table of **only the rows you would change**:
| key | verdict | corrected Khmer | reason |
verdict = FIX (wrong/awkward/inconsistent/placeholder issue); omit already-correct rows.
Then **Glossary decisions** (your standard Khmer term per domain word), then **Red flags**
(any ambiguous English source or misused placeholder). Corrected Khmer copy-paste ready.

---

## NEWEST SET — added since this doc was last generated, NEVER reviewed (63 keys)

These are the highest priority: the review-workspace strings from the latest UI passes
(raw/edited view toggle, confidence show/hide, dismiss-all, page-text blocks, find, copy,
auto-resolved DPI/engine readouts) plus the engine-picker grouping (Local / Cloud /
Experimental) and Labs-mode strings from the settings-drawer redesign. Look hardest here.

| key | English | Current Khmer |
|---|---|---|
| `engine_label_qwen_ardb` | Qwen ARDB (unreliable, trial) | Qwen ARDB (មិនទាន់ជឿទុកចិត្ត សាកល្បង) |
| `engine_guidance_qwen_ardb` | Frequently fails to produce valid output. Shown for comparison, not real extraction. | ភាគច្រើនបរាជ័យក្នុងការផលិតលទ្ធផលត្រឹមត្រូវ។ បង្ហាញសម្រាប់ប្រៀបធៀប មិនមែនសម្រាប់ការស្រង់ទិន្នន័យពិតប្រាកដទេ។ |
| `raw_output_fallback_note` | Model output — could not be read as text or tables. Shown as produced. | លទ្ធផលម៉ូដែល — មិនអាចអានជាអត្ថបទ ឬតារាងបានទេ។ បង្ហាញតាមដែលបានផលិត។ |
| `step_finetune_slow` | running the fine-tune (may take several minutes) | កំពុងដំណើរការម៉ូដែលដែលបានបង្វឹក (អាចចំណាយពេលច្រើននាទី) |
| `engine_label_gemma_ardb` | ARDB specialist (trial) | ជំនាញ ARDB (សាកល្បង) |
| `engine_guidance_gemma_ardb` | Tuned on ARDB bulletins. Under evaluation — may return incomplete rows. | បានបង្វឹកលើព្រឹត្តិប័ត្រ ARDB។ កំពុងវាយតម្លៃ — អាចត្រឡប់ជួរដេកមិនពេញលេញ។ |
| `engine_trial` | Trial | សាកល្បង |
| `engine_group_local` | Local (on this device) | ក្នុងម៉ាស៊ីន (លើឧបករណ៍នេះ) |
| `engine_group_cloud` | Cloud | ពពក (Cloud) |
| `engine_group_local_short` | Local | ក្នុងម៉ាស៊ីន |
| `engine_group_cloud_short` | Cloud | ពពក |
| `engine_group_experimental` | Experimental | ពិសោធន៍ |
| `engine_recommended` | Recommended | ណែនាំ |
| `contains_selected_engine` | contains the selected engine | មានម៉ាស៊ីនដែលបានជ្រើស |
| `labs_mode` | Labs engines | ម៉ាស៊ីនពិសោធន៍ |
| `labs_mode_tip` | Show custom ARDB fine-tuned models (experimental) | បង្ហាញម៉ូដែល ARDB ដែលបានបង្វឹកពិសេស (ពិសោធន៍) |
| `engine_label_auto` | Automatic | ស្វ័យប្រវត្តិ |
| `engine_guidance_auto` | Picks the best engine for each document. | ជ្រើសម៉ាស៊ីនល្អបំផុតសម្រាប់ឯកសារនីមួយៗ។ |
| `engine_label_surya` | Standard | ស្តង់ដារ |
| `engine_guidance_surya` | Best all-round, fastest. Use for number-heavy or wide tables. | ល្អបំផុតជាទូទៅ លឿនបំផុត។ សម្រាប់តារាងច្រើនលេខ ឬធំទូលាយ។ |
| `engine_label_surya_kiri` | Khmer-text specialist | ជំនាញអក្សរខ្មែរ |
| `engine_guidance_surya_kiri` | Strongest on Khmer-text-heavy narrow tables (ARDB bulletins). Slower. | ខ្លាំងបំផុតលើតារាងតូចចង្អៀតដែលមានអក្សរខ្មែរច្រើន (ព្រឹត្តិប័ត្រ ARDB)។ យឺតជាង។ |
| `engine_label_surya_kiri_vlm` | Best structure (slow) | រចនាសម្ព័ន្ធល្អបំផុត (យឺត) |
| `engine_guidance_surya_kiri_vlm` | Keeps spanning headers intact and upgrades Khmer cells when safe. Slowest. | រក្សាក្បាលតារាងលាតសន្ធឹងឲ្យនៅដដែល ហើយកែលម្អក្រឡាខ្មែរពេលមានសុវត្ថិភាព។ យឺតបំផុត។ |
| `engine_label_gemini` | Cloud (Gemini) | ពពក (Gemini) |
| `engine_guidance_gemini` | Sends the page image to Google. Do not use for confidential documents. | ផ្ញើរូបទំព័រទៅ Google។ កុំប្រើសម្រាប់ឯកសារសម្ងាត់។ |
| `raw_view` | Raw OCR | OCR ដើម |
| `raw_banner` | Raw OCR · read only | OCR ដើម · អានតែប៉ុណ្ណោះ |
| `raw_view_tip` | Show the original OCR reading, read-only. Turn off to edit. | បង្ហាញលទ្ធផល OCR ដើម ក្នុងទម្រង់អានតែប៉ុណ្ណោះ។ បិទដើម្បីកែ។ |
| `conf_toggle` | Confidence | ទំនុកចិត្ត |
| `conf_toggle_tip` | Show or hide the marks on cells the recogniser was unsure of | បង្ហាញ ឬលាក់សញ្ញាលើក្រឡាដែលកម្មវិធីស្គាល់អក្សរមិនប្រាកដ |
| `ocr_original_tt` | OCR: {v} | OCR: {v} |
| `dismiss_all` | Dismiss all | លុបចេញទាំងអស់ |
| `dismiss_all_confirm` | Dismiss all {n} issues from this list? The cells stay flagged until fixed. | លុបចេញបញ្ហាទាំង {n} ចេញពីបញ្ជីនេះ? ក្រឡានៅតែត្រូវបានសម្គាល់រហូតដល់កែរួច។ |
| `auto_resolved_dpi` | Auto → {n} DPI | ស្វ័យប្រវត្តិ → {n} DPI |
| `auto_resolved_dpi_tip` | This document was rendered at {n} DPI. | ឯកសារនេះត្រូវបានបង្ហាញក្នុងកម្រិត {n} DPI។ |
| `auto_resolved_engine` | Auto → {v} | ស្វ័យប្រវត្តិ → {v} |
| `auto_resolved_engine_tip` | The router read this document and used the {v} engine. | ប្រព័ន្ធជ្រើសរើសបានអានឯកសារនេះ ហើយបានប្រើប្រាស់ម៉ាស៊ីន {v}។ |
| `block_copy` | Copy text | ចម្លងអត្ថបទ |
| `block_edit` | Edit this block | កែប្រែប្លុកនេះ |
| `block_edit_done` | Done editing | កែប្រែរួចរាល់ |
| `block_edit_raw` | Edit in Raw | កែប្រែក្នុងអត្ថបទឆៅ |
| `block_focus` | Show this text block on the page | បង្ហាញប្លុកអត្ថបទនេះលើទំព័រ |
| `block_open` | Show this block in Page text | បង្ហាញប្លុកនេះក្នុងអត្ថបទទំព័រ |
| `block_untitled` | Block | ប្លុក |
| `blocks_unmatched` | Block details could not be matched to this text — showing the text only. | មិនអាចផ្គូផ្គងព័ត៌មានប្លុកជាមួយអត្ថបទនេះបានទេ — បង្ហាញតែអត្ថបទប៉ុណ្ណោះ។ |
| `no_text_blocks` | No text outside the tables on this page. | គ្មានអត្ថបទនៅខាងក្រៅតារាងលើទំព័រនេះទេ។ |
| `text_blocks_count` | {n} blocks | {n} ប្លុក |
| `view_raw` | Raw | អត្ថបទឆៅ |
| `view_structured` | Blocks | ប្លុក |
| `view_text_mode` | Page text view | ទិដ្ឋភាពអត្ថបទលើទំព័រ |
| `filter_all` | All {n} | ទាំងអស់ {n} |
| `filter_empty` | No blocks in this band. | គ្មានប្លុកក្នុងកម្រិតនេះទេ។ |
| `find_matches_aria` | Search matches on this page | លទ្ធផលស្វែងរកក្នុងទំព័រនេះ |
| `find_next` | Next match | លទ្ធផលបន្ទាប់ |
| `find_prev` | Previous match | លទ្ធផលមុន |
| `find_none` | none | រកមិនឃើញ |
| `copied` | Copied | បានចម្លង |
| `copy_all` | Copy all | ចម្លងទាំងអស់ |
| `copy_failed` | Could not copy — select the text and copy manually. | មិនអាចចម្លងបានទេ — សូមជ្រើសរើសអត្ថបទ រួចចម្លងដោយដៃ។ |
| `remove_action` | Remove | លុបឯកសារ |
| `retry` | Retry | ព្យាយាមម្តងទៀត |
| `page_load_failed` | Could not load this page: {e} | មិនអាចផ្ទុកទំព័រនេះបានទេ: {e} |

## PRIORITY SET — most recently added, least reviewed (52 keys)

| key | English | Current Khmer |
|---|---|---|
| `zoom_group` | Zoom controls | ការគ្រប់គ្រងការពង្រីក |
| `zoom_in` | Zoom in | ពង្រីក |
| `zoom_out` | Zoom out | បង្រួម |
| `zoom_reset` | Reset to 100% | កំណត់ត្រឡប់ 100% |
| `focus_table` | Focus table | ផ្តោតលើតារាង |
| `focus_table_none` | No table selected | មិនបានជ្រើសតារាង |
| `band_check` | Check | ត្រួតពិនិត្យ |
| `band_skim` | Skim | មើលរំលង |
| `band_clean` | Clean | ស្អាត |
| `band_aria_check` | {n} need checking (under 80%) | {n} ត្រូវត្រួតពិនិត្យ (ក្រោម 80%) |
| `band_aria_skim` | {n} to skim (80 to 95%) | {n} ត្រូវមើលរំលង (80 ដល់ 95%) |
| `band_aria_clean` | {n} clean (over 95%) | {n} ស្អាត (លើ 95%) |
| `find_btn` | Find | ស្វែងរក |
| `find_tip` | Find & replace across this page (Ctrl/Cmd-F) | ស្វែងរក និងជំនួសក្នុងទំព័រនេះ (Ctrl/Cmd-F) |
| `triage_aria_check` | Check band: {n} cells under 80% — jump to next | ក្រុមត្រួតពិនិត្យ៖ {n} ក្រឡាក្រោម 80% — លោតទៅបន្ទាប់ |
| `triage_aria_skim` | Skim band: {n} cells 80 to 95% — jump to next | ក្រុមមើលរំលង៖ {n} ក្រឡា 80 ដល់ 95% — លោតទៅបន្ទាប់ |
| `triage_aria_clean` | Clean band: {n} cells over 95% — jump to next | ក្រុមស្អាត៖ {n} ក្រឡាលើ 95% — លោតទៅបន្ទាប់ |
| `canvas_aria` | Document page — arrow keys pan, plus and minus zoom, 0 fits | ទំព័រឯកសារ — គ្រាប់ចុចព្រួញអូស បូក/ដក ពង្រីក 0 សម​ |
| `check_tilted_minor` | Page is tilted — deskewing might help | ទំព័រទ្រេត — ការតម្រង់អាចជួយបាន |
| `check_tilted_major` | Page is severely tilted — deskewing will help | ទំព័រទ្រេតខ្លាំង — ការតម្រង់នឹងជួយ |
| `check_stamps_minor` | Stamp ink detected — stamp removal might help | រកឃើញទឹកថ្នាំត្រា — ការលុបត្រាអាចជួយបាន |
| `check_stamps_major` | Heavy stamp ink detected — stamp removal is recommended | ទឹកថ្នាំត្រាច្រើន — គួរតែលុបត្រាចេញ |
| `check_contrast_minor` | Low contrast detected — contrast enhancement might help | កម្រិតពណ៌ទាប — ការបង្កើនកម្រិតពណ៌អាចជួយបាន |
| `check_contrast_major` | Poor contrast detected — contrast enhancement will help | កម្រិតពណ៌ទាបខ្លាំង — ការបង្កើនកម្រិតពណ៌នឹងជួយ |
| `dpi_auto` | Auto | ស្វ័យប្រវត្តិ |
| `dpi_auto_tip` | Reads the document and picks 200, or 300 for faint or low-resolution scans. | អានឯកសារ ហើយជ្រើស 200 ឬ 300 សម្រាប់ស្កេនស្រាល ឬគុណភាពទាប។ |
| `auto_applied` | Auto: Applied | ស្វ័យប្រវត្តិ៖ បានអនុវត្ត |
| `auto_off` | Auto: Off | ស្វ័យប្រវត្តិ៖ បិទ |
| `delete_all` | Delete all | លុបទាំងអស់ |
| `cancel` | Cancel | បោះបង់ |
| `step_layout` | finding the layout | កំពុងរកប្លង់ទំព័រ |
| `step_text` | reading the text | កំពុងអានអក្សរ |
| `step_tables` | reading the tables | កំពុងអានតារាង |
| `busy_elsewhere` | Another document is being extracted right now. | ឯកសារមួយទៀតកំពុងត្រូវបានស្រង់ទិន្នន័យ។ |
| `delete_all_title` | Remove all documents? | លុបឯកសារទាំងអស់មែនទេ? |
| `delete_all_confirm` | All {n} documents leave the workspace. Results and edits are discarded. | ឯកសារទាំង {n} នឹងចាកចេញពីកន្លែងធ្វើការ។ លទ្ធផល និងការកែសម្រួលនឹងបាត់។ |
| `delete_all_action` | Remove all | លុបទាំងអស់ |
| `scan_toast_clean` | Scan check: pages look good — standard settings kept. | ការត្រួតពិនិត្យស្កេន៖ ទំព័រមើលទៅល្អ រក្សាការកំណត់ស្តង់ដារ។ |
| `scan_toast_active` | Scan check: {n} cleanup(s) suggested for this document. | ការត្រួតពិនិត្យស្កេន៖ ស្នើការរៀបចំ {n} សម្រាប់ឯកសារនេះ។ |
| `scan_toast_open` | Review | ពិនិត្យ |
| `stopped_msg` | Extraction stopped — press r or Run to start again. | បានបញ្ឈប់ការទាញយក — ចុច r ឬ Run ដើម្បីចាប់ផ្តើមម្តងទៀត។ |
| `tele_on` | {x} — on (auto) | {x} — បើក (ស្វ័យប្រវត្តិ) |
| `tele_off` | {x} — off (auto) | {x} — បិទ (ស្វ័យប្រវត្តិ) |
| `tele_tip` | See this setting | មើលការកំណត់នេះ |
| `preview_hint` | Preview — pages as uploaded. Press r or Run extraction to read them. | មើលជាមុន — ទំព័រដូចដែលបានផ្ទុកឡើង។ ចុច r ឬ Run extraction ដើម្បីអានវា។ |
| `view_single` | Single page | មួយទំព័រ |
| `view_grid` | Grid | ក្រឡា |
| `grid_tip` | Switch between one page and an overview of every page | ប្តូររវាងមួយទំព័រ និងទិដ្ឋភាពរួមនៃគ្រប់ទំព័រ |
| `grid_include_page` | Include page {n} in the run | បញ្ចូលទំព័រ {n} ក្នុងការដំណើរការ |
| `scope_list_option` | Selected pages ({n}) | ទំព័រដែលបានជ្រើស ({n}) |
| `status_stopped` | Stopped | បានបញ្ឈប់ |
| `dup_doc_notice` | Already in the queue: {name} — kept the existing copy and its results. | មានក្នុងជួររួចហើយ៖ {name} — រក្សាច្បាប់ចាស់ និងលទ្ធផលរបស់វា។ |

---

## FULL SET — all remaining UI strings (267 keys)

| key | English | Current Khmer |
|---|---|---|
| `view_options` | View | មើល |
| `view_options_tip` | Confidence tints and text size | ពណ៌ទំនុកចិត្ត និងទំហំអក្សរ |
| `text_size` | Text size | ទំហំអក្សរ |
| `toggle_on` | On | បើក |
| `toggle_off` | Off | បិទ |
| `conf_regions_scope` | Regions | តំបន់ |
| `dismiss_all_title` | Dismiss all issues? | លុបចេញបញ្ហាទាំងអស់មែនទេ? |
| `app_title` | Khmer Document Extraction | ការទាញយកទិន្នន័យពីឯកសារខ្មែរ |
| `backend_ready` | OCR backend is running | ម៉ាស៊ីន OCR កំពុងដំណើរការ |
| `backend_off` | OCR backend not running | ម៉ាស៊ីន OCR មិនទាន់ដំណើរការទេ |
| `notes` | Notes ({n}) | កំណត់ចំណាំ ({n}) |
| `notes_tip` | Things the pipeline noticed while reading this document | អ្វីដែលប្រព័ន្ធបានសង្កេតឃើញពេលអានឯកសារនេះ |
| `processing_notes` | Processing notes | កំណត់ចំណាំពេលដំណើរការ |
| `notes_intro` | Things the pipeline noticed while reading this document — worth a look, not necessarily wrong. | អ្វីដែលប្រព័ន្ធបានសង្កេតឃើញពេលអានឯកសារនេះ — គួរពិនិត្យមើល ប៉ុន្តែមិនប្រាកដថាខុសទេ។ |
| `view_page_n` | view page {n} | មើលទំព័រ {n} |
| `issues_n` | Issues ({n}) | បញ្ហា ({n}) |
| `no_issues` | No issues | គ្មានបញ្ហា |
| `issues_tip` | Low-confidence cells to review (n / p to step through) | ក្រឡាទំនុកចិត្តទាបដែលត្រូវពិនិត្យ (ចុច n / p ដើម្បីរំកិល) |
| `settings` | Settings | ការកំណត់ |
| `settings_tip` | How pages are cleaned and read — applies to the next run | របៀបសម្អាត និងអានទំព័រ — អនុវត្តចាប់ពីការដំណើរការបន្ទាប់ |
| `theme_tip_light` | Theme: light (click to change) | រូបរាង៖ ភ្លឺ (ចុចដើម្បីប្តូរ) |
| `theme_tip_dark` | Theme: dark (click to change) | រូបរាង៖ ងងឹត (ចុចដើម្បីប្តូរ) |
| `theme_tip_system` | Theme: follow system (click to change) | រូបរាង៖ តាមប្រព័ន្ធ (ចុចដើម្បីប្តូរ) |
| `lang_toggle_tip` | ប្តូរទៅភាសាខ្មែរ | Switch to English |
| `shortcuts` | Keyboard shortcuts | គ្រាប់ចុចផ្លូវកាត់ |
| `shortcuts_tip` | Keyboard shortcuts (?) | គ្រាប់ចុចផ្លូវកាត់ (?) |
| `ks_run` | Run / re-run extraction | ដំណើរការ / ដំណើរការទាញយកម្តងទៀត |
| `ks_pages` | Previous / next page | ទំព័រមុន / បន្ទាប់ |
| `ks_issue` | Next / previous issue (low-confidence cell) | បញ្ហាបន្ទាប់ / មុន (ក្រឡាទំនុកចិត្តទាប) |
| `ks_find` | Find & replace across all tables | ស្វែងរក និងជំនួសក្នុងតារាងទាំងអស់ |
| `ks_undo` | Undo / redo in the focused table | មិនធ្វើវិញ / ធ្វើវិញ ក្នុងតារាងដែលកំពុងជ្រើស |
| `ks_rowmenu_key` | Right-click a row | ចុចម៉ៅស៍ស្តាំលើជួរដេក |
| `ks_rowmenu` | Insert or delete rows | បញ្ចូល ឬលុបជួរដេក |
| `ks_esc` | Close panels | បិទផ្ទាំង |
| `ks_overlay` | This overlay | ផ្ទាំងនេះ |
| `err_unreachable` | Cannot reach the extraction server — check that it is still running. | មិនអាចភ្ជាប់ទៅម៉ាស៊ីនមេទាញយកបានទេ — សូមពិនិត្យថាវានៅដំណើរការ។ |
| `stale_notice` | Settings have changed since these results were made — re-run to apply them. | ការកំណត់បានផ្លាស់ប្តូរក្រោយពេលលទ្ធផលនេះធ្វើឡើង — សូមដំណើរការម្តងទៀតដើម្បីអនុវត្ត។ |
| `rerun_now` | Re-run now | ដំណើរការឥឡូវ |
| `ready_msg` | Ready — press “Run extraction” in the top bar. | រួចរាល់ — ចុច “ដំណើរការទាញយក” នៅរបារខាងលើ។ |
| `failed_msg` | The last run did not finish — adjust and retry from the top bar. | ការដំណើរការចុងក្រោយមិនបានបញ្ចប់ទេ — កែសម្រួល រួចព្យាយាមម្តងទៀតពីរបារខាងលើ។ |
| `working` | Working… | កំពុងដំណើរការ… |
| `loading_tables` | Loading tables… | កំពុងផ្ទុកតារាង… |
| `empty_title` | Extract tables from Khmer documents | ទាញយកតារាងពីឯកសារខ្មែរ |
| `empty_sub` | Verified numbers, straight into Excel. | តួលេខដែលបានផ្ទៀងផ្ទាត់ ចូល Excel ផ្ទាល់។ |
| `step_upload` | Upload a bulletin PDF or scan | បញ្ចូលឯកសារ PDF ឬស្កេន |
| `step_run` | Run the extraction | ដំណើរការទាញយក |
| `step_review` | Review flagged cells, then export | ពិនិត្យក្រឡាដែលបានសម្គាល់ រួចនាំចេញ |
| `upload_documents` | Upload documents | បញ្ចូលឯកសារ |
| `stage_read` | Reading the document… | កំពុងអានឯកសារ… |
| `stage_clean` | Cleaning the pages… | កំពុងសម្អាតទំព័រ… |
| `stage_ocr` | Finding text & tables… | កំពុងស្វែងរកអត្ថបទ និងតារាង… |
| `stage_tidy` | Tidying the text… | កំពុងរៀបចំអត្ថបទ… |
| `stage_export` | Preparing your files… | កំពុងរៀបចំឯកសាររបស់អ្នក… |
| `page_of` | page {a}/{b} | ទំព័រ {a}/{b} |
| `eta_s` | ~{s}s left | ~នៅសល់ {s} វិនាទី |
| `eta_ms` | ~{m}m {s}s left | ~នៅសល់ {m} នាទី {s} វិនាទី |
| `stop` | Stop | បញ្ឈប់ |
| `stopping` | Stopping… | កំពុងបញ្ឈប់… |
| `stop_tip` | Stop the extraction (finishes the current page, then cancels) | បញ្ឈប់ការទាញយក (បញ្ចប់ទំព័របច្ចុប្បន្នសិន រួចបោះបង់) |
| `rerun` | Re-run | ដំណើរការម្តងទៀត |
| `rerun_tip` | Run again with the selected engine | ដំណើរការម្តងទៀតដោយម៉ាស៊ីនស្គាល់អក្សរដែលបានជ្រើស |
| `engine_aria` | Recognition engine | ម៉ាស៊ីនស្គាល់អក្សរ |
| `extracting` | Extracting… | កំពុងទាញយក… |
| `export_results` | Export results | នាំចេញលទ្ធផល |
| `retry_extraction` | Retry extraction | ព្យាយាមទាញយកម្តងទៀត |
| `run_extraction` | Run extraction | ដំណើរការទាញយក |
| `n_unverified` | {n} unverified | {n} មិនទាន់ផ្ទៀងផ្ទាត់ |
| `export_ok_tip` | All tables verified — export away. | តារាងទាំងអស់បានផ្ទៀងផ្ទាត់ — នាំចេញបានទាំងស្រុង។ |
| `export_warn_prefix` | Export is always allowed — but  | ការនាំចេញអាចធ្វើបានជានិច្ច — ប៉ុន្តែ  |
| `warn_tables_one` | {n} table not yet verified | តារាង {n} មិនទាន់ផ្ទៀងផ្ទាត់ |
| `warn_tables_other` | {n} tables not yet verified | តារាង {n} មិនទាន់ផ្ទៀងផ្ទាត់ |
| `warn_cells_one` | {n} low-confidence cell unreviewed | ក្រឡាទំនុកចិត្តទាប {n} មិនទាន់ពិនិត្យ |
| `warn_cells_other` | {n} low-confidence cells unreviewed | ក្រឡាទំនុកចិត្តទាប {n} មិនទាន់ពិនិត្យ |
| `and` |  and  |  និង  |
| `other_formats` | Other formats | ទម្រង់ផ្សេងទៀត |
| `other_formats_aria` | Other export formats | ទម្រង់នាំចេញផ្សេងទៀត |
| `combine_title` | Tables that continue across pages | តារាងដែលបន្តពីទំព័រមួយទៅទំព័រមួយ |
| `combine_join` | Join into one table | បញ្ចូលគ្នាជាតារាងតែមួយ |
| `combine_join_hint` | One sheet, header once — paste straight into Excel. | សន្លឹកតែមួយ ក្បាលតារាងតែម្តង — បិទភ្ជាប់ចូល Excel ផ្ទាល់។ |
| `combine_keep` | Keep one table per page | រក្សាតារាងមួយក្នុងមួយទំព័រ |
| `combine_keep_hint` | Matches the pages exactly. | ត្រូវនឹងទំព័រយ៉ាងជាក់លាក់។ |
| `fmt_xlsx` | Excel (.xlsx) | Excel (.xlsx) |
| `fmt_json` | JSON | JSON |
| `fmt_txt` | Text report | របាយការណ៍អត្ថបទ |
| `add_documents` | Add documents | បន្ថែមឯកសារ |
| `uploading` | Uploading… | កំពុងបញ្ចូល… |
| `show_queue` | Show document queue | បង្ហាញជួរឯកសារ |
| `hide_queue` | Hide document queue | លាក់ជួរឯកសារ |
| `drop_to_add` | Drop to add | ទម្លាក់ដើម្បីបន្ថែម |
| `run_all` | Run all ({n}) | ដំណើរការទាំងអស់ ({n}) |
| `running_all` | Running all… | កំពុងដំណើរការទាំងអស់… |
| `run_all_tip` | Run every unprocessed document, one after another | ដំណើរការឯកសារដែលមិនទាន់ដំណើរការទាំងអស់ ម្តងមួយៗ |
| `export_all` | Export all | នាំចេញទាំងអស់ |
| `export_all_tip_warn` | One zip with every finished document. {n} table(s) across the queue are not verified yet. | ហ្ស៊ីបមួយមានឯកសាររួចរាល់ទាំងអស់។ តារាង {n} ក្នុងជួរមិនទាន់ផ្ទៀងផ្ទាត់ទេ។ |
| `export_all_tip_ok` | One zip with every finished document's results — all tables verified. | ហ្ស៊ីបមួយមានលទ្ធផលឯកសាររួចរាល់ទាំងអស់ — តារាងទាំងអស់បានផ្ទៀងផ្ទាត់។ |
| `no_docs_1` | No documents yet. | មិនទាន់មានឯកសារទេ។ |
| `no_docs_2` | Add a bulletin PDF, or drop it here. | បន្ថែមឯកសារ PDF ឬទម្លាក់វានៅទីនេះ។ |
| `pages_kb` | {p} page(s) · {kb} KB | {p} ទំព័រ · {kb} KB |
| `status_queued` | queued | រង់ចាំ |
| `status_running` | running | កំពុងដំណើរការ |
| `status_done` | done | រួចរាល់ |
| `status_error` | error | បញ្ហា |
| `verified_count` | {a}/{b} verified | បានផ្ទៀងផ្ទាត់ {a}/{b} |
| `remove_doc` | Remove document | លុបឯកសារ |
| `remove_confirm` | Remove “{name}”? Its results and any edits will be discarded. | លុប “{name}”? លទ្ធផល និងការកែប្រែទាំងអស់នឹងបាត់បង់។ |
| `queue_count_tip` | {n} document(s) in the queue | ឯកសារ {n} ក្នុងជួរ |
| `page_n_of` | Page {i} / {n} | ទំព័រ {i} / {n} |
| `prev_page` | Previous page | ទំព័រមុន |
| `next_page` | Next page | ទំព័របន្ទាប់ |
| `fit` | Fit | ពេញផ្ទាំង |
| `fit_tip` | Fit the whole page in view | បង្ហាញទំព័រទាំងមូល |
| `actual_size` | Actual size | ទំហំពិត |
| `variant_processed` | Cleaned | បានសម្អាត |
| `variant_original` | Original | ដើម |
| `variant_processed_tip` | The cleaned page the recogniser read | ទំព័រដែលបានសម្អាត ដែលប្រព័ន្ធបានអាន |
| `variant_original_tip` | The original scan | ស្កេនដើម |
| `rendition_aria` | Page rendition | ទម្រង់ទំព័រ |
| `overlay_conf` | Confidence boxes | ប្រអប់ទំនុកចិត្ត |
| `overlay_regions` | Region types | ប្រភេទតំបន់ |
| `overlay_none` | No boxes | គ្មានប្រអប់ |
| `overlay_aria` | Overlay mode | របៀបប្រអប់ |
| `overlay_tip` | What the colored boxes mean | អត្ថន័យនៃប្រអប់ពណ៌ |
| `loupe` | Loupe | កែវពង្រីក |
| `loupe_tip` | Magnifier — inspect glyphs at 3× without zooming the page | កែវពង្រីក — ពិនិត្យអក្សរ 3× ដោយមិនពង្រីកទំព័រ |
| `legend_high` | ≥80% solid | ≥80% បន្ទាត់តាន់ |
| `legend_mid` | 50–80% | 50–80% |
| `legend_low` | <50% dashed | <50% បន្ទាត់ដាច់ៗ |
| `open_table` | Open table {tid} | បើកតារាង {tid} |
| `open_table_tip` | {tid} — click to open this table | {tid} — ចុចដើម្បីបើកតារាងនេះ |
| `tables_one` | table | តារាង |
| `tables_other` | tables | តារាង |
| `blocks_one` | text block | ប្លុកអត្ថបទ |
| `blocks_other` | text blocks | ប្លុកអត្ថបទ |
| `n_lowconf` | {n} low-confidence | ទំនុកចិត្តទាប {n} |
| `verify_page` | Verify page | ផ្ទៀងផ្ទាត់ទំព័រ |
| `verify_page_tip` | Mark every table on this page as reviewed | សម្គាល់តារាងទាំងអស់ក្នុងទំព័រនេះថាបានពិនិត្យ |
| `legend_check` | conf — check | ត្រូវពិនិត្យ |
| `legend_check_pct` | <80% | <80% |
| `legend_skim` | 80–95% skim | 80–95% មើលរហ័ស |
| `size_tip` | Content text size (Khmer legibility) | ទំហំអក្សរខ្លឹមសារ (ភាពច្បាស់អក្សរខ្មែរ) |
| `size_smaller` | Smaller content text | អក្សរតូចជាង |
| `size_larger` | Larger content text | អក្សរធំជាង |
| `find_ph` | Find… | ស្វែងរក… |
| `replace_ph` | Replace with… | ជំនួសដោយ… |
| `replace_all_btn` | Replace in all tables | ជំនួសក្នុងតារាងទាំងអស់ |
| `replace_all_tip` | Replace in every table of this document | ជំនួសក្នុងគ្រប់តារាងនៃឯកសារនេះ |
| `undo_replace` | Undo replace | មិនធ្វើការជំនួសវិញ |
| `undo_replace_tip` | Restore the tables to before the replace | ស្តារតារាងទៅមុនពេលជំនួស |
| `close_find` | Close find and replace | បិទការស្វែងរក និងជំនួស |
| `replace_confirm` | Replace “{a}” with “{b}” in every table of this document — including pages you have not reviewed?\n\nYou can undo this immediately afterwards. | ជំនួស “{a}” ដោយ “{b}” ក្នុងគ្រប់តារាងនៃឯកសារនេះ — រួមទាំងទំព័រដែលអ្នកមិនទាន់ពិនិត្យ?\n\nអ្នកអាចមិនធ្វើវិញភ្លាមៗក្រោយពេលនេះ។ |
| `replaced_msg` | Replaced {n} occurrence(s) in {t} table(s). | បានជំនួស {n} កន្លែង ក្នុងតារាង {t}។ |
| `no_matches` | No matches found. | រកមិនឃើញទេ។ |
| `replace_undone` | Replace undone. | បានមិនធ្វើការជំនួសវិញ។ |
| `replace_failed` | Replace failed: {e} | ការជំនួសបរាជ័យ៖ {e} |
| `undo_failed` | Undo failed: {e} | ការមិនធ្វើវិញបរាជ័យ៖ {e} |
| `intro_title` | Checking a table: | ការពិនិត្យតារាង៖ |
| `intro_body` | tinted cells are the ones the recogniser was unsure of — compare each against the page image on the left, correct it here, then mark the table  | ក្រឡាដែលមានពណ៌ គឺក្រឡាដែលប្រព័ន្ធមិនប្រាកដ — ប្រៀបធៀបនឹងរូបទំព័រខាងឆ្វេង កែនៅទីនេះ រួចចុច  |
| `intro_verify` | Verify | ផ្ទៀងផ្ទាត់ |
| `intro_tail` | . Export stays available at any point; it tells you how much is still unverified. | ។ ការនាំចេញអាចធ្វើបានគ្រប់ពេល ហើយវាបង្ហាញចំនួនដែលមិនទាន់ផ្ទៀងផ្ទាត់។ |
| `dismiss_tip` | Dismiss this tip | បិទគន្លឹះនេះ |
| `no_tables` | No tables on this page. | គ្មានតារាងក្នុងទំព័រនេះទេ។ |
| `page_text` | Page text | អត្ថបទទំព័រ |
| `unsaved_suffix` |  — unsaved |  — មិនទាន់រក្សាទុក |
| `save_text` | Save text | រក្សាទុកអត្ថបទ |
| `verify` | Verify | ផ្ទៀងផ្ទាត់ |
| `verified` | Verified | បានផ្ទៀងផ្ទាត់ |
| `verify_tip_on` | Marked verified — click to unmark | បានផ្ទៀងផ្ទាត់ — ចុចដើម្បីដកចេញ |
| `verify_tip_off` | Mark this table as reviewed | សម្គាល់តារាងនេះថាបានពិនិត្យ |
| `edited` | Edited | បានកែ |
| `saving` | saving… | កំពុងរក្សាទុក… |
| `not_saved` | Not saved to the server — try that again | មិនបានរក្សាទុកទៅម៉ាស៊ីនមេទេ — សូមព្យាយាមម្តងទៀត |
| `undo` | Undo | មិនធ្វើវិញ |
| `redo` | Redo | ធ្វើវិញ |
| `undo_tip` | Undo (Cmd-Z) | មិនធ្វើវិញ (Cmd-Z) |
| `redo_tip` | Redo (Shift-Cmd-Z) | ធ្វើវិញ (Shift-Cmd-Z) |
| `diff` | Diff | ភាពខុសគ្នា |
| `diff_tip` | Highlight cells that differ from the OCR result | បន្លិចក្រឡាដែលខុសពីលទ្ធផល OCR |
| `row` | Row | ជួរដេក |
| `row_tip` | Add a row at the end | បន្ថែមជួរដេកនៅចុង |
| `csv_tip` | Download just this table as CSV | ទាញយកតារាងនេះជា CSV |
| `reset` | Reset | កំណត់ឡើងវិញ |
| `reset_tip` | Discard all edits to this table | បោះបង់ការកែទាំងអស់ក្នុងតារាងនេះ |
| `insert_above` | Insert row above | បញ្ចូលជួរដេកខាងលើ |
| `insert_below` | Insert row below | បញ្ចូលជួរដេកខាងក្រោម |
| `delete_row` | Delete row | លុបជួរដេក |
| `np_step` | n / p to step | n / p ដើម្បីរំកិល |
| `close_issues` | Close issues panel | បិទផ្ទាំងបញ្ហា |
| `all_confident` | No flagged cells. | គ្មានក្រឡាដែលបានសម្គាល់។ |
| `reason_numeric_mismatch` | Row total doesn't add up | ផលបូកជួរដេកមិនត្រូវគ្នា |
| `reason_sequence_illegal` | Illegal Khmer character sequence | លំដាប់អក្សរខ្មែរមិនត្រឹមត្រូវ |
| `reason_digit_mixed` | Mixed Khmer and Arabic digits | លាយលេខខ្មែរ និងលេខអារ៉ាប់ |
| `reason_numeric_unparseable` | Number can't be read | មិនអាចអានតួលេខបានទេ |
| `reason_structure_ragged` | Ragged row structure | រចនាសម្ព័ន្ធជួរដេកមិនស្មើ |
| `reason_low_conf` | Low recognition confidence | ទំនុកចិត្តនៃការស្គាល់អក្សរទាប |
| `empty_cell` | empty cell | ក្រឡាទទេ |
| `issue_loc` | row {r}, col {c} | ជួរដេក {r}, ជួរឈរ {c} |
| `issue_page` | page {p} | ទំព័រ {p} |
| `scan_check_title` | Scan check | ការត្រួតពិនិត្យស្កេន |
| `scan_off_list` | Scan check — turned off {labels} for this document. | ការត្រួតពិនិត្យស្កេន — បានបិទ {labels} សម្រាប់ឯកសារនេះ។ |
| `scan_clean` | Scan check — this document looks good; standard settings kept. | ការត្រួតពិនិត្យស្កេន — ឯកសារនេះមើលទៅល្អ រក្សាការកំណត់ស្តង់ដារ។ |
| `details` | Details | លម្អិត |
| `check_straight` | Pages are straight | ទំព័រត្រង់ហើយ |
| `check_no_stamps` | No stamps found | រកមិនឃើញត្រាទេ |
| `check_soft_scan` | Scan is soft — sharpening will help | ស្កេនព្រិល — ការធ្វើឲ្យច្បាស់នឹងជួយ |
| `check_already_sharp` | Scan is sharp — sharpening turned off | ស្កេនច្បាស់ស្រាប់ — បានបិទការធ្វើឲ្យច្បាស់ |
| `check_good_contrast` | Contrast is already good — enhancement turned off | កម្រិតពណ៌ល្អស្រាប់ — បានបិទការបង្កើន |
| `check_table_shading_default` | Table backgrounds flattened automatically | ស្រមោលតារាងត្រូវបានតម្រូវដោយស្វ័យប្រវត្តិ |
| `settings_subtitle` | Applies to the next run | អនុវត្តចាប់ពីការដំណើរការបន្ទាប់ |
| `extraction_settings` | Extraction settings | ការកំណត់ការទាញយក |
| `engine_section` | Recognition engine | ម៉ាស៊ីនស្គាល់អក្សរ |
| `engine_hint` | Which model reads the pages — applies to the next run. | ម៉ូដែលណាដែលអានទំព័រ — អនុវត្តចាប់ពីការដំណើរការបន្ទាប់។ |
| `more_menu` | More | ច្រើនទៀត |
| `ks_verify` | Verify / unverify the focused table | ផ្ទៀងផ្ទាត់ / ដកការផ្ទៀងផ្ទាត់ តារាងដែលកំពុងជ្រើស |
| `close_settings` | Close settings | បិទការកំណត់ |
| `dpi` | Scan quality (DPI) | គុណភាពស្កេន (DPI) |
| `pages` | Pages | ទំព័រ |
| `all_pages` | All pages | ទំព័រទាំងអស់ |
| `single_page` | Single page | ទំព័រតែមួយ |
| `page_range` | Page range | ចន្លោះទំព័រ |
| `first_page` | First page | ទំព័រដំបូង |
| `last_page` | Last page | ទំព័រចុងក្រោយ |
| `range_error` | The last page is before the first — fix the range before running. | ទំព័រចុងក្រោយនៅមុនទំព័រដំបូង — សូមកែចន្លោះមុនដំណើរការ។ |
| `single_error` | This document has only {n} page(s). | ឯកសារនេះមានតែ {n} ទំព័រប៉ុណ្ណោះ។ |
| `page_cleanup` | Preprocessing | ការរៀបចំមុនដំណើរការ |
| `flag_deskew` | Deskew | តម្រង់ទំព័រ |
| `hint_deskew` | Straightens a scan that was fed in crooked. | តម្រង់ស្កេនដែលបញ្ចូលទ្រេត។ |
| `flag_stamps` | Remove stamps | លុបត្រា |
| `hint_stamps` | Erases signature stamps that sit over the numbers. | លុបត្រា និងហត្ថលេខាដែលគ្របលើតួលេខ។ |
| `flag_sharpen` | Sharpen | ធ្វើឲ្យច្បាស់ |
| `hint_sharpen` | Crispens soft or faxed scans. | ធ្វើឲ្យស្កេនព្រិលច្បាស់ឡើង។ |
| `flag_contrast` | Enhance contrast | បង្កើនកម្រិតពណ៌ |
| `hint_contrast` | Evens out faded or unevenly lit pages. | តម្រូវទំព័រស្លេក ឬពន្លឺមិនស្មើ។ |
| `flag_tablebg` | Normalise table backgrounds | តម្រូវផ្ទៃខាងក្រោយតារាង |
| `hint_tablebg` | Flattens coloured table shading so text reads cleanly. | លុបស្រមោលពណ៌ក្នុងតារាង ដើម្បីឲ្យអក្សរច្បាស់។ |
| `output` | Export settings | ការកំណត់នាំចេញ |
| `flag_repair` | Repair table structure | ជួសជុលរចនាសម្ព័ន្ធតារាង |
| `hint_repair` | Fills ragged rows the recogniser left uneven. | បំពេញជួរដេកមិនស្មើដែលប្រព័ន្ធទុកចោល។ |
| `flag_numerals` | Convert Khmer numerals to Arabic | បម្លែងលេខខ្មែរទៅលេខអារ៉ាប់ |
| `hint_numerals` | Writes ១២៣ as 123 in exports. | សរសេរ ១២៣ ជា 123 ក្នុងការនាំចេញ។ |
| `join_note` | Joining tables that continue across pages is chosen when you export, not here — so review always stays linked to the page each row came from. | ការបញ្ចូលតារាងបន្តទំព័រ ត្រូវជ្រើសពេលនាំចេញ មិនមែននៅទីនេះទេ — ដើម្បីឲ្យការពិនិត្យនៅភ្ជាប់នឹងទំព័រដើមជានិច្ច។ |
| `settings_footer` | Changes apply to the next run. If results were made with different settings, a “settings changed” notice appears until you re-run. | ការផ្លាស់ប្តូរអនុវត្តចាប់ពីការដំណើរការបន្ទាប់។ បើលទ្ធផលធ្វើដោយការកំណត់ផ្សេង នឹងមានការជូនដំណឹងរហូតដល់ដំណើរការម្តងទៀត។ |
| `palette_tip` | Command palette (⌘K) | ផ្ទាំងបញ្ជា (⌘K) |
| `palette_placeholder` | Type a command or search… | វាយបញ្ជា ឬស្វែងរក… |
| `palette_empty` | No matching commands | គ្មានបញ្ជាដែលត្រូវគ្នា |
| `group_documents` | Documents | ឯកសារ |
| `group_pages` | Pages | ទំព័រ |
| `group_issues` | Issues | បញ្ហា |
| `group_actions` | Actions | សកម្មភាព |
| `cmd_goto_page` | Go to page {n} | ទៅទំព័រ {n} |
| `cmd_goto_issue` | Go to issue {n} of {total} | ទៅបញ្ហាទី {n} ក្នុងចំណោម {total} |
| `cmd_open_settings` | Open extraction settings | បើកការកំណត់ការទាញយក |
| `cmd_open_issues` | Open issues panel | បើកផ្ទាំងបញ្ហា |
| `cmd_toggle_queue` | Show / hide document queue | បង្ហាញ / លាក់ជួរឯកសារ |
| `cmd_theme` | Cycle theme (light / dark / system) | ប្តូររូបរាង (ភ្លឺ / ងងឹត / តាមប្រព័ន្ធ) |
| `cmd_language` | Switch language — English ⇄ ខ្មែរ | ប្តូរភាសា — English ⇄ ខ្មែរ |
| `cmd_export_xlsx` | Export Excel (.xlsx) | នាំចេញ Excel (.xlsx) |
| `cmd_export_json` | Export JSON | នាំចេញ JSON |
| `cmd_export_zip` | Export everything (.zip) | នាំចេញទាំងអស់ (.zip) |
| `cmd_turn_on` | Turn on: {x} | បើក៖ {x} |
| `cmd_turn_off` | Turn off: {x} | បិទ៖ {x} |
| `cmd_engine` | Use engine: {x} | ប្រើម៉ាស៊ីន៖ {x} |
| `ks_palette` | Command palette | ផ្ទាំងបញ្ជា |
| `dismiss_issue` | Dismiss from this list (the cell stays flagged until fixed) | លុបចេញពីបញ្ជីនេះ (ក្រឡានៅតែត្រូវបានសម្គាល់រហូតដល់កែរួច) |
| `dismiss_badge` | Dismiss | លុបចេញ |
| `pill_on` | On | បើក |
| `pill_off` | Off | បិទ |
| `pill_toggle_tip` | Click to flip this for the next run | ចុចដើម្បីបិទបើកសម្រាប់ដំណើរការបន្ទាប់ |
