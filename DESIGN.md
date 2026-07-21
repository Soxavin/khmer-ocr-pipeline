---
name: Khmer OCR Review Workspace
description: A high-agency command deck for turning scanned Khmer financial documents into verified spreadsheets.
colors:
  primary: "#1565c0"
  primary-strong: "#0f4c94"
  primary-soft: "oklch(0.945 0.028 255)"
  surface: "#ffffff"
  rail: "oklch(0.972 0.006 255)"
  canvas: "oklch(0.916 0.011 255)"
  raised: "#ffffff"
  ink: "oklch(0.24 0.022 255)"
  ink-2: "oklch(0.42 0.026 255)"
  ink-3: "oklch(0.52 0.024 255)"
  line: "oklch(0.895 0.011 255)"
  line-strong: "oklch(0.856 0.014 255)"
  ok: "#16a34a"
  ok-soft: "oklch(0.962 0.024 155)"
  warn: "#d97706"
  warn-soft: "oklch(0.972 0.022 85)"
  danger: "#dc2626"
  danger-soft: "oklch(0.962 0.018 25)"
  conf-high: "#16a34a"
  conf-mid: "#f59e0b"
  conf-low: "#dc2626"
typography:
  title:
    fontFamily: "Inter Variable, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Inter Variable, system-ui, sans-serif"
    fontSize: "0.84375rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: "Inter Variable, system-ui, sans-serif"
    fontSize: "0.78125rem"
    fontWeight: 500
    lineHeight: 1.45
    letterSpacing: "normal"
  micro:
    fontFamily: "Inter Variable, system-ui, sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "normal"
  data-khmer:
    fontFamily: "Noto Sans Khmer, Khmer OS, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.9
    letterSpacing: "normal"
rounded:
  md: "6px"
  lg: "8px"
  xl: "12px"
  full: "9999px"
spacing:
  seam: "6px"
  sm: "8px"
  md: "12px"
  lg: "16px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#ffffff"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "32px"
  button-primary-hover:
    backgroundColor: "{colors.primary-strong}"
    textColor: "#ffffff"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.md}"
    padding: "0 10px"
    height: "28px"
  button-danger:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.danger}"
    rounded: "{rounded.md}"
    padding: "0 10px"
    height: "28px"
  chip:
    backgroundColor: "{colors.ok-soft}"
    textColor: "{colors.ink}"
    rounded: "{rounded.full}"
    padding: "0 10px"
    height: "24px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "0 8px"
    height: "28px"
  panel-main:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.xl}"
  panel-floating:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.ink}"
    rounded: "{rounded.xl}"
  kbd:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.md}"
    height: "18px"
---

# Design System: Khmer OCR Review Workspace

## 1. Overview

**Creative North Star: "The Command Deck"**

This is an instrument, not a document. An analyst sits at a desk with a scanned Khmer price bulletin and one job: get the numbers out with certainty that nothing was silently mis-read. The interface earns its keep by being a finely tuned tool — crisp, dense, and laser-focused — that disappears into that task. The reference frame is Linear and Raycast, not a marketing dashboard: government-grade reliability delivered with the spatial, keyboard-first responsiveness of a modern developer tool.

Depth is carried by **layered zones, not decoration**. The whole surface is one semantic layer over a single blue hue (GDDE blue, OKLCH h≈255): zones separate by stepped backgrounds — recessed `canvas` for the page image, a cooler `rail` for the queue, bright `surface` for cards — so structure reads without a single heavy border or drop shadow doing the work. Cards barely lift; overlays clearly float; that two-tier elevation is the entire z-vocabulary. Color is rationed hard: the one blue means focus, selection, and "the system is acting," and the confidence trio (green/amber/red) means exactly what the model believes about a cell. Nothing on screen is tinted for flavor.

The system explicitly rejects three things, in PRODUCT.md's own words: **bloated marketing dashboards** (heavy gradients, hero illustrations, empty card grids), **sterile legacy portals** (cramped flat tables, tiny unreadable type, rigid un-nested sheets), and **playful consumer apps** (rounded-everything, emoji, mascot energy). Trust is the product; every visual signal is calibrated and honest, or it is cut.

**Key Characteristics:**
- One hue (blue h≈255), one accent job: focus / selection / system-acting. Everything else is a tinted neutral or a semantic status.
- Layered-zone depth: background steps (canvas → rail → surface → raised) replace borders and shadows as the primary structure signal.
- Compact, sub-Tailwind type ramp (11 → 12.5 → 13.5 → 15px) for command density without noise.
- Khmer content is first-class and visually distinct from chrome: a dedicated face with 1.9 line-height, an adjustable size, and grid rows that grow rather than clip.
- Snappy, single-purpose motion (75–200 ms, expo-out), fully suppressed under `prefers-reduced-motion`.

## 2. Colors

A restrained, single-hue system: one blue does all the accent work, a hue-255 neutral ramp carries every surface and line, and three status families speak only when the data warrants it.

### Primary
- **GDDE Blue** (`#1565c0`): The only blue, and the only decoration-free accent. It marks primary actions, the current selection, links, focus rings, and the "system is acting" pulse (the header progress line, flash-to-evidence). Its scarcity is what makes it legible as a signal.
- **GDDE Blue Deep** (`#0f4c94`): Hover and active state for the primary action only.
- **Selection Wash** (`oklch(0.945 0.028 255)`): The soft primary tint behind a selected queue row, a chosen segmented-control option, or an edited (diff) cell. Never a decorative fill.

### Neutral
- **Ink** (`oklch(0.24 0.022 255)`): Primary text and iconography. A blue-tinted near-black, never pure gray.
- **Ink-2** (`oklch(0.42 0.026 255)`): Secondary text — safe on every surface. Default color for button labels and captions.
- **Ink-3** (`oklch(0.52 0.024 255)`): Muted text, ≥4.5:1 on white but reserved for non-load-bearing labels only.
- **Surface** (`#ffffff`): The bright content layer — header, table cards, panels.
- **Rail** (`oklch(0.972 0.006 255)`): The cooler second neutral for the queue rail and panel/grid headers. This is the "sidebar layer" the product register calls for.
- **Canvas** (`oklch(0.916 0.011 255)`): The recessed ground the page-viewer sits in, so the image reads as inset under the cards.
- **Raised** (`#ffffff`): Popovers, menus, dialogs — the same white as surface but paired with overlay elevation to lift it off the workbench.
- **Line** (`oklch(0.895 0.011 255)`) / **Line-Strong** (`oklch(0.856 0.014 255)`): Hairline seams and control borders respectively. Borders are a last resort; zones separate by background step first.

### Tertiary — Status & Confidence
- **OK Green** (`#16a34a`), **Warn Amber** (`#d97706`), **Danger Red** (`#dc2626`): Semantic states only — verified, processing notes, errors — each with a soft tint (`ok-soft`, `warn-soft`, `danger-soft`) for backgrounds.
- **Confidence Trio** — **High** (`#16a34a`), **Mid** (`#f59e0b`), **Low** (`#dc2626`): The recognizer's own belief about a cell, mirrored 1:1 from the Python pipeline (`webapp/components.py`). This is the trust palette; it is never used for anything but confidence.

### Named Rules
**The One Blue Rule.** There is exactly one accent hue and it is never spent on decoration. If a blue element is not a primary action, the current selection, a focus ring, or the system actively working, it is wrong.

**The Zones-Not-Borders Rule.** Structure is carried by the background step between canvas, rail, surface, and raised. Reach for a border only when two same-zone surfaces must be told apart; reach for a shadow only to float an overlay.

**The Honest-Trust Rule.** Confidence and status colors are calibrated signals, never reassurance. A green mark means verified; an amber cell means the model is unsure. Never tint a cell to look confident, and never soften a real warning.

## 3. Typography

**Chrome Font:** Inter Variable (with `system-ui`, `-apple-system`, sans fallback)
**Content Font:** Noto Sans Khmer (with `Khmer OS`, `system-ui` fallback)
**Label/Mono accent:** Inter Variable for keycaps; `font-variant-numeric: tabular-nums` is global — this is a numbers product and digits line up everywhere by default.

**Character:** One well-tuned sans carries all chrome — headings, buttons, labels, data — pitched half a step smaller and tighter than stock Tailwind for command density. The one deliberate pairing is functional, not stylistic: Khmer document content renders in its own face at generous leading, so the analyst can always tell the *material* (the extracted text) from the *instrument* (the UI around it).

### Hierarchy
- **Title** (600, 15px / `0.9375rem`, tracking −0.01em): Panel and card headers ("Documents", "Extraction settings"), the app title. The ceiling of the chrome scale — there is no display tier; product UI does not shout.
- **Body** (400, 13.5px / `0.84375rem`, line-height 1.5): Default reading size for hints, descriptions, and menu rows.
- **Label** (500, 12.5px / `0.78125rem`): Control labels, chips, secondary captions.
- **Micro** (600, 11px / `0.6875rem`): Count badges, status-dot labels, keycaps — the smallest legible tier, weight-boosted to hold up.
- **Data — Khmer** (400, 14px adjustable, line-height 1.9): Extracted document content in table cells and the text view. Size is user-adjustable (A− / A+) because stacked subscript consonants and diacritics need room; rows grow to fit and never clip.

### Named Rules
**The Instrument-vs-Material Rule.** UI chrome is Inter and never renders Khmer document content; document content is Noto Sans Khmer and never styles a button or a label. The moment the two faces blur, the analyst loses the line between what the tool says and what the document says.

**The Breathing-Khmer Rule.** Khmer content is never set below 1.9 line-height, and its rows resize with content. Subscript stacks that touch or clip are a correctness bug, not a cosmetic one — this is a legibility product.

## 4. Elevation

Depth is primarily **tonal**, not cast. Zones are read by their background step (canvas → rail → surface → raised), and that layering does most of the structural work. Shadows are a deliberate two-tier minority vocabulary: cards *barely* lift so the workbench stays calm, and only true overlays cast enough to read as floating above everything else. Under `prefers-reduced-motion` all entrance animation is cut, but the elevation itself is static and always present.

### Shadow Vocabulary
- **Raised** (`box-shadow: 0 1px 2px oklch(0.24 0.022 255 / 0.06)`): The whisper-lift on resting cards and panels. Just enough to detach a `surface` card from the `canvas` behind it.
- **Overlay** (`0 1px 3px …/0.08, 0 8px 24px …/0.13`): Popovers, menus, drawers, the settings sheet — anything that appears over the workbench and must clearly float.
- **Modal** (`0 2px 6px …/0.08, 0 16px 48px …/0.22`): Dialogs and the command palette — the top of the stack, casting hardest.

### Named Rules
**The Two-Tier Lift Rule.** Resting surfaces get `raised` (barely there) and nothing more. Shadow depth is earned only by leaving the plane: `overlay` for contextual pop-ups, `modal` for the command palette and dialogs. There is no third resting elevation.

## 5. Components

Every control composes from one shared vocabulary (`frontend/src/ui.ts`); per-instance styling drift is treated as the enemy. All interactive elements carry default / hover / focus / active / disabled, and press-feedback is a uniform `active:scale-[0.98]`.

### Buttons
- **Shape:** Gently rounded (6px, `rounded-md`). Never pill-shaped, never square.
- **Primary:** Solid GDDE Blue fill, white text, semibold, 32px tall, with a faint colored shadow (`shadow-md shadow-primary/10`) that deepens on hover to `primary-strong`. The one loud element per context — the single morphing action Upload → Run → Export.
- **Secondary (default button):** White surface, 1px `line-strong` border, `ink-2` label, 28px tall; hover fills to `rail` and darkens text to `ink`. This is the workhorse.
- **Danger:** Same chassis as secondary but a `danger/40` border and `danger-ink` text; hover fills `danger-soft`. Used for Stop and destructive confirms only.
- **Icon button:** 28px square, borderless, `ink-3` glyph, hover fills `rail` — for dense toolbars where a label would be noise.
- **Hover / Focus / Active:** 150 ms ease-out on color/border/shadow; a 2px `primary` focus-visible outline on every variant; `scale-0.98` on press.

### Chips
- **Style:** Fully rounded (`rounded-full`), 24px tall, 12.5px medium label. Tinted by role — `ok-soft` / `warn-soft` / `danger-soft` grounds with matching `-ink` text.
- **State:** Read-only status (verified count, processing notes, issue count). Clickable chips open the relevant drawer; they are indicators first, controls second.

### Cards / Containers
- **Corner Style:** 12px (`rounded-xl`) for the three workbench panels and floating decks; 6–8px for inner controls.
- **Background:** `surface` for content cards, `raised` for floating panels.
- **Shadow Strategy:** `raised` at rest (see Elevation). Floating panels (`panel-floating`) add `overlay` plus a `backdrop-blur-md` translucency for the command-deck feel.
- **Border:** `line-strong/60` hairline — present but secondary to the zone step behind it.
- **Internal Padding:** Built on a 6px seam rhythm — the workbench uses `p-1.5` (6px) gaps between panels, mandatory `h-10` panel headers, and explicit 6px margins rather than flex `gap` (so a collapsed `w-0` drawer leaves no phantom gutter).

### Inputs / Fields
- **Style:** 28px tall, `line-strong` border, white `surface`, 6px radius, `ink-3` placeholder.
- **Focus:** Border shifts to `primary` plus the shared 2px focus-visible outline. No glow.
- **Segmented controls:** The recurring pattern for small either/or choices (Cleaned⇄Original, DPI, Single⇄Grid) — a bordered row of buttons where the active option takes `primary-soft` fill and `primary-strong` text.

### Navigation & Command
- **Collapsible queue rail:** The left rail folds from `w-64` to a slim `w-11` icon strip (remembered per machine) so the review panes get the width — Design Principle 2 (Dynamic Workspace Layout) in the flesh.
- **Command palette (⌘K):** The keyboard-first spine. A `modal`-elevation sheet with fuzzy search over every workspace action; the interface is fully operable without it, but it is the fastest path.
- **Keycaps (`kbd`):** 18px-tall bordered chips, mono, `ink-2`, with a `raised` shadow — shortcuts are surfaced visibly in menus and the help dialog, never hidden.

### Signature Component — The Confidence Grid
The AG-Grid table is where trust becomes visible. Cells the model read with low confidence carry a faint red wash *and* a dotted red underline (the signal must survive color-blindness and a washed-out projector); mid-confidence carries a faint amber wash. Edited cells take the `primary-soft` diff tint with an inset border — "edited" is the analyst's own mark, not a warning. Tints are deliberately faint: on a bad page hundreds of cells flag at once, and loud tints would turn the grid into wallpaper. Rows grow with the Khmer line-height; nothing is ever truncated behind an ellipsis.

## 6. Do's and Don'ts

### Do:
- **Do** spend the one blue (`#1565c0`) only on primary actions, current selection, focus rings, and the system-acting pulse. Everything else is a tinted neutral or a semantic status.
- **Do** separate zones by background step (canvas → rail → surface → raised) before reaching for a border or a shadow.
- **Do** keep chrome in Inter and document content in Noto Sans Khmer at ≥1.9 line-height, always distinguishable.
- **Do** compose every control from `frontend/src/ui.ts`; give each one default / hover / focus / active / disabled.
- **Do** keep motion to 75–200 ms, expo-out, single-purpose (menus rise from their trigger, drawers slide from their edge, backdrops fade), and fully suppressed under `prefers-reduced-motion`.
- **Do** keep confidence and status tints faint and honest — calibrated to survive volume and a bad projector.
- **Do** take structural risks when a view is crowded (collapse the rail, pull up a resizable split, nest tools in a fly-out drawer) — this is authorized in PRODUCT.md's Implementation Mandate.

### Don't:
- **Don't** build **bloated marketing-dashboard** energy: no heavy gradients, no hero illustrations, no empty card grids wasting the workbench.
- **Don't** slide toward a **sterile legacy portal**: no cramped flat tables, no tiny unreadable type, no rigid un-nested sheets.
- **Don't** go **playful-consumer**: no rounded-everything, no emoji as section markers, no mascot energy.
- **Don't** introduce a second accent hue, or tint anything blue for decoration. One blue, one job.
- **Don't** use `border-left`/`border-right` greater than 1px as a colored accent stripe on cards, list items, or alerts.
- **Don't** use gradient text (`background-clip: text`) or glassmorphism as decoration — the single deliberate `backdrop-blur` on floating panels is the only permitted blur.
- **Don't** set Khmer content below 1.9 line-height or let a cell clip a subscript stack — that is a correctness bug.
- **Don't** render document content in Inter or UI labels in the Khmer face; the instrument and the material never share a typeface.
- **Don't** add a third resting elevation or cast heavy shadows on flat cards — resting surfaces get `raised` and nothing more.
