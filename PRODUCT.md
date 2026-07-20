# Product

## Register

product

## Platform

web

## Users

Data analysts at GDDE (General Department of Digital Economy, Ministry of Economy and Finance, Cambodia). They are domain experts and Excel-native, not software enthusiasts. Their context: reviewing scanned or PDF Khmer financial documents (e.g. ARDB daily price bulletins) at a desk, needing the numbers out and into Excel with confidence that nothing was silently mis-read. Secondary audience: mentors and stakeholders seeing the tool in a projected demo.

## Product Purpose

A review workspace for a Khmer OCR pipeline: upload documents, run extraction, review and correct the recognized tables against the page image, and export trustworthy JSON/CSV/Excel. Success is an analyst finishing a bulletin in minutes with every low-confidence cell checked — trusting the export enough to use it downstream without re-keying.

## Positioning

The only tool that turns Khmer financial documents into verified, analyst-approved spreadsheets — with every uncertain cell flagged and linked back to the exact spot on the page.

## Brand Personality

A high-agency command deck: crisp, fluid, and laser-focused. Precision is our aesthetic. We build for professional speed and tactical focus, blending government-grade reliability with the ultra-responsive, spatial layout of modern developer tools (like Linear or Raycast). The interface feels like a finely tuned instrument—satisfying to use, deeply structured, and visually engaging.

## Anti-references

- **Bloated marketing dashboards:** Heavy gradients, useless hero illustrations, and empty card grids that waste screen real estate.
- **Sterile legacy portals:** Cramped, flat tables, tiny unreadable type, and un-nested layout sheets that feel static and rigid.
- **Playful consumer apps:** Rounded-everything, emoji-heavy, mascot energy.

## Design Principles

1. **Trust is the product.** Every visual signal (confidence tints, diff view, verify marks) must be honest and calibrated — never decorate, never reassure falsely.
2. **Dynamic Workspace Layout.** The analyst is in a multi-step flow (Upload → Run → Export). The UI should actively adapt to these states—collapsing sidebars, pulling up resizable side-by-side comparison split views, and shifting panels smoothly to maximize the data density currently in focus.
3. **Khmer text is first-class.** Stacked subscripts and diacritics render with room to breathe (dedicated face, generous line-height, adjustable size) — legibility is non-negotiable.
4. **Interactive Ground Truth.** Review always links back to the source region on the page. Hover, click, and keyboard focus states must feel immediate, tactile, and highly responsive.
5. **Polished Command Density.** High-density layout using spatial depth (subtle dark/light layer borders, modern translucent panel blurs, clean keyboard shortcut tags) to organize complex data without visual noise.

## Accessibility & Inclusion

Solid basics as the working bar: body text ≥4.5:1 contrast, full keyboard operability (shortcuts are accelerators, never the only path), `prefers-reduced-motion` respected on all animation. No formal WCAG audit targeted.

---

## AI Implementation Mandate (Crucial)

When Claude (or any AI engine) is generating UI components or modifying layouts for this workspace, you are explicitly authorized and expected to:
1. **Take structural risks:** If a view is crowded, do not just shrink the padding. Propose layout shifts (e.g., converting static headers into collapsible command bars, nesting secondary tools inside contextual fly-out drawers, or implementing adjustable split-panes).
2. **Utilize depth and polish:** Use subtle borders, layered container shadows, and keyboard command styling to make the interface feel like a premium, modern web instrument.
3. **Design for motion:** Use snappy, hardware-accelerated micro-transitions (scale-downs on click, smooth drawer slide-ins, container height animations) to make the workspace feel alive and responsive.