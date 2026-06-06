---
name: markdown-to-paged-html
description: Convert Markdown notes, reports, research summaries, plans, meeting records, or long-form documents into a clean self-contained vertical paginated HTML report with A4-like portrait pages, readable tables, code blocks, and print-friendly styling.
---

# Markdown To Paged HTML

## Overview

Use this skill to convert Markdown into a polished vertical paginated HTML report. The default output should feel like a compact, calm reading document or printable report, not a horizontal PPT deck.

The bundled script preserves the Markdown structure instead of aggressively summarizing it. Tables remain tables, code blocks remain code blocks, links remain clickable, and empty section-title pages are merged into the following content page as a small chapter context label. Treat the script output as a first pass: for long reports, refine the generated HTML so pages are information-dense without becoming overlong.

## Workflow

1. Read the source Markdown and identify the title, audience, key sections, tables, code examples, checklists, and reference lists.
2. Generate the HTML with `scripts/build_paged_html.py`.
3. For long reports, inspect the generated page structure and refine the HTML when the default output creates sparse pages, inconsistent heading sizes, or pure title pages.
4. Open or inspect the generated HTML. Check especially:
   - no blank pages that only contain a section title;
   - no run of sparse pages caused by splitting every heading into its own page; merge short neighboring sections by hierarchy and shared purpose;
   - no pure chapter-title pages when the page header/chapter context already names the section;
   - same-level Markdown headings have stable visual sizes regardless of whether they are the first title on a page or a merged title inside a page;
   - short same-section blocks such as `P0`/`P1` task tables, short checklists, and compact reading lists are merged onto the same page when this does not make the page too long;
   - code blocks have readable contrast, with no nested inline-code background inside dark blocks;
   - long tables scroll horizontally on small screens rather than breaking the layout;
   - page numbers are correct after empty-section merging.
5. Manually refine when the source needs narrative restructuring, custom visual treatment, or compact report pagination beyond the script default.

## Quick Start

Run the bundled script from this skill directory:

```bash
python3 scripts/build_paged_html.py input.md --output output.html
```

Useful options:

```bash
python3 scripts/build_paged_html.py input.md --output output.html --title "Quarterly Review"
python3 scripts/build_paged_html.py input.md --output output.html --theme ink
python3 scripts/build_paged_html.py input.md --output output.html --max-pages 24
```

The script uses only Python standard library modules. It writes a self-contained HTML file with embedded CSS and no external scripts, fonts, or network assets.

`--max-pages` is a target page budget for compact pagination, not a truncation limit. The script should preserve all source content and merge short neighboring sections more aggressively when a target is provided.

## Output Style

- Vertical A4-like pages stacked in a scrollable report.
- Cover page with title and compact overview/metadata from the Markdown lead-in.
- For paper-reading notes, move a Markdown `Metadata` section onto the cover as a compact metadata block and omit the standalone `Metadata` page.
- On the cover page, vertically center the title/subtitle text group on the page, but keep the text block itself left-aligned like a printed report cover.
- Do not use generic generator subtitles such as `A vertical paginated HTML report generated from Markdown.` or similar tool/process descriptions. The cover subtitle should be meaningful source content, or omitted when no suitable source summary exists.
- Do not use `Reading Verdict` content, including its `Relevance` item, as a paper-reading cover subtitle. Treat Verdict content as internal reading-process metadata.
- White paper surface, restrained color bar, subtle shadow, generous margins.
- Section labels and page folios in the upper area.
- Empty `##` chapter pages are merged into nearby content as chapter context instead of rendered as standalone pages.
- Do not make every heading its own page. Keep pages compact by combining short neighboring subsections within the same chapter.
- Use heading hierarchy to decide page grouping: keep substantial top-level `##` sections as page anchors, but merge short sibling `##` sections when they form one conceptual unit; merge `###` and `####` subsections under their parent by default unless their content is large enough to justify a page break.
- Preserve heading hierarchy visually: `##`, `###`, and `####` must keep distinct, stable sizes. A `####` heading must not become large just because it starts a page.
- If a page header or chapter context already shows the current section, do not add a separate page just to repeat that section title.
- Merge short `P0`/`P1` tables and bullet/checklist groups under the same parent section when the resulting page remains readable.
- Split pages only when content would become visually overlong, especially around large tables, long code blocks, or substantial project descriptions.
- Inline code gets a light background; fenced code blocks get one dark background and reset nested `code` styling so text remains legible.
- Print CSS uses portrait A4 pages.

## Quality Bar

- Preserve important numbers, dates, names, constraints, decisions, risks, links, code examples, and next steps.
- For paper-reading notes that contain a `Reading Verdict` section, treat the full section as internal reading metadata by default: do not render `Depth reached`, `Decision`, `Relevance`, pass labels, or other process-oriented judgment in the public-facing body or cover.
- For paper-reading notes that contain a `Metadata` section, render that metadata compactly on the cover and remove the standalone metadata page.
- Preserve tables when they are readable; wrap them in horizontal scroll containers for narrow screens.
- Use direct page titles from the Markdown headings, but keep visual size tied to the original Markdown level, not to page position.
- Avoid sparse pages: combine short tables, lists, and small subsections where they share a chapter or parent section. If multiple consecutive pages each contain only a short list, a small table, or a few paragraphs, merge them according to heading hierarchy and rename the combined page title to reflect the grouped content.
- Avoid overlong pages: if a page contains multiple large tables/code blocks or feels taller than the A4-like page intent, split at a natural heading.
- Keep the output self-contained: no external CSS, fonts, scripts, or network assets.
- Avoid slide controls, horizontal deck navigation, and PPT-like full-screen slide behavior unless the user explicitly asks for that.

## Resource

- `scripts/build_paged_html.py`: Converts Markdown into a self-contained vertical paginated HTML report.
