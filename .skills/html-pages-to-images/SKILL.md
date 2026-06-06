---
name: html-pages-to-images
description: Export paginated or presentation-style HTML files into one image per page or slide, saved in a separate output folder. Use when Codex needs to render browser-viewable HTML decks, slide pages, paged reports, or .html presentations as PNG/JPEG image sequences for sharing, thumbnails, visual QA, or downstream conversion.
---

# HTML Pages To Images

## Overview

Use this skill to render a paginated HTML file and save each page as a separate image. It is designed for self-contained HTML reports where pages are represented by repeated `.page` elements, with automatic fallback for older deck-style `.slide` elements.

## Workflow

1. Identify the source HTML file and choose an output directory. Default to `<html-name>_images` beside the source file.
2. Use `scripts/export_html_pages.js` to render the file in Chromium and capture each page.
3. Prefer PNG for visual fidelity. Use JPEG only when the user asks for smaller files.
4. Review the generated folder and spot-check at least the first and last image when visual quality matters.

## Quick Start

From this skill directory:

```bash
node scripts/export_html_pages.js /path/to/presentation.html
```

Useful options:

```bash
node scripts/export_html_pages.js /path/to/presentation.html --output-dir /path/to/images
node scripts/export_html_pages.js /path/to/report.html --selector ".page" --width 1080 --height 1440 --scale 2
node scripts/export_html_pages.js /path/to/report.html --selector ".page" --no-clip
node scripts/export_html_pages.js /path/to/presentation.html --selector ".slide" --width 1440 --height 900 --scale 2
node scripts/export_html_pages.js /path/to/presentation.html --format jpeg --quality 92
```

Output files are named `page-001.png`, `page-002.png`, and so on.

## Dependency

The script requires Node.js and Playwright:

```bash
npm install playwright
npx playwright install chromium
```

If Playwright is already available in the project or workspace, use it directly. If it is missing, install it before running the export script.

## Capture Rules

- Default page selector is `.page`.
- The script captures one page at a time by temporarily hiding sibling pages and marking the current page active.
- If the default `.page` selector finds nothing, the script automatically falls back to `.slide`.
- It hides common presentation controls (`.controls`, `nav[aria-label*="控制"]`, `nav[aria-label*="Slide"]`) unless `--show-controls` is passed.
- By default, it screenshots only the active page element, equivalent to `--clip-selector` using the resolved page selector (`.page` or fallback `.slide`).
- Pass `--no-clip` to capture the full browser viewport instead, preserving report/deck background and framing.
- Pass `--clip-selector "<selector>"` to capture a different element.
- If no matching pages are found, it captures the document body once as `page-001`.

## Quality Bar

- Use a stable viewport such as `1440x900` for landscape decks or `1080x1440` for portrait decks.
- Use `--scale 2` for crisp images unless file size is a concern.
- Keep generated images in their own folder; do not scatter screenshots beside source assets.
- Re-run with a larger width/height if text wraps differently than in the browser.

## Resource

- `scripts/export_html_pages.js`: Uses Playwright to capture each HTML page/slide as an image sequence.
