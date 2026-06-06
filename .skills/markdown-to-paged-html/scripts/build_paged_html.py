#!/usr/bin/env python3
"""Build a clean vertical paginated HTML report from a Markdown file."""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Page:
    title: str
    level: int
    body: list[str] = field(default_factory=list)
    chapter_context: str = ""


THEMES = {
    "report": {
        "bg": "#eef1f4",
        "paper": "#ffffff",
        "ink": "#172026",
        "muted": "#66727d",
        "line": "#dce3e8",
        "accent": "#0f766e",
        "accent2": "#b42318",
        "soft": "#f6f8f9",
    },
    "ink": {
        "bg": "#f4f5f5",
        "paper": "#ffffff",
        "ink": "#171717",
        "muted": "#626262",
        "line": "#deded8",
        "accent": "#2563eb",
        "accent2": "#0f766e",
        "soft": "#f7f7f4",
    },
    "warm": {
        "bg": "#f2f0ea",
        "paper": "#fffdf8",
        "ink": "#23201c",
        "muted": "#6f6a61",
        "line": "#ded6c8",
        "accent": "#8b5e34",
        "accent2": "#3f6c51",
        "soft": "#faf7ef",
    },
}


def detect_language(text: str) -> str:
    return "zh-CN" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"&lt;(https?://[^&]+)&gt;", r'<a href="\1">\1</a>', escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^\s*](?:[^*\n]*[^\s*])?)\*", r"<em>\1</em>", escaped)
    return escaped


def plain_text(markup: str) -> str:
    return re.sub(r"\s+", " ", strip_tags(markup)).strip()


def render_blocks(lines: list[str]) -> str:
    parts: list[str] = []
    paragraph: list[str] = []
    code_lines: list[str] = []
    in_code = False
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(item.strip() for item in paragraph)
            parts.append(f"<p>{inline_markdown(text)}</p>")
            paragraph = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                code_lines = []
                in_code = True
            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)), 6)
            parts.append(f'<h{level} class="md-h{level}">{inline_markdown(heading.group(2).strip())}</h{level}>')
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            if rows:
                parts.append(render_table(rows))
            continue

        unordered = re.match(r"^[-*+]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if unordered or ordered:
            flush_paragraph()
            is_ordered = bool(ordered)
            tag = "ol" if is_ordered else "ul"
            items: list[str] = []
            while index < len(lines):
                item_line = lines[index].strip()
                match = re.match(r"^\d+[.)]\s+(.+)$", item_line) if is_ordered else re.match(
                    r"^[-*+]\s+(.+)$", item_line
                )
                if not match:
                    break
                items.append(match.group(1))
                index += 1
            parts.append(f"<{tag}>" + "".join(f"<li>{inline_markdown(item)}</li>" for item in items) + f"</{tag}>")
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            parts.append(f"<blockquote>{inline_markdown(stripped.lstrip('>').strip())}</blockquote>")
            index += 1
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    if in_code and code_lines:
        parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(parts)


def render_table(rows: list[list[str]]) -> str:
    rendered_rows = []
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        rendered_rows.append("<tr>" + "".join(f"<{tag}>{inline_markdown(cell)}</{tag}>" for cell in row) + "</tr>")
    return '<div class="table-wrap"><table>' + "\n".join(rendered_rows) + "</table></div>"


def parse_markdown(markdown: str) -> tuple[str, list[str], list[Page], list[tuple[str, str]]]:
    title = ""
    lead: list[str] = []
    pages: list[Page] = []
    current: Page | None = None
    in_fence = False

    def push_current() -> None:
        nonlocal current
        if current is not None:
            pages.append(current)
            current = None

    for line in markdown.splitlines():
        stripped = line.strip()
        heading = None if in_fence else re.match(r"^(#{1,3})\s+(.+?)\s*$", line)

        if heading:
            level = len(heading.group(1))
            heading_title = heading.group(2).strip()
            if level == 1 and not title:
                title = strip_tags(inline_markdown(heading_title))
            else:
                push_current()
                current = Page(heading_title, level)
        else:
            if current is None:
                lead.append(line)
            else:
                current.body.append(line)

        if stripped.startswith("```"):
            in_fence = not in_fence

    push_current()
    if not title:
        for line in markdown.splitlines():
            candidate = strip_tags(inline_markdown(line)).strip()
            if candidate:
                title = candidate[:90]
                break
    lead, pages, is_paper_notes = extract_internal_reading_verdict(lead, pages)
    cover_meta, pages = extract_cover_metadata(pages) if is_paper_notes else ([], pages)
    return title or "Markdown Report", lead, merge_empty_title_pages(pages), cover_meta


def extract_internal_reading_verdict(lead: list[str], pages: list[Page]) -> tuple[list[str], list[Page], bool]:
    """Hide process-oriented paper-reading metadata from public HTML output.

    The paper-reading skill uses a `Reading Verdict` section to track internal
    reading depth, relevance, and decisions. Keep the entire section internal.
    """

    kept: list[Page] = []
    found_verdict = False
    for page in pages:
        if page.title.strip().lower() != "reading verdict":
            kept.append(page)
            continue
        found_verdict = True
    return lead, kept, found_verdict


def extract_cover_metadata(pages: list[Page]) -> tuple[list[tuple[str, str]], list[Page]]:
    """Move a paper-reading Metadata section onto the cover."""

    kept: list[Page] = []
    metadata: list[tuple[str, str]] = []
    for page in pages:
        if page.title.strip().lower() != "metadata":
            kept.append(page)
            continue
        for line in page.body:
            match = re.match(r"\s*[-*+]\s*([^:]+)\s*:\s*(.+)\s*$", line)
            if match:
                metadata.append((match.group(1).strip(), match.group(2).strip()))
    return metadata, kept


def merge_empty_title_pages(pages: list[Page]) -> list[Page]:
    merged: list[Page] = []
    current_chapter = ""
    for page in pages:
        body_markup = render_blocks(page.body)
        empty_body = len(re.sub(r"\s+", "", plain_text(body_markup))) == 0
        if empty_body:
            if page.level <= 2:
                current_chapter = page.title
            continue
        if page.level <= 2:
            current_chapter = page.title
        elif current_chapter and not page.chapter_context:
            page.chapter_context = current_chapter
        merged.append(page)
    return merged


def page_weight(page: Page) -> int:
    """Rough visual weight used for compact pagination.

    The goal is not exact browser layout; it is to avoid sparse one-heading pages
    while keeping table/code-heavy pages from becoming unwieldy.
    """

    nonempty_lines = [line for line in page.body if line.strip()]
    tables = sum(1 for line in page.body if line.strip().startswith("|"))
    code_fences = sum(1 for line in page.body if line.strip().startswith("```"))
    list_items = sum(1 for line in page.body if re.match(r"\s*(?:[-*+]|\d+[.)])\s+", line))
    headings = sum(1 for line in page.body if re.match(r"\s*#{1,6}\s+", line))
    return (
        max(1, len(plain_text(render_blocks(page.body))) // 95)
        + len(nonempty_lines) // 5
        + tables * 2
        + code_fences * 3
        + list_items // 4
        + headings
        + 3
    )


def merge_page_group(title: str, pages: list[Page]) -> Page:
    primary = pages[0]
    merged_level = 2 if primary.chapter_context and title == primary.chapter_context else primary.level
    merged_context = "" if primary.chapter_context == title else primary.chapter_context
    merged = Page(title=title, level=merged_level, chapter_context=merged_context)
    for page in pages:
        if page.title != title:
            # Render as a stable in-page subsection instead of a fresh page title.
            merged.body.append(f"{'#' * page.level} {page.title}")
        merged.body.extend(page.body)
        merged.body.append("")
    return merged


def compact_pages(pages: list[Page], max_pages: int | None = None) -> list[Page]:
    if not pages:
        return pages

    # Total pages includes the cover; the compact target applies to content pages.
    target_content_pages = max(1, max_pages - 1) if max_pages and max_pages > 1 else None
    default_limit = 52
    target_limit = max(default_limit, sum(page_weight(page) for page in pages) // target_content_pages + 22) if target_content_pages else default_limit

    compacted: list[Page] = []
    group: list[Page] = []
    group_weight = 0
    group_title = ""
    group_context = ""

    def flush() -> None:
        nonlocal group, group_weight, group_title, group_context
        if group:
            compacted.append(merge_page_group(group_title or group[0].title, group))
        group = []
        group_weight = 0
        group_title = ""
        group_context = ""

    for page in pages:
        weight = page_weight(page)
        same_context = page.chapter_context and (page.chapter_context == group_context or page.chapter_context == group_title)
        same_chapter = bool(group and page.level >= 3 and (same_context or group[-1].level >= 3))
        short_section = page.level <= 2 and weight <= 12
        can_merge_short_section = bool(
            group
            and short_section
            and not group_context
            and not page.chapter_context
            and group_weight + weight <= target_limit
        )
        can_merge_detail = bool(group and same_chapter and group_weight + weight <= target_limit)

        if not group:
            group = [page]
            group_weight = weight
            group_title = page.chapter_context or page.title
            group_context = page.chapter_context
            continue

        if can_merge_detail or can_merge_short_section:
            group.append(page)
            group_weight += weight
            if page.chapter_context and not group_context:
                group_context = page.chapter_context
            if group_title == group[0].title and page.chapter_context:
                group_title = page.chapter_context
            continue

        flush()
        group = [page]
        group_weight = weight
        group_title = page.chapter_context or page.title
        group_context = page.chapter_context

    flush()
    return compacted


def make_cover_meta(lead: list[str], fallback: str) -> str:
    lead_markup = render_blocks(lead)
    lead_text = plain_text(lead_markup)
    if lead_text:
        return lead_text[:170]
    return fallback


def render_page(page: Page, index: int, total: int) -> str:
    title = inline_markdown(page.title)
    label = "Section" if page.level <= 2 or page.chapter_context else "Detail"
    context = (
        f'\n  <div class="chapter-context">{inline_markdown(page.chapter_context)}</div>'
        if page.chapter_context and page.chapter_context != page.title
        else ""
    )
    return f"""
<section class="page {'chapter' if page.level <= 2 else 'detail'}">
  <div class="folio">{index:02d} / {total:02d}</div>
  <div class="section-label">{label}</div>{context}
  <h2 class="page-title md-h{page.level}">{title}</h2>
  <div class="content">
{render_blocks(page.body)}
  </div>
</section>"""


def render_cover_metadata(metadata: list[tuple[str, str]]) -> str:
    if not metadata:
        return ""

    def render_value(value: str) -> str:
        code_url = re.fullmatch(r"`(https?://[^`]+)`", value)
        plain_url = re.fullmatch(r"(https?://\S+)", value)
        url = (code_url or plain_url).group(1) if code_url or plain_url else ""
        return f'<a href="{html.escape(url)}">{html.escape(url)}</a>' if url else inline_markdown(value)

    rows = "".join(
        f"\n      <dt>{inline_markdown(label)}</dt>\n      <dd>{render_value(value)}</dd>"
        for label, value in metadata
    )
    return f'\n    <dl class="cover-meta">{rows}\n    </dl>'


def render_html(
    title: str,
    lead: list[str],
    pages: list[Page],
    theme_name: str,
    cover_metadata: list[tuple[str, str]],
) -> str:
    theme = THEMES[theme_name]
    total = len(pages) + 1
    meta = make_cover_meta(lead, "")
    lang = detect_language(title + " " + meta)
    page_html = "\n".join(render_page(page, index, total) for index, page in enumerate(pages, start=2))
    subtitle = f'\n    <p class="subtitle">{inline_markdown(meta)}</p>' if meta and not cover_metadata else ""
    metadata = render_cover_metadata(cover_metadata)
    cover = f"""
<section class="page cover">
  <div class="folio">01 / {total:02d}</div>
  <div class="cover-body">
    <div class="eyebrow">Markdown Report</div>
    <h1>{inline_markdown(title)}</h1>{subtitle}{metadata}
  </div>
</section>"""
    return f"""<!doctype html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{report_css(theme)}</style>
</head>
<body>
  <main class="report">
{cover}
{page_html}
  </main>
</body>
</html>
"""


def report_css(theme: dict[str, str]) -> str:
    return f"""
:root {{
  --bg: {theme['bg']};
  --paper: {theme['paper']};
  --ink: {theme['ink']};
  --muted: {theme['muted']};
  --line: {theme['line']};
  --accent: {theme['accent']};
  --accent-2: {theme['accent2']};
  --soft: {theme['soft']};
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif;
  line-height: 1.55;
}}
a {{ color: var(--accent); text-decoration: none; border-bottom: 1px solid color-mix(in srgb, var(--accent) 28%, transparent); }}
img {{ max-width: 100%; height: auto; }}
code {{
  padding: 0.06rem 0.28rem;
  border: 1px solid var(--line);
  background: var(--soft);
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em;
}}
.report {{
  display: grid;
  gap: 22px;
  padding: 28px 16px 44px;
}}
.page {{
  position: relative;
  width: min(920px, 100%);
  min-height: 1080px;
  margin: 0 auto;
  padding: 62px 64px 58px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 6px;
  box-shadow: 0 18px 48px rgba(23, 32, 38, 0.10);
  overflow: hidden;
}}
.page::before {{
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 7px;
  background: linear-gradient(90deg, var(--accent), #2f5f98 55%, var(--accent-2));
}}
.folio {{
  position: absolute;
  top: 24px;
  right: 36px;
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
}}
.chapter-context {{
  margin-top: 10px;
  color: var(--muted);
  font-size: 0.96rem;
  font-weight: 680;
}}
.eyebrow, .section-label {{
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 760;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}}
.cover {{
  text-align: left;
}}
.cover-body {{
  position: absolute;
  top: 50%;
  left: 64px;
  width: calc(100% - 128px);
  transform: translateY(-50%);
}}
h1, h2, h3, h4 {{
  margin: 0;
  letter-spacing: 0;
  color: var(--ink);
}}
h1 {{
  max-width: 680px;
  margin-top: 20px;
  font-size: 3.6rem;
  line-height: 1.05;
  font-weight: 780;
}}
.page-title {{
  max-width: 720px;
  margin-top: 12px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--line);
  line-height: 1.2;
}}
.md-h2 {{
  font-size: 1.85rem;
  line-height: 1.2;
  font-weight: 760;
}}
.content .md-h2 {{
  margin: 22px 0 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
}}
.md-h3 {{
  margin: 20px 0 8px;
  font-size: 1.05rem;
  line-height: 1.35;
  font-weight: 760;
  color: var(--accent);
}}
.md-h4 {{
  margin: 16px 0 6px;
  font-size: 1rem;
  color: var(--accent);
  font-weight: 760;
}}
.subtitle {{
  max-width: 640px;
  margin: 22px 0 0;
  color: var(--muted);
  font-size: 1.14rem;
}}
.cover-meta {{
  display: grid;
  grid-template-columns: 112px minmax(0, 1fr);
  gap: 7px 16px;
  max-width: 700px;
  margin: 30px 0 0;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.91rem;
  line-height: 1.42;
}}
.cover-meta dt {{
  color: var(--accent);
  font-size: 0.73rem;
  font-weight: 760;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.cover-meta dd {{
  min-width: 0;
  margin: 0;
}}
.cover-meta a {{
  overflow-wrap: anywhere;
}}
.content {{
  margin-top: 20px;
  font-size: 0.95rem;
}}
p {{
  margin: 0 0 11px;
  color: color-mix(in srgb, var(--ink) 88%, white);
}}
ul, ol {{
  margin: 7px 0 13px;
  padding-left: 1.2rem;
}}
li {{ margin: 4px 0; }}
blockquote {{
  margin: 24px 0;
  padding: 18px 22px;
  border-left: 4px solid var(--accent);
  background: var(--soft);
  color: var(--ink);
  font-size: 1.03rem;
}}
.table-wrap {{
  width: 100%;
  overflow-x: auto;
  margin: 11px 0 17px;
  border: 1px solid var(--line);
  border-radius: 6px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  line-height: 1.36;
}}
th, td {{
  padding: 8px 9px;
  vertical-align: top;
  text-align: left;
  border-bottom: 1px solid var(--line);
}}
th {{
  color: var(--accent);
  background: var(--soft);
  font-weight: 760;
}}
tr:last-child td {{ border-bottom: 0; }}
pre code {{
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  font-size: inherit;
}}
pre {{
  margin: 11px 0 16px;
  padding: 13px 14px;
  white-space: pre-wrap;
  word-break: break-word;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #111827;
  color: #eef2f7;
  font-size: 0.82rem;
  line-height: 1.42;
}}
@media print {{
  @page {{ size: A4 portrait; margin: 0; }}
  body {{ background: #fff; }}
  .report {{ display: block; padding: 0; }}
  .page {{
    width: 210mm;
    min-height: 297mm;
    margin: 0;
    border: 0;
    border-radius: 0;
    box-shadow: none;
    page-break-after: always;
  }}
}}
@media (max-width: 720px) {{
  .report {{ padding: 14px 10px 32px; gap: 18px; }}
  .page {{ min-height: auto; padding: 58px 24px 36px; }}
  .cover-body {{ left: 24px; width: calc(100% - 48px); }}
  .cover-meta {{ grid-template-columns: 1fr; gap: 3px; margin-top: 24px; }}
  .cover-meta dd {{ margin-bottom: 5px; }}
  .folio {{ top: 22px; right: 24px; }}
  h1 {{ font-size: 2.55rem; }}
  .md-h2 {{ font-size: 1.55rem; }}
  .md-h3 {{ font-size: 1rem; }}
  table {{ font-size: 0.82rem; }}
  th, td {{ padding: 9px 10px; }}
}}
"""


def build(input_path: Path, output_path: Path, title_override: str | None, max_pages: int | None, theme: str) -> None:
    markdown = input_path.read_text(encoding="utf-8")
    title, lead, pages, cover_metadata = parse_markdown(markdown)
    if title_override:
        title = title_override
    pages = compact_pages(pages, max_pages)
    output_path.write_text(render_html(title, lead, pages, theme, cover_metadata), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source Markdown file")
    parser.add_argument("--output", "-o", type=Path, help="Output HTML file")
    parser.add_argument("--title", help="Override the generated title")
    parser.add_argument("--max-pages", type=int, default=0, help="Target total pages including cover; 0 means automatic compact pagination")
    parser.add_argument("--max-slides", type=int, help="Deprecated alias for --max-pages")
    parser.add_argument("--theme", choices=sorted(THEMES), default="report", help="Visual theme")
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".html")
    max_pages = args.max_pages or args.max_slides or 0
    build(args.input, output, args.title, max_pages, args.theme)
    print(output)


if __name__ == "__main__":
    main()
