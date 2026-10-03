"""Manuscript export: markdown to standalone HTML or a basic text PDF."""

import html
import re
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

_LINK = re.compile(r"\[([^\]]+)\]\(([A-Za-z][A-Za-z0-9+.-]*://[^)\s]+)\)")
_URL = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://[^\s<]+)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
_CODE = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_UNORDERED = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^\s*\d+\.\s+(.*)$")
_CITATION_LINE = re.compile(r"^\[(\d+|P\d+)\]\s+")


def markdown_to_html(markdown: str, title: str) -> str:
    body = render_markdown_body(markdown or "")
    safe_title = html.escape(title or "Manuscript")
    return (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{safe_title}</title><style>"
        "body{font-family:Georgia,'Times New Roman',serif;max-width:860px;margin:2.5rem auto;padding:0 1rem;line-height:1.65;color:#1f2328;background:#fff}"
        "h1{font-size:1.9rem;line-height:1.25;margin:0 0 1rem}h2{margin:2rem 0 .6rem;font-size:1.35rem}h3{margin:1.4rem 0 .5rem}"
        "p{margin:.6rem 0}ul,ol{margin:.4rem 0 .9rem 1.3rem}li{margin:.2rem 0}"
        "code{background:#f3f4f6;border-radius:4px;padding:.1rem .3rem;font-family:Consolas,monospace;font-size:.9em}"
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:6px}"
        "blockquote{border-left:4px solid #d0d7de;margin:.8rem 0;padding:.2rem .9rem;color:#57606a;background:#fafbfc}"
        "table{border-collapse:collapse;margin:1rem 0;font-family:system-ui,sans-serif;font-size:.88rem;width:100%}"
        "th,td{border:1px solid #d0d7de;padding:.35rem .55rem;text-align:left;vertical-align:top}th{background:#f6f8fa}"
        "a{color:#0a58ca;word-break:break-word}.citation{font-size:.92rem}"
        f"</style></head><body>{body}</body></html>"
    )


def render_markdown_body(markdown: str) -> str:
    chunks = [_RENDERERS.get(kind, _render_nothing)(lines) for kind, lines in _blocks(markdown.splitlines())]
    return "".join(chunks) or "<p></p>"


def _line_kind(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("```"):
        return "fence"
    if stripped.startswith("|") and stripped.endswith("|"):
        return "table"
    if not stripped:
        return "blank"
    if _HEADING.match(stripped):
        return "heading"
    if _UNORDERED.match(line):
        return "ul"
    if _ORDERED.match(line):
        return "ol"
    if stripped.startswith(">"):
        return "quote"
    if _CITATION_LINE.match(stripped):
        return "citation"
    return "paragraph"


def _blocks(lines: List[str]) -> Iterator[Tuple[str, List[str]]]:
    """Group lines into (kind, lines) blocks; consecutive lines of a grouping kind share a block."""
    kind, block = "", []
    for raw in lines:
        line = raw.rstrip()
        line_kind = _line_kind(line)
        if kind == "code":
            if line_kind == "fence":
                yield "code", block
                kind, block = "", []
            else:
                block.append(line)
            continue
        if block and (line_kind != kind or kind not in _GROUPED):
            yield kind, block
            block = []
        kind = "code" if line_kind == "fence" else line_kind
        if line_kind != "fence":
            block.append(line)
    if block:
        yield kind, block


def _render_nothing(lines: List[str]) -> str:
    return ""


def _render_code(lines: List[str]) -> str:
    return f"<pre><code>{html.escape(chr(10).join(lines))}</code></pre>"


def _render_paragraph(lines: List[str]) -> str:
    return f"<p>{format_inline(' '.join(line.strip() for line in lines).strip())}</p>"


def _render_list(kind: str, pattern: re.Pattern) -> Callable[[List[str]], str]:
    def render(lines: List[str]) -> str:
        items = "".join(f"<li>{format_inline(pattern.match(line).group(1).strip())}</li>" for line in lines)
        return f"<{kind}>{items}</{kind}>"

    return render


def _render_table(lines: List[str]) -> str:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "--") for c in r)]
    if not rows:
        return ""
    head = "".join(f"<th>{format_inline(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{format_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_heading(lines: List[str]) -> str:
    heading = _HEADING.match(lines[0].strip())
    level = len(heading.group(1))
    return f"<h{level}>{format_inline(heading.group(2).strip())}</h{level}>"


def _render_quote(lines: List[str]) -> str:
    return f"<blockquote>{format_inline(lines[0].strip()[1:].strip())}</blockquote>"


def _render_citation(lines: List[str]) -> str:
    return f"<p class='citation'>{format_inline(lines[0].strip())}</p>"


_GROUPED = {"paragraph", "ul", "ol", "table", "blank"}
_RENDERERS: Dict[str, Callable[[List[str]], str]] = {
    "code": _render_code,
    "paragraph": _render_paragraph,
    "ul": _render_list("ul", _UNORDERED),
    "ol": _render_list("ol", _ORDERED),
    "table": _render_table,
    "heading": _render_heading,
    "quote": _render_quote,
    "citation": _render_citation,
}


def format_inline(text: str) -> str:
    if not text:
        return ""
    parts: List[str] = []
    last = 0
    for match in _LINK.finditer(text):
        parts.append(_inline_markup(text[last : match.start()]))
        url = html.escape(match.group(2), quote=True)
        parts.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{html.escape(match.group(1))}</a>')
        last = match.end()
    parts.append(_inline_markup(text[last:]))
    return "".join(parts)


def _inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = _CODE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC.sub(r"<em>\1</em>", escaped)

    def link(match: re.Match) -> str:
        url, trailer = match.group(1), ""
        while url and url[-1] in ".,);]":
            trailer = url[-1] + trailer
            url = url[:-1]
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>{trailer}'

    return _URL.sub(link, escaped)


def markdown_to_pdf(path: Path, markdown: str, title: str) -> None:
    lines = [title or "Manuscript", ""]
    for line in (markdown or "").splitlines():
        text = line.replace("\t", "    ").strip()
        lines.extend(_wrap(text, 95) if text else [""])
    pages = [lines[i : i + 52] for i in range(0, len(lines), 52)] or [[""]]
    path.write_bytes(_build_pdf(pages))


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    rows: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            rows.append(current)
            current = word
    rows.append(current)
    return rows


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf(pages: List[List[str]]) -> bytes:
    objects = {1: b"<< /Type /Catalog /Pages 2 0 R >>"}
    page_ids = [3 + 2 * i for i in range(len(pages))]
    font_id = 3 + 2 * len(pages)
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode()
    for page_id, page_lines in zip(page_ids, pages):
        stream_lines = ["BT", "/F1 11 Tf", "50 770 Td", "14 TL"]
        for line in page_lines:
            stream_lines.extend([f"({_pdf_escape(line)}) Tj", "T*"])
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1")
        objects[page_id + 1] = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {page_id + 1} 0 R >>"
        ).encode()
    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    doc = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for obj_id in range(1, font_id + 1):
        offsets[obj_id] = len(doc)
        doc.extend(f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n")
    xref = len(doc)
    doc.extend(f"xref\n0 {font_id + 1}\n0000000000 65535 f \n".encode())
    for obj_id in range(1, font_id + 1):
        doc.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode())
    doc.extend(f"trailer\n<< /Size {font_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(doc)
