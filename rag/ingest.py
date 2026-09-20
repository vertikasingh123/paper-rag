"""PDF -> clean text -> chunks that still know what page they came from.

Page provenance is the whole point. An answer that says "the ablation in
Table 3 (p. 7)" is checkable; an answer with no page is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import RAGConfig

# A heading looks like "3", "3.1", "IV." followed by a short Title Case phrase,
# or one of the standard unnumbered section names.
_NUMBERED_HEADING = re.compile(
    r"^\s*((?:\d+|[IVXLC]+)(?:\.\d+)*\.?)\s+([A-Z][^.!?]{2,60})\s*$"
)
_NAMED_HEADING = re.compile(
    r"^\s*(abstract|introduction|related work|background|methods?|methodology|"
    r"materials and methods|experiments?|results?|discussion|conclusions?|"
    r"limitations|acknowledgements?|references|bibliography|appendix)\s*$",
    re.IGNORECASE,
)
_REFERENCES_HEADING = re.compile(
    r"^\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(references|bibliography|works cited)\s*$",
    re.IGNORECASE,
)


@dataclass
class Page:
    number: int  # 1-indexed, as printed in a PDF reader
    text: str


@dataclass
class Chunk:
    id: int
    text: str
    pages: list[int] = field(default_factory=list)
    section: str | None = None

    @property
    def page_label(self) -> str:
        if not self.pages:
            return "?"
        if len(self.pages) == 1:
            return f"p. {self.pages[0]}"
        return f"pp. {self.pages[0]}-{self.pages[-1]}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "pages": self.pages,
            "section": self.section,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(id=d["id"], text=d["text"], pages=d["pages"], section=d.get("section"))


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def extract_pages(pdf_path: str | Path) -> list[Page]:
    """Pull raw text out of each page, keeping the page number."""
    reader = PdfReader(str(pdf_path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append(Page(number=i, text=raw))
    return pages


def clean_text(text: str, dehyphenate: bool = True) -> str:
    """Undo the damage PDF layout does to a paragraph.

    Two-column academic PDFs come out with a hard newline at the end of every
    visual line, and words split across lines with a hyphen. Left alone, both
    wreck embedding quality, because the chunk text stops looking like prose.
    """
    if dehyphenate:
        # "hyphen-\nation" -> "hyphenation", but keep the hyphen when the
        # second fragment starts with a capital or a digit ("non-Gaussian",
        # "COVID-19"), where the hyphen is part of the word rather than an
        # artefact of line wrapping.
        text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)
        text = re.sub(r"(\w)-\s*\n\s*([A-Z0-9])", r"\1-\2", text)

    out: list[str] = []
    # Parallel flags: a line we must never append wrapped text onto, because
    # doing so swallows the heading into the paragraph below it and destroys
    # both the section labels and the chunk boundaries.
    standalone: list[bool] = []

    for line in text.split("\n"):
        stripped = line.rstrip()
        if not stripped:
            out.append("")
            standalone.append(True)
            continue
        # Headings and short lines (list items, table rows, captions) stay put.
        if _looks_like_heading(stripped) or len(stripped) < 40:
            out.append(stripped)
            standalone.append(True)
            continue
        # A long line that does not end a sentence is a wrapped line: join it
        # back onto the paragraph it belongs to.
        if out and out[-1] and not standalone[-1] and not _ends_sentence(out[-1]):
            out[-1] = out[-1] + " " + stripped.lstrip()
        else:
            out.append(stripped)
            standalone.append(False)

    joined = "\n".join(out)
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def _ends_sentence(line: str) -> bool:
    return bool(re.search(r"[.!?:;]\"?\s*$", line))


def _looks_like_heading(line: str) -> bool:
    return bool(_NUMBERED_HEADING.match(line) or _NAMED_HEADING.match(line))


def _heading_text(line: str) -> str:
    m = _NUMBERED_HEADING.match(line)
    if m:
        return f"{m.group(1).rstrip('.')} {m.group(2).strip()}"
    m = _NAMED_HEADING.match(line)
    if m:
        return m.group(1).strip().title()
    return line.strip()


def truncate_at_references(pages: list[Page]) -> list[Page]:
    """Cut the document at the bibliography.

    Searches from the back so a forward reference in the body ("see References")
    does not trigger it, and refuses to cut if that would drop more than half
    the paper (which usually means a false positive).
    """
    total_chars = sum(len(p.text) for p in pages)
    for page in reversed(pages):
        lines = page.text.split("\n")
        for idx, line in enumerate(lines):
            if not _REFERENCES_HEADING.match(line):
                continue
            kept_before = sum(len(p.text) for p in pages if p.number < page.number)
            kept_here = len("\n".join(lines[:idx]))
            if total_chars and (kept_before + kept_here) / total_chars < 0.5:
                return pages  # suspicious - leave the document alone
            out = [p for p in pages if p.number < page.number]
            head = "\n".join(lines[:idx]).strip()
            if head:
                out.append(Page(number=page.number, text=head))
            return out
    return pages


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def _flatten(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join pages into one string and record (start, end, page_number) spans."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    sep = "\n\n"
    for page in pages:
        text = page.text
        spans.append((cursor, cursor + len(text), page.number))
        parts.append(text)
        cursor += len(text) + len(sep)
    return sep.join(parts), spans


def _pages_for_span(start: int, end: int, spans) -> list[int]:
    if not spans:
        return []
    hits = [num for (s, e, num) in spans if start < e and end > s]
    return hits if hits else [spans[0][2]]


def _section_index(document: str) -> list[tuple[int, str]]:
    """Offsets of every detected heading, so chunks can be labelled."""
    index: list[tuple[int, str]] = []
    offset = 0
    for line in document.split("\n"):
        if _looks_like_heading(line):
            index.append((offset, _heading_text(line)))
        offset += len(line) + 1
    return index


def _section_for(offset: int, index: list[tuple[int, str]]) -> str | None:
    current = None
    for pos, name in index:
        if pos <= offset:
            current = name
        else:
            break
    return current


def chunk_pages(pages: list[Page], config: RAGConfig) -> list[Chunk]:
    """Clean, flatten and split the document, carrying page + section labels."""
    cleaned = [
        Page(number=p.number, text=clean_text(p.text, config.dehyphenate)) for p in pages
    ]
    cleaned = [p for p in cleaned if p.text.strip()]
    if config.drop_references:
        cleaned = truncate_at_references(cleaned)
    if not cleaned:
        return []

    document, spans = _flatten(cleaned)
    sections = _section_index(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    pieces = splitter.split_text(document)

    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        start = _locate(document, piece, cursor)
        end = start + len(piece)
        cursor = max(cursor, start + 1)
        chunks.append(
            Chunk(
                id=i,
                text=piece.strip(),
                pages=_pages_for_span(start, end, spans),
                section=_section_for(start, sections),
            )
        )
    return _merge_stragglers([c for c in chunks if c.text], config.min_chunk_chars)


def _merge_stragglers(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Fold tiny leftover chunks into their predecessor and renumber."""
    if not chunks:
        return []
    merged: list[Chunk] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk.text) < min_chars and merged:
            previous = merged[-1]
            previous.text = f"{previous.text} {chunk.text}".strip()
            previous.pages = sorted(set(previous.pages) | set(chunk.pages))
        else:
            merged.append(chunk)
    for new_id, chunk in enumerate(merged):
        chunk.id = new_id
    return merged


def _locate(document: str, piece: str, cursor: int) -> int:
    """Find where a chunk sits in the document.

    The splitter strips whitespace, so an exact find can miss. Fall back to a
    distinctive prefix, then to the cursor, rather than mislabelling the page.
    """
    idx = document.find(piece, cursor)
    if idx != -1:
        return idx
    probe = piece[:60].strip()
    if probe:
        idx = document.find(probe, cursor)
        if idx != -1:
            return idx
        idx = document.find(probe)
        if idx != -1:
            return idx
    return cursor


def load_pdf(pdf_path: str | Path, config: RAGConfig) -> list[Chunk]:
    """One-call convenience: PDF path in, chunks out."""
    return chunk_pages(extract_pages(pdf_path), config)
