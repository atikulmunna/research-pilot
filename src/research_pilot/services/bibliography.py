"""Deterministic citation formatting (a code task, no model call)."""

from typing import Iterable, List

from ..state.models import Paper


def format_authors(authors: List[str]) -> str:
    names = [a for a in authors if a]
    if not names:
        return "Unknown authors"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."


def format_reference(paper: Paper) -> str:
    year = paper.year or "n.d."
    venue = f" *{paper.venue}*." if paper.venue else ""
    if paper.doi:
        link = f" https://doi.org/{paper.doi}"
    elif paper.arxiv_id:
        link = f" https://arxiv.org/abs/{paper.arxiv_id}"
    else:
        link = f" {paper.url}" if paper.url else ""
    synthetic = " [synthetic placeholder source]" if paper.synthetic else ""
    return f"[{paper.id}] {format_authors(paper.authors)} ({year}). {paper.title}.{venue}{link}{synthetic}"


def bibliography(papers: Iterable[Paper]) -> List[str]:
    return [format_reference(p) for p in sorted(papers, key=lambda p: p.id)]
