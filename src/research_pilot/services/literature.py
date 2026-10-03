"""Academic literature search: OpenAlex, Semantic Scholar, arXiv and a mock provider.

Retrieval, deduplication and citation chasing are deterministic code (no model calls).
Every returned paper carries a traceable identifier from the provider that found it.
"""

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Protocol

import requests

from ..state.models import Paper
from ..state.research_state import paper_keys
from .parser import DocumentParser

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENALEX_FIELDS = (
    "id,doi,title,display_name,publication_year,authorships,abstract_inverted_index,"
    "cited_by_count,referenced_works,primary_location,best_oa_location,ids,type"
)
S2_FIELDS = "paperId,title,year,authors,abstract,venue,citationCount,externalIds,url,openAccessPdf"


class LiteratureProvider(Protocol):
    name: str

    def search(self, query: str, limit: int) -> List[Paper]: ...

    def references(self, paper: Paper, limit: int) -> List[Paper]: ...

    def citing(self, paper: Paper, limit: int) -> List[Paper]: ...


def reconstruct_abstract(inverted: Dict[str, List[int]] | None) -> str:
    if not inverted:
        return ""
    positions: Dict[int, str] = {}
    for word, idxs in inverted.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def _arxiv_from_doi(doi: str) -> str:
    match = re.search(r"10\.48550/arxiv\.(.+)$", doi or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


class OpenAlexProvider:
    name = "openalex"
    base = "https://api.openalex.org"

    def __init__(self, api_key: str = "", mailto: str = "", timeout: float = 30.0):
        self.api_key = api_key
        self.mailto = mailto
        self.timeout = timeout

    def _get(self, path: str, params: Dict[str, str]) -> Dict:
        query = dict(params)
        query["select"] = OPENALEX_FIELDS
        if self.api_key:
            query["api_key"] = self.api_key
        if self.mailto:
            query["mailto"] = self.mailto
        response = requests.get(f"{self.base}{path}", params=query, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, limit: int) -> List[Paper]:
        data = self._get("/works", {"search": query, "per-page": str(limit)})
        return [self.parse(row) for row in data.get("results", [])]

    def resolve_doi(self, doi: str) -> Paper | None:
        clean = doi.lower().replace("https://doi.org/", "").strip()
        data = self._get("/works", {"filter": f"doi:{clean}", "per-page": "1"})
        rows = data.get("results", [])
        return self.parse(rows[0]) if rows else None

    def references(self, paper: Paper, limit: int) -> List[Paper]:
        ids = [ref.rsplit("/", 1)[-1] for ref in paper.references[:50]]
        if not ids:
            return []
        data = self._get("/works", {"filter": "openalex_id:" + "|".join(ids), "per-page": "50", "sort": "cited_by_count:desc"})
        return [self.parse(row) for row in data.get("results", [])][:limit]

    def citing(self, paper: Paper, limit: int) -> List[Paper]:
        if not paper.openalex_id:
            return []
        short = paper.openalex_id.rsplit("/", 1)[-1]
        data = self._get("/works", {"filter": f"cites:{short}", "per-page": str(limit), "sort": "cited_by_count:desc"})
        return [self.parse(row) for row in data.get("results", [])]

    @staticmethod
    def parse(row: Dict) -> Paper:
        doi = (row.get("doi") or "").replace("https://doi.org/", "")
        primary = row.get("primary_location") or {}
        best = row.get("best_oa_location") or {}
        source = primary.get("source") or {}
        landing = primary.get("landing_page_url") or ""
        arxiv_id = _arxiv_from_doi(doi)
        if not arxiv_id and "arxiv.org/abs/" in landing:
            arxiv_id = landing.rsplit("/abs/", 1)[-1]
        return Paper(
            title=str(row.get("title") or row.get("display_name") or "").strip(),
            year=row.get("publication_year"),
            authors=[
                (a.get("author") or {}).get("display_name", "")
                for a in (row.get("authorships") or [])
                if (a.get("author") or {}).get("display_name")
            ][:12],
            venue=str(source.get("display_name") or ""),
            abstract=reconstruct_abstract(row.get("abstract_inverted_index")),
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=str(row.get("id") or ""),
            url=landing or (f"https://doi.org/{doi}" if doi else str(row.get("id") or "")),
            pdf_url=str(best.get("pdf_url") or primary.get("pdf_url") or ""),
            cited_by_count=int(row.get("cited_by_count") or 0),
            references=list(row.get("referenced_works") or []),
            source="openalex",
        )


class SemanticScholarProvider:
    name = "semantic_scholar"
    base = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = "", timeout: float = 30.0, retries: int = 2):
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    def _get(self, path: str, params: Dict[str, str]) -> Dict:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        for attempt in range(self.retries + 1):
            response = requests.get(f"{self.base}{path}", params=params, headers=headers, timeout=self.timeout)
            if response.status_code == 429 and attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        return {}

    def search(self, query: str, limit: int) -> List[Paper]:
        data = self._get("/paper/search", {"query": query, "limit": str(limit), "fields": S2_FIELDS})
        return [self.parse(row) for row in data.get("data") or []]

    def references(self, paper: Paper, limit: int) -> List[Paper]:
        if not paper.s2_id:
            return []
        data = self._get(f"/paper/{paper.s2_id}/references", {"limit": str(limit), "fields": S2_FIELDS})
        return [self.parse(row.get("citedPaper") or {}) for row in data.get("data") or [] if row.get("citedPaper")]

    def citing(self, paper: Paper, limit: int) -> List[Paper]:
        if not paper.s2_id:
            return []
        data = self._get(f"/paper/{paper.s2_id}/citations", {"limit": str(limit), "fields": S2_FIELDS})
        return [self.parse(row.get("citingPaper") or {}) for row in data.get("data") or [] if row.get("citingPaper")]

    @staticmethod
    def parse(row: Dict) -> Paper:
        external = row.get("externalIds") or {}
        pdf = row.get("openAccessPdf") or {}
        return Paper(
            title=str(row.get("title") or "").strip(),
            year=row.get("year"),
            authors=[a.get("name", "") for a in row.get("authors") or [] if a.get("name")][:12],
            venue=str(row.get("venue") or ""),
            abstract=str(row.get("abstract") or ""),
            doi=str(external.get("DOI") or ""),
            arxiv_id=str(external.get("ArXiv") or ""),
            s2_id=str(row.get("paperId") or ""),
            url=str(row.get("url") or ""),
            pdf_url=str(pdf.get("url") or ""),
            cited_by_count=int(row.get("citationCount") or 0),
            source="semantic_scholar",
        )


class ArxivProvider:
    name = "arxiv"
    base = "https://export.arxiv.org/api/query"
    min_interval_s = 3.0

    def __init__(self, timeout: float = 30.0, retries: int = 2):
        self.timeout = timeout
        self.retries = retries
        self._last_call = 0.0

    def _query(self, params: Dict[str, str]) -> List[Paper]:
        for attempt in range(self.retries + 1):
            wait = self.min_interval_s * (attempt + 1) - (time.monotonic() - self._last_call)
            if wait > 0 and self._last_call:
                time.sleep(wait)
            try:
                response = requests.get(self.base, params=params, timeout=self.timeout)
            except requests.Timeout:
                self._last_call = time.monotonic()
                if attempt < self.retries:
                    continue
                raise
            self._last_call = time.monotonic()
            if response.status_code in {429, 500, 502, 503} and attempt < self.retries:
                continue
            response.raise_for_status()
            return self.parse_feed(response.text)
        return []

    def search(self, query: str, limit: int) -> List[Paper]:
        terms = [t for t in re.findall(r"[A-Za-z0-9\-]+", query) if len(t) > 2][:6]
        search = " AND ".join(f"all:{t}" for t in terms) or f"all:{query}"
        return self._query({"search_query": search, "max_results": str(limit), "sortBy": "relevance"})

    def resolve(self, arxiv_id: str) -> Paper | None:
        rows = self._query({"id_list": arxiv_id, "max_results": "1"})
        return rows[0] if rows else None

    def references(self, paper: Paper, limit: int) -> List[Paper]:
        return []

    def citing(self, paper: Paper, limit: int) -> List[Paper]:
        return []

    @staticmethod
    def parse_feed(xml_text: str) -> List[Paper]:
        root = ET.fromstring(xml_text)
        out = []
        for entry in root.findall(f"{ATOM}entry"):
            raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
            arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else ""
            if not arxiv_id:
                continue
            pdf_url = ""
            for link in entry.findall(f"{ATOM}link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
            published = entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated") or ""
            out.append(
                Paper(
                    title=" ".join((entry.findtext(f"{ATOM}title") or "").split()),
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    authors=[" ".join((a.findtext(f"{ATOM}name") or "").split()) for a in entry.findall(f"{ATOM}author")][:12],
                    venue=(entry.findtext(f"{ARXIV_NS}journal_ref") or "arXiv").strip(),
                    abstract=" ".join((entry.findtext(f"{ATOM}summary") or "").split()),
                    doi=(entry.findtext(f"{ARXIV_NS}doi") or "").strip(),
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=pdf_url,
                    source="arxiv",
                )
            )
        return out


class MockLiteratureProvider:
    """Deterministic synthetic papers for offline runs and tests. Always flagged synthetic."""

    name = "mock"
    approaches = ("retrieval", "contrastive", "graph-based", "curriculum", "ensemble", "distillation", "probing", "benchmark")

    def search(self, query: str, limit: int) -> List[Paper]:
        return [self._paper(query, idx) for idx in range(limit)]

    def references(self, paper: Paper, limit: int) -> List[Paper]:
        return [self._paper(f"foundations of {paper.title[:40]}", idx) for idx in range(min(limit, 2))]

    def citing(self, paper: Paper, limit: int) -> List[Paper]:
        return [self._paper(f"follow-up to {paper.title[:40]}", idx) for idx in range(min(limit, 2))]

    def _paper(self, query: str, idx: int) -> Paper:
        digest = hashlib.sha1(f"{query}|{idx}".encode()).hexdigest()
        approach = self.approaches[int(digest[:2], 16) % len(self.approaches)]
        topic = " ".join(query.split()[:6])
        return Paper(
            title=f"A {approach} approach to {topic} ({digest[:4]})",
            year=2018 + int(digest[2:4], 16) % 8,
            authors=[f"Author {digest[4:6].upper()}", f"Author {digest[6:8].upper()}"],
            venue=("NeurIPS", "ICML", "ACL", "ICLR", "arXiv")[int(digest[8:10], 16) % 5],
            abstract=(
                f"We study {topic} with a {approach} method. Experiments on two benchmarks show gains over "
                f"strong baselines, although results degrade under distribution shift and the method assumes clean labels."
            ),
            doi=f"10.0000/mock.{digest[:10]}",
            url=f"mock://paper/{digest[:10]}",
            cited_by_count=int(digest[10:13], 16) % 900,
            references=[f"mock:{digest[13:17]}"],
            source="mock",
            synthetic=True,
        )


DOI_PATTERN = re.compile(r"^(https?://doi\.org/)?10\.\d{4,}/", flags=re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$", flags=re.IGNORECASE)
PROVIDER_ALIASES = {"s2": "semantic_scholar", "semanticscholar": "semantic_scholar"}
PROVIDER_FACTORIES = {
    "openalex": lambda s: OpenAlexProvider(api_key=s.openalex_api_key, mailto=s.openalex_mailto),
    "semantic_scholar": lambda s: SemanticScholarProvider(api_key=s.semantic_scholar_api_key),
    "arxiv": lambda s: ArxivProvider(),
    "mock": lambda s: MockLiteratureProvider(),
}


class LiteratureSearch:
    def __init__(self, providers: Iterable[LiteratureProvider], parser: DocumentParser | None = None):
        self.providers = list(providers)
        self.parser = parser or DocumentParser()
        self.warnings: List[str] = []

    @classmethod
    def from_settings(cls, settings) -> "LiteratureSearch":
        providers: List[LiteratureProvider] = []
        for raw in settings.literature_providers.split(","):
            name = PROVIDER_ALIASES.get(raw.strip().lower(), raw.strip().lower())
            if not name:
                continue
            if name not in PROVIDER_FACTORIES:
                raise ValueError(f"Unknown literature provider: {name}")
            providers.append(PROVIDER_FACTORIES[name](settings))
        return cls(providers)

    @property
    def provider_names(self) -> List[str]:
        return [p.name for p in self.providers]

    def search(self, query: str, limit: int) -> List[Paper]:
        results: List[Paper] = []
        for provider in self.providers:
            try:
                results.extend(provider.search(query, limit))
            except Exception as exc:
                self.warnings.append(f"{provider.name} search failed for '{query}': {type(exc).__name__}")
        return dedupe(results)

    def expand(self, paper: Paper, limit: int) -> Dict[str, List[Paper]]:
        out: Dict[str, List[Paper]] = {"references": [], "citing": []}
        for provider in self.providers:
            if provider.name != paper.source:
                continue
            try:
                out["references"] = provider.references(paper, limit)
                out["citing"] = provider.citing(paper, limit)
            except Exception as exc:
                self.warnings.append(f"{provider.name} citation lookup failed for {paper.id}: {type(exc).__name__}")
        return out

    def resolve(self, identifier: str) -> Paper | None:
        """Find a seed paper by DOI, arXiv id or, failing that, by title search."""
        ident = identifier.strip()
        if DOI_PATTERN.match(ident):
            openalex = next((p for p in self.providers if isinstance(p, OpenAlexProvider)), None)
            if openalex is not None:
                return self._try_resolve(openalex, lambda: openalex.resolve_doi(ident), ident)
        arxiv = ARXIV_PATTERN.match(ident)
        if arxiv:
            for provider, lookup in self._arxiv_lookups(arxiv.group(1)):
                found = self._try_resolve(provider, lookup, ident)
                if found is not None:
                    return found
        hits = self.search(ident, 1)
        return hits[0] if hits else None

    def _arxiv_lookups(self, arxiv_id: str):
        for provider in self.providers:
            if isinstance(provider, ArxivProvider):
                yield provider, lambda p=provider: p.resolve(arxiv_id)
            elif isinstance(provider, OpenAlexProvider):
                yield provider, lambda p=provider: p.resolve_doi(f"10.48550/arxiv.{arxiv_id}")

    def _try_resolve(self, provider: LiteratureProvider, lookup, ident: str) -> Paper | None:
        try:
            return lookup()
        except Exception as exc:
            self.warnings.append(f"{provider.name} could not resolve seed '{ident}': {type(exc).__name__}")
            return None

    def fulltext(self, paper: Paper, max_chars: int = 30000) -> str:
        if not paper.pdf_url or paper.synthetic:
            return ""
        try:
            response = requests.get(paper.pdf_url, timeout=60)
            response.raise_for_status()
            doc = self.parser.parse(response.content, self.parser.detect_type(paper.pdf_url, response.content), paper.pdf_url)
            return doc["content"][:max_chars]
        except Exception as exc:
            self.warnings.append(f"full text unavailable for {paper.id}: {type(exc).__name__}")
            return ""

    def drain_warnings(self) -> List[str]:
        out, self.warnings = self.warnings, []
        return out


def dedupe(papers: Iterable[Paper]) -> List[Paper]:
    """Merge duplicates found by different providers (DOI, arXiv id, provider ids, title)."""
    out: List[Paper] = []
    index: Dict[str, Paper] = {}
    for paper in papers:
        if not paper.title:
            continue
        keys = paper_keys(paper)
        match = next((index[k] for k in keys if k in index), None)
        if match is None:
            out.append(paper)
            match = paper
        else:
            for attr in ("doi", "arxiv_id", "openalex_id", "s2_id", "pdf_url", "abstract", "venue"):
                if not getattr(match, attr) and getattr(paper, attr):
                    setattr(match, attr, getattr(paper, attr))
            match.cited_by_count = max(match.cited_by_count, paper.cited_by_count)
            if not match.references and paper.references:
                match.references = paper.references
        for key in paper_keys(match):
            index[key] = match
    return out
