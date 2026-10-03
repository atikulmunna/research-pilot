import math
import re
from typing import Any, Dict, List

from ..services.literature import dedupe
from ..state.models import ClaimVerification, ExtractionBatch, LiteratureClustering, Paper, QueryPlan, TaskEnvelope, now_iso
from .base import Agent, clip, unique

_WORD = re.compile(r"[a-z0-9]{3,}")
TARGETED_MIN_NEW = 6


def _terms(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def _extraction_block(paper: Paper, fulltext: str | None) -> str:
    text = fulltext or paper.abstract or "(no abstract available)"
    limit = 6000 if fulltext is not None else 1800
    return (
        f"### {paper.id}\nTitle: {paper.title}\nYear: {paper.year or 'n.d.'} | Venue: {paper.venue or 'unknown'}\n"
        f"Text ({'full text excerpt' if fulltext is not None else 'abstract'}):\n{clip(text, limit)}"
    )


class LiteratureIntelligence(Agent):
    key = "literature"
    name = "Literature Intelligence"

    def run(self, task: TaskEnvelope):
        project = self.project()
        field_map = self.store.field_map()
        lit_map = self.store.literature_map()
        focus = str(task.inputs.get("focus", "") or "")
        searched = set(q.lower() for q in lit_map.get("queries", []))

        if task.inputs.get("queries"):
            queries = list(task.inputs["queries"])
        else:
            hint = ""
            if field_map and field_map.literature_queries:
                hint = "Field Scout suggestions:\n" + "\n".join(f"- {q}" for q in field_map.literature_queries)
            plan = self.ask(
                "literature.search_queries",
                f"{self.project_block()}\n\n{hint}\n\nFocus for this round: {focus or 'core prior work and competing approaches'}\n"
                f"Already searched: {', '.join(sorted(searched)) or 'nothing yet'}\n\n"
                "Propose 3 to 6 new academic search queries that find seminal work, recent work and competing approaches.",
                QueryPlan,
                {"topic": project.topic, "focus": focus},
            )
            queries = plan.queries
            if field_map and not lit_map.get("rounds"):
                queries = [*queries, *field_map.literature_queries]
        queries = [q for q in unique(queries) if q.lower() not in searched][:6]

        targeted = bool(task.inputs.get("queries") or focus)
        added = self._retrieve(queries, project.topic + " " + (self.state.proposal().text if self.state.proposal() else ""), targeted)
        chained = self._chase_citations()
        fulltexts = self._fulltexts()
        extracted = self._extract(fulltexts)
        clustering = self._cluster()
        verified = self._verify_claims(fulltexts)

        papers = self.state.papers()
        lit_map = self.store.literature_map()
        lit_map["providers"] = self.deps.literature.provider_names
        lit_map["queries"] = unique([*lit_map.get("queries", []), *queries])
        lit_map["rounds"] = int(lit_map.get("rounds", 0)) + 1
        lit_map["papers_total"] = len(papers)
        lit_map["papers_extracted"] = sum(1 for p in papers if p.extracted)
        lit_map["synthetic_papers"] = sum(1 for p in papers if p.synthetic)
        lit_map["clusters"] = [c.name for c in clustering.clusters] if clustering else []
        lit_map["warnings"] = [*lit_map.get("warnings", []), *self.deps.literature.drain_warnings()][-50:]
        lit_map["updated_at"] = now_iso()
        self.store.save_literature_map(lit_map)
        self.state.save_graph()

        findings = [
            f"Round {lit_map['rounds']}: {len(queries)} queries, {len(added)} new papers, {chained} via citation chains ({len(papers)} total)",
            f"{extracted} papers extracted; {len(clustering.clusters) if clustering else 0} methodology clusters",
        ]
        if verified:
            findings.append(f"{verified} claims checked against full text")
        contradictions = list(clustering.contradictions) if clustering else []
        return self.result(
            task,
            findings=findings,
            contradictions=contradictions,
            uncertainties=lit_map["warnings"][-5:],
            artifacts=["literature/literature_map.yaml", "literature/clusters/clusters.yaml", "literature/citations/citation_graph.yaml"],
        )

    # ------------------------------------------------------------ retrieval (code)

    def _retrieve(self, queries: List[str], reference_text: str, targeted: bool) -> List[Paper]:
        reserve = self.settings.literature_citation_seed_k * 2
        budget = self.settings.literature_max_papers - len(self.state.papers()) - reserve
        if targeted:
            budget = max(budget, TARGETED_MIN_NEW)
        if budget <= 0 or not queries:
            return []
        candidates: List[Paper] = []
        for query in queries:
            for paper in self.deps.literature.search(query, self.settings.literature_results_per_query):
                paper.found_by = [f"literature: {query}"]
                candidates.append(paper)
        candidates = dedupe(candidates)
        ref_terms = _terms(reference_text)
        newest = max((p.year or 0 for p in candidates), default=0)

        def score(paper: Paper) -> float:
            overlap = len(ref_terms & _terms(paper.title + " " + paper.abstract)) / max(1, len(ref_terms))
            citations = math.log1p(paper.cited_by_count) / math.log1p(5000)
            recency = 1.0 - min(1.0, max(0, newest - (paper.year or newest)) / 10) if newest else 0.5
            return 0.5 * min(1.0, overlap * 3) + 0.3 * min(1.0, citations) + 0.2 * recency

        candidates.sort(key=score, reverse=True)
        return self.state.add_papers(candidates[:budget], found_by="literature")

    def _chase_citations(self) -> int:
        lit_map = self.store.literature_map()
        expanded = set(lit_map.get("expanded", []))
        count = 0
        for paper in self._citation_seeds(expanded):
            count += self._expand(paper)
            expanded.add(paper.id)
        self._link_known_references()

        lit_map["expanded"] = sorted(expanded)
        self.store.save_literature_map(lit_map)
        edges = [{"from": e.source, "to": e.target} for e in self.state.graph.edges if e.relation == "cites"]
        self.store.save_citation_graph({"edges": edges, "expanded": sorted(expanded)})
        return count

    def _citation_seeds(self, expanded: set) -> List[Paper]:
        papers = self.state.papers()
        seeds = [p for p in papers if p.seed and p.id not in expanded]
        seeds += sorted((p for p in papers if p.id not in expanded and not p.seed), key=lambda p: p.cited_by_count, reverse=True)
        return seeds[: self.settings.literature_citation_seed_k]

    def _expand(self, paper: Paper) -> int:
        """Add a paper's references and citing papers within the paper budget; returns how many were new."""
        budget = self.settings.literature_max_papers - len(self.state.papers())
        added = 0
        for relation, related in self.deps.literature.expand(paper, self.settings.literature_citation_limit).items():
            for other in related:
                other.found_by = [f"{'reference of' if relation == 'references' else 'cites'} {paper.id}"]
            if budget > 0:
                new = self.state.add_papers(related[:budget], found_by=f"citations of {paper.id}")
                added += len(new)
                budget -= len(new)
            self._link_citations(paper, relation, related)
        return added

    def _link_citations(self, paper: Paper, relation: str, related: List[Paper]) -> None:
        for other in related:
            stored = self.state.find_paper(other)
            if stored is None:
                continue
            citing, cited = (paper.id, stored.id) if relation == "references" else (stored.id, paper.id)
            self.state.add_citation(citing, cited)

    def _link_known_references(self) -> None:
        """Add citation edges for references that point at papers already in the corpus."""
        index = {p.openalex_id.rsplit("/", 1)[-1]: p.id for p in self.state.papers() if p.openalex_id}
        links = ((p.id, index.get(ref.rsplit("/", 1)[-1])) for p in self.state.papers() for ref in p.references)
        for source, target in links:
            if target:
                self.state.add_citation(source, target)

    def _fulltexts(self) -> Dict[str, str]:
        k = self.settings.literature_fulltext_top_k
        if k <= 0:
            return {}
        out: Dict[str, str] = {}
        candidates = [p for p in self.state.papers() if not p.extracted and p.pdf_url and not p.synthetic]
        for paper in sorted(candidates, key=lambda p: p.cited_by_count, reverse=True)[:k]:
            text = self.deps.literature.fulltext(paper)
            if text:
                out[paper.id] = text
        return out

    # ------------------------------------------------------------ extraction (lite)

    def _extract(self, fulltexts: Dict[str, str]) -> int:
        pending = [p for p in self.state.papers() if not p.extracted]
        size = max(1, self.settings.literature_extraction_batch)
        for start in range(0, len(pending), size):
            batch = pending[start : start + size]
            extractions = self._extract_batch(batch, fulltexts)
            for paper in batch:
                self._apply_extraction(paper, extractions.get(paper.id), fulltexts)
        return len(pending)

    def _extract_batch(self, batch: List[Paper], fulltexts: Dict[str, str]) -> Dict[str, Any]:
        blocks = [_extraction_block(paper, fulltexts.get(paper.id)) for paper in batch]
        out = self.ask(
            "literature.extraction",
            f"{self.project_block()}\n\nExtract structured information for each paper below. Return one entry per paper id.\n\n"
            + "\n\n".join(blocks),
            ExtractionBatch,
            {"papers": [{"id": p.id, "title": p.title, "abstract": p.abstract} for p in batch]},
        )
        by_id = {x.id.strip(): x for x in out.papers}
        for unknown in set(by_id) - {p.id for p in batch}:
            self.state.violation("no_fabricated_evidence", f"extraction returned unknown paper id '{unknown}'")
        return by_id

    def _apply_extraction(self, paper: Paper, extraction: Any, fulltexts: Dict[str, str]) -> None:
        if extraction is not None:
            for field, value in extraction.model_dump(exclude={"id"}).items():
                setattr(paper, field, value)
            paper.claims = extraction.claims
        paper.extracted = True
        paper.text_basis = "fulltext" if paper.id in fulltexts else ("abstract" if paper.abstract else "metadata")
        self.state.update_paper(paper)

    def _cluster(self) -> LiteratureClustering | None:
        papers = [p for p in self.state.papers() if p.extracted]
        if len(papers) < 2:
            return self.store.clustering()
        lines = "\n".join(f"- {p.id}: {clip(p.title, 120)} | method: {clip(p.method, 160)}" for p in papers)
        clustering = self.ask(
            "literature.clustering",
            f"Topic: {self.project().topic}\n\nGroup these papers into methodology clusters, name the current state of the art, "
            f"note redundant approaches and contradictory findings.\n\n{lines}",
            LiteratureClustering,
            {"papers": [{"id": p.id, "title": p.title, "method": p.method} for p in papers]},
        )
        known = {p.id for p in papers}
        for cluster in clustering.clusters:
            cluster.paper_ids = self.state.filter_refs(cluster.paper_ids, known, f"cluster '{cluster.name}'")
        clustering.state_of_the_art = self.state.filter_refs(clustering.state_of_the_art, known, "state_of_the_art")
        by_paper = {pid: c.name for c in clustering.clusters for pid in c.paper_ids}
        for paper in papers:
            if by_paper.get(paper.id) and paper.cluster != by_paper[paper.id]:
                paper.cluster = by_paper[paper.id]
                self.state.update_paper(paper)
        self.store.save_clustering(clustering)
        return clustering

    def _verify_claims(self, fulltexts: Dict[str, str]) -> int:
        papers = [p for p in self.state.papers() if p.id in fulltexts and p.claims]
        if not papers:
            return 0
        blocks = "\n\n".join(
            f"### {p.id}\nClaims:\n" + "\n".join(f"- {c.text}" for c in p.claims) + f"\nText excerpt:\n{clip(fulltexts[p.id], 5000)}"
            for p in papers
        )
        out = self.ask(
            "literature.claim_verification",
            "For each claim, decide whether the text shows experiments that demonstrate it, only asserts it, or does not contain it.\n\n" + blocks,
            ClaimVerification,
            {"papers": [{"id": p.id, "claims": [c.text for c in p.claims]} for p in papers]},
        )
        by_paper: Dict[str, Dict[str, str]] = {}
        for check in out.checks:
            by_paper.setdefault(check.paper_id, {})[check.claim.strip().lower()] = check.verdict
        for paper in papers:
            verdicts = by_paper.get(paper.id, {})
            for claim in paper.claims:
                verdict = verdicts.get(claim.text.strip().lower())
                if verdict:
                    claim.basis = "demonstrated" if verdict == "demonstrated" else "claimed"
            self.state.update_paper(paper)
        self.store.write_yaml("literature/claim_verification.yaml", out)
        return len(out.checks)
