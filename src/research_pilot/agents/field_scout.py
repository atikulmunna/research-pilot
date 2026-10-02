from collections import Counter

from ..services.literature import dedupe
from ..state.models import FieldMap, QueryPlan, TaskEnvelope
from .base import Agent, clip, unique


class FieldScout(Agent):
    key = "field_scout"
    name = "Field Scout"

    def run(self, task: TaskEnvelope):
        project = self.project()
        proposal = self.state.proposal()
        context = {"topic": project.topic, "question": project.question, "proposal": proposal.text if proposal else ""}
        base = self.project_block()

        plan = self.ask(
            "field.search_queries",
            f"{base}\n\nGenerate 4 to 6 broad academic search queries that together cover the field around this topic, plus common synonyms.",
            QueryPlan,
            context,
        )
        queries = unique([project.topic, *plan.queries])[:6]
        per_query = max(3, self.settings.literature_results_per_query // 2)
        hits = []
        for query in queries:
            for paper in self.deps.literature.search(query, per_query):
                paper.found_by = [f"field_scout: {query}"]
                hits.append(paper)
        hits = dedupe(hits)

        seeds = []
        for ident in project.seed_papers:
            paper = self.deps.literature.resolve(ident)
            if paper is not None:
                paper.seed = True
                paper.found_by = [f"seed: {ident}"]
                seeds.append(paper)

        seminal = sorted(hits, key=lambda p: p.cited_by_count, reverse=True)[:10]
        added = self.state.add_papers(seeds + seminal, found_by="field_scout")

        web_hits = []
        if self.deps.web.enabled:
            try:
                web_hits = self.deps.web.search(f"{project.topic} recent advances", 5)
            except Exception as exc:
                self.deps.literature.warnings.append(f"web search failed: {type(exc).__name__}")

        hit_lines = "\n".join(
            f"- {clip(p.title, 150)} ({p.year or 'n.d.'}, {p.venue or 'unknown venue'}, cited {p.cited_by_count})"
            for p in sorted(hits, key=lambda p: p.cited_by_count, reverse=True)[:30]
        )
        web_lines = "\n".join(f"- {clip(w['title'], 120)}: {clip(w['snippet'], 200)}" for w in web_hits)
        field_map = self.ask(
            "field.map",
            f"{base}\n\nBroad search results (titles only, not yet analysed):\n{hit_lines or '(none)'}"
            + (f"\n\nRecent web snippets:\n{web_lines}" if web_lines else "")
            + "\n\nBuild the field map. Also propose 4 to 8 targeted literature_queries for the in-depth review.",
            FieldMap,
            {**context, "hits": [p.title for p in hits[:10]]},
        )

        authors = Counter(a for p in hits for a in p.authors[:3] if a)
        venues = Counter(p.venue for p in hits if p.venue)
        newest = max((p.year or 0 for p in hits), default=0)
        field_map.key_research_groups = [f"{name} ({count} papers)" for name, count in authors.most_common(8) if count > 1]
        field_map.venues = [f"{name} ({count})" for name, count in venues.most_common(8)]
        field_map.recent_developments = [
            f"{clip(p.title, 140)} ({p.year})" for p in hits if newest and (p.year or 0) >= newest - 1
        ][:6] + [f"[web] {clip(w['title'], 120)}" for w in web_hits]
        field_map.provenance = (
            f"Lite model over {len(hits)} search hits from {', '.join(self.deps.literature.provider_names)}. "
            "Key groups, venues and recent developments are computed from search metadata; other fields are "
            "model-proposed pointers to be verified by the literature review."
        )
        self.store.save_field_map(field_map)

        lit_map = self.store.literature_map()
        lit_map["providers"] = self.deps.literature.provider_names
        lit_map.setdefault("queries", [])
        lit_map["queries"] = unique([*lit_map["queries"], *queries])
        lit_map["warnings"] = [*lit_map.get("warnings", []), *self.deps.literature.drain_warnings()]
        self.store.save_literature_map(lit_map)

        return self.result(
            task,
            findings=[
                f"{len(field_map.subdomains)} subdomains, {len(field_map.common_methods)} common methods, "
                f"{len(field_map.datasets)} datasets, {len(field_map.open_questions)} open questions",
                f"{len(hits)} broad search hits; {len(added)} seminal or seed papers added to the literature base",
            ],
            uncertainties=["Field map items from model knowledge are provisional until the literature review confirms them."],
            artifacts=["field/field_map.yaml", "field/terminology.yaml"],
        )
