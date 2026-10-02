from ..state.models import GapAnalysis, NoveltyAssessment, SearchRecord, TaskEnvelope, now_iso
from .base import Agent, as_yaml, clip, paper_digest, ranked_papers

MIN_PAPERS_FOR_NOVELTY = 15


class GapNoveltyAnalyst(Agent):
    key = "gap_novelty"
    name = "Gap & Novelty Analyst"

    def run(self, task: TaskEnvelope):
        if task.inputs.get("mode") == "novelty":
            return self._novelty(task)
        return self._gaps(task)

    def search_record(self) -> SearchRecord:
        lit_map = self.store.literature_map()
        return SearchRecord(
            providers=list(lit_map.get("providers", [])),
            queries=list(lit_map.get("queries", [])),
            papers_checked=len(self.state.papers()),
            rounds=int(lit_map.get("rounds", 0)),
            warnings=list(lit_map.get("warnings", []))[-5:],
        )

    def _gaps(self, task: TaskEnvelope):
        papers = ranked_papers(p for p in self.state.papers() if p.extracted)[:30]
        field_map = self.store.field_map()
        clustering = self.store.clustering()
        context_parts = [self.project_block(), "Literature corpus (most relevant first):", paper_digest(papers)]
        if clustering:
            context_parts.append("Methodology clusters:\n" + as_yaml([{"name": c.name, "papers": c.paper_ids} for c in clustering.clusters]))
            if clustering.contradictions:
                context_parts.append("Contradictory findings noted:\n" + "\n".join(f"- {c}" for c in clustering.contradictions))
        if field_map:
            context_parts.append("Open questions from the field map:\n" + "\n".join(f"- {q}" for q in field_map.open_questions[:8]))
            context_parts.append("Neglected areas:\n" + "\n".join(f"- {q}" for q in field_map.neglected_areas[:6]))
        record = self.search_record()
        context_parts.append(
            f"Search coverage so far: {record.papers_checked} papers from {', '.join(record.providers)} over {record.rounds} round(s)."
        )
        analysis = self.ask(
            "gap.discovery",
            "\n\n".join(context_parts)
            + "\n\nIdentify 2 to 5 concrete research gaps grounded in these papers, with supporting paper ids. "
            "Say whether literature coverage is sufficient; if not, give the missing search queries.",
            GapAnalysis,
            {"papers": [{"id": p.id, "title": p.title, "limitations": p.limitations} for p in papers]},
        )
        for gap in analysis.gaps:
            gap.search_record = record
        stored = self.state.set_gaps(
            analysis.gaps,
            {
                "coverage_assessment": analysis.coverage_assessment,
                "coverage_sufficient": analysis.coverage_sufficient,
                "missing_literature_queries": analysis.missing_literature_queries,
                "search_record": record,
                "created_at": now_iso(),
            },
        )
        return self.result(
            task,
            findings=[f"{g.id}: {clip(g.description, 140)} ({g.novelty_strength})" for g in stored],
            uncertainties=[] if analysis.coverage_sufficient else [f"Coverage insufficient: {analysis.coverage_assessment}"],
            artifacts=["gaps/research_gaps.yaml"],
            data={
                "coverage_sufficient": analysis.coverage_sufficient,
                "missing_queries": analysis.missing_literature_queries,
                "gap_ids": [g.id for g in stored],
            },
        )

    def _novelty(self, task: TaskEnvelope):
        proposal = self.state.proposal()
        papers = ranked_papers(p for p in self.state.papers() if p.extracted)[:20]
        record = self.search_record()
        gaps = self.state.gaps()
        user = "\n\n".join(
            [
                self.project_block(),
                "Closest papers in the searched corpus:",
                paper_digest(papers),
                "Identified gaps:\n" + "\n".join(f"- {g.id}: {clip(g.description, 200)}" for g in gaps),
                f"Searched: {record.papers_checked} papers via {', '.join(record.providers)}; queries: {'; '.join(record.queries[:12])}",
                "Assess the novelty of the current proposal against this corpus. Name the closest prior work by paper id "
                "and explain why it does or does not already cover the idea.",
            ]
        )
        assessment = self.ask("novelty.analysis", user, NoveltyAssessment, {"papers": [p.id for p in papers], "proposal": proposal.text if proposal else ""})
        assessment.proposal_version = proposal.version if proposal else 0
        assessment.search_record = record
        warnings = []
        if record.papers_checked < MIN_PAPERS_FOR_NOVELTY:
            warnings.append(f"Novelty judged against only {record.papers_checked} papers; treat it as provisional.")
            assessment.confidence = min(assessment.confidence, 0.5)
        if any(p.synthetic for p in self.state.papers()):
            warnings.append("The corpus contains synthetic mock papers, so this novelty judgment is illustrative only.")
            assessment.confidence = min(assessment.confidence, 0.3)
        if assessment.verdict == "novel" and not assessment.closest_prior_work:
            warnings.append("Claimed novel without naming any closest prior work; the claim is weakly grounded.")
            assessment.confidence = min(assessment.confidence, 0.5)
        assessment.warnings = warnings
        stored = self.state.add_novelty(assessment)
        return self.result(
            task,
            findings=[
                f"Novelty verdict for proposal v{stored.proposal_version}: {stored.verdict} "
                f"({stored.novelty_strength}, confidence {stored.confidence:.2f})"
            ],
            uncertainties=warnings,
            artifacts=["proposal/novelty_analysis.yaml"],
            data={"verdict": stored.verdict, "strength": stored.novelty_strength, "confidence": stored.confidence},
        )
