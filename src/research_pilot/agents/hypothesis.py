from ..state.models import HypothesisSet, TaskEnvelope
from .base import Agent, as_yaml, clip, paper_digest, ranked_papers

HYPOTHESIS_FIELDS = {
    "id",
    "version",
    "statement",
    "independent_variables",
    "dependent_variables",
    "controls",
    "expected_outcome",
    "falsification_criteria",
    "assumptions",
    "status",
}


class HypothesisDesigner(Agent):
    key = "hypothesis_designer"
    name = "Hypothesis Designer"

    def run(self, task: TaskEnvelope):
        mode = task.inputs.get("mode", "design")
        if mode == "revise":
            return self._revise(task)
        if mode == "refine":
            return self._refine(task)
        if mode == "follow_up":
            return self._follow_up(task)
        return self._design(task)

    def _context(self) -> str:
        gaps = self.state.gaps()
        novelty = self.state.latest_novelty()
        papers = ranked_papers(p for p in self.state.papers() if p.extracted)[:12]
        parts = [
            self.project_block(),
            "Research gaps:\n" + ("\n".join(f"- {g.id}: {clip(g.description, 220)}" for g in gaps) or "(none)"),
            "Most relevant papers:\n" + paper_digest(papers, detail=False),
        ]
        if novelty:
            parts.append(f"Novelty: {novelty.verdict} ({novelty.novelty_strength}). Differentiation: {clip(novelty.differentiation, 400)}")
        criticisms = self.state.open_criticisms("proposal")
        if criticisms:
            parts.append("Open criticisms and the tests they ask for:\n" + "\n".join(f"- {c.id} [{c.severity}] {clip(c.description, 160)} -> test: {clip(c.proposed_test, 160)}" for c in criticisms))
        existing = self.state.current_hypotheses()
        if existing:
            parts.append("Existing hypotheses (do not duplicate):\n" + as_yaml([h.model_dump(include=HYPOTHESIS_FIELDS) for h in existing]))
        return "\n\n".join(parts)

    def _context_ids(self) -> dict:
        return {"gaps": [g.id for g in self.state.gaps()], "papers": [p.id for p in ranked_papers(self.state.papers())[:5]]}

    def _design(self, task: TaskEnvelope):
        out = self.ask(
            "hypothesis.design",
            self._context()
            + "\n\nDecompose the research question and write 2 to 4 explicit, falsifiable hypotheses. Give each a priority "
            "(1 = central contribution). Every central claim of the proposal should become a testable hypothesis.",
            HypothesisSet,
            {**self._context_ids(), "mode": "design"},
        )
        stored = [self.state.add_hypothesis(h) for h in out.hypotheses[:4] if h.statement]
        if out.question_decomposition:
            self.store.write_yaml("hypotheses/question_decomposition.yaml", out.question_decomposition)
        return self.result(
            task,
            findings=[f"{h.key} (priority {h.priority}): {clip(h.statement, 160)}" for h in stored],
            artifacts=["hypotheses/hypotheses.yaml"],
            data={"hypothesis_ids": [h.id for h in stored]},
        )

    def _refine(self, task: TaskEnvelope):
        """Pre-registration refinement after design critique. Versions are kept and logged."""
        ids = set(task.inputs.get("targets", []))
        criticisms = [c for c in self.state.open_criticisms("hypotheses") if not ids or c.target_id in ids or not c.target_id]
        targets = [h for h in self.state.current_hypotheses(include_closed=False) if not ids or h.id in ids]
        revised = []
        for hyp in targets:
            relevant = [c for c in criticisms if c.target_id in {hyp.id, ""}]
            if not relevant:
                continue
            out = self.ask(
                "hypothesis.design",
                self._context()
                + f"\n\nRevise hypothesis {hyp.id} to address these criticisms (keep the same id):\n"
                + "\n".join(f"- {c.id} [{c.severity}] {c.description} -> {c.proposed_test}" for c in relevant)
                + "\n\nCurrent version:\n"
                + as_yaml(hyp),
                HypothesisSet,
                {**self._context_ids(), "mode": "revise", "hypothesis": hyp.model_dump()},
            )
            if not out.hypotheses:
                continue
            new = self.state.revise_hypothesis(
                hyp.id,
                out.hypotheses[0],
                reason="pre-registration refinement addressing " + ", ".join(c.id for c in relevant),
                triggering_evidence=[c.id for c in relevant],
                decision_id=task.inputs.get("decision_id", ""),
            )
            self.state.resolve_criticisms([c.id for c in relevant])
            revised.append(new)
        return self.result(
            task,
            findings=[f"{h.key} revised from {h.revision_of}" for h in revised] or ["No hypothesis needed revision"],
            artifacts=["hypotheses/hypotheses.yaml"],
            data={"hypothesis_ids": [h.id for h in revised]},
        )

    def _revise(self, task: TaskEnvelope):
        hyp = self.state.hypothesis(task.inputs["hypothesis_id"])
        evidence = list(task.inputs.get("evidence", []))
        observations = []
        for run_id in evidence:
            interp = self.store.interpretation(run_id)
            if interp:
                observations.append(f"{run_id}: verdict {interp.verdict}. {clip(interp.verdict_rationale, 400)}")
        out = self.ask(
            "hypothesis.design",
            self._context()
            + f"\n\nRevise hypothesis {hyp.id} in light of the observed results. Keep the same id. The original version "
            "stays on record; state precisely what changed and why.\n\nCurrent version:\n"
            + as_yaml(hyp)
            + "\n\nReason for revision: "
            + str(task.inputs.get("reason", ""))
            + ("\n\nObserved results:\n" + "\n".join(observations) if observations else ""),
            HypothesisSet,
            {**self._context_ids(), "mode": "revise", "hypothesis": hyp.model_dump()},
        )
        if not out.hypotheses:
            return self.result(task, status="failed", error="no revised hypothesis returned")
        new = self.state.revise_hypothesis(
            hyp.id,
            out.hypotheses[0],
            reason=str(task.inputs.get("reason", "revision after results")),
            triggering_evidence=evidence,
            decision_id=task.inputs.get("decision_id", ""),
        )
        return self.result(
            task,
            findings=[f"{new.key} supersedes {new.revision_of}: {clip(new.statement, 160)}"],
            artifacts=["hypotheses/hypotheses.yaml"],
            data={"hypothesis_ids": [new.id]},
        )

    def _follow_up(self, task: TaskEnvelope):
        statement = str(task.inputs.get("statement", ""))
        out = self.ask(
            "hypothesis.design",
            self._context()
            + f"\n\nFormalise this follow-up idea as one new testable hypothesis:\n{statement}\n\nRationale: {task.inputs.get('rationale', '')}",
            HypothesisSet,
            {**self._context_ids(), "mode": "follow_up", "statement": statement},
        )
        stored = [self.state.add_hypothesis(h) for h in out.hypotheses[:1] if h.statement]
        for hyp in stored:
            for ref in task.inputs.get("evidence", []):
                if ref in self.state.graph.nodes and self.state.graph.nodes[ref].type == "result":
                    self.state.graph.add_edge(ref, hyp.key, "suggests", "science.interpretation")
        self.state.save_graph()
        return self.result(
            task,
            findings=[f"{h.key}: {clip(h.statement, 160)}" for h in stored],
            artifacts=["hypotheses/hypotheses.yaml"],
            data={"hypothesis_ids": [h.id for h in stored]},
        )
