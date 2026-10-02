from ..state.models import Interpretation, TaskEnvelope, now_iso
from .base import Agent, as_yaml, clip, paper_digest, ranked_papers
from .hypothesis import HYPOTHESIS_FIELDS
from .quant_analyst import comparison_table, summary_table


class ScientificInterpreter(Agent):
    key = "interpreter"
    name = "Scientific Interpreter"

    def run(self, task: TaskEnvelope):
        registry = self.state.experiments
        run = registry.run(task.inputs["run_id"])
        spec = registry.spec(run.spec_key)
        quant = self.store.quant_result(run.id)
        hyp = next(
            (h for h in self.state.hypotheses() if h.id == spec.hypothesis_id and h.version == spec.hypothesis_version),
            self.state.hypothesis(spec.hypothesis_id),
        )
        prior = [i for i in self.store.interpretations() if i.hypothesis_id.split("@")[0] == hyp.id and i.run_id != run.id]
        papers = ranked_papers(p for p in self.state.papers() if p.extracted)[:8]

        primary = [c for c in quant.comparisons if c.primary]
        can_support = bool(primary) and all(c.significant and c.direction_ok for c in primary)
        can_refute = bool(primary) and (
            any(c.significant and not c.direction_ok for c in primary) or all(not c.direction_ok for c in primary)
        )
        parts = [
            f"Hypothesis {hyp.key}:\n{as_yaml(hyp.model_dump(include=HYPOTHESIS_FIELDS))}",
            f"Experiment {spec.key} ({spec.kind}): {spec.objective}\nExpected: {spec.expected_outcomes}\nFailure criteria: {spec.failure_criteria}",
            f"Run {run.id}: {len(run.records)} records, seeds {run.seeds}" + (" [SYNTHETIC RESULTS]" if run.synthetic else ""),
            "Per-arm summaries:\n" + summary_table(quant),
            "Comparisons:\n" + comparison_table(quant),
            "Anomalies:\n" + ("\n".join(f"- {a}" for a in quant.anomalies) or "(none)"),
            "Interpretation constraints:\n" + "\n".join(f"- {c}" for c in quant.interpretation_constraints),
            "Related literature:\n" + paper_digest(papers),
        ]
        if run.deviations:
            parts.append("Protocol deviations:\n" + "\n".join(f"- {d}" for d in run.deviations))
        if prior:
            parts.append("Earlier interpretations for this hypothesis:\n" + "\n".join(f"- {i.run_id}: {i.verdict}. {clip(i.verdict_rationale, 200)}" for i in prior))
        parts.append(
            "Interpret these results. Compare them against the falsification criteria, give competing explanations with "
            "discriminating follow-ups, and propose follow-up hypotheses. Tag every statement."
        )
        out = self.ask(
            "science.interpretation",
            "\n\n".join(parts),
            Interpretation,
            {"run_id": run.id, "hypothesis_id": hyp.id, "primary_significant": can_support},
        )
        out.run_id = run.id
        out.hypothesis_id = hyp.key
        out.created_at = now_iso()
        adjustments = []
        if out.verdict == "supported" and not can_support:
            adjustments.append("verdict downgraded from supported to inconclusive: not every primary comparison is significant in the expected direction after Holm correction")
            out.verdict = "inconclusive"
        if out.verdict == "refuted" and not can_refute:
            adjustments.append("verdict downgraded from refuted to inconclusive: the primary comparisons do not show a reliable effect in the wrong direction")
            out.verdict = "inconclusive"
        allowed = {r.id for r in registry.runs()} | self.state.paper_ids()
        for statement in out.statements:
            statement.evidence_refs = self.state.filter_refs(statement.evidence_refs, allowed, f"interpretation of {run.id}")
            if statement.tag == "evidence_backed" and not statement.evidence_refs:
                statement.tag = "inference"
                adjustments.append(f"statement re-tagged as inference (no valid evidence ids): {clip(statement.text, 100)}")
        if run.synthetic:
            adjustments.append("results are synthetic; the verdict is illustrative and adds no evidence")
        out.adjustments = adjustments
        self.store.save_interpretation(out)

        self.state.record_result(run, out.verdict)
        current = self.state.hypothesis(spec.hypothesis_id)
        if current is not None and current.version == spec.hypothesis_version:
            claim_state = self.state.claim_state(f"C{current.key}")
            status = {"SUPPORTED": "supported", "CONTRADICTED": "refuted"}.get(claim_state, "active")
            if status != current.status and current.status != "abandoned":
                self.state.set_hypothesis_status(current.id, status)

        follow_ups = [{"type": "FORM_HYPOTHESIS", "statement": f.statement, "rationale": f.rationale, "evidence": [run.id]} for f in out.follow_up_hypotheses[:2]]
        follow_ups += [
            {"type": "INVESTIGATE_FAILURE", "target": run.id, "rationale": e.follow_up}
            for e in out.competing_explanations
            if e.follow_up and e.plausibility >= 0.3
        ][:2]
        return self.result(
            task,
            findings=[f"{run.id} -> {hyp.key}: {out.verdict}. {clip(out.verdict_rationale, 200)}"],
            contradictions=list(out.contradicting_evidence),
            uncertainties=[f"Competing explanation: {e.explanation}" for e in out.competing_explanations] + adjustments,
            follow_up_tasks=follow_ups,
            artifacts=[f"analysis/interpretations/{run.id}.yaml"],
            data={"verdict": out.verdict, "hypothesis": hyp.key},
        )
