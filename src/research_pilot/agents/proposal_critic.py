from ..state.models import Assessment, TaskEnvelope
from .base import Agent, as_yaml, clip


class ProposalCritic(Agent):
    key = "proposal_critic"
    name = "Proposal Critic"

    def run(self, task: TaskEnvelope):
        target = task.inputs.get("target", "proposal")
        if target == "proposal":
            return self._proposal(task)
        return self._design(task, target, list(task.inputs.get("ids", [])))

    def _proposal(self, task: TaskEnvelope):
        proposal = self.state.proposal()
        novelty = self.state.latest_novelty()
        gaps = self.state.gaps()
        parts = [self.project_block()]
        if novelty:
            parts.append(
                f"Novelty analysis: verdict {novelty.verdict}, strength {novelty.novelty_strength}, confidence {novelty.confidence:.2f}\n"
                f"Differentiation: {clip(novelty.differentiation, 600)}\n"
                + "\n".join(f"- closest: {w.paper_id}: overlap {clip(w.overlap, 150)}; difference {clip(w.difference, 150)}" for w in novelty.closest_prior_work)
                + ("\nWarnings: " + "; ".join(novelty.warnings) if novelty.warnings else "")
            )
        if gaps:
            parts.append("Research gaps:\n" + "\n".join(f"- {g.id}: {clip(g.description, 200)}" for g in gaps))
        previous = self.state.open_criticisms("proposal")
        if previous:
            parts.append("Criticisms of earlier versions:\n" + "\n".join(f"- {c.id} [{c.severity}] {clip(c.description, 160)}" for c in previous))
        parts.append(
            "Attack this proposal as a hostile but fair reviewer would. Every serious issue needs a proposed test. "
            "Verdict: proceed, revise, pivot or kill."
        )
        assessment = self.ask(
            "critique.proposal",
            "\n\n".join(parts),
            Assessment,
            {"target": "proposal", "revision": (proposal.version - 1) if proposal else 0},
        )
        stored = self.state.add_assessment(assessment, f"proposal@v{proposal.version if proposal else 0}")
        return self._summarize(task, stored)

    def _design(self, task: TaskEnvelope, target: str, ids: list):
        hypotheses = self.state.current_hypotheses(include_closed=False)
        parts = [self.project_block()]
        parts.append("Hypotheses:\n" + as_yaml([h.model_dump(include={"id", "version", "statement", "falsification_criteria", "controls", "assumptions", "competing_hypotheses"}) for h in hypotheses]))
        if target == "experiments":
            specs = [s for s in self.state.experiments.latest_specs() if not ids or s.id in ids]
            parts.append(
                "Experiment designs to review:\n"
                + as_yaml([s.model_dump(exclude={"status", "change_log", "critique_notes", "created_at"}) for s in specs])
            )
            parts.append(
                "Check whether these experiments actually test the hypotheses: baselines, controls, leakage, confounders, "
                "statistical power and failure criteria. Set target_id to the experiment id for each issue."
            )
        else:
            parts.append(
                "Check every hypothesis for testability, falsifiability, hidden assumptions and confounds. "
                "Set target_id to the hypothesis id for each issue."
            )
        assessment = self.ask(
            "critique.design",
            "\n\n".join(parts),
            Assessment,
            {"target": target, "ids": ids or ([h.id for h in hypotheses] if target == "hypotheses" else [])},
        )
        stored = self.state.add_assessment(assessment, target)
        return self._summarize(task, stored)

    def _summarize(self, task: TaskEnvelope, assessment: Assessment):
        counts = {s: sum(1 for i in assessment.issues if i.severity == s) for s in ("fatal", "major", "minor")}
        serious = [i for i in assessment.issues if i.severity in {"fatal", "major"}]
        return self.result(
            task,
            findings=[f"{assessment.id} on {assessment.target}: verdict {assessment.verdict}; {counts['fatal']} fatal, {counts['major']} major, {counts['minor']} minor"],
            recommendations=[f"{i.id}: {i.proposed_test}" for i in serious if i.proposed_test],
            artifacts=["proposal/criticisms.yaml"],
            data={
                "assessment_id": assessment.id,
                "verdict": assessment.verdict,
                "serious_issue_ids": [i.id for i in serious],
                "serious_targets": sorted({i.target_id for i in serious if i.target_id}),
                "fatal": counts["fatal"],
            },
        )
