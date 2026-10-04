"""Research Planner: candidate generation and validation in code, judgment on the strong tier."""

from collections import defaultdict
from typing import Dict, List

from ..state.models import PLANNER_ACTIONS, Action, PlannerOutput, ProposalRevision, ReviewIssue, TaskEnvelope
from .base import Agent, clip

RUN_ACTIONS = {"RUN_EXPERIMENT", "REPRODUCE_BASELINE", "RUN_ABLATION", "RUN_ROBUSTNESS_TEST"}
KIND_FOR_ACTION = {"REPRODUCE_BASELINE": "baseline_reproduction", "RUN_ABLATION": "ablation", "RUN_ROBUSTNESS_TEST": "robustness"}
NEEDS_HYPOTHESIS = {"REVISE_HYPOTHESIS", "DESIGN_EXPERIMENT", "REPRODUCE_BASELINE", "RUN_ABLATION", "RUN_ROBUSTNESS_TEST"}

CATALOGUE = """Available actions (type: target):
- SEARCH_LITERATURE: a search query
- RUN_EXPERIMENT: an approved experiment id (E#) that has not run yet, or one that deserves a replication with fresh seeds
- REPRODUCE_BASELINE / RUN_ABLATION / RUN_ROBUSTNESS_TEST: a hypothesis id (H#); a new experiment of that kind is designed, critiqued and run
- INVESTIGATE_FAILURE: a run id (R###) that failed or produced an inconclusive or surprising result
- REVISE_HYPOTHESIS: a hypothesis id; creates a new logged version (the original stays on record)
- FORM_HYPOTHESIS: the statement of a new follow-up hypothesis
- DESIGN_EXPERIMENT: a hypothesis id that lacks an adequate experiment
- REVISE_PROPOSAL: no target; revise or pivot the proposal
- WRITE_PAPER: no target; draft or revise the manuscript (a red-team review follows automatically)
- REQUEST_REVIEW: no target; red-team review of the current manuscript
- STOP: no target; finish because the evidence is as good as the budget allows
- KILL: no target; abandon the research idea
Score each candidate from 0 to 1 on scientific_value, uncertainty_reduction, feasibility and cost (1 = most expensive).
Priority is computed by code as scientific_value x uncertainty_reduction x feasibility / cost."""


def priority(action: Action) -> float:
    return round(action.scientific_value * action.uncertainty_reduction * action.feasibility / max(action.cost, 0.05), 4)


def rank_actions(actions: List[Action]) -> List[Action]:
    """Highest priority first, one action per (type, target), marked as planner-made."""
    seen = set()
    ranked = []
    for action in sorted(actions, key=priority, reverse=True):
        key = (action.type, action.target)
        if key in seen:
            continue
        seen.add(key)
        action.priority = priority(action)
        action.origin = "planner"
        ranked.append(action)
    return ranked


def _hint(type_: str, target: str, rationale: str, value: float, unc: float, feas: float = 0.8, cost: float = 0.4, addresses=None) -> Action:
    return Action(type=type_, target=target, rationale=rationale, scientific_value=value, uncertainty_reduction=unc, feasibility=feas, cost=cost, addresses=addresses or [])


def _review_hint(issue: ReviewIssue, addressed: bool, first_hypothesis: str | None, runs_left: int) -> Action:
    if addressed:
        return _hint("WRITE_PAPER", "", f"work done for review {issue.id}; revise the manuscript and request re-review", 0.8, 0.5, 0.9, 0.3, [issue.id])
    if issue.required_action == "experiment" and first_hypothesis and runs_left > 0:
        return _hint("RUN_ABLATION", first_hypothesis, f"review {issue.id}: {clip(issue.description, 120)}", 0.7, 0.6, 0.6, 0.6, [issue.id])
    if issue.required_action == "literature":
        return _hint("SEARCH_LITERATURE", clip(issue.suggestion or issue.description, 100), f"review {issue.id}", 0.5, 0.5, 0.9, 0.2, [issue.id])
    return _hint("WRITE_PAPER", "", f"review {issue.id}: {clip(issue.description, 120)}", 0.6, 0.4, 0.9, 0.3, [issue.id])


class ResearchPlanner(Agent):
    key = "planner"
    name = "Research Planner"

    def run(self, task: TaskEnvelope):
        if task.inputs.get("mode") == "revise_proposal":
            return self._revise_proposal(task)
        return self._plan(task)

    # ------------------------------------------------------------ planning

    def _plan(self, task: TaskEnvelope):
        budget = dict(task.inputs.get("budget", {}))
        hints = self.hints(budget)
        evidence_sufficient = self._tested_all() and not any(h.type == "RUN_EXPERIMENT" for h in hints)
        out = self._ask_planner(budget, hints, evidence_sufficient)
        rejected: List[str] = []
        ranked = rank_actions(self._valid_actions(out, hints, budget, evidence_sufficient, rejected))
        abandoned = self._abandon(out, task.task_id)
        if ranked:
            self._log_plan(ranked, out, rejected, task.task_id)
        return self.result(
            task,
            findings=[clip(out.assessment, 400)] if out.assessment else [],
            recommendations=[f"{a.type} {a.target} (priority {a.priority})".strip() for a in ranked],
            uncertainties=list(out.blockers) + [f"rejected candidate: {r}" for r in rejected],
            artifacts=["roadmap/roadmap.yaml", "decisions/decision_log.yaml"],
            data={"actions": [a.model_dump(mode="json") for a in ranked], "abandoned": abandoned},
        )

    def _ask_planner(self, budget: Dict, hints: List[Action], evidence_sufficient: bool) -> PlannerOutput:
        summary = self.state_summary(budget)
        hint_lines = "\n".join(
            f"- {h.type} {h.target}: {h.rationale} (value {h.scientific_value}, uncertainty {h.uncertainty_reduction}, feasibility {h.feasibility}, cost {h.cost})"
            for h in hints
        )
        try:
            return self.ask(
                "research.planning",
                f"{summary}\n\nCandidate actions computed from the state (you may add others):\n{hint_lines or '(none)'}\n\n{CATALOGUE}\n\n"
                "Propose up to 5 ranked candidate actions, decide whether evidence suffices for writing, and list any hypotheses to abandon.",
                PlannerOutput,
                {"hints": [h.model_dump(mode="json") for h in hints], "evidence_sufficient": evidence_sufficient},
            )
        except Exception as exc:
            return PlannerOutput(assessment=f"Planner model unavailable ({type(exc).__name__}); using state-derived candidates.")

    def _valid_actions(self, out: PlannerOutput, hints: List[Action], budget: Dict, evidence_sufficient: bool, rejected: List[str]) -> List[Action]:
        """The model's valid candidates, else the valid state-derived hints, plus drafting once evidence suffices."""
        actions = [a for a in (self.validate(c, budget, rejected) for c in out.candidates) if a is not None]
        if not actions:
            actions = [a for a in (self.validate(h, budget, rejected) for h in hints) if a is not None]
        wants_paper = (out.evidence_sufficient_for_paper or evidence_sufficient) and not self.store.manuscript()
        if wants_paper and not any(a.type == "WRITE_PAPER" for a in actions):
            actions.append(Action(type="WRITE_PAPER", rationale="evidence judged sufficient for drafting", scientific_value=0.8, uncertainty_reduction=0.5, feasibility=0.9, cost=0.3))
        return actions

    def _abandon(self, out: PlannerOutput, task_id: str) -> List[str]:
        abandoned = []
        for request in out.abandon:
            hyp = self.state.hypothesis(request.hypothesis_id)
            if hyp is None or hyp.status in {"abandoned", "superseded"}:
                continue
            decision = self.state.log_decision(
                "abandon_hypothesis",
                f"Abandon {hyp.key}",
                request.reason,
                evidence=[r.id for r in self.state.experiments.runs() if r.hypothesis_id == hyp.id],
                agent_inputs=[task_id],
                effects=["no further experiments for this hypothesis"],
            )
            self.state.set_hypothesis_status(hyp.id, "abandoned")
            abandoned.append(f"{hyp.key} ({decision.id})")
        return abandoned

    def _log_plan(self, ranked: List[Action], out: PlannerOutput, rejected: List[str], task_id: str) -> None:
        top = ranked[0]
        hotspots = self.state.graph.uncertainty_hotspots(self.settings.allow_synthetic_evidence)
        self.state.log_decision(
            "plan",
            f"Next action: {top.type} {top.target}".strip(),
            top.rationale or out.assessment,
            evidence=[h["claim"] for h in hotspots[:3]],
            alternatives=[f"{a.type} {a.target} (priority {a.priority})".strip() for a in ranked[1:6]] + [f"rejected: {r}" for r in rejected[:5]],
            agent_inputs=[task_id],
            effects=[f"{len(ranked)} actions queued on the roadmap"],
        )

    def validate(self, action: Action, budget: Dict, rejected: List[str]) -> Action | None:
        action = action.model_copy(deep=True)
        action.target = action.target.strip()
        if action.type not in PLANNER_ACTIONS:
            rejected.append(f"{action.type or '?'} is not a known action")
            return None
        if action.type in RUN_ACTIONS and int(budget.get("runs_left", 1)) <= 0:
            rejected.append(f"{action.type} {action.target}: experiment budget exhausted")
            return None
        specs = {s.id: s for s in self.state.experiments.latest_specs()}
        target = action.target.split("@")[0]
        if action.type == "RUN_EXPERIMENT":
            return self._validate_run(action, specs.get(target), target, rejected)
        if action.type in KIND_FOR_ACTION and target in specs:
            spec = specs[target]
            if spec.kind == KIND_FOR_ACTION[action.type] and spec.status == "approved" and not self.state.experiments.runs_for(spec.key):
                return action.model_copy(update={"type": "RUN_EXPERIMENT", "target": spec.id})
            target = spec.hypothesis_id
        return self._validate_target(action, target, rejected)

    def _validate_run(self, action: Action, spec, target: str, rejected: List[str]) -> Action | None:
        if spec is None and self._active_hypothesis(target):
            return action.model_copy(update={"type": "DESIGN_EXPERIMENT", "target": target})
        if spec is None or spec.status != "approved":
            rejected.append(f"RUN_EXPERIMENT {action.target}: not an approved experiment")
            return None
        if len(self.state.experiments.runs_for(spec.key)) >= self.settings.max_runs_per_spec:
            rejected.append(f"RUN_EXPERIMENT {spec.id}: run limit reached")
            return None
        action.target = spec.id
        return action

    def _validate_target(self, action: Action, target: str, rejected: List[str]) -> Action | None:
        """Check the target of an action that does not run an experiment directly."""
        if action.type in NEEDS_HYPOTHESIS:
            return self._accept_target(action, target, self._active_hypothesis(target), "no active hypothesis with that id", rejected)
        if action.type == "INVESTIGATE_FAILURE":
            return self._accept_target(action, target, self.state.experiments.run(target) is not None, "unknown run", rejected)
        if action.type in {"FORM_HYPOTHESIS", "SEARCH_LITERATURE"}:
            target = action.target or clip(action.rationale, 160)
            return self._accept_target(action, target, bool(target), "missing statement or query", rejected, label=action.type)
        if action.type == "REQUEST_REVIEW" and not self.store.manuscript():
            return action.model_copy(update={"type": "WRITE_PAPER", "target": ""})
        action.target = ""
        return action

    @staticmethod
    def _accept_target(action: Action, target: str, valid: bool, reason: str, rejected: List[str], label: str = "") -> Action | None:
        if not valid:
            rejected.append(f"{label or f'{action.type} {action.target}'}: {reason}")
            return None
        action.target = target
        return action

    def _active_hypothesis(self, hypothesis_id: str) -> bool:
        hyp = self.state.hypothesis(hypothesis_id) if hypothesis_id else None
        return hyp is not None and hyp.status not in {"abandoned", "superseded"}

    def _tested_all(self) -> bool:
        hyps = self.state.current_hypotheses(include_closed=False)
        completed = {(r.hypothesis_id, r.spec_key) for r in self.state.experiments.runs() if r.status == "completed"}
        specs = self.state.experiments.latest_specs()
        for hyp in hyps:
            keys = {s.key for s in specs if s.hypothesis_id == hyp.id and s.hypothesis_version == hyp.version}
            if not any((hyp.id, key) in completed for key in keys):
                return False
        return bool(hyps)

    # ------------------------------------------------------------ deterministic candidates

    def hints(self, budget: Dict) -> List[Action]:
        registry = self.state.experiments
        by_spec: Dict[str, list] = defaultdict(list)
        for run in registry.runs():
            by_spec[run.spec_key].append(run)
        runs_left = int(budget.get("runs_left", 1))
        hyps = {h.id: h for h in self.state.current_hypotheses(include_closed=False)}
        specs = [s for s in registry.latest_specs() if s.hypothesis_id in hyps and s.hypothesis_version == hyps[s.hypothesis_id].version]
        manuscript = bool(self.store.manuscript())
        out = [
            *self._experiment_hints(specs, hyps, by_spec, runs_left),
            *self._hypothesis_hints(specs, hyps, by_spec, runs_left),
            *self._review_hints(next(iter(hyps), None), runs_left),
            *self._writing_hints(runs_left, manuscript),
        ]
        if not out:
            out.append(_hint("WRITE_PAPER" if not manuscript else "STOP", "", "no further informative action is available", 0.5, 0.3, 1.0, 0.2))
        for action in out:
            action.priority = priority(action)
        return sorted(out, key=lambda a: a.priority, reverse=True)

    def _experiment_hints(self, specs: List, hyps: Dict, by_spec: Dict[str, list], runs_left: int) -> List[Action]:
        out = []
        for spec in specs:
            spec_runs = by_spec.get(spec.key, [])
            if spec.status == "approved" and not spec_runs and runs_left > 0:
                hyp = hyps[spec.hypothesis_id]
                out.append(_hint("RUN_EXPERIMENT", spec.id, f"{spec.kind} experiment for {hyp.id} (priority {hyp.priority}) is approved but not run", 0.9 if spec.kind == "primary" else 0.7, 0.9 - 0.1 * (hyp.priority - 1)))
            failed = [r for r in spec_runs if r.status == "failed"]
            if failed and not any(r.status == "completed" for r in spec_runs):
                out.append(_hint("INVESTIGATE_FAILURE", failed[-1].id, f"{spec.key} failed: {clip(failed[-1].error, 120)}", 0.7, 0.6, 0.6, 0.5))
        return out

    def _hypothesis_hints(self, specs: List, hyps: Dict, by_spec: Dict[str, list], runs_left: int) -> List[Action]:
        out = []
        for hyp in hyps.values():
            hyp_specs = [s for s in specs if s.hypothesis_id == hyp.id]
            if not hyp_specs:
                out.append(_hint("DESIGN_EXPERIMENT", hyp.id, f"{hyp.key} has no experiment for its current version", 0.8, 0.8, 0.8, 0.3))
                continue
            claim_state = self.state.claim_state(f"C{hyp.key}")
            completed = [r for s in hyp_specs for r in by_spec.get(s.key, []) if r.status == "completed"]
            if claim_state == "PARTIALLY_SUPPORTED" and runs_left > 0 and not any(s.kind == "robustness" for s in hyp_specs):
                out.append(_hint("RUN_ROBUSTNESS_TEST", hyp.id, f"{hyp.key} rests on a single supporting result", 0.7, 0.7, 0.7, 0.5))
            interp = self.store.interpretation(completed[-1].id) if completed else None
            if interp and interp.verdict == "inconclusive":
                out.append(_hint("INVESTIGATE_FAILURE", completed[-1].id, f"{completed[-1].id} was inconclusive for {hyp.key}", 0.6, 0.6, 0.7, 0.4))
        return out

    def _review_hints(self, first_hypothesis: str | None, runs_left: int) -> List[Action]:
        addressed = self.addressed_issues()
        return [
            _review_hint(issue, issue.id in addressed, first_hypothesis, runs_left)
            for issue in self.open_review_issues()
            if issue.severity != "minor"
        ]

    def _writing_hints(self, runs_left: int, manuscript: bool) -> List[Action]:
        out = []
        if self._tested_all() and not manuscript:
            out.append(_hint("WRITE_PAPER", "", "every active hypothesis has at least one completed experiment", 0.8, 0.5, 0.9, 0.3))
        if runs_left <= 0 and not manuscript:
            out.append(_hint("WRITE_PAPER", "", "experiment budget exhausted; write up the evidence that exists", 0.7, 0.4, 0.9, 0.3))
        if manuscript and not any(i.severity in {"critical", "major"} for i in self.open_review_issues()):
            out.append(_hint("STOP", "", "manuscript reviewed with no open major issues", 0.6, 0.2, 1.0, 0.1))
        return out

    def addressed_issues(self) -> set:
        """Open review issues that already have work done for them since they were raised."""
        done = [a for a in self.state.roadmap() if a.status == "done"]
        out = {iid for a in done for iid in a.addresses}
        specs = self.state.experiments.latest_specs()
        for report in self.store.reviews():
            newer_specs = any(s.created_at > report.created_at for s in specs)
            for issue in report.issues:
                if issue.required_action == "experiment" and newer_specs:
                    out.add(issue.id)
        return out

    def open_review_issues(self):
        return [issue for report in self.store.reviews() for issue in report.issues if issue.status == "open"]

    def state_summary(self, budget: Dict) -> str:
        project = self.project()
        novelty = self.state.latest_novelty()
        registry = self.state.experiments
        lines = [
            f"Phase: {project.phase}. Step {budget.get('step', project.step)} of {budget.get('max_steps', self.settings.max_steps)}. "
            f"Experiment runs left: {budget.get('runs_left', '?')}. Review rounds used: {len(self.store.reviews())} of {self.settings.max_review_rounds}.",
            self.project_block(),
        ]
        if budget.get("cost_cap"):
            left = budget["cost_cap"] - budget.get("cost_spent", 0.0)
            lines.insert(1, f"Cost: ${budget.get('cost_spent', 0.0):.2f} spent of a ${budget['cost_cap']:.2f} budget (${left:.2f} left). Weigh each action's cost against what is left.")
        if novelty:
            lines.append(f"Novelty of proposal v{novelty.proposal_version}: {novelty.verdict} ({novelty.novelty_strength}, confidence {novelty.confidence:.2f})")
        lines.append("Hypotheses:")
        for hyp in self.state.current_hypotheses():
            state = self.state.claim_state(f"C{hyp.key}")
            lines.append(f"- {hyp.key} [status {hyp.status}, evidence {state}, priority {hyp.priority}] {clip(hyp.statement, 180)}")
        lines.append("Experiments:")
        for spec in registry.latest_specs():
            runs = registry.runs_for(spec.key)
            run_text = ", ".join(
                f"{r.id} {r.status}" + (f" ({self.store.interpretation(r.id).verdict})" if self.store.interpretation(r.id) else "") + (" synthetic" if r.synthetic else "")
                for r in runs
            )
            lines.append(f"- {spec.key} [{spec.status}, {spec.kind}, tests {spec.hypothesis_id}@v{spec.hypothesis_version}] runs: {run_text or 'none'}")
        graph = self.state.graph
        allow = self.settings.allow_synthetic_evidence
        hotspots = graph.uncertainty_hotspots(allow)
        if hotspots:
            lines.append("Uncertainty hotspots (evidence graph): " + "; ".join(f"{h['claim']} {h['state']} u={h['uncertainty']}" for h in hotspots))
        single = graph.claims_with_single_support(allow)
        if single:
            lines.append("Claims resting on one source: " + ", ".join(s["claim"] for s in single[:8]))
        contradicted = graph.contradicted_claims(allow)
        if contradicted:
            lines.append("Contradicted or contested claims: " + ", ".join(c["claim"] for c in contradicted[:8]))
        issues = self.open_review_issues()
        if issues:
            lines.append("Open red-team issues:")
            lines.extend(f"- {i.id} [{i.severity}, {i.dimension}, needs {i.required_action}] {clip(i.description, 160)}" for i in issues[:10])
        follow_ups = [f for i in self.store.interpretations() for f in i.follow_up_hypotheses][-4:]
        if follow_ups:
            lines.append("Follow-up hypotheses suggested by interpretations:\n" + "\n".join(f"- {clip(f.statement, 160)}" for f in follow_ups))
        completion = project.completion or {}
        if completion.get("checks"):
            failing = [c["name"] for c in completion["checks"] if not c["passed"]]
            lines.append("Failing completion checks: " + (", ".join(failing) or "none"))
        lines.append(f"Manuscript: {'drafted' if self.store.manuscript() else 'not started'}.")
        return "\n".join(lines)

    # ------------------------------------------------------------ proposal revision

    def _revise_proposal(self, task: TaskEnvelope):
        proposal = self.state.proposal()
        novelty = self.state.latest_novelty()
        criticisms = self.state.open_criticisms("proposal")
        parts = [self.project_block()]
        if novelty:
            parts.append(f"Novelty: {novelty.verdict} ({novelty.novelty_strength}). {clip(novelty.differentiation, 500)}")
        parts.append("Open criticisms:\n" + "\n".join(f"- {c.id} [{c.severity}] {c.description} -> {c.proposed_test}" for c in criticisms))
        parts.append(
            "Revise the proposal so that it survives these criticisms, or pivot to an adjacent idea within the same topic "
            "if the original cannot be defended. Keep what is defensible."
        )
        out = self.ask("proposal.revision", "\n\n".join(parts), ProposalRevision, {"proposal": proposal.text if proposal else ""})
        if not out.revised_proposal:
            return self.result(task, status="failed", error="no revised proposal returned")
        decision = self.state.log_decision(
            "pivot" if out.action == "pivot" else "proposal_revision",
            f"{'Pivot' if out.action == 'pivot' else 'Revise'} proposal v{proposal.version} to v{proposal.version + 1}",
            out.rationale,
            evidence=[c.id for c in criticisms] + ([f"novelty: {novelty.verdict}"] if novelty else []),
            alternatives=["keep the current proposal", "kill the project"],
            agent_inputs=[task.task_id],
            effects=["re-run focused literature search, novelty analysis and proposal critique"],
        )
        new = self.state.add_proposal_version(out.revised_proposal, out.rationale, decision.id)
        self.state.resolve_criticisms([c.id for c in criticisms], status="superseded")
        return self.result(
            task,
            findings=[f"Proposal v{new.version} ({out.action}): {clip(out.rationale, 200)}"] + [f"change: {c}" for c in out.changes[:5]],
            artifacts=["proposal/proposal.yaml", "decisions/decision_log.yaml"],
            data={"version": new.version, "action": out.action, "decision_id": decision.id, "focus": clip(out.revised_proposal, 200)},
        )
