"""Research Orchestrator: the control plane above the 12 agents.

The orchestrator does not do research itself. It keeps a persistent action queue
(roadmap), routes each action to the responsible agent, follows the lifecycle gates of the
plan (field map, literature loop, proposal validation, hypothesis and design critique),
hands control to the Research Planner for the adaptive research loop, enforces budgets,
records decisions, and decides when the work is complete.
"""

import time
from dataclasses import asdict
from threading import Event
from typing import Any, Callable, Dict, List

from .agents import AGENT_CLASSES, AgentDeps
from .config import Settings
from .llm.router import CallRecord, ModelRouter
from .services.executor import make_executor
from .services.literature import LiteratureSearch
from .services.web_search import WebSearch
from .state.models import Action, AgentResult, Project, TaskEnvelope, now_iso
from .state.research_state import ResearchState
from .state.store import ProjectStore

SIGNIFICANT = {
    "RUN_EXPERIMENT",
    "REPRODUCE_BASELINE",
    "RUN_ABLATION",
    "RUN_ROBUSTNESS_TEST",
    "INVESTIGATE_FAILURE",
    "REVISE_PROPOSAL",
    "SEARCH_LITERATURE",
}
# Share of the cost or token budget after which the manuscript is written before anything else.
ENDGAME_SHARE = 0.7


class AgentFailure(RuntimeError):
    pass


def create_project(
    settings: Settings,
    topic: str,
    question: str = "",
    proposal: str = "",
    seed_papers: List[str] | None = None,
    constraints: List[str] | None = None,
    title: str = "",
) -> ProjectStore:
    project = Project(
        id="",
        title=(title or topic)[:120],
        topic=topic,
        question=question,
        seed_papers=list(seed_papers or []),
        constraints=list(constraints or []),
    )
    store = ProjectStore.create(settings.workspace_dir, project)
    ResearchState(store, settings).add_proposal_version(proposal or question or topic, rationale="initial proposal")
    return store


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        store: ProjectStore,
        router: ModelRouter | None = None,
        literature: LiteratureSearch | None = None,
        executor: Any = None,
        web: WebSearch | None = None,
        on_event: Callable[[Dict[str, Any]], None] | None = None,
        cancel_event: Event | None = None,
    ):
        self.settings = settings
        self.store = store
        self.state = ResearchState(store, settings)
        self.router = router or ModelRouter(settings)
        self.router.listeners.append(self._on_llm_call)
        self.literature = literature or LiteratureSearch.from_settings(settings)
        self.executor = executor or make_executor(settings)
        self.web = web or WebSearch(settings.web_search_provider, settings.tavily_api_key, settings.serpapi_api_key)
        deps = AgentDeps(self.state, self.router, settings, self.literature, self.executor, self.web)
        self.agents = {cls.key: cls(deps) for cls in AGENT_CLASSES}
        self.on_event = on_event
        self.cancel_event = cancel_event or Event()
        self.project = store.load_project()
        self.roadmap: List[Action] = store.roadmap()
        self._task_calls: List[Dict[str, Any]] = []
        self._current_task = ""
        self._finished = False

    # ------------------------------------------------------------ main loop

    def run(self, max_steps: int | None = None) -> Project:
        limit = max_steps or self.settings.max_steps
        started = time.monotonic()
        base_seconds = self.project.usage.seconds
        if self.project.status == "completed":
            return self.project
        retry_failed = self.project.status == "failed"
        self.project.status = "running"
        self.project.error = ""
        self._save()
        self._emit("run_started", step=self.project.step, max_steps=limit)
        try:
            if not self._prepare_resume(retry_failed):
                return self.project
            while not self._finished:
                self.project.usage.seconds = base_seconds + (time.monotonic() - started)
                if self.cancel_event.is_set():
                    self.project.status = "paused"
                    self.project.stop_reason = "cancelled by user"
                    break
                budget_reason = self._budget_exhausted()
                if budget_reason:
                    self._finalize("budget_exhausted", budget_reason)
                    break
                if self.project.step >= limit:
                    self._finalize("step_limit", f"step limit {limit} reached")
                    break
                self._maybe_endgame(limit)
                action = self._next_action()
                self._execute(action)
                if self.project.status == "awaiting_experiments":
                    break
        except Exception as exc:
            self.project.status = "failed"
            self.project.error = f"{type(exc).__name__}: {exc}"[:1000]
            self._emit("run_failed", error=self.project.error)
        finally:
            self.project.usage.seconds = base_seconds + (time.monotonic() - started)
            self._save()
            self._emit("run_stopped", status=self.project.status, outcome=self.project.outcome, step=self.project.step)
        return self.project

    def _prepare_resume(self, retry_failed: bool = False) -> bool:
        # An action still marked running was interrupted (the process was stopped); run it again.
        for action in self.roadmap:
            if action.status == "running":
                action.status = "pending"
        if retry_failed:
            failed = next((a for a in reversed(self.roadmap) if a.status == "failed"), None)
            if failed is not None:
                params = {k: v for k, v in failed.params.items() if k != "attempts"}
                self._insert_front([self._new(failed.type, failed.target, origin="orchestrator", rationale=failed.rationale, **params)])
        registry = self.state.experiments
        runs = registry.runs()
        awaiting = [r for r in runs if r.status == "awaiting_execution"]
        unanalyzed = [r for r in runs if r.status == "completed" and self.store.quant_result(r.id) is None]
        front: List[Action] = [self._new("ANALYZE_RUN", r.id) for r in unanalyzed]
        if awaiting and self.executor.name != "manual":
            front = [self._new("RUN_EXPERIMENT", r.experiment_id, run_id=r.id) for r in awaiting] + front
        elif awaiting and not unanalyzed:
            self.project.status = "awaiting_experiments"
            self.project.stop_reason = "waiting for results of " + ", ".join(r.id for r in awaiting)
            return False
        if front:
            if not self._pending("PLAN"):
                front.append(self._new("PLAN"))
            self._insert_front(front)
        return True

    def _next_action(self) -> Action:
        for action in self.roadmap:
            if action.status == "pending" and action.origin != "planner":
                return action
        planned = [a for a in self.roadmap if a.status == "pending" and a.origin == "planner"]
        if planned:
            return max(planned, key=lambda a: a.priority)
        action = self._new("SCOUT_FIELD" if not self.roadmap else "PLAN")
        self.roadmap.append(action)
        self._save_roadmap()
        return action

    def _execute(self, action: Action) -> None:
        action.status = "running"
        self._save_roadmap()
        self.project.step += 1
        self._emit("action_started", action_id=action.id, type=action.type, target=action.target, origin=action.origin)
        handler = getattr(self, f"_do_{action.type.lower()}")
        try:
            follow = handler(action) or []
            if action.status == "running":
                action.status = "done"
        except Exception as exc:
            action.status = "failed"
            action.result = f"{type(exc).__name__}: {exc}"[:500]
            attempts = int(action.params.get("attempts", 0)) + 1
            if attempts >= 2:
                action.finished_at = now_iso()
                self._save_roadmap()
                raise
            self._emit("action_retry", action_id=action.id, error=action.result)
            follow = [self._new(action.type, action.target, origin=action.origin, rationale=action.rationale, **{**action.params, "attempts": attempts})]
        action.finished_at = now_iso()
        if action.origin == "planner" and action.type in SIGNIFICANT and not self._finished:
            self._supersede_planned()
            if not any(f.type == "PLAN" for f in follow):
                follow.append(self._new("PLAN"))
        self._insert_after(action, follow)
        self._save()
        self._emit("action_finished", action_id=action.id, type=action.type, status=action.status, result=action.result)

    # ------------------------------------------------------------ lifecycle handlers

    def _do_scout_field(self, action: Action):
        self._phase("initialization")
        result = self._task("field_scout", "SCOUT_FIELD", "Map the research field", {})
        action.result = "; ".join(result.findings)
        return [self._new("SEARCH_LITERATURE", lifecycle=True)]

    def _do_search_literature(self, action: Action):
        lifecycle = bool(action.params.get("lifecycle"))
        if lifecycle:
            self._phase("literature_discovery")
        queries = list(action.params.get("queries") or ([action.target] if action.origin == "planner" and action.target else []))
        result = self._task(
            "literature",
            "SEARCH_LITERATURE",
            "Build and extend the literature evidence base",
            {"focus": action.params.get("focus", action.rationale), "queries": queries},
        )
        action.result = "; ".join(result.findings)
        if lifecycle:
            return [self._new("ANALYZE_GAPS", lifecycle=True)]
        if action.origin == "planner":
            return [self._new("ASSESS_NOVELTY")]
        return []

    def _do_analyze_gaps(self, action: Action):
        result = self._task("gap_novelty", "ANALYZE_GAPS", "Identify research gaps in the literature", {"mode": "gaps"})
        action.result = "; ".join(result.findings[:3])
        if not action.params.get("lifecycle"):
            return []
        rounds = int(self.store.literature_map().get("rounds", 0))
        if not result.data.get("coverage_sufficient", True) and rounds < self.settings.max_literature_rounds:
            decision = self.state.log_decision(
                "literature_loop",
                f"Run literature round {rounds + 1}",
                "the gap analysis judged literature coverage insufficient",
                evidence=result.data.get("gap_ids", []),
                alternatives=["proceed to novelty analysis with current coverage"],
                agent_inputs=[result.task_id],
                effects=["additional targeted searches"],
            )
            return [self._new("SEARCH_LITERATURE", lifecycle=True, queries=result.data.get("missing_queries", []), decision_id=decision.id)]
        return [self._new("ASSESS_NOVELTY", gate=True)]

    def _do_assess_novelty(self, action: Action):
        if action.params.get("gate"):
            self._phase("proposal_validation")
        result = self._task("gap_novelty", "ASSESS_NOVELTY", "Assess the novelty of the current proposal", {"mode": "novelty"})
        action.result = "; ".join(result.findings)
        return [self._new("CRITIQUE_PROPOSAL")] if action.params.get("gate") else []

    def _do_critique_proposal(self, action: Action):
        self._phase("proposal_validation")
        result = self._task("proposal_critic", "CRITIQUE_PROPOSAL", "Attack the proposal before reviewers do", {"target": "proposal"})
        action.result = "; ".join(result.findings)
        verdict = result.data.get("verdict", "proceed")
        novelty = self.state.latest_novelty()
        revisions = len(self.store.proposal_versions()) - 1
        evidence = [result.data.get("assessment_id", ""), *result.data.get("serious_issue_ids", [])]
        needs_revision = verdict in {"revise", "pivot"} or (novelty is not None and novelty.verdict == "already_done")
        if needs_revision and revisions < self.settings.max_proposal_revisions:
            self.state.log_decision(
                "proposal_gate",
                "Revise the proposal before forming hypotheses",
                f"critic verdict '{verdict}'" + (f", novelty '{novelty.verdict}'" if novelty else ""),
                evidence=evidence,
                alternatives=["proceed with the known risks", "kill the project"],
                agent_inputs=[result.task_id],
                effects=["proposal revision, focused literature search, novelty re-check, new critique"],
            )
            return [self._new("REVISE_PROPOSAL", gate=True)]
        if verdict == "kill" and novelty is not None and novelty.verdict == "already_done" and novelty.confidence >= 0.6:
            return [self._new("KILL", rationale="critic and novelty analysis agree the idea is already covered and cannot be defended")]
        self.state.log_decision(
            "proposal_gate",
            "Proceed to hypothesis formation",
            f"critic verdict '{verdict}'" + (" (revision budget used)" if needs_revision else ""),
            evidence=evidence,
            alternatives=["revise the proposal again", "kill the project"],
            agent_inputs=[result.task_id],
            effects=["open criticisms are carried into hypothesis and experiment design"],
        )
        return [self._new("FORM_HYPOTHESIS", mode="design")]

    def _do_revise_proposal(self, action: Action):
        self._phase("proposal_validation")
        result = self._task("planner", "REVISE_PROPOSAL", "Revise or pivot the proposal", {"mode": "revise_proposal"})
        action.result = "; ".join(result.findings[:2])
        return [
            self._new("SEARCH_LITERATURE", focus=result.data.get("focus", "")),
            self._new("ASSESS_NOVELTY"),
            self._new("CRITIQUE_PROPOSAL"),
        ]

    def _do_form_hypothesis(self, action: Action):
        mode = action.params.get("mode") or ("follow_up" if action.origin == "planner" else "design")
        if mode == "design":
            self._phase("hypothesis_formation")
        inputs: Dict[str, Any] = {"mode": mode}
        if mode == "follow_up":
            inputs.update(statement=action.target or action.params.get("statement", ""), rationale=action.rationale, evidence=action.params.get("evidence", []))
        if mode == "refine":
            inputs.update(targets=action.params.get("targets", []), decision_id=action.params.get("decision_id", ""))
        result = self._task("hypothesis_designer", "FORM_HYPOTHESIS", f"Hypothesis work ({mode})", inputs)
        action.result = "; ".join(result.findings[:4])
        ids = result.data.get("hypothesis_ids", [])
        if mode == "design":
            return [self._new("CRITIQUE_DESIGN", target="hypotheses", ids=ids)]
        if mode == "follow_up" and ids:
            return [self._new("DESIGN_EXPERIMENT", ids[0], mode="request", kind="primary")]
        return []

    def _do_revise_hypothesis(self, action: Action):
        runs = [r.id for r in self.state.experiments.runs() if r.hypothesis_id == action.target and r.status == "completed"]
        decision = self.state.log_decision(
            "hypothesis_revision",
            f"Revise {action.target}",
            action.rationale,
            evidence=runs,
            alternatives=["keep the hypothesis", "abandon the hypothesis"],
            effects=["new hypothesis version; the original remains on record and needs new experiments"],
        )
        result = self._task(
            "hypothesis_designer",
            "REVISE_HYPOTHESIS",
            f"Revise hypothesis {action.target} after results",
            {"mode": "revise", "hypothesis_id": action.target, "reason": action.rationale, "evidence": runs, "decision_id": decision.id},
        )
        action.result = "; ".join(result.findings)
        return [self._new("DESIGN_EXPERIMENT", mode="design")]

    def _do_critique_design(self, action: Action):
        target = action.params.get("target", action.target or "experiments")
        ids = list(action.params.get("ids", []))
        result = self._task("proposal_critic", "CRITIQUE_DESIGN", f"Check that the {target} are testable and sound", {"target": target, "ids": ids})
        action.result = "; ".join(result.findings)
        serious = result.data.get("serious_issue_ids", [])
        if target == "hypotheses":
            self._phase("hypothesis_formation")
            follow = []
            if serious:
                decision = self.state.log_decision(
                    "design_gate",
                    "Refine hypotheses before experimental design",
                    f"{len(serious)} serious criticism(s) of the hypotheses",
                    evidence=serious,
                    alternatives=["design experiments for the hypotheses as written"],
                    agent_inputs=[result.task_id],
                    effects=["new hypothesis versions (pre-registration)"],
                )
                follow.append(self._new("FORM_HYPOTHESIS", mode="refine", targets=result.data.get("serious_targets", []), decision_id=decision.id))
            follow.append(self._new("DESIGN_EXPERIMENT", mode="design", after=action.params.get("after")))
            return follow
        if serious:
            return [self._new("DESIGN_EXPERIMENT", mode="revise", ids=ids, serious=serious, after=action.params.get("after"))]
        return self._approve(ids, [], action.params.get("after"), result.task_id)

    def _do_design_experiment(self, action: Action):
        mode = action.params.get("mode") or ("request" if action.target else "design")
        if mode == "design":
            self._phase("experimental_design")
        inputs: Dict[str, Any] = {"mode": mode, "ids": action.params.get("ids", [])}
        if mode == "request":
            inputs.update(hypothesis_id=action.target, kind=action.params.get("kind", "primary"), rationale=action.rationale)
        result = self._task("experiment_designer", "DESIGN_EXPERIMENT", f"Experiment design ({mode})", inputs)
        action.result = "; ".join(result.findings[:4])
        ids = result.data.get("experiment_ids", [])
        after = action.params.get("after")
        if after is None:
            after = [{"type": "RUN_APPROVED"}] if action.params.get("run") else ([{"type": "PLAN"}] if mode == "design" else [])
        if mode == "revise":
            return self._approve(action.params.get("ids") or ids, action.params.get("serious", []), after, result.task_id)
        if not ids:
            return self._expand_after(after, [])
        return [self._new("CRITIQUE_DESIGN", target="experiments", ids=ids, after=after)]

    def _approve(self, ids: List[str], serious: List[str], after, task_id: str):
        drafts = [s for s in self.state.experiments.latest_specs() if s.status == "draft" and (not ids or s.id in ids)]
        approved = []
        for spec in drafts:
            notes = [f"{c.id} [{c.severity}] {c.description}" for c in self.state.open_criticisms("experiments") if c.target_id in {spec.id, ""}]
            approved.append(self.state.approve_experiment(spec.key, notes))
        if approved:
            self.state.log_decision(
                "design_approval",
                "Approve " + ", ".join(s.key for s in approved),
                "design critique passed" + (" after revision" if serious else ""),
                evidence=serious,
                alternatives=["revise the designs again"],
                agent_inputs=[task_id],
                effects=["approved experiments are frozen; any later change creates a new version"],
            )
        self._phase("experimental_execution")
        return self._expand_after(after or [{"type": "PLAN"}], [s.id for s in approved])

    def _expand_after(self, after, approved_ids: List[str]) -> List[Action]:
        out = []
        for item in after or []:
            if item.get("type") == "RUN_APPROVED":
                out.extend(self._new("RUN_EXPERIMENT", spec_id) for spec_id in approved_ids)
            elif item.get("type") == "PLAN" and self._pending("PLAN"):
                continue
            else:
                out.append(self._new(item["type"], item.get("target", "")))
        return out

    # ------------------------------------------------------------ research loop handlers

    def _do_run_experiment(self, action: Action):
        run_id = action.params.get("run_id")
        if not run_id and len(self.state.experiments.runs()) >= self.settings.max_experiment_runs:
            action.status = "skipped"
            action.result = "experiment budget exhausted"
            return []
        self._phase("experimental_execution" if not self.store.quant_results() else "research_loop")
        inputs = {"run_id": run_id} if run_id else {"experiment_id": action.target, "previous_failure": action.params.get("previous_failure")}
        result = self._task("experiment_engineer", "RUN_EXPERIMENT", f"Implement and run {action.target or run_id}", inputs, allow_failure=True)
        action.result = "; ".join(result.findings) or result.error
        status = result.data.get("status")
        run_id = result.data.get("run_id", run_id)
        if status == "awaiting_execution":
            self.state.log_decision(
                "blocker",
                f"Pause for manual execution of {run_id}",
                "the experiment executor is manual: generated code needs human review and execution",
                evidence=[run_id],
                agent_inputs=[result.task_id],
                effects=["project paused until results are ingested"],
            )
            self.project.status = "awaiting_experiments"
            self.project.stop_reason = f"waiting for results of {run_id}"
            return []
        if status == "completed":
            return [self._new("ANALYZE_RUN", run_id)]
        action.status = "failed"
        return []

    def _do_analyze_run(self, action: Action):
        self._phase("research_loop")
        quant = self._task("quant_analyst", "ANALYZE_RUN", f"Quantify what {action.target} demonstrates", {"run_id": action.target})
        interp = self._task("interpreter", "INTERPRET_RUN", f"Interpret {action.target}", {"run_id": action.target})
        action.result = "; ".join(quant.findings[:2] + interp.findings)
        return []

    def _kind_run(self, action: Action, kind: str):
        return [self._new("DESIGN_EXPERIMENT", action.target, rationale=action.rationale, mode="request", kind=kind, run=True)]

    def _do_reproduce_baseline(self, action: Action):
        return self._kind_run(action, "baseline_reproduction")

    def _do_run_ablation(self, action: Action):
        return self._kind_run(action, "ablation")

    def _do_run_robustness_test(self, action: Action):
        return self._kind_run(action, "robustness")

    def _do_investigate_failure(self, action: Action):
        registry = self.state.experiments
        run = registry.run(action.target)
        spec = registry.spec(run.spec_key)
        if run.status == "failed":
            if len(registry.runs_for(spec.key)) < self.settings.max_runs_per_spec:
                return [self._new("RUN_EXPERIMENT", spec.id, previous_failure=run.id)]
            action.result = f"{spec.key} reached its run limit; no retry"
            return []
        interp = self.store.interpretation(run.id)
        leads = [e.follow_up for e in (interp.competing_explanations if interp else []) if e.follow_up]
        rationale = "; ".join([action.rationale, *leads[:2]]).strip("; ")
        return [self._new("DESIGN_EXPERIMENT", spec.hypothesis_id, rationale=rationale, mode="request", kind="validation", run=True)]

    def _do_plan(self, action: Action):
        runs = self.state.experiments.runs()
        budget = {
            "step": self.project.step,
            "max_steps": self.settings.max_steps,
            "runs_left": self.settings.max_experiment_runs - len(runs),
        }
        result = self._task("planner", "PLAN", "Decide what the swarm should do next", {"budget": budget})
        self._supersede_planned()
        for raw in result.data.get("actions", []):
            planned = Action.model_validate(raw)
            planned.id = self.store.next_id("A", width=3)
            planned.status = "pending"
            planned.origin = "planner"
            planned.created_at = now_iso()
            self.roadmap.append(planned)
        action.result = "; ".join(result.recommendations[:3])
        self._save_roadmap()
        if not result.data.get("actions"):
            return [self._new("STOP", rationale="planner found no further informative action")]
        return []

    def _do_write_paper(self, action: Action):
        self._phase("paper_production")
        result = self._task("paper_architect", "WRITE_PAPER", "Turn the evidence into a defensible manuscript", {})
        action.result = "; ".join(result.findings)
        return [self._new("REQUEST_REVIEW", final=bool(action.params.get("final")))]

    def _do_request_review(self, action: Action):
        if not self.store.manuscript():
            return [self._new("WRITE_PAPER", final=bool(action.params.get("final")))]
        self._phase("paper_production")
        result = self._task("red_team", "REQUEST_REVIEW", "Simulate skeptical peer review", {})
        action.result = "; ".join(result.findings)
        completion = self._completion_check()
        self.agents["paper_architect"].render_manuscript(self._status_line(completion))
        rounds = len(self.store.reviews())
        if completion["ready"]:
            return [self._new("STOP", reason="publication_ready")]
        if action.params.get("final") or rounds >= self.settings.max_review_rounds:
            return [self._new("STOP", reason="review_rounds_exhausted" if not action.params.get("final") else "final_review")]
        self._supersede_planned()
        return [] if self._pending("PLAN") else [self._new("PLAN")]

    def _do_stop(self, action: Action):
        if not self.store.manuscript():
            return [self._new("WRITE_PAPER", final=True)]
        self._finalize("auto", action.rationale or action.params.get("reason", "stop requested"))
        return []

    def _do_kill(self, action: Action):
        self.state.log_decision(
            "kill",
            "Abandon the research idea",
            action.rationale,
            evidence=[a.id for a in self.store.assessments()[-2:]],
            alternatives=["revise the proposal", "continue with known risks"],
            effects=["no further research actions"],
        )
        self._finalize("killed", action.rationale)
        return []

    # ------------------------------------------------------------ completion and budgets

    def _completion_check(self) -> Dict[str, Any]:
        allow = self.settings.allow_synthetic_evidence
        claim_map = self.store.claim_map()
        claims = self.state.refresh_claim_states(claim_map).claims if claim_map else []
        central = [c for c in claims if c.importance == "central" and c.kind in {"contribution", "finding"}]
        exceeding = [c.id for c in claims if c.kind in {"contribution", "finding"} and c.state not in {"SUPPORTED", "PARTIALLY_SUPPORTED"}]
        registry = self.state.experiments
        hyps = {h.id: h for h in self.state.current_hypotheses(include_closed=False)}
        required = [s for s in registry.latest_specs() if s.status == "approved" and s.hypothesis_id in hyps and s.hypothesis_version == hyps[s.hypothesis_id].version]
        incomplete = [s.key for s in required if not any(r.status == "completed" for r in registry.runs_for(s.key))]
        reports = self.store.reviews()
        open_serious = [i.id for r in reports for i in r.issues if i.status == "open" and i.severity in {"critical", "major"}]
        draft = self.store.draft()
        qa = self.store.read_yaml("paper/qa.yaml", {}) or {}
        completed = [r for r in registry.runs() if r.status == "completed"]
        synthetic = [r.id for r in completed if r.synthetic]
        reproducible = bool(completed) and all(r.code_version and r.seeds and r.config for r in completed) and (allow or not synthetic)
        supported = [c for c in central if c.state == "SUPPORTED"]
        checks = [
            {"name": "manuscript_written", "passed": bool(self.store.manuscript()), "detail": ""},
            {"name": "central_claims_supported", "passed": bool(central) and len(supported) == len(central), "detail": f"{len(supported)} of {len(central)} central claims SUPPORTED"},
            {"name": "claims_within_evidence", "passed": not exceeding, "detail": ", ".join(exceeding)},
            {"name": "required_experiments_complete", "passed": not incomplete, "detail": ", ".join(incomplete)},
            {"name": "reviewer_objections_addressed", "passed": bool(reports) and not open_serious, "detail": ", ".join(open_serious) or ("no review yet" if not reports else "")},
            {"name": "limitations_documented", "passed": bool(draft and draft.limitations.strip()), "detail": ""},
            {"name": "citations_verified", "passed": not qa.get("unknown_citations") and (allow or not qa.get("synthetic_citations")), "detail": ", ".join(qa.get("unknown_citations", []) + qa.get("synthetic_citations", []))},
            {"name": "results_reproducible", "passed": reproducible, "detail": ("synthetic runs: " + ", ".join(synthetic)) if synthetic else ""},
        ]
        completion = {"ready": all(c["passed"] for c in checks), "checks": checks, "checked_at": now_iso()}
        self.project.completion = completion
        return completion

    def _status_line(self, completion: Dict[str, Any]) -> str:
        if completion.get("ready"):
            return "Publication-ready (all completion checks pass)"
        failing = [c["name"] for c in completion.get("checks", []) if not c["passed"]]
        return "Draft with open issues (failing checks: " + ", ".join(failing) + ")"

    def _budget_exhausted(self) -> str:
        usage = self.project.usage
        if self.settings.max_total_tokens and usage.total_tokens >= self.settings.max_total_tokens:
            return f"token budget {self.settings.max_total_tokens} reached"
        if self.settings.max_cost_usd and usage.cost_usd >= self.settings.max_cost_usd:
            return f"cost budget ${self.settings.max_cost_usd} reached"
        if self.settings.max_seconds and usage.seconds >= self.settings.max_seconds:
            return f"time budget {self.settings.max_seconds}s reached"
        return ""

    def _maybe_endgame(self, limit: int) -> None:
        if self.store.manuscript() or self._pending("WRITE_PAPER") or self._pending("REQUEST_REVIEW"):
            return
        usage, s = self.project.usage, self.settings
        if limit - self.project.step <= 2:
            reason = f"{limit - self.project.step} step(s) left and no manuscript yet"
        elif s.max_cost_usd and usage.cost_usd >= ENDGAME_SHARE * s.max_cost_usd:
            reason = f"${usage.cost_usd:.2f} of the ${s.max_cost_usd:.2f} cost budget spent and no manuscript yet"
        elif s.max_total_tokens and usage.total_tokens >= ENDGAME_SHARE * s.max_total_tokens:
            reason = f"{usage.total_tokens} of {s.max_total_tokens} budgeted tokens used and no manuscript yet"
        else:
            return
        self.state.log_decision(
            "endgame",
            "Write the manuscript before the budget runs out",
            reason,
            effects=["remaining research actions are deferred"],
        )
        self._insert_front([self._new("WRITE_PAPER", final=True)])

    def _finalize(self, outcome: str, reason: str) -> None:
        completion = self._completion_check()
        if self.store.draft():
            self.agents["paper_architect"].render_manuscript(self._status_line(completion))
        if outcome == "auto":
            outcome = "publication_ready" if completion["ready"] else ("draft_with_open_issues" if self.store.manuscript() else "no_manuscript")
        failing = [c["name"] for c in completion["checks"] if not c["passed"]]
        self.state.log_decision(
            "completion",
            f"Finish with outcome '{outcome}'",
            reason,
            evidence=failing,
            effects=["research loop stopped"],
        )
        for action in self.roadmap:
            if action.status == "pending":
                action.status = "skipped"
        self._save_roadmap()
        self.project.outcome = outcome
        self.project.stop_reason = reason
        self.project.status = "completed"
        self.project.phase = "completed"
        self._finished = True
        self._save()

    # ------------------------------------------------------------ agents and events

    def _task(self, agent_key: str, task_type: str, objective: str, inputs: Dict[str, Any], allow_failure: bool = False) -> AgentResult:
        task = TaskEnvelope(
            task_id=self.store.next_id("T", width=3),
            type=task_type,
            agent=agent_key,
            objective=objective,
            inputs={k: v for k, v in inputs.items() if v not in (None, "", [])},
            constraints=list(self.project.constraints),
        )
        self._current_task = task.task_id
        self._task_calls = []
        self._emit("task_started", task_id=task.task_id, agent=agent_key, type=task_type, objective=objective)
        start = time.perf_counter()
        try:
            result = self.agents[agent_key].run(task)
        except Exception as exc:
            result = AgentResult(task_id=task.task_id, agent=agent_key, status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        result.duration_s = round(time.perf_counter() - start, 3)
        result.llm_calls = list(self._task_calls)
        result.violations = self.state.drain_violations()
        self.store.log_activity({"timestamp": now_iso(), "event": "task", "task": task.model_dump(mode="json"), "result": result.model_dump(mode="json")})
        self._emit("task_finished", task_id=task.task_id, agent=agent_key, status=result.status, findings=result.findings[:3], violations=len(result.violations))
        self._current_task = ""
        self._save()
        if result.status == "failed" and not allow_failure:
            raise AgentFailure(f"{agent_key} failed: {result.error}")
        return result

    def _on_llm_call(self, record: CallRecord) -> None:
        usage = self.project.usage
        usage.llm_calls += 1
        usage.prompt_tokens += record.prompt_tokens
        usage.completion_tokens += record.completion_tokens
        usage.total_tokens += record.total_tokens
        usage.cost_usd += record.cost_usd
        usage.escalations += int(record.escalated)
        tier = usage.by_tier.setdefault(record.tier, {"calls": 0, "total_tokens": 0, "cost_usd": 0.0, "failures": 0})
        tier["calls"] += 1
        tier["total_tokens"] += record.total_tokens
        tier["cost_usd"] += record.cost_usd
        tier["failures"] += int(not record.success)
        entry = {**asdict(record), "task_id": self._current_task, "step": self.project.step}
        self.store.log_llm_call(entry)
        self._task_calls.append({k: entry[k] for k in ("task", "tier", "provider", "model", "reasoning_effort", "total_tokens", "success", "escalated")})

    def _emit(self, event: str, **data: Any) -> None:
        payload = {"timestamp": now_iso(), "event": event, "project": self.project.id, "phase": self.project.phase, **data}
        if event != "task_finished":
            self.store.log_activity(payload)
        if self.on_event:
            try:
                self.on_event(payload)
            except Exception:
                pass

    def _phase(self, phase: str) -> None:
        if self.project.phase != phase:
            self.project.phase = phase
            self._emit("phase", phase=phase)

    # ------------------------------------------------------------ roadmap helpers

    def _new(self, type_: str, target: str = "", origin: str = "orchestrator", rationale: str = "", **params: Any) -> Action:
        return Action(
            id=self.store.next_id("A", width=3),
            type=type_,
            target=target,
            rationale=rationale,
            origin=origin,
            status="pending",
            params={k: v for k, v in params.items() if v is not None},
            created_at=now_iso(),
        )

    def _pending(self, type_: str) -> bool:
        return any(a.status == "pending" and a.type == type_ for a in self.roadmap)

    def _insert_after(self, anchor: Action, actions: List[Action]) -> None:
        if not actions:
            self._save_roadmap()
            return
        index = next((i for i, a in enumerate(self.roadmap) if a.id == anchor.id), len(self.roadmap) - 1)
        self.roadmap[index + 1 : index + 1] = actions
        self._save_roadmap()

    def _insert_front(self, actions: List[Action]) -> None:
        index = next((i for i, a in enumerate(self.roadmap) if a.status == "pending"), len(self.roadmap))
        self.roadmap[index:index] = actions
        self._save_roadmap()

    def _supersede_planned(self) -> None:
        for action in self.roadmap:
            if action.status == "pending" and action.origin == "planner":
                action.status = "superseded"

    def _save_roadmap(self) -> None:
        self.state.save_roadmap(self.roadmap)

    def _save(self) -> None:
        self.store.save_project(self.project)
