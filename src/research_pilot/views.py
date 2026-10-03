"""Read-only views of a project, shared by the CLI, the API and the dashboard."""

from collections import defaultdict
from typing import Any, Callable, Dict, List

from pydantic import BaseModel

from .config import Settings
from .llm.router import ModelRouter
from .llm.tasks import TASKS
from .services.executor import check_coverage, load_records
from .state.research_state import ResearchState
from .state.store import ProjectStore

AGENT_NAMES = {
    "field_scout": "Field Scout",
    "literature": "Literature Intelligence",
    "gap_novelty": "Gap & Novelty Analyst",
    "proposal_critic": "Proposal Critic",
    "hypothesis_designer": "Hypothesis Designer",
    "experiment_designer": "Experiment Designer",
    "experiment_engineer": "Experiment Engineer",
    "quant_analyst": "Quantitative Analyst",
    "interpreter": "Scientific Interpreter",
    "planner": "Research Planner",
    "paper_architect": "Paper Architect",
    "red_team": "Red-Team Reviewer",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def _state(store: ProjectStore, settings: Settings) -> ResearchState:
    return ResearchState(store, settings)


def overview(store: ProjectStore, settings: Settings) -> Dict[str, Any]:
    state = _state(store, settings)
    project = store.load_project()
    activity = store.activity()
    return {
        "project": jsonable(project),
        "counts": _counts(store, state),
        "agents": _agent_activity(store, activity),
        "active_agent": _active_agent(project.status, activity),
        "max_steps": settings.max_steps,
        "next_actions": [jsonable(a) for a in store.roadmap() if a.status == "pending"][:8],
        "evidence": state.graph.summary(settings.allow_synthetic_evidence),
        "has_manuscript": bool(store.manuscript()),
    }


def _agent_activity(store: ProjectStore, activity: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per agent: finished tasks, the latest result and model calls by tier."""
    agents: Dict[str, Dict[str, Any]] = {
        key: {"name": name, "tasks": 0, "last": "", "last_status": "", "last_finding": "", "tiers": {}} for key, name in AGENT_NAMES.items()
    }
    for row in activity:
        result = row.get("result", {}) if row.get("event") == "task" else {}
        agent = agents.get(result.get("agent"))
        if agent is None:
            continue
        agent["tasks"] += 1
        agent["last"] = row.get("timestamp", "")
        agent["last_status"] = result.get("status", "")
        agent["last_finding"] = (result.get("findings") or [""])[0]
    for call in store.llm_calls():
        spec = TASKS.get(call.get("task", ""))
        agent = agents.get(spec.issued_by) if spec else None
        if agent is not None:
            agent["tiers"][call["tier"]] = agent["tiers"].get(call["tier"], 0) + 1
    return agents


def _active_agent(status: str, activity: List[Dict[str, Any]]) -> str:
    """The agent of the latest started task, if that task has not finished yet."""
    if status != "running":
        return ""
    latest = next((row for row in reversed(activity) if row.get("event") in {"task", "task_started"}), None)
    return latest.get("agent", "") if latest and latest.get("event") == "task_started" else ""


def _counts(store: ProjectStore, state: ResearchState) -> Dict[str, int]:
    registry = state.experiments
    runs = registry.runs()
    reviews = store.reviews()
    proposal = state.proposal()
    return {
        "papers": len(store.papers()),
        "gaps": len(state.gaps()),
        "proposal_versions": proposal.version if proposal else 0,
        "hypotheses": len(state.current_hypotheses()),
        "hypothesis_versions": len(state.hypotheses()),
        "experiments": len(registry.latest_specs()),
        "runs": len(runs),
        "runs_completed": sum(1 for r in runs if r.status == "completed"),
        "runs_failed": sum(1 for r in runs if r.status == "failed"),
        "runs_awaiting": sum(1 for r in runs if r.status == "awaiting_execution"),
        "decisions": len(store.decisions()),
        "review_rounds": len(reviews),
        "open_review_issues": sum(1 for r in reviews for i in r.issues if i.status == "open"),
        "violations": len(store.read_jsonl("logs/violations.jsonl")),
    }


def _hypotheses(store: ProjectStore, settings: Settings):
    state = _state(store, settings)
    out = []
    for hyp in state.hypotheses():
        row = jsonable(hyp)
        row["evidence_state"] = state.claim_state(f"C{hyp.key}")
        out.append(row)
    return out


def _experiments(store: ProjectStore, settings: Settings):
    registry = _state(store, settings).experiments
    return {"specs": jsonable(registry.specs()), "runs": [jsonable(r) for r in registry.runs()]}


def _analysis(store: ProjectStore, settings: Settings):
    return {"quantitative": jsonable(store.quant_results()), "interpretations": jsonable(store.interpretations())}


SECTIONS: Dict[str, Callable[[ProjectStore, Settings], Any]] = {
    "field": lambda s, _: jsonable(s.field_map()),
    "papers": lambda s, _: jsonable(s.papers()),
    "literature": lambda s, _: {"map": s.literature_map(), "clusters": jsonable(s.clustering()), "citations": s.citation_graph()},
    "gaps": lambda s, _: {**jsonable(s.gap_analysis_meta()), "gaps": jsonable(s.gaps())},
    "proposal": lambda s, _: jsonable(s.proposal_versions()),
    "novelty": lambda s, _: jsonable(s.novelty_analyses()),
    "critique": lambda s, _: jsonable(s.assessments()),
    "hypotheses": _hypotheses,
    "experiments": _experiments,
    "analysis": _analysis,
    "decisions": lambda s, _: jsonable(s.decisions()),
    "roadmap": lambda s, _: jsonable(s.roadmap()),
    "claims": lambda s, _: jsonable(s.claim_map()),
    "reviews": lambda s, _: jsonable(s.reviews()),
    "qa": lambda s, _: s.read_yaml("paper/qa.yaml", {}),
    "manuscript": lambda s, _: s.manuscript(),
    "activity": lambda s, _: s.activity(limit=200),
    "violations": lambda s, _: s.read_jsonl("logs/violations.jsonl"),
}


def section(store: ProjectStore, settings: Settings, name: str) -> Any:
    if name not in SECTIONS:
        raise KeyError(f"Unknown section '{name}'. Choose from: {', '.join(SECTIONS)}")
    return SECTIONS[name](store, settings)


EVIDENCE_QUERIES = {
    "single-support": "claims_with_single_support",
    "contradicted": "contradicted_claims",
    "untested": "untested_hypotheses",
    "unsupported-statements": "unsupported_statements",
    "weak-literature": "weak_literature_claims",
    "hotspots": "uncertainty_hotspots",
}


def evidence_report(store: ProjectStore, settings: Settings, include_graph: bool = False) -> Dict[str, Any]:
    graph = _state(store, settings).graph
    allow = settings.allow_synthetic_evidence
    queries = {}
    for label, method in EVIDENCE_QUERIES.items():
        fn = getattr(graph, method)
        queries[label] = fn() if method in {"untested_hypotheses", "weak_literature_claims"} else fn(allow)
    states = {cid: state for cid, state in graph.claim_states(allow).items() if not cid.startswith("LC-")}
    report = {"summary": graph.summary(allow), "claim_states": states, "queries": queries}
    if include_graph:
        report["graph"] = graph.to_dict()
    return report


def metrics_report(store: ProjectStore) -> Dict[str, Any]:
    project = store.load_project()
    by_task: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost_usd": 0.0, "failures": 0, "escalations": 0, "tier": "", "models": set()})
    for call in store.llm_calls():
        row = by_task[call["task"]]
        row["calls"] += 1
        row["tokens"] += int(call.get("total_tokens", 0))
        row["cost_usd"] += float(call.get("cost_usd", 0.0))
        row["failures"] += int(not call.get("success", True))
        row["escalations"] += int(bool(call.get("escalated")))
        row["tier"] = call.get("tier", row["tier"])
        row["models"].add(f"{call.get('provider')}/{call.get('model')}")
    tasks = []
    for task, row in sorted(by_task.items(), key=lambda kv: -kv[1]["calls"]):
        spec = TASKS.get(task)
        tasks.append(
            {
                "task": task,
                "label": spec.label if spec else task,
                "difficulty": spec.difficulty.label if spec else "",
                **{k: v for k, v in row.items() if k != "models"},
                "models": sorted(row["models"]),
            }
        )
    return {"usage": jsonable(project.usage), "tasks": tasks}


def routing_table(settings: Settings) -> List[Dict[str, Any]]:
    return ModelRouter(settings).routing_table()


def ingest_results(store: ProjectStore, settings: Settings, run_id: str, text: str) -> Dict[str, Any]:
    state = _state(store, settings)
    registry = state.experiments
    run = registry.run(run_id)
    if run is None:
        raise KeyError(f"Unknown run: {run_id}")
    records = load_records(text)
    spec = registry.spec(run.spec_key)
    problems = check_coverage(spec, run.seeds, records) if spec else []
    run = registry.ingest(run_id, records, problems)
    return {"run_id": run.id, "status": run.status, "records": len(run.records), "problems": problems, "synthetic": run.synthetic}
