import subprocess
import sys

from research_pilot.agents import AgentDeps
from research_pilot.agents.interpreter import ScientificInterpreter
from research_pilot.agents.quant_analyst import analyze_run
from research_pilot.llm.mock import MockResponder
from research_pilot.llm.router import ModelRouter
from research_pilot.orchestrator import Orchestrator, create_project
from research_pilot.services.executor import SimulatedExecutor, parse_result_lines
from research_pilot.state.models import ExperimentSpec, Hypothesis, MetricSpec, RunRecord, TaskEnvelope
from research_pilot.state.research_state import ResearchState
from research_pilot.views import overview


def run_project(settings, **kw):
    store = create_project(settings, "retrieval augmented generation robustness", question="Does RAG stay robust under shift?")
    events = []
    project = Orchestrator(settings, store, on_event=events.append, **kw).run()
    return store, project, events


def test_full_swarm_run_with_mock_models(mock_settings):
    settings = mock_settings()
    store, project, events = run_project(settings)
    assert project.status == "completed", project.error
    assert project.outcome == "draft_with_open_issues"

    view = overview(store, settings)
    assert all(agent["tasks"] > 0 for agent in view["agents"].values()), view["agents"]
    assert {"lite", "strong", "coding"} <= set(project.usage.by_tier)
    assert view["counts"]["violations"] == 0

    kinds = [d.kind for d in store.decisions()]
    for kind in ("proposal_gate", "proposal_revision", "design_gate", "design_approval", "plan", "completion"):
        assert kind in kinds, kinds
    state = ResearchState(store, settings)
    assert len(store.proposal_versions()) == 2
    assert any(h.status == "superseded" for h in state.hypotheses())
    assert any(s.version > 1 for s in state.experiments.specs())
    assert len(store.reviews()) == 2 and store.reviews()[0].issues[0].status == "resolved"

    manuscript = store.manuscript()
    assert "## Claim-Evidence Map" in manuscript and "synthetic" in manuscript
    checks = {c["name"]: c["passed"] for c in project.completion["checks"]}
    assert checks["manuscript_written"] and not checks["results_reproducible"] and not checks["central_claims_supported"]
    assert {e["event"] for e in events} >= {"phase", "task_started", "task_finished", "action_started"}


class DecisiveExecutor(SimulatedExecutor):
    """Synthetic results with a large, consistent effect for the method arm."""

    def execute(self, run_dir, spec, seeds, run_id):
        records = [
            RunRecord(arm=arm, seed=seed, metrics={m.name: (0.9 if i == 0 else 0.6) + 0.001 * seed for m in spec.metrics})
            for i, arm in enumerate(spec.arms)
            for seed in seeds
        ]
        outcome = super().execute(run_dir, spec, seeds, run_id)
        outcome.records = records
        return outcome


def test_swarm_reaches_publication_ready_when_evidence_suffices(mock_settings):
    settings = mock_settings(allow_synthetic_evidence=True)
    store, project, _ = run_project(settings, executor=DecisiveExecutor())
    assert project.status == "completed", project.error
    failing = [c for c in project.completion["checks"] if not c["passed"]]
    assert project.outcome == "publication_ready", failing
    assert "Publication-ready" in store.manuscript()


def test_manual_execution_pauses_and_resumes(mock_settings):
    settings = mock_settings(experiment_executor="manual")
    store = create_project(settings, "manual flow topic")
    project = Orchestrator(settings, store).run()
    assert project.status == "awaiting_experiments"
    state = ResearchState(store, settings)
    (run,) = [r for r in state.experiments.runs() if r.status == "awaiting_execution"]
    run_dir = store.path(f"experiments/runs/{run.id}")
    lines = []
    for seed in run.seeds:
        out = subprocess.run([sys.executable, "run.py", "--seed", str(seed), "--output-dir", "artifacts"], cwd=run_dir, capture_output=True, text=True, check=True)
        lines += out.stdout.splitlines()
    ingested = state.experiments.ingest(run.id, parse_result_lines(lines), [])
    assert ingested.synthetic, "results of mock-generated code stay synthetic"
    project = Orchestrator(settings, store).run()
    assert store.quant_result(run.id) is not None and store.interpretation(run.id) is not None
    assert project.step > 0 and project.status in {"awaiting_experiments", "completed"}


def test_interpreter_cannot_claim_support_without_significance(mock_settings):
    settings = mock_settings()
    store = create_project(settings, "guard topic")
    state = ResearchState(store, settings)
    state.add_hypothesis(Hypothesis(statement="method helps"))
    state.add_experiment(ExperimentSpec(hypothesis_id="H1", objective="o", method="m", baselines=["b"], metrics=[MetricSpec(name="acc", primary=True)]))
    spec = state.approve_experiment("E1")
    run = state.experiments.create_run(spec, "print()", "subprocess", [0, 1, 2])
    records = [RunRecord(arm=arm, seed=s, metrics={"acc": 0.5 + 0.01 * s}) for arm in ("m", "b") for s in range(3)]
    run = state.experiments.complete(run, records, synthetic=False)
    store.save_quant_result(analyze_run(run, spec, 0.05))

    class OverclaimingMock(MockResponder):
        def _science_interpretation(self, c):
            return {"verdict": "supported", "statements": [{"text": "proven", "tag": "evidence_backed", "evidence_refs": ["R999"]}]}

    router = ModelRouter(settings, mock=OverclaimingMock())
    agent = ScientificInterpreter(AgentDeps(state, router, settings, None, None, None))
    result = agent.run(TaskEnvelope(task_id="T1", type="INTERPRET_RUN", agent="interpreter", objective="", inputs={"run_id": run.id}))
    interp = store.interpretation(run.id)
    assert interp.verdict == "inconclusive" and result.data["verdict"] == "inconclusive"
    assert interp.statements[0].tag == "inference" and interp.statements[0].evidence_refs == []
    assert len(interp.adjustments) == 2
    assert state.claim_state("CH1@v1") == "HYPOTHESIS"


def test_failed_project_resumes_from_failed_action(mock_settings):
    settings = mock_settings()
    store = create_project(settings, "resilience topic")

    class FlakyMock(MockResponder):
        broken = True

        def _critique_proposal(self, c):
            if FlakyMock.broken:
                raise RuntimeError("provider outage")
            return super()._critique_proposal(c)

    project = Orchestrator(settings, store, router=ModelRouter(settings, mock=FlakyMock())).run()
    assert project.status == "failed" and "outage" in project.error
    FlakyMock.broken = False
    project = Orchestrator(settings, store, router=ModelRouter(settings, mock=FlakyMock())).run()
    assert project.status == "completed", project.error
