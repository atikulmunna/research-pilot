from types import SimpleNamespace

from research_pilot.agents import AgentDeps
from research_pilot.agents.base import match_by_id
from research_pilot.agents.experiment_designer import ExperimentDesigner
from research_pilot.agents.hypothesis import HypothesisDesigner
from research_pilot.llm.router import ModelRouter
from research_pilot.orchestrator import create_project
from research_pilot.state.models import Assessment, Criticism, ExperimentSpec, Hypothesis, MetricSpec, TaskEnvelope
from research_pilot.state.research_state import ResearchState


def test_match_by_id_aligns_by_id_then_position():
    h1, h2, anonymous = SimpleNamespace(id="H1"), SimpleNamespace(id="H2"), SimpleNamespace(id="")
    assert match_by_id([h2, h1], ["H1", "H2"]) == [h1, h2]
    assert match_by_id([anonymous, h2], ["H1", "H2"]) == [anonymous, h2]
    assert match_by_id([h2], ["H1", "H2"]) == [None, h2]


def setup(mock_settings):
    settings = mock_settings()
    state = ResearchState(create_project(settings, "batching topic"), settings)
    router = ModelRouter(settings)
    calls = []
    router.listeners.append(lambda record: calls.append(record.task))
    return state, AgentDeps(state, router, settings, None, None, None), calls


def task(**inputs):
    return TaskEnvelope(task_id="T1", type="X", agent="a", objective="", inputs=inputs)


def test_refining_several_hypotheses_takes_one_call(mock_settings):
    state, deps, calls = setup(mock_settings)
    for statement in ("a", "b", "c"):
        state.add_hypothesis(Hypothesis(statement=statement))
    issues = [Criticism(target_id=h, description="vague") for h in ("H1", "H2")]
    state.add_assessment(Assessment(issues=issues, verdict="revise"), "hypotheses")

    result = HypothesisDesigner(deps).run(task(mode="refine", targets=["H1", "H2"]))

    assert calls == ["hypothesis.design"]
    assert sorted(result.data["hypothesis_ids"]) == ["H1", "H2"]
    assert {h.key for h in state.current_hypotheses()} == {"H1@v2", "H2@v2", "H3@v1"}
    assert state.open_criticisms("hypotheses") == []


def test_revising_several_designs_takes_one_call(mock_settings):
    state, deps, calls = setup(mock_settings)
    for statement in ("a", "b"):
        state.add_hypothesis(Hypothesis(statement=statement))
    for hypothesis_id in ("H1", "H2"):
        state.add_experiment(ExperimentSpec(hypothesis_id=hypothesis_id, objective="o", method="m", baselines=["b"], metrics=[MetricSpec(name="acc")]))
    issues = [Criticism(target_id=e, severity="major", description="missing baseline") for e in ("E1", "E2")]
    state.add_assessment(Assessment(issues=issues, verdict="revise"), "experiments")

    result = ExperimentDesigner(deps).run(task(mode="revise", ids=["E1", "E2"]))

    assert calls == ["experiment.design"]
    assert sorted(result.data["experiment_ids"]) == ["E1", "E2"]
    assert {s.key for s in state.experiments.latest_specs()} == {"E1@v2", "E2@v2"}
