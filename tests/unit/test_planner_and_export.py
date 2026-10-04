from research_pilot.agents import AgentDeps
from research_pilot.agents.planner import ResearchPlanner, priority
from research_pilot.llm.router import ModelRouter
from research_pilot.orchestrator import create_project
from research_pilot.services.bibliography import format_reference
from research_pilot.services.export import markdown_to_html, markdown_to_pdf
from research_pilot.state.models import Action, ExperimentSpec, Hypothesis, MetricSpec, Paper
from research_pilot.state.research_state import ResearchState


def planner_for(settings):
    store = create_project(settings, "topic")
    state = ResearchState(store, settings)
    return ResearchPlanner(AgentDeps(state, ModelRouter(settings), settings, None, None, None)), state


def test_priority_formula():
    action = Action(type="RUN_EXPERIMENT", scientific_value=0.8, uncertainty_reduction=0.5, feasibility=0.5, cost=0.2)
    assert priority(action) == 1.0
    assert priority(action.model_copy(update={"cost": 0.0})) == 4.0


def test_planner_validation_rules(mock_settings):
    planner, state = planner_for(mock_settings())
    state.add_hypothesis(Hypothesis(statement="h"))
    state.add_experiment(ExperimentSpec(hypothesis_id="H1", objective="o", method="m", baselines=["b"], metrics=[MetricSpec(name="acc")]))
    rejected = []
    budget = {"runs_left": 5}
    assert planner.validate(Action(type="RUN_EXPERIMENT", target="E1"), budget, rejected) is None
    state.approve_experiment("E1")
    assert planner.validate(Action(type="RUN_EXPERIMENT", target="E1@v1"), budget, rejected).target == "E1"
    assert planner.validate(Action(type="RUN_EXPERIMENT", target="H1"), budget, rejected).type == "DESIGN_EXPERIMENT"
    assert planner.validate(Action(type="RUN_ABLATION", target="H9"), budget, rejected) is None
    assert planner.validate(Action(type="RUN_EXPERIMENT", target="E1"), {"runs_left": 0}, rejected) is None
    assert planner.validate(Action(type="REQUEST_REVIEW"), budget, rejected).type == "WRITE_PAPER"
    assert planner.validate(Action(type="DANCE"), budget, rejected) is None
    assert len(rejected) == 4


def test_planner_hints_cover_unrun_and_undesigned_work(mock_settings):
    planner, state = planner_for(mock_settings())
    state.add_hypothesis(Hypothesis(statement="h1", priority=1))
    state.add_hypothesis(Hypothesis(statement="h2", priority=2))
    state.add_experiment(ExperimentSpec(hypothesis_id="H1", objective="o", method="m", baselines=["b"], metrics=[MetricSpec(name="acc")]))
    state.approve_experiment("E1")
    hints = {(h.type, h.target) for h in planner.hints({"runs_left": 3})}
    assert ("RUN_EXPERIMENT", "E1") in hints
    assert ("DESIGN_EXPERIMENT", "H2") in hints


def test_markdown_export_renders_tables_and_citations(tmp_path):
    markdown = "# Title\n\n| a | b |\n|---|---|\n| 1 | **2** |\n\n- item\n\n[P001] Author (2020). https://doi.org/10.1/x"
    html = markdown_to_html(markdown, "Paper")
    assert "<table>" in html and "<strong>2</strong>" in html and "<li>item</li>" in html
    assert "class='citation'" in html and '<a href="https://doi.org/10.1/x"' in html
    out = tmp_path / "paper.pdf"
    markdown_to_pdf(out, markdown + "\nunicode: ≤", "Paper")
    assert out.read_bytes().startswith(b"%PDF-1.4")


def test_reference_formatting_marks_synthetic_sources():
    paper = Paper(id="P001", title="T", year=2021, authors=["A", "B", "C"], venue="ICML", doi="10.1/x")
    assert format_reference(paper) == "[P001] A et al. (2021). T. *ICML*. https://doi.org/10.1/x"
    assert "synthetic" in format_reference(paper.model_copy(update={"synthetic": True}))


def test_paper_writer_sees_every_run_deviation(mock_settings):
    from research_pilot.agents.paper_architect import PaperArchitect

    settings = mock_settings()
    _, state = planner_for(settings)
    state.add_hypothesis(Hypothesis(statement="h"))
    spec = state.add_experiment(ExperimentSpec(hypothesis_id="H1", objective="o", method="m", baselines=["b"], metrics=[MetricSpec(name="acc")]))
    state.approve_experiment(spec.id)
    deviation = "reviewer code repair: the solver changed so the gate converges; thresholds unchanged " + "x" * 300
    state.experiments.create_run(state.experiments.spec("E1"), "print()", "manual", [0], deviations=[deviation])
    architect = PaperArchitect(AgentDeps(state, ModelRouter(settings), settings, None, None, None))
    assert f"deviations from the protocol: {deviation}" in architect._evidence_block()
    assert deviation in architect._reproducibility()


def test_planner_hints_a_primary_design_when_only_validation_exists(mock_settings):
    planner, state = planner_for(mock_settings())
    state.add_hypothesis(Hypothesis(statement="h"))
    gate = state.add_experiment(ExperimentSpec(hypothesis_id="H1", kind="validation", objective="gate", method="m", baselines=["b"], metrics=[MetricSpec(name="acc")]))
    state.approve_experiment(gate.id)

    def untested(hints):
        return [h for h in hints if h.type == "DESIGN_EXPERIMENT" and h.target == "H1" and "no primary experiment" in h.rationale]

    assert untested(planner.hints({"runs_left": 5}))
    state.add_experiment(ExperimentSpec(hypothesis_id="H1", kind="primary", objective="test", method="m", baselines=["b"], metrics=[MetricSpec(name="acc")]))
    assert not untested(planner.hints({"runs_left": 5}))
