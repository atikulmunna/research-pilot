"""Failure-safety rules: no fabricated evidence, no silent protocol changes, logged revisions."""

import pytest

from research_pilot.orchestrator import create_project
from research_pilot.state.models import ClaimMap, ExperimentSpec, Hypothesis, ManuscriptClaim, MetricSpec, Paper, RunRecord
from research_pilot.state.registry import RegistryError
from research_pilot.state.research_state import ResearchState
from research_pilot.state.store import ProjectStore


@pytest.fixture
def state(mock_settings):
    settings = mock_settings()
    store = create_project(settings, "test topic", question="does it work?")
    return ResearchState(store, settings)


def paper(title, **kw):
    return Paper(title=title, source=kw.pop("source", "openalex"), doi=kw.pop("doi", f"10.1/{abs(hash(title))}"), **kw)


def spec(hypothesis_id="H1"):
    return ExperimentSpec(hypothesis_id=hypothesis_id, objective="test", method="m", baselines=["b"], metrics=[MetricSpec(name="acc", primary=True)])


def test_project_layout_matches_plan(state):
    root = state.store.root
    for rel in ("field", "literature/papers", "gaps", "proposal", "hypotheses", "experiments/specs", "experiments/runs", "analysis/quantitative", "decisions", "roadmap", "paper/manuscript", "reviews/red_team"):
        assert (root / rel).is_dir(), rel
    assert state.proposal().version == 1 and state.proposal().text == "does it work?"
    assert ProjectStore.resolve_id(state.settings.workspace_dir, "latest") == root.name


def test_project_and_run_ids_cannot_escape_the_workspace(state):
    workspace = state.settings.workspace_dir
    for bad in ("../outside", r"..\outside", "a/b", "Upper", "x" * 90):
        with pytest.raises(FileNotFoundError, match="Invalid project id"):
            ProjectStore.open(workspace, bad)
    assert ProjectStore.open(workspace, state.store.root.name).root == state.store.root
    assert state.experiments.run("../../outside") is None


def test_papers_need_a_traceable_source(state):
    added = state.add_papers([paper("Real paper"), Paper(title="Invented paper", source="llm")], found_by="q")
    assert [p.id for p in added] == ["P001"]
    assert any("traceable" in v for v in state.drain_violations())


def test_duplicate_papers_are_merged(state):
    state.add_papers([paper("A study of things", doi="10.1/x")], found_by="q1")
    again = state.add_papers([paper("A Study of Things!", doi="", arxiv_id="2401.00001")], found_by="q2")
    assert again == []
    stored = state.papers()[0]
    assert stored.arxiv_id == "2401.00001" and stored.found_by == ["q1", "q2"]


def test_unknown_references_are_dropped_and_logged(state):
    state.add_papers([paper("Real paper")], found_by="q")
    assert state.check_paper_refs(["P001", "P999"], "test") == ["P001"]
    assert "P999" in state.drain_violations()[0]
    hyp = state.add_hypothesis(Hypothesis(statement="x", motivated_by=["P001", "G42"]))
    assert hyp.motivated_by == ["P001"]


def test_decorated_ids_are_kept_and_unknown_ones_dropped(state):
    state.add_papers([paper("Real paper")], found_by="q")
    refs = ["P001: Real paper - strong baseline", "[P001]", "P999: Invented - note", "P0011 trailing digits"]
    assert state.check_paper_refs(refs, "test") == ["P001"]
    violations = state.drain_violations()
    assert len(violations) == 2 and "P999" in violations[0]


def test_hypothesis_revision_keeps_original_and_records_reason(state):
    h1 = state.add_hypothesis(Hypothesis(statement="original"))
    h2 = state.revise_hypothesis(h1.id, Hypothesis(statement="revised"), reason="result R001 was inconclusive", triggering_evidence=["R001"], decision_id="D001")
    versions = [h for h in state.hypotheses() if h.id == h1.id]
    assert [(h.version, h.status) for h in versions] == [(1, "superseded"), (2, "active")]
    assert h2.revision_of == "H1@v1" and "D001" in h2.revision_reason and h2.triggering_evidence == ["R001"]
    assert state.graph.nodes["CH1@v1"].attrs["status"] == "superseded"


def test_experiments_for_unknown_hypotheses_are_rejected(state):
    assert state.add_experiment(spec("H9")) is None
    assert state.drain_violations()


def test_specs_are_versioned_and_runs_need_approval(state):
    state.add_hypothesis(Hypothesis(statement="h"))
    e1 = state.add_experiment(spec())
    assert e1.key == "E1@v1" and e1.seeds == [0, 1, 2]
    with pytest.raises(RegistryError):
        state.experiments.create_run(e1, "print()", "simulated", [0])
    e2 = state.revise_experiment("E1", spec().model_copy(update={"baselines": ["b", "strong"]}), reason="CR1-1")
    assert e2.key == "E1@v2" and e2.change_log[-1] == "v2: CR1-1"
    assert state.experiments.spec("E1@v1").status == "superseded"
    approved = state.approve_experiment("E1")
    assert approved.status == "approved"
    with pytest.raises(RegistryError):
        state.experiments.approve("E1@v1")


def test_results_are_immutable_and_failed_runs_are_kept(state):
    state.add_hypothesis(Hypothesis(statement="h"))
    state.add_experiment(spec())
    approved = state.approve_experiment("E1")
    failed = state.experiments.create_run(approved, "print()", "subprocess", [0])
    state.experiments.mark(failed, "failed", "boom")
    run = state.experiments.create_run(approved, "print()", "subprocess", [0])
    state.experiments.complete(run, [RunRecord(arm="m", seed=0, metrics={"acc": 1.0})], synthetic=False)
    with pytest.raises(RegistryError):
        state.experiments.complete(run, [], synthetic=False)
    assert [r.status for r in state.experiments.runs()] == ["failed", "completed"]


def test_manuscript_claims_are_validated_and_states_derived(state):
    state.add_papers([paper("Background paper"), paper("Another paper")], found_by="q")
    state.add_hypothesis(Hypothesis(statement="h"))
    claim_map = ClaimMap(
        claims=[
            ManuscriptClaim(text="we win", kind="contribution", importance="central", hypothesis_ids=["H1", "H7"], sections=["results"]),
            ManuscriptClaim(text="prior work", kind="background", paper_ids=["P001", "P002", "P404"]),
        ]
    )
    stored = state.set_manuscript_claims(claim_map)
    assert stored.claims[0].hypothesis_ids == ["H1"] and stored.claims[0].state == "HYPOTHESIS"
    assert stored.claims[1].paper_ids == ["P001", "P002"] and stored.claims[1].state == "SUPPORTED"
    assert len(state.drain_violations()) == 2


def test_decisions_are_logged(state):
    decision = state.log_decision("plan", "Run E1", "highest priority", alternatives=["search literature"])
    assert decision.id == "D001"
    assert state.store.decisions()[0].alternatives_considered == ["search literature"]


def test_writes_retry_while_the_target_is_locked(state, monkeypatch):
    from research_pilot.state import store as store_module

    real_replace, calls = store_module.os.replace, []

    def locked_twice(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise PermissionError("held open by a reader")
        real_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", locked_twice)
    monkeypatch.setattr(store_module.time, "sleep", lambda _: None)
    state.store.write_text("notes/retry.txt", "saved")
    assert state.store.read_text("notes/retry.txt") == "saved" and len(calls) == 3


def test_a_persistent_lock_raises_and_leaves_no_temp_file(state, monkeypatch):
    from research_pilot.state import store as store_module

    def always_locked(src, dst):
        raise PermissionError("held open by a reader")

    monkeypatch.setattr(store_module.os, "replace", always_locked)
    monkeypatch.setattr(store_module.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        state.store.write_text("notes/locked.txt", "lost")
    assert not list(state.store.path("notes").glob("*.tmp"))


def test_reads_retry_while_the_file_is_being_replaced(state, monkeypatch):
    from pathlib import Path

    from research_pilot.state import store as store_module

    state.store.write_text("notes/read.txt", "saved")
    real_read, calls = Path.read_text, []

    def locked_twice(self, *args, **kwargs):
        calls.append(self.name)
        if len(calls) <= 2:
            raise PermissionError("being replaced by a writer")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", locked_twice)
    monkeypatch.setattr(store_module.time, "sleep", lambda _: None)
    assert state.store.read_text("notes/read.txt") == "saved" and len(calls) == 3
