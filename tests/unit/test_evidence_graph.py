import pytest

from research_pilot.state.evidence import EvidenceGraph


def graph_with_hypothesis():
    g = EvidenceGraph()
    g.add_node("H1@v1", "hypothesis", "method helps")
    g.add_node("CH1@v1", "claim", "method helps", kind="contribution", hypothesis_claim=True)
    g.add_edge("CH1@v1", "H1@v1", "motivates")
    g.add_node("E1@v1", "experiment", "test")
    g.add_edge("H1@v1", "E1@v1", "tested_by")
    return g


def add_result(g, run_id, relation="supports", synthetic=False):
    g.add_node(run_id, "result", run_id, synthetic=synthetic)
    g.add_edge("E1@v1", run_id, "produces")
    g.add_edge(run_id, "CH1@v1", relation)


def test_untested_hypothesis_claim_is_hypothesis():
    g = graph_with_hypothesis()
    assert g.derive_state("CH1@v1")[0] == "HYPOTHESIS"
    assert g.untested_hypotheses()[0]["reason"] == "not_run"


def test_one_result_is_partial_two_are_supported():
    g = graph_with_hypothesis()
    add_result(g, "R001")
    assert g.derive_state("CH1@v1")[0] == "PARTIALLY_SUPPORTED"
    assert g.claims_with_single_support()[0]["source"] == "R001"
    add_result(g, "R002")
    assert g.derive_state("CH1@v1")[0] == "SUPPORTED"
    assert g.untested_hypotheses() == []


def test_refuting_evidence():
    g = graph_with_hypothesis()
    add_result(g, "R001", "refutes")
    assert g.derive_state("CH1@v1")[0] == "CONTRADICTED"
    add_result(g, "R002")
    state, detail = g.derive_state("CH1@v1")
    assert state == "PARTIALLY_SUPPORTED" and detail["contested"]
    assert g.contradicted_claims()[0]["claim"] == "CH1@v1"


def test_synthetic_results_do_not_count_unless_allowed():
    g = graph_with_hypothesis()
    add_result(g, "R001", synthetic=True)
    add_result(g, "R002", synthetic=True)
    state, detail = g.derive_state("CH1@v1")
    assert state == "HYPOTHESIS" and detail["ignored_synthetic"] == ["R001", "R002"]
    assert g.derive_state("CH1@v1", allow_synthetic=True)[0] == "SUPPORTED"


def test_literature_alone_cannot_establish_a_contribution():
    g = EvidenceGraph()
    g.add_node("C1", "claim", "our method is better", kind="contribution")
    for pid in ("P001", "P002", "P003"):
        g.add_node(pid, "paper", pid)
        g.add_edge(pid, "C1", "supports")
    state, detail = g.derive_state("C1")
    assert state == "HYPOTHESIS" and "untested" in detail["reason"]
    g.add_node("C2", "claim", "background", kind="background")
    g.add_edge("P001", "C2", "supports")
    g.add_edge("P002", "C2", "supports")
    assert g.derive_state("C2")[0] == "SUPPORTED"


def test_manuscript_claim_inherits_from_hypothesis_claim_and_statements():
    g = graph_with_hypothesis()
    g.add_node("C1", "claim", "paper claim", kind="contribution", manuscript=True)
    g.add_edge("CH1@v1", "C1", "supports")
    g.add_node("S-results", "statement", "results")
    g.add_edge("C1", "S-results", "appears_in")
    assert g.derive_state("C1")[0] == "HYPOTHESIS"
    assert g.unsupported_statements()[0]["statement"] == "S-results"
    add_result(g, "R001")
    add_result(g, "R002")
    assert g.derive_state("C1")[0] == "SUPPORTED"
    assert g.unsupported_statements() == []
    assert g.experiments_for_claim("C1") == ["E1@v1"]


def test_weak_literature_claims_and_hotspots():
    g = graph_with_hypothesis()
    g.add_node("P001", "paper", "p", evidence_quality="high")
    g.add_node("LC-P001-1", "claim", "x works", kind="literature")
    g.add_edge("P001", "LC-P001-1", "supports", basis="claimed")
    assert g.weak_literature_claims()[0]["claim"] == "LC-P001-1"
    hot = g.uncertainty_hotspots()
    assert hot[0]["claim"] == "CH1@v1" and hot[0]["pending_experiments"] == ["E1@v1"]


def test_edges_require_known_nodes_and_relations():
    g = EvidenceGraph()
    g.add_node("A", "paper")
    with pytest.raises(ValueError):
        g.add_edge("A", "missing", "supports")
    g.add_node("B", "claim")
    with pytest.raises(ValueError):
        g.add_edge("A", "B", "likes")


def test_round_trip_and_superseded_nodes_are_ignored():
    g = graph_with_hypothesis()
    g.nodes["H1@v1"].attrs["status"] = "superseded"
    g.nodes["CH1@v1"].attrs["status"] = "superseded"
    restored = EvidenceGraph.from_dict(g.to_dict())
    assert restored.untested_hypotheses() == []
    assert "claim" not in restored.summary()["nodes"]
