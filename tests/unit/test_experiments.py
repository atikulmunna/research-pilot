import json
import sys

from research_pilot.agents.quant_analyst import analyze_run
from research_pilot.llm.mock import MockResponder
from research_pilot.llm.json_utils import extract_code_block
from research_pilot.services.executor import (
    ManualExecutor,
    SimulatedExecutor,
    SubprocessExecutor,
    check_coverage,
    load_records,
    parse_result_lines,
)
from research_pilot.state.models import ExperimentRun, ExperimentSpec, MetricSpec, RunRecord

SPEC = ExperimentSpec(
    id="E1",
    hypothesis_id="H1",
    objective="test",
    method="proposed",
    baselines=["baseline"],
    metrics=[MetricSpec(name="acc", primary=True, min_effect=0.01)],
    seeds=[0, 1, 2],
)


def script_for(spec):
    return extract_code_block(MockResponder().respond("experiment.coding", {"spec": spec.model_dump(mode="json")}))


def test_parse_and_load_records():
    lines = ["noise", 'RESULT_JSON: {"arm": "a", "seed": 1, "metrics": {"acc": "0.5", "bad": "x"}}', '{"arm": "b", "seed": 1, "metrics": {}}']
    records = parse_result_lines(lines)
    assert [(r.arm, r.metrics) for r in records] == [("a", {"acc": 0.5}), ("b", {})]
    as_list = load_records(json.dumps({"records": [{"arm": "a", "seed": 0, "metrics": {"acc": 1}}]}))
    assert as_list[0].metrics == {"acc": 1.0}


def test_coverage_reports_missing_seeds_and_metrics():
    records = [RunRecord(arm="proposed", seed=0, metrics={"acc": 1.0})]
    problems = check_coverage(SPEC, [0, 1], records)
    assert any("proposed" in p and "[1]" in p for p in problems)
    assert any("baseline" in p for p in problems)


def test_subprocess_executor_runs_generated_script(tmp_path):
    (tmp_path / "run.py").write_text(script_for(SPEC), encoding="utf-8")
    outcome = SubprocessExecutor(timeout_s=60, python=sys.executable).execute(tmp_path, SPEC, [0, 1, 2], "R001")
    assert outcome.status == "completed", outcome.error
    assert len(outcome.records) == 6 and not outcome.synthetic
    assert (tmp_path / "stdout.log").exists()


def test_subprocess_executor_reports_failures(tmp_path):
    (tmp_path / "run.py").write_text("raise SystemExit('broken pipeline')", encoding="utf-8")
    outcome = SubprocessExecutor(timeout_s=60, python=sys.executable).execute(tmp_path, SPEC, [0], "R001")
    assert outcome.status == "failed" and "broken pipeline" in outcome.error


def test_manual_and_simulated_executors(tmp_path):
    assert ManualExecutor().execute(tmp_path, SPEC, [0], "R001").status == "awaiting_execution"
    assert "experiments ingest" in (tmp_path / "README.md").read_text(encoding="utf-8")
    simulated = SimulatedExecutor().execute(tmp_path, SPEC, [0, 1, 2], "R001")
    assert simulated.synthetic and len(simulated.records) == 6


def run_with(values_method, values_base, synthetic=False):
    records = [RunRecord(arm="proposed", seed=i, metrics={"acc": v}) for i, v in enumerate(values_method)]
    records += [RunRecord(arm="baseline", seed=i, metrics={"acc": v}) for i, v in enumerate(values_base)]
    return ExperimentRun(id="R001", experiment_id="E1", spec_version=1, seeds=[0, 1, 2], records=records, synthetic=synthetic)


def test_quant_analysis_clear_win():
    result = analyze_run(run_with([0.80, 0.81, 0.82], [0.70, 0.71, 0.72]), SPEC, 0.05)
    comparison = result.comparisons[0]
    assert comparison.primary and comparison.significant and comparison.direction_ok and comparison.practically_significant
    assert comparison.stable_without_best_seed is True
    assert any("very large effect" in a for a in result.anomalies)


def test_quant_analysis_flags_cherry_picking_and_synthetic():
    result = analyze_run(run_with([0.70, 0.71, 0.95], [0.70, 0.705, 0.71], synthetic=True), SPEC, 0.05)
    comparison = result.comparisons[0]
    assert not comparison.significant
    assert any("synthetic" in a for a in result.anomalies)
    assert any("unstable" in a for a in result.anomalies)


def test_quant_analysis_lower_is_better():
    spec = SPEC.model_copy(update={"metrics": [MetricSpec(name="acc", higher_is_better=False, primary=True)]})
    result = analyze_run(run_with([0.10, 0.11, 0.12], [0.30, 0.31, 0.32]), spec, 0.05)
    assert result.comparisons[0].direction_ok and result.comparisons[0].diff < 0


def test_metrics_match_by_identifier_when_names_carry_descriptions():
    spec = SPEC.model_copy(update={"metrics": [MetricSpec(name="acc (held-out, per replicate)", primary=True)]})
    result = analyze_run(run_with([0.80, 0.81, 0.82], [0.70, 0.71, 0.72]), spec, 0.05)
    assert result.comparisons and result.comparisons[0].primary and result.comparisons[0].significant
    assert not [a for a in result.anomalies if a.startswith("coverage: metrics never reported")]
