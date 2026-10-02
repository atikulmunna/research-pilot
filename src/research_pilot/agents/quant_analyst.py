"""Quantitative Analyst: statistics in code (data.processing, stats.calculation) plus a lite-tier summary."""

from collections import defaultdict
from typing import Dict, List, Tuple

from ..services import stats
from ..services.executor import check_coverage
from ..state.models import ArmSummary, Comparison, ExperimentRun, ExperimentSpec, QuantResult, StatsSummary, TaskEnvelope
from .base import Agent


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def analyze_run(run: ExperimentRun, spec: ExperimentSpec, alpha: float) -> QuantResult:
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    seeds_by: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for record in run.records:
        for metric, value in record.metrics.items():
            values[(metric, record.arm)].append(value)
            seeds_by[(metric, record.arm)].append(record.seed)

    summaries: List[ArmSummary] = []
    for (metric, arm), vals in sorted(values.items()):
        low, high = stats.mean_ci(vals)
        summaries.append(
            ArmSummary(metric=metric, arm=arm, n=len(vals), mean=stats.mean(vals), std=stats.stdev(vals), ci_low=low, ci_high=high, min=min(vals), max=max(vals))
        )

    primary = spec.primary_metric()
    comparisons: List[Comparison] = []
    anomalies: List[str] = []
    for metric in spec.metrics:
        method_vals = values.get((metric.name, spec.method), [])
        if not method_vals:
            continue
        for baseline in spec.baselines:
            base_vals = values.get((metric.name, baseline), [])
            if not base_vals:
                continue
            test = stats.welch(method_vals, base_vals)
            sign = 1.0 if metric.higher_is_better else -1.0
            diff = test["diff"]
            base_mean = stats.mean(base_vals)
            g = stats.hedges_g(method_vals, base_vals)
            if metric.min_effect is not None:
                practical = abs(diff) >= metric.min_effect
            else:
                practical = g is not None and abs(g) >= 0.5
            stable = None
            if len(method_vals) >= 3:
                best = max(method_vals) if metric.higher_is_better else min(method_vals)
                trimmed = list(method_vals)
                trimmed.remove(best)
                loo = stats.welch(trimmed, base_vals)
                stable = loo["p_value"] is not None and loo["p_value"] < alpha and sign * loo["diff"] > 0
            comparisons.append(
                Comparison(
                    metric=metric.name,
                    method=spec.method,
                    baseline=baseline,
                    higher_is_better=metric.higher_is_better,
                    primary=primary is not None and metric.name == primary.name,
                    diff=diff,
                    rel_diff=(diff / abs(base_mean)) if base_mean else None,
                    ci_low=test["ci_low"],
                    ci_high=test["ci_high"],
                    t=test["t"],
                    df=test["df"],
                    p_value=test["p_value"],
                    hedges_g=g,
                    direction_ok=sign * diff > 0,
                    practically_significant=practical,
                    stable_without_best_seed=stable,
                )
            )

    adjusted = stats.holm([c.p_value for c in comparisons])
    for comparison, p_adj in zip(comparisons, adjusted):
        comparison.p_adjusted = p_adj
        comparison.significant = p_adj is not None and p_adj < alpha

    for summary in summaries:
        if summary.n < 3:
            anomalies.append(f"{summary.arm}/{summary.metric}: only {summary.n} seed(s); too few for reliable inference")
        cv = stats.coefficient_of_variation(values[(summary.metric, summary.arm)])
        if cv is not None and cv > 0.1:
            anomalies.append(f"{summary.arm}/{summary.metric}: unstable across seeds (CV {cv:.2f})")
        vals = values[(summary.metric, summary.arm)]
        if len(set(vals)) == 1 and len(vals) > 1:
            anomalies.append(f"{summary.arm}/{summary.metric}: identical value on every seed; check that the seed is actually used")
        if summary.max == 1.0 or summary.min == 0.0:
            anomalies.append(f"{summary.arm}/{summary.metric}: perfect or zero score observed; check for evaluation leakage")
    for comparison in comparisons:
        label = f"{comparison.method} vs {comparison.baseline} on {comparison.metric}"
        if comparison.hedges_g is not None and abs(comparison.hedges_g) > 3:
            anomalies.append(f"{label}: very large effect (g={comparison.hedges_g:.1f}); rule out leakage or an evaluation bug")
        if comparison.p_value is not None and comparison.p_value < alpha and not comparison.significant:
            anomalies.append(f"{label}: significant before but not after Holm correction for {len(comparisons)} comparisons")
        if comparison.significant and comparison.stable_without_best_seed is False:
            anomalies.append(f"{label}: improvement depends on the single best seed")
        if comparison.significant and comparison.direction_ok and not comparison.practically_significant:
            anomalies.append(f"{label}: statistically significant but below the practical-significance threshold")
    anomalies.extend(f"coverage: {p}" for p in check_coverage(spec, run.seeds, run.records))
    if run.synthetic:
        anomalies.append("Results are synthetic (simulated execution or mock-generated code); they are not evidence.")

    constraints = [
        f"n = {min((s.n for s in summaries), default=0)} to {max((s.n for s in summaries), default=0)} seeds per arm; p-values use Welch's t-test with Holm correction across {len(comparisons)} comparisons (alpha {alpha}).",
    ]
    if primary is not None:
        constraints.append(f"Only '{primary.name}' is the pre-registered primary metric; other metrics are secondary.")
    return QuantResult(
        run_id=run.id,
        experiment_id=run.spec_key,
        hypothesis_id=f"{spec.hypothesis_id}@v{spec.hypothesis_version}",
        alpha=alpha,
        synthetic=run.synthetic,
        summaries=summaries,
        comparisons=comparisons,
        anomalies=anomalies,
        interpretation_constraints=constraints,
    )


def comparison_table(result: QuantResult) -> str:
    rows = ["| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |", "|---|---|---|---|---|---|---|---|"]
    for c in result.comparisons:
        rows.append(
            f"| {c.metric}{' (primary)' if c.primary else ''} | {c.method} vs {c.baseline} | {_fmt(c.diff)} | "
            f"[{_fmt(c.ci_low)}, {_fmt(c.ci_high)}] | {_fmt(c.p_adjusted, 4)} | {_fmt(c.hedges_g, 2)} | "
            f"{'yes' if c.significant else 'no'} | {'yes' if c.practically_significant else 'no'} |"
        )
    return "\n".join(rows)


def summary_table(result: QuantResult) -> str:
    rows = ["| metric | arm | n | mean | std | 95% CI |", "|---|---|---|---|---|---|"]
    for s in result.summaries:
        rows.append(f"| {s.metric} | {s.arm} | {s.n} | {_fmt(s.mean)} | {_fmt(s.std)} | [{_fmt(s.ci_low)}, {_fmt(s.ci_high)}] |")
    return "\n".join(rows)


class QuantitativeAnalyst(Agent):
    key = "quant_analyst"
    name = "Quantitative Analyst"

    def run(self, task: TaskEnvelope):
        registry = self.state.experiments
        run = registry.run(task.inputs["run_id"])
        spec = registry.spec(run.spec_key)
        result = analyze_run(run, spec, self.settings.significance_alpha)

        summary = self.ask(
            "stats.summary",
            f"Experiment {spec.key} ({spec.kind}) for hypothesis {spec.hypothesis_id}.\n\nPer-arm summaries:\n{summary_table(result)}\n\n"
            f"Comparisons:\n{comparison_table(result)}\n\nAnomalies detected by code:\n"
            + ("\n".join(f"- {a}" for a in result.anomalies) or "(none)")
            + "\n\nSummarise what these numbers establish and list the constraints on interpreting them.",
            StatsSummary,
            {"comparisons": [c.model_dump() for c in result.comparisons], "run_id": run.id},
        )
        result.summary = summary.summary
        result.interpretation_constraints = list(dict.fromkeys([*result.interpretation_constraints, *summary.interpretation_constraints]))
        self.store.save_quant_result(result)

        primary = [c for c in result.comparisons if c.primary]
        return self.result(
            task,
            findings=[
                f"{c.metric}: {c.method} vs {c.baseline} diff {c.diff:+.4f}, p_adj {_fmt(c.p_adjusted)}, g {_fmt(c.hedges_g, 2)}"
                for c in primary
            ],
            uncertainties=result.anomalies,
            artifacts=[f"analysis/quantitative/{run.id}.yaml"],
            data={
                "primary_significant": bool(primary) and all(c.significant and c.direction_ok for c in primary),
                "anomalies": len(result.anomalies),
            },
        )
