"""Quantitative Analyst: statistics in code (data.processing, stats.calculation) plus a lite-tier summary."""

from collections import defaultdict
from typing import Dict, List, Tuple

from ..services import stats
from ..services.executor import check_coverage
from ..state.models import ArmSummary, Comparison, ExperimentRun, ExperimentSpec, MetricSpec, QuantResult, StatsSummary, TaskEnvelope
from .base import Agent


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


Values = Dict[Tuple[str, str], List[float]]
# Limits for what goes into model prompts; the stored analysis keeps every value.
MAX_SUMMARY_ROWS = 60
MAX_NAMED_EXTRAS = 30
MAX_ANOMALY_LINES = 25


def _group_values(run: ExperimentRun) -> Values:
    values: Values = defaultdict(list)
    for record in run.records:
        for metric, value in record.metrics.items():
            values[(metric, record.arm)].append(value)
    return values


def _summaries(values: Values) -> List[ArmSummary]:
    out = []
    for (metric, arm), vals in sorted(values.items()):
        low, high = stats.mean_ci(vals)
        out.append(ArmSummary(metric=metric, arm=arm, n=len(vals), mean=stats.mean(vals), std=stats.stdev(vals), ci_low=low, ci_high=high, min=min(vals), max=max(vals)))
    return out


def _stable_without_best_seed(method_vals: List[float], base_vals: List[float], metric: MetricSpec, alpha: float) -> bool | None:
    """Does the improvement survive dropping the method's single best seed?"""
    if len(method_vals) < 3:
        return None
    trimmed = list(method_vals)
    trimmed.remove(max(method_vals) if metric.higher_is_better else min(method_vals))
    loo = stats.welch(trimmed, base_vals)
    sign = 1.0 if metric.higher_is_better else -1.0
    return loo["p_value"] is not None and loo["p_value"] < alpha and sign * loo["diff"] > 0


def _compare(spec: ExperimentSpec, metric: MetricSpec, baseline: str, method_vals: List[float], base_vals: List[float], alpha: float) -> Comparison:
    test = stats.welch(method_vals, base_vals)
    diff = test["diff"]
    base_mean = stats.mean(base_vals)
    g = stats.hedges_g(method_vals, base_vals)
    practical = abs(diff) >= metric.min_effect if metric.min_effect is not None else g is not None and abs(g) >= 0.5
    primary = spec.primary_metric()
    return Comparison(
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
        direction_ok=(diff if metric.higher_is_better else -diff) > 0,
        practically_significant=practical,
        stable_without_best_seed=_stable_without_best_seed(method_vals, base_vals, metric, alpha),
    )


def _comparisons(values: Values, spec: ExperimentSpec, alpha: float) -> List[Comparison]:
    """Method against every baseline on every metric, with Holm correction across all of them."""
    by_arm: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for (name, arm), vals in values.items():
        by_arm[arm][name] = vals
    out = []
    for metric in spec.metrics:
        method_vals = metric.lookup(by_arm[spec.method]) or []
        for baseline in spec.baselines:
            base_vals = metric.lookup(by_arm[baseline]) or []
            if method_vals and base_vals:
                out.append(_compare(spec, metric, baseline, method_vals, base_vals, alpha))
    for comparison, p_adj in zip(out, stats.holm([c.p_value for c in out])):
        comparison.p_adjusted = p_adj
        comparison.significant = p_adj is not None and p_adj < alpha
    return out


def _arm_anomalies(summary: ArmSummary, vals: List[float]) -> List[str]:
    label = f"{summary.arm}/{summary.metric}"
    out = []
    if summary.n < 3:
        out.append(f"{label}: only {summary.n} seed(s); too few for reliable inference")
    cv = stats.coefficient_of_variation(vals)
    if cv is not None and cv > 0.1:
        out.append(f"{label}: unstable across seeds (CV {cv:.2f})")
    if len(set(vals)) == 1 and len(vals) > 1:
        out.append(f"{label}: identical value on every seed; check that the seed is actually used")
    if summary.max == 1.0 or summary.min == 0.0:
        out.append(f"{label}: perfect or zero score observed; check for evaluation leakage")
    return out


def _comparison_anomalies(comparison: Comparison, total: int, alpha: float) -> List[str]:
    label = f"{comparison.method} vs {comparison.baseline} on {comparison.metric}"
    out = []
    if comparison.hedges_g is not None and abs(comparison.hedges_g) > 3:
        out.append(f"{label}: very large effect (g={comparison.hedges_g:.1f}); rule out leakage or an evaluation bug")
    if comparison.p_value is not None and comparison.p_value < alpha and not comparison.significant:
        out.append(f"{label}: significant before but not after Holm correction for {total} comparisons")
    if comparison.significant and comparison.stable_without_best_seed is False:
        out.append(f"{label}: improvement depends on the single best seed")
    if comparison.significant and comparison.direction_ok and not comparison.practically_significant:
        out.append(f"{label}: statistically significant but below the practical-significance threshold")
    return out


def analyze_run(run: ExperimentRun, spec: ExperimentSpec, alpha: float) -> QuantResult:
    values = _group_values(run)
    summaries = _summaries(values)
    comparisons = _comparisons(values, spec, alpha)
    anomalies = [a for s in summaries for a in _arm_anomalies(s, values[(s.metric, s.arm)])]
    anomalies += [a for c in comparisons for a in _comparison_anomalies(c, len(comparisons), alpha)]
    anomalies += [f"coverage: {p}" for p in check_coverage(spec, run.seeds, run.records)]
    if run.synthetic:
        anomalies.append("Results are synthetic (simulated execution or mock-generated code); they are not evidence.")

    constraints = [
        f"n = {min((s.n for s in summaries), default=0)} to {max((s.n for s in summaries), default=0)} seeds per arm; p-values use Welch's t-test with Holm correction across {len(comparisons)} comparisons (alpha {alpha}).",
    ]
    primary = spec.primary_metric()
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
    """Per-arm summaries of the compared (pre-registered) metrics. Other reported metrics are only named, because
    scripts can report hundreds of per-cell values and every row would go into a model prompt."""
    declared = {c.metric for c in result.comparisons} | {c.metric.split(" (")[0].strip() for c in result.comparisons}
    shown = [s for s in result.summaries if s.metric in declared] if declared else result.summaries[:MAX_SUMMARY_ROWS]
    rows = ["| metric | arm | n | mean | std | 95% CI |", "|---|---|---|---|---|---|"]
    rows += [f"| {s.metric} | {s.arm} | {s.n} | {_fmt(s.mean)} | {_fmt(s.std)} | [{_fmt(s.ci_low)}, {_fmt(s.ci_high)}] |" for s in shown]
    extra = sorted({s.metric for s in result.summaries} - {s.metric for s in shown})
    if extra:
        names = ", ".join(extra[:MAX_NAMED_EXTRAS]) + (f", and {len(extra) - MAX_NAMED_EXTRAS} more" if len(extra) > MAX_NAMED_EXTRAS else "")
        rows.append(f"\n{len(extra)} further reported metrics are summarised in analysis/quantitative/{result.run_id}.yaml: {names}")
    return "\n".join(rows)


def anomaly_lines(result: QuantResult) -> str:
    """The code-detected anomalies as a bullet list, capped for prompts; the full list stays in the stored result."""
    lines = [f"- {a}" for a in result.anomalies[:MAX_ANOMALY_LINES]]
    if len(result.anomalies) > MAX_ANOMALY_LINES:
        lines.append(f"- and {len(result.anomalies) - MAX_ANOMALY_LINES} more in analysis/quantitative/{result.run_id}.yaml")
    return "\n".join(lines) or "(none)"


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
            f"Comparisons:\n{comparison_table(result)}\n\nAnomalies detected by code:\n{anomaly_lines(result)}"
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
