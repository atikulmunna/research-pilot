from ..state.models import ExperimentPlan, TaskEnvelope
from .base import Agent, as_yaml, clip
from .hypothesis import HYPOTHESIS_FIELDS

SPEC_EXCLUDE = {"status", "change_log", "critique_notes", "created_at", "hypothesis_version"}


class ExperimentDesigner(Agent):
    key = "experiment_designer"
    name = "Experiment Designer"

    def run(self, task: TaskEnvelope):
        mode = task.inputs.get("mode", "design")
        if mode == "revise":
            return self._revise(task)
        if mode == "request":
            return self._request(task)
        return self._design(task)

    def _context(self, hypotheses) -> str:
        field_map = self.store.field_map()
        parts = [self.project_block()]
        parts.append("Hypotheses:\n" + as_yaml([h.model_dump(include=HYPOTHESIS_FIELDS) for h in hypotheses]))
        if field_map:
            parts.append(
                "Field context: datasets " + ", ".join(field_map.datasets[:8])
                + "; benchmarks " + ", ".join(field_map.benchmarks[:8])
                + "; metrics " + ", ".join(field_map.metrics[:8])
            )
        papers = [p for p in self.state.papers() if p.extracted and p.baselines]
        baselines = sorted({b for p in papers for b in p.baselines})[:15]
        if baselines:
            parts.append("Baselines used in the literature: " + ", ".join(baselines))
        return "\n\n".join(parts)

    def _design(self, task: TaskEnvelope):
        with_specs = {(s.hypothesis_id, s.hypothesis_version) for s in self.state.experiments.latest_specs() if s.status != "superseded"}
        targets = [h for h in self.state.current_hypotheses(include_closed=False) if (h.id, h.version) not in with_specs]
        if not targets:
            return self.result(task, findings=["Every active hypothesis already has an experiment"], data={"experiment_ids": []})
        plan = self.ask(
            "experiment.design",
            self._context(targets)
            + "\n\nDesign the minimum set of experiments that could falsify each hypothesis: one primary experiment per "
            "hypothesis, plus an ablation only where a hypothesis is about a specific component.",
            ExperimentPlan,
            {"hypotheses": [h.id for h in targets], "mode": "design"},
        )
        return self._store(task, plan)

    def _request(self, task: TaskEnvelope):
        hyp = self.state.hypothesis(str(task.inputs.get("hypothesis_id", "")))
        if hyp is None:
            return self.result(task, status="failed", error=f"unknown hypothesis {task.inputs.get('hypothesis_id')}")
        kind = task.inputs.get("kind", "primary")
        existing = [s for s in self.state.experiments.latest_specs() if s.hypothesis_id == hyp.id]
        plan = self.ask(
            "experiment.design",
            self._context([hyp])
            + "\n\nExisting experiments for this hypothesis:\n"
            + (as_yaml([s.model_dump(exclude=SPEC_EXCLUDE) for s in existing]) if existing else "(none)")
            + f"\n\nDesign exactly one new '{kind}' experiment for {hyp.id}. Reason: {task.inputs.get('rationale', '')}",
            ExperimentPlan,
            {"hypotheses": [hyp.id], "mode": "request", "kind": kind},
        )
        for spec in plan.experiments:
            spec.hypothesis_id = hyp.id
            spec.kind = kind if kind else spec.kind
        plan.experiments = plan.experiments[:1]
        return self._store(task, plan)

    def _store(self, task: TaskEnvelope, plan: ExperimentPlan):
        stored = []
        for spec in plan.experiments:
            if not spec.metrics or not spec.method:
                self.state.violation("protocol", f"experiment for {spec.hypothesis_id} dropped: no method arm or metrics")
                continue
            if not any(m.primary for m in spec.metrics):
                spec.metrics[0].primary = True
            created = self.state.add_experiment(spec)
            if created:
                stored.append(created)
        return self.result(
            task,
            findings=[f"{s.key} ({s.kind}) for {s.hypothesis_id}: {clip(s.objective, 140)}" for s in stored],
            artifacts=[f"experiments/specs/{s.id}.v{s.version}.yaml" for s in stored],
            data={"experiment_ids": [s.id for s in stored]},
        )

    def _revise(self, task: TaskEnvelope):
        ids = list(task.inputs.get("ids", []))
        criticisms = [c for c in self.state.open_criticisms("experiments") if c.severity in {"fatal", "major"}]
        specs = [s for s in self.state.experiments.latest_specs() if s.status == "draft" and (not ids or s.id in ids)]
        revised = []
        for spec in specs:
            relevant = [c for c in criticisms if c.target_id in {spec.id, ""}]
            if not relevant:
                continue
            hyp = self.state.hypothesis(spec.hypothesis_id)
            plan = self.ask(
                "experiment.design",
                self._context([hyp] if hyp else [])
                + f"\n\nRevise experiment {spec.id} (keep the same id and hypothesis) to address:\n"
                + "\n".join(f"- {c.id} [{c.severity}] {c.description} -> {c.proposed_test}" for c in relevant)
                + "\n\nCurrent design:\n"
                + as_yaml(spec.model_dump(exclude=SPEC_EXCLUDE)),
                ExperimentPlan,
                {"hypotheses": [spec.hypothesis_id], "mode": "revise", "existing": [spec.model_dump(mode="json")]},
            )
            if not plan.experiments:
                continue
            candidate = plan.experiments[0]
            if not candidate.metrics or not candidate.method:
                continue
            if not any(m.primary for m in candidate.metrics):
                candidate.metrics[0].primary = True
            new = self.state.revise_experiment(spec.id, candidate, reason="addresses " + ", ".join(c.id for c in relevant))
            self.state.resolve_criticisms([c.id for c in relevant])
            revised.append(new)
        return self.result(
            task,
            findings=[f"{s.key} revised" for s in revised] or ["No design needed revision"],
            artifacts=[f"experiments/specs/{s.id}.v{s.version}.yaml" for s in revised],
            data={"experiment_ids": [s.id for s in revised]},
        )
