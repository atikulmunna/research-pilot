from typing import List

from ..llm.json_utils import extract_code_block, parse_json_object
from ..services.executor import RESULT_PREFIX, ExecutionOutcome, check_coverage
from ..state.models import ExperimentRun, ExperimentSpec, ImplementationNotes, TaskEnvelope
from .base import Agent, as_yaml, clip
from .experiment_designer import SPEC_EXCLUDE

CONTRACT = f"""Script contract (mandatory):
- Invocation: python run.py --seed <int> --output-dir <path>
- One invocation is one independent replicate. The swarm runs the script once per seed and computes statistics across
  seeds, so draw all data, splits and initialisations from --seed and never loop over replicate seeds inside the script.
- Write any artifacts (plots, checkpoints) under --output-dir with the seed in each file name; seeds share that directory.
- For EVERY arm (the method arm and each baseline arm, using the exact arm names) print exactly one line to stdout:
  {RESULT_PREFIX} {{"arm": "<arm name>", "seed": <seed>, "metrics": {{"<metric name>": <float>, ...}}}}
- Report every metric named in the spec, using the exact metric names.
- Exit with a non-zero status on failure. Never print fabricated or hard-coded results.

Reply with the full script in one ```python block, followed by one ```json block with keys:
dataset_version, model_version, compute_estimate, assumptions (list), deviations (list of any departures from the spec)."""


class ExperimentEngineer(Agent):
    key = "experiment_engineer"
    name = "Experiment Engineer"

    def run(self, task: TaskEnvelope):
        registry = self.state.experiments
        if task.inputs.get("run_id"):
            run = registry.run(task.inputs["run_id"])
            return self._execute(task, run, registry.spec(run.spec_key))
        spec = registry.spec(str(task.inputs["experiment_id"]))
        refusal = self._refusal(spec, task.inputs["experiment_id"])
        if refusal:
            return self.result(task, status="failed", error=refusal)
        offset = 1000 * len(registry.runs_for(spec.key))
        seeds = [s + offset for s in (spec.seeds or list(range(self.settings.default_seeds)))]
        run = self._generate_run(spec, seeds, self._failure_note(task.inputs.get("previous_failure")))
        if offset:
            run.deviations.append(f"replication run with fresh seeds {seeds}")
        return self._execute(task, run, spec)

    def _refusal(self, spec: ExperimentSpec | None, requested: str) -> str:
        if spec is None or spec.status != "approved":
            return f"{requested} is not an approved experiment"
        previous = self.state.experiments.runs_for(spec.key)
        if len(previous) >= self.settings.max_runs_per_spec:
            return f"{spec.key} already has {len(previous)} runs (limit {self.settings.max_runs_per_spec})"
        return ""

    def _failure_note(self, failed_ref: str | None) -> str:
        failed = self.state.experiments.run(failed_ref) if failed_ref else None
        if failed is None:
            return ""
        return f"\n\nA previous run ({failed.id}) of this experiment failed with:\n{clip(failed.error, 1500)}\nAvoid that failure."

    def _generate_run(self, spec: ExperimentSpec, seeds: List[int], failure_note: str) -> ExperimentRun:
        route = self.router.resolve("experiment.coding")
        text = self.ask_text(
            "experiment.coding",
            f"{self.project_block()}\n\nApproved experiment spec ({spec.key}):\n{as_yaml(spec.model_dump(exclude=SPEC_EXCLUDE))}\n\n{CONTRACT}{failure_note}",
            {"spec": spec.model_dump(mode="json")},
        )
        notes = ImplementationNotes.model_validate(parse_json_object(text.split("```python", 1)[-1]) or {})
        return self.state.experiments.create_run(
            spec,
            code=extract_code_block(text) or text,
            executor=self.deps.executor.name,
            seeds=seeds,
            dataset_version=notes.dataset_version,
            model_version=notes.model_version,
            compute=notes.compute_estimate,
            deviations=[f"declared by engineer: {d}" for d in notes.deviations],
            code_generator=f"{route.provider}/{route.model}",
        )

    def _execute(self, task: TaskEnvelope, run: ExperimentRun, spec: ExperimentSpec):
        """Run the script, repairing it after a failure up to the configured number of times."""
        registry = self.state.experiments
        run_dir = self.store.path(registry.run_dir(run.id))
        synthetic_code = run.code_generator.startswith("mock")
        for attempt in range(self.settings.experiment_max_repairs + 1):
            outcome = self.deps.executor.execute(run_dir, spec, run.seeds, run.id)
            if outcome.status == "awaiting_execution":
                return self._awaiting(task, run, spec)
            if outcome.status == "completed":
                return self._completed(task, run, outcome, synthetic_code, spec)
            run = registry.record_attempt(run, "failed", error=outcome.error, duration_s=outcome.duration_s)
            self.store.write_yaml(f"analysis/errors/{run.id}.attempt{len(run.attempts)}.yaml", {"run_id": run.id, "error": outcome.error})
            if attempt < self.settings.experiment_max_repairs:
                run = self._repair(run, spec, outcome.error)
        registry.mark(run, "failed", error=outcome.error)
        return self.result(
            task,
            status="failed",
            findings=[f"{run.id} failed after {len(run.attempts)} attempt(s)"],
            error=clip(outcome.error, 500),
            data={"run_id": run.id, "status": "failed"},
        )

    def _awaiting(self, task: TaskEnvelope, run: ExperimentRun, spec: ExperimentSpec):
        registry = self.state.experiments
        registry.mark(run, "awaiting_execution")
        return self.result(
            task,
            status="blocked",
            findings=[f"{run.id} for {spec.key} is waiting for manual execution: see {registry.run_dir(run.id)}/README.md"],
            artifacts=[f"{registry.run_dir(run.id)}/run.py", f"{registry.run_dir(run.id)}/README.md"],
            data={"run_id": run.id, "status": "awaiting_execution"},
        )

    def _completed(self, task: TaskEnvelope, run: ExperimentRun, outcome: ExecutionOutcome, synthetic_code: bool, spec: ExperimentSpec):
        registry = self.state.experiments
        for problem in check_coverage(spec, run.seeds, outcome.records):
            run.deviations.append(f"incomplete results: {problem}")
        run = registry.record_attempt(run, "completed", duration_s=outcome.duration_s)
        run = registry.complete(run, outcome.records, synthetic=outcome.synthetic or synthetic_code, duration_s=outcome.duration_s, artifacts=outcome.artifacts)
        return self.result(
            task,
            findings=[f"{run.id} completed: {len(run.records)} records over seeds {run.seeds}" + (" (synthetic)" if run.synthetic else "")],
            uncertainties=list(run.deviations),
            artifacts=[f"{registry.run_dir(run.id)}/run.yaml", f"experiments/results/{run.id}.yaml"],
            data={"run_id": run.id, "status": "completed"},
        )

    def _repair(self, run: ExperimentRun, spec: ExperimentSpec, error: str) -> ExperimentRun:
        registry = self.state.experiments
        text = self.ask_text(
            "experiment.debugging",
            f"The experiment script for {spec.key} failed.\n\nError:\n{clip(error, 3000)}\n\n"
            f"Current script:\n```python\n{clip(self.store.read_text(registry.run_dir(run.id) + '/run.py'), 12000)}\n```\n\n"
            "Fix the bug without changing the experimental protocol.\n\n" + CONTRACT,
            {"spec": spec.model_dump(mode="json"), "error": error},
        )
        code = extract_code_block(text) or text
        return registry.update_code(run, code, deviation=f"code repair after failed attempt {len(run.attempts)}: {clip(error.splitlines()[0] if error else 'unknown error', 160)}")
