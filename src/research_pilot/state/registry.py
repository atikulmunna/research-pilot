"""Experiment registry: versioned specs, append-only runs, immutable results.

Rules enforced here:
- A spec is never edited in place. Any change creates a new version with a change log.
- Runs are never deleted. Failed runs and every attempt are retained.
- A completed result is written once and cannot be overwritten (no cherry-picking).
"""

import hashlib
import re
from typing import Dict, List

from .models import ExperimentRun, ExperimentSpec, RunAttempt, RunRecord, now_iso
from .store import ProjectStore, load_yaml


RUN_ID = re.compile(r"R\d{1,6}")


class RegistryError(RuntimeError):
    pass


def code_version(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()[:12]


class ExperimentRegistry:
    def __init__(self, store: ProjectStore):
        self.store = store

    # ------------------------------------------------------------ specs

    def specs(self) -> List[ExperimentSpec]:
        folder = self.store.path("experiments/specs")
        items = [ExperimentSpec.model_validate(load_yaml(p.read_text(encoding="utf-8"))) for p in folder.glob("*.yaml")]
        items.sort(key=lambda s: (int(s.id[1:]) if s.id[1:].isdigit() else 0, s.version))
        return items

    def latest_specs(self) -> List[ExperimentSpec]:
        latest: Dict[str, ExperimentSpec] = {}
        for spec in self.specs():
            current = latest.get(spec.id)
            if current is None or spec.version > current.version:
                latest[spec.id] = spec
        return list(latest.values())

    def spec(self, ref: str) -> ExperimentSpec | None:
        exp_id, _, version = ref.partition("@v")
        candidates = [s for s in self.specs() if s.id == exp_id]
        if version:
            candidates = [s for s in candidates if str(s.version) == version]
        return max(candidates, key=lambda s: s.version) if candidates else None

    def add_spec(self, spec: ExperimentSpec, hypothesis_version: int) -> ExperimentSpec:
        spec = spec.model_copy(deep=True)
        spec.id = self.store.next_id("E")
        spec.version = 1
        spec.hypothesis_version = hypothesis_version
        spec.status = "draft"
        spec.created_at = now_iso()
        spec.change_log = [f"v1 created: {spec.objective[:120]}"]
        self._write(spec)
        return spec

    def revise_spec(self, exp_id: str, revised: ExperimentSpec, reason: str) -> ExperimentSpec:
        current = self.spec(exp_id)
        if current is None:
            raise RegistryError(f"Unknown experiment: {exp_id}")
        new = revised.model_copy(deep=True)
        new.id = current.id
        new.version = current.version + 1
        new.hypothesis_id = current.hypothesis_id
        new.hypothesis_version = current.hypothesis_version
        new.status = "draft"
        new.created_at = now_iso()
        new.change_log = [*current.change_log, f"v{new.version}: {reason}"]
        new.critique_notes = list(current.critique_notes)
        superseded = current.model_copy(update={"status": "superseded"})
        self._write(superseded)
        self._write(new)
        return new

    def approve(self, ref: str, notes: List[str] | None = None) -> ExperimentSpec:
        spec = self.spec(ref)
        if spec is None:
            raise RegistryError(f"Unknown experiment: {ref}")
        if spec.status == "superseded":
            raise RegistryError(f"{spec.key} is superseded and cannot be approved")
        approved = spec.model_copy(update={"status": "approved", "critique_notes": [*spec.critique_notes, *(notes or [])]})
        self._write(approved)
        return approved

    def _write(self, spec: ExperimentSpec) -> None:
        self.store.write_yaml(f"experiments/specs/{spec.id}.v{spec.version}.yaml", spec)

    # ------------------------------------------------------------ runs

    def runs(self) -> List[ExperimentRun]:
        folder = self.store.path("experiments/runs")
        items = [ExperimentRun.model_validate(load_yaml(p.read_text(encoding="utf-8"))) for p in folder.glob("*/run.yaml")]
        items.sort(key=lambda r: r.id)
        return items

    def run(self, run_id: str) -> ExperimentRun | None:
        if not RUN_ID.fullmatch(run_id or ""):
            return None
        data = self.store.read_yaml(f"experiments/runs/{run_id}/run.yaml")
        return ExperimentRun.model_validate(data) if data else None

    def runs_for(self, spec_key: str) -> List[ExperimentRun]:
        return [r for r in self.runs() if r.spec_key == spec_key]

    def run_dir(self, run_id: str) -> str:
        return f"experiments/runs/{run_id}"

    def create_run(
        self,
        spec: ExperimentSpec,
        code: str,
        executor: str,
        seeds: List[int],
        dataset_version: str = "",
        model_version: str = "",
        compute: str = "",
        deviations: List[str] | None = None,
        code_generator: str = "",
    ) -> ExperimentRun:
        if spec.status != "approved":
            raise RegistryError(f"{spec.key} must be approved before it can run (status: {spec.status})")
        run = ExperimentRun(
            id=self.store.next_id("R", width=3),
            experiment_id=spec.id,
            spec_version=spec.version,
            hypothesis_id=spec.hypothesis_id,
            code_version=code_version(code),
            dataset_version=dataset_version,
            model_version=model_version,
            config={
                "method": spec.method,
                "baselines": spec.baselines,
                "datasets": spec.datasets,
                "metrics": [m.model_dump() for m in spec.metrics],
                "kind": spec.kind,
            },
            seeds=list(seeds),
            executor=executor,
            code_generator=code_generator,
            compute=compute,
            deviations=list(deviations or []),
        )
        self.store.write_text(f"{self.run_dir(run.id)}/run.py", code)
        self._save(run)
        return run

    def update_code(self, run: ExperimentRun, code: str, deviation: str) -> ExperimentRun:
        """Record a code repair. The protocol (spec) is unchanged; the change is logged."""
        self._guard_open(run)
        attempt = len(run.attempts)
        self.store.write_text(f"{self.run_dir(run.id)}/run.attempt{attempt}.py", self.store.read_text(f"{self.run_dir(run.id)}/run.py"))
        self.store.write_text(f"{self.run_dir(run.id)}/run.py", code)
        run.code_version = code_version(code)
        run.deviations.append(deviation)
        self._save(run)
        return run

    def record_attempt(self, run: ExperimentRun, status: str, error: str = "", duration_s: float = 0.0) -> ExperimentRun:
        self._guard_open(run)
        run.attempts.append(
            RunAttempt(attempt=len(run.attempts) + 1, code_version=run.code_version, status=status, error=error[:2000], duration_s=duration_s)
        )
        run.status = "running" if status == "running" else run.status
        self._save(run)
        return run

    def mark(self, run: ExperimentRun, status: str, error: str = "") -> ExperimentRun:
        self._guard_open(run)
        run.status = status
        run.error = error[:2000]
        self._save(run)
        return run

    def complete(
        self,
        run: ExperimentRun,
        records: List[RunRecord],
        synthetic: bool,
        duration_s: float = 0.0,
        artifacts: List[str] | None = None,
    ) -> ExperimentRun:
        self._guard_open(run)
        result_rel = f"experiments/results/{run.id}.yaml"
        if self.store.path(result_rel).exists():
            raise RegistryError(f"Results for {run.id} already recorded; results are immutable")
        run.records = list(records)
        run.synthetic = synthetic
        run.duration_s = duration_s
        run.artifacts = list(artifacts or [])
        run.status = "completed"
        run.error = ""
        run.completed_at = now_iso()
        self._save(run)
        self.store.write_yaml(
            result_rel,
            {
                "run_id": run.id,
                "experiment": run.spec_key,
                "code_version": run.code_version,
                "seeds": run.seeds,
                "synthetic": synthetic,
                "records": records,
            },
        )
        return run

    def ingest(self, run_id: str, records: List[RunRecord], problems: List[str]) -> ExperimentRun:
        """Record results produced outside the swarm (manual execution)."""
        run = self.run(run_id)
        if run is None:
            raise RegistryError(f"Unknown run: {run_id}")
        if run.status not in {"awaiting_execution", "failed"}:
            raise RegistryError(f"{run_id} is {run.status}; only runs awaiting execution can be ingested")
        if not records:
            raise RegistryError("No result records found. Expected RESULT_JSON lines or a JSON list of {arm, seed, metrics}.")
        run.deviations.extend(f"incomplete results: {p}" for p in problems)
        run = self.record_attempt(run, "completed")
        return self.complete(run, records, synthetic=run.code_generator.startswith("mock"))

    def _guard_open(self, run: ExperimentRun) -> None:
        stored = self.run(run.id)
        if stored is not None and stored.status == "completed":
            raise RegistryError(f"{run.id} is completed; completed runs are immutable")

    def _save(self, run: ExperimentRun) -> None:
        self.store.write_yaml(f"{self.run_dir(run.id)}/run.yaml", run)
