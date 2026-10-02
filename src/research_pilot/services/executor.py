"""Experiment execution backends.

The script contract every generated experiment follows:

    python run.py --seed <int> --output-dir <path>

For each arm it prints one line:

    RESULT_JSON: {"arm": "<arm name>", "seed": <int>, "metrics": {"<metric>": <float>}}

Backends:
- manual (default): writes the script and instructions; a human runs it and ingests results.
- subprocess: runs the script locally with a timeout. This is NOT a security sandbox, so
  only enable it for code you are prepared to run on this machine.
- simulated: produces deterministic synthetic numbers for demos and tests. Results are
  flagged synthetic and never count as evidence unless explicitly allowed.
"""

import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from ..state.models import ExperimentSpec, RunRecord

RESULT_PREFIX = "RESULT_JSON:"


@dataclass
class ExecutionOutcome:
    status: str
    records: List[RunRecord] = field(default_factory=list)
    synthetic: bool = False
    duration_s: float = 0.0
    error: str = ""
    artifacts: List[str] = field(default_factory=list)


def parse_result_lines(lines: Iterable[str]) -> List[RunRecord]:
    records: List[RunRecord] = []
    for line in lines:
        text = line.strip()
        if text.startswith(RESULT_PREFIX):
            text = text[len(RESULT_PREFIX) :].strip()
        elif not text.startswith("{"):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "arm" not in data or "metrics" not in data:
            continue
        metrics = {}
        for key, value in (data.get("metrics") or {}).items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        records.append(RunRecord(arm=str(data["arm"]), seed=int(data.get("seed", 0)), metrics=metrics))
    return records


def load_records(text: str) -> List[RunRecord]:
    """Parse ingested results: RESULT_JSON lines, JSON lines, a JSON list, or {"records": [...]}."""
    stripped = (text or "").strip()
    if stripped[:1] in "[{":
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if isinstance(data, list):
            return parse_result_lines(json.dumps(item) for item in data)
    return parse_result_lines(stripped.splitlines())


def check_coverage(spec: ExperimentSpec, seeds: List[int], records: List[RunRecord]) -> List[str]:
    """Report declared arm/seed/metric combinations that produced no result."""
    problems = []
    seen = {(r.arm, r.seed) for r in records}
    for arm in spec.arms:
        missing = [s for s in seeds if (arm, s) not in seen]
        if missing:
            problems.append(f"arm '{arm}' missing seeds {missing}")
    declared = {m.name for m in spec.metrics}
    reported = {k for r in records for k in r.metrics}
    if declared - reported:
        problems.append(f"metrics never reported: {sorted(declared - reported)}")
    return problems


class ManualExecutor:
    name = "manual"

    def execute(self, run_dir: Path, spec: ExperimentSpec, seeds: List[int], run_id: str) -> ExecutionOutcome:
        commands = "\n".join(f"python run.py --seed {s} --output-dir artifacts >> results.jsonl" for s in seeds)
        (run_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# Run {run_id} ({spec.key})",
                    "",
                    f"Objective: {spec.objective}",
                    "",
                    "Review `run.py`, then run it for every seed from this directory:",
                    "",
                    "```bash",
                    commands,
                    "```",
                    "",
                    "Then ingest the results and resume the project:",
                    "",
                    "```bash",
                    f"research-pilot experiments ingest <project> {run_id} results.jsonl",
                    "research-pilot run <project>",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return ExecutionOutcome(status="awaiting_execution")


class SubprocessExecutor:
    name = "subprocess"

    def __init__(self, timeout_s: int = 900, python: str = ""):
        self.timeout_s = timeout_s
        self.python = python or sys.executable

    def execute(self, run_dir: Path, spec: ExperimentSpec, seeds: List[int], run_id: str) -> ExecutionOutcome:
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        records: List[RunRecord] = []
        start = time.perf_counter()
        for seed in seeds:
            try:
                proc = subprocess.run(
                    [self.python, "run.py", "--seed", str(seed), "--output-dir", str(artifacts)],
                    cwd=run_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                return ExecutionOutcome(status="failed", records=records, duration_s=time.perf_counter() - start, error=f"seed {seed}: timed out after {self.timeout_s}s")
            with (run_dir / "stdout.log").open("a", encoding="utf-8") as out:
                out.write(f"--- seed {seed} ---\n{proc.stdout}")
            with (run_dir / "stderr.log").open("a", encoding="utf-8") as err:
                err.write(f"--- seed {seed} ---\n{proc.stderr}")
            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
                return ExecutionOutcome(status="failed", records=records, duration_s=time.perf_counter() - start, error=f"seed {seed}: exit {proc.returncode}\n{tail}")
            seed_records = parse_result_lines(proc.stdout.splitlines())
            if not seed_records:
                return ExecutionOutcome(status="failed", records=records, duration_s=time.perf_counter() - start, error=f"seed {seed}: no {RESULT_PREFIX} lines in output")
            records.extend(seed_records)
        produced = sorted(str(p.relative_to(run_dir)) for p in artifacts.rglob("*") if p.is_file())
        return ExecutionOutcome(status="completed", records=records, duration_s=time.perf_counter() - start, artifacts=produced)


class SimulatedExecutor:
    name = "simulated"

    def execute(self, run_dir: Path, spec: ExperimentSpec, seeds: List[int], run_id: str) -> ExecutionOutcome:
        records = []
        effect_size = {"primary": 0.04, "ablation": 0.03, "validation": 0.02}.get(spec.kind, 0.01)
        for arm_index, arm in enumerate(spec.arms):
            for seed in seeds:
                metrics = {}
                for metric in spec.metrics:
                    digest = hashlib.sha256(f"{spec.id}|{spec.kind}|{metric.name}".encode()).hexdigest()
                    base = 0.6 + (int(digest[:4], 16) % 200) / 1000.0
                    effect = effect_size if arm_index == 0 else 0.0
                    rng = random.Random(f"{digest}|{arm}|{seed}")
                    value = base + effect + rng.gauss(0, 0.01)
                    metrics[metric.name] = round(value if metric.higher_is_better else 1.0 - value, 6)
                records.append(RunRecord(arm=arm, seed=seed, metrics=metrics))
        return ExecutionOutcome(status="completed", records=records, synthetic=True)


def make_executor(settings):
    name = (settings.experiment_executor or "manual").strip().lower()
    if name == "manual":
        return ManualExecutor()
    if name == "subprocess":
        return SubprocessExecutor(timeout_s=settings.experiment_timeout_s, python=settings.experiment_python)
    if name == "simulated":
        return SimulatedExecutor()
    raise ValueError(f"Unknown experiment executor: {name}")
