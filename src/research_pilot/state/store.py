"""On-disk project workspace, laid out as the shared research state described in the plan."""

import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Type, TypeVar

import yaml
from pydantic import BaseModel

from .models import (
    Action,
    Assessment,
    ClaimMap,
    Decision,
    Draft,
    FieldMap,
    Gap,
    Hypothesis,
    Interpretation,
    LiteratureClustering,
    NoveltyAssessment,
    Paper,
    Project,
    ProposalVersion,
    QuantResult,
    ReviewReport,
    now_iso,
)

M = TypeVar("M", bound=BaseModel)
REPLACE_ATTEMPTS = 10
PROJECT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,80}")
log = logging.getLogger(__name__)
_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_Dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


def load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)

LAYOUT = (
    "field",
    "literature/papers",
    "literature/clusters",
    "literature/citations",
    "gaps",
    "proposal",
    "hypotheses",
    "experiments/specs",
    "experiments/runs",
    "experiments/results",
    "experiments/artifacts",
    "analysis/quantitative",
    "analysis/errors",
    "analysis/interpretations",
    "decisions",
    "roadmap",
    "paper/figures",
    "paper/tables",
    "paper/manuscript",
    "paper/supplementary",
    "reviews/red_team",
    "evidence",
    "logs",
)


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:limit].rstrip("-") or "project")


def _dump(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, list):
        return [_dump(item) for item in data]
    if isinstance(data, dict):
        return {key: _dump(value) for key, value in data.items()}
    return data


def _replace_with_retry(tmp: Path, path: Path) -> None:
    """Move tmp over path. On Windows a replace fails while a reader (the dashboard, antivirus) holds the file open."""
    for attempt in range(REPLACE_ATTEMPTS - 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    try:
        os.replace(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = RLock()
        self._papers: Dict[str, Paper] | None = None

    # ------------------------------------------------------------ lifecycle

    @classmethod
    def create(cls, workspace: str | Path, project: Project) -> "ProjectStore":
        base = Path(workspace)
        if not project.id:
            project.id = f"{slugify(project.title or project.topic)}-{secrets.token_hex(2)}"
        root = base / project.id
        if root.exists():
            raise FileExistsError(f"Project already exists: {root}")
        for rel in LAYOUT:
            (root / rel).mkdir(parents=True, exist_ok=True)
        store = cls(root)
        store.save_project(project)
        return store

    @classmethod
    def open(cls, workspace: str | Path, project_ref: str) -> "ProjectStore":
        project_id = cls.resolve_id(workspace, project_ref)
        root = Path(workspace) / project_id
        if not (root / "project.yaml").exists():
            raise FileNotFoundError(f"Project not found: {project_ref}")
        return cls(root)

    @staticmethod
    def list_projects(workspace: str | Path) -> List[Project]:
        base = Path(workspace)
        if not base.exists():
            return []
        out = []
        for path in base.glob("*/project.yaml"):
            try:
                out.append(Project.model_validate(load_yaml(path.read_text(encoding="utf-8"))))
            except Exception as exc:
                log.warning("skipping unreadable project %s: %s", path.parent.name, exc)
        out.sort(key=lambda p: p.updated_at, reverse=True)
        return out

    @classmethod
    def resolve_id(cls, workspace: str | Path, project_ref: str) -> str:
        ref = (project_ref or "").strip()
        if ref in {"", "latest"}:
            projects = cls.list_projects(workspace)
            if not projects:
                raise FileNotFoundError("No projects found.")
            return projects[0].id
        if not PROJECT_ID.fullmatch(ref):
            raise FileNotFoundError(f"Invalid project id: {project_ref!r}")
        if (Path(workspace) / ref / "project.yaml").exists():
            return ref
        matches = [p.id for p in cls.list_projects(workspace) if p.id.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"Project not found: {project_ref}")

    # ------------------------------------------------------------ raw io

    def path(self, rel: str) -> Path:
        return self.root / rel

    def read_yaml(self, rel: str, default: Any = None) -> Any:
        path = self.path(rel)
        if not path.exists():
            return default
        data = load_yaml(path.read_text(encoding="utf-8"))
        return default if data is None else data

    def write_yaml(self, rel: str, data: Any) -> None:
        text = yaml.dump(_dump(data), Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=110)
        self.write_text(rel, text)

    def read_text(self, rel: str, default: str = "") -> str:
        path = self.path(rel)
        return path.read_text(encoding="utf-8") if path.exists() else default

    def write_text(self, rel: str, text: str) -> None:
        path = self.path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(3)}.tmp")
        with self._lock:
            tmp.write_text(text, encoding="utf-8")
            _replace_with_retry(tmp, path)

    def append_jsonl(self, rel: str, record: Dict[str, Any]) -> None:
        path = self.path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_dump(record), ensure_ascii=False, default=str) + "\n")

    def read_jsonl(self, rel: str, limit: int | None = None) -> List[Dict[str, Any]]:
        path = self.path(rel)
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-limit:] if limit else rows

    def _load_list(self, rel: str, model: Type[M]) -> List[M]:
        return [model.model_validate(row) for row in (self.read_yaml(rel, []) or [])]

    def _load_one(self, rel: str, model: Type[M]) -> M | None:
        data = self.read_yaml(rel)
        return model.model_validate(data) if data else None

    # ------------------------------------------------------------ ids

    def next_id(self, prefix: str, width: int = 0) -> str:
        with self._lock:
            counters = self.read_yaml(".counters.yaml", {}) or {}
            value = int(counters.get(prefix, 0)) + 1
            counters[prefix] = value
            self.write_yaml(".counters.yaml", counters)
        return f"{prefix}{value:0{width}d}" if width else f"{prefix}{value}"

    # ------------------------------------------------------------ project

    def load_project(self) -> Project:
        return Project.model_validate(self.read_yaml("project.yaml"))

    def save_project(self, project: Project) -> None:
        project.updated_at = now_iso()
        self.write_yaml("project.yaml", project)

    # ------------------------------------------------------------ field and literature

    def field_map(self) -> FieldMap | None:
        return self._load_one("field/field_map.yaml", FieldMap)

    def save_field_map(self, field_map: FieldMap) -> None:
        self.write_yaml("field/field_map.yaml", field_map)
        self.write_yaml("field/terminology.yaml", field_map.terminology)

    def _paper_cache(self) -> Dict[str, Paper]:
        # Write-through cache: this instance is the only writer during a run.
        if self._papers is None:
            folder = self.path("literature/papers")
            files = sorted(folder.glob("*.yaml")) if folder.exists() else []
            self._papers = {p.stem: Paper.model_validate(load_yaml(p.read_text(encoding="utf-8"))) for p in files}
        return self._papers

    def papers(self) -> List[Paper]:
        return [p.model_copy(deep=True) for _, p in sorted(self._paper_cache().items())]

    def paper(self, paper_id: str) -> Paper | None:
        cached = self._paper_cache().get(paper_id)
        return cached.model_copy(deep=True) if cached else None

    def save_paper(self, paper: Paper) -> None:
        self.write_yaml(f"literature/papers/{paper.id}.yaml", paper)
        self._paper_cache()[paper.id] = paper.model_copy(deep=True)

    def literature_map(self) -> Dict[str, Any]:
        return self.read_yaml("literature/literature_map.yaml", {}) or {}

    def save_literature_map(self, data: Dict[str, Any]) -> None:
        self.write_yaml("literature/literature_map.yaml", data)

    def clustering(self) -> LiteratureClustering | None:
        return self._load_one("literature/clusters/clusters.yaml", LiteratureClustering)

    def save_clustering(self, clustering: LiteratureClustering) -> None:
        self.write_yaml("literature/clusters/clusters.yaml", clustering)

    def citation_graph(self) -> Dict[str, Any]:
        return self.read_yaml("literature/citations/citation_graph.yaml", {"edges": []}) or {"edges": []}

    def save_citation_graph(self, data: Dict[str, Any]) -> None:
        self.write_yaml("literature/citations/citation_graph.yaml", data)

    # ------------------------------------------------------------ gaps, proposal, critique

    def gaps(self) -> List[Gap]:
        data = self.read_yaml("gaps/research_gaps.yaml", {}) or {}
        return [Gap.model_validate(row) for row in data.get("gaps", [])]

    def gap_analysis_meta(self) -> Dict[str, Any]:
        data = self.read_yaml("gaps/research_gaps.yaml", {}) or {}
        return {k: v for k, v in data.items() if k != "gaps"}

    def save_gaps(self, gaps: List[Gap], meta: Dict[str, Any]) -> None:
        self.write_yaml("gaps/research_gaps.yaml", {**meta, "gaps": gaps})

    def proposal_versions(self) -> List[ProposalVersion]:
        return self._load_list("proposal/proposal.yaml", ProposalVersion)

    def save_proposal_versions(self, versions: List[ProposalVersion]) -> None:
        self.write_yaml("proposal/proposal.yaml", versions)

    def assessments(self) -> List[Assessment]:
        return self._load_list("proposal/criticisms.yaml", Assessment)

    def save_assessments(self, items: List[Assessment]) -> None:
        self.write_yaml("proposal/criticisms.yaml", items)

    def novelty_analyses(self) -> List[NoveltyAssessment]:
        return self._load_list("proposal/novelty_analysis.yaml", NoveltyAssessment)

    def save_novelty_analyses(self, items: List[NoveltyAssessment]) -> None:
        self.write_yaml("proposal/novelty_analysis.yaml", items)

    # ------------------------------------------------------------ hypotheses

    def hypotheses(self) -> List[Hypothesis]:
        return self._load_list("hypotheses/hypotheses.yaml", Hypothesis)

    def save_hypotheses(self, items: List[Hypothesis]) -> None:
        self.write_yaml("hypotheses/hypotheses.yaml", items)

    # ------------------------------------------------------------ analysis

    def quant_result(self, run_id: str) -> QuantResult | None:
        return self._load_one(f"analysis/quantitative/{run_id}.yaml", QuantResult)

    def save_quant_result(self, result: QuantResult) -> None:
        self.write_yaml(f"analysis/quantitative/{result.run_id}.yaml", result)

    def quant_results(self) -> List[QuantResult]:
        return [QuantResult.model_validate(load_yaml(p.read_text(encoding="utf-8"))) for p in sorted(self.path("analysis/quantitative").glob("*.yaml"))]

    def interpretation(self, run_id: str) -> Interpretation | None:
        return self._load_one(f"analysis/interpretations/{run_id}.yaml", Interpretation)

    def save_interpretation(self, item: Interpretation) -> None:
        self.write_yaml(f"analysis/interpretations/{item.run_id}.yaml", item)

    def interpretations(self) -> List[Interpretation]:
        return [Interpretation.model_validate(load_yaml(p.read_text(encoding="utf-8"))) for p in sorted(self.path("analysis/interpretations").glob("*.yaml"))]

    # ------------------------------------------------------------ decisions and roadmap

    def decisions(self) -> List[Decision]:
        return self._load_list("decisions/decision_log.yaml", Decision)

    def save_decisions(self, items: List[Decision]) -> None:
        self.write_yaml("decisions/decision_log.yaml", items)

    def roadmap(self) -> List[Action]:
        return self._load_list("roadmap/roadmap.yaml", Action)

    def save_roadmap(self, items: Iterable[Action]) -> None:
        self.write_yaml("roadmap/roadmap.yaml", list(items))

    # ------------------------------------------------------------ paper and reviews

    def claim_map(self) -> ClaimMap | None:
        return self._load_one("paper/claims.yaml", ClaimMap)

    def save_claim_map(self, claim_map: ClaimMap) -> None:
        self.write_yaml("paper/claims.yaml", claim_map)

    def draft(self) -> Draft | None:
        return self._load_one("paper/manuscript/draft.yaml", Draft)

    def save_draft(self, draft: Draft) -> None:
        self.write_yaml("paper/manuscript/draft.yaml", draft)

    def manuscript(self) -> str:
        return self.read_text("paper/manuscript/manuscript.md")

    def save_manuscript(self, markdown: str) -> None:
        self.write_text("paper/manuscript/manuscript.md", markdown)

    def reviews(self) -> List[ReviewReport]:
        folder = self.path("reviews/red_team")
        return [ReviewReport.model_validate(load_yaml(p.read_text(encoding="utf-8"))) for p in sorted(folder.glob("review_*.yaml"))]

    def save_review(self, report: ReviewReport) -> None:
        self.write_yaml(f"reviews/red_team/review_{report.round:02d}.yaml", report)

    # ------------------------------------------------------------ evidence graph and logs

    def graph_data(self) -> Dict[str, Any]:
        return self.read_yaml("evidence/evidence_graph.yaml", {"nodes": [], "edges": []}) or {"nodes": [], "edges": []}

    def save_graph_data(self, data: Dict[str, Any]) -> None:
        self.write_yaml("evidence/evidence_graph.yaml", data)

    def log_activity(self, record: Dict[str, Any]) -> None:
        self.append_jsonl("logs/activity.jsonl", record)

    def activity(self, limit: int | None = None) -> List[Dict[str, Any]]:
        return self.read_jsonl("logs/activity.jsonl", limit=limit)

    def log_llm_call(self, record: Dict[str, Any]) -> None:
        self.append_jsonl("logs/llm_calls.jsonl", record)

    def llm_calls(self, limit: int | None = None) -> List[Dict[str, Any]]:
        return self.read_jsonl("logs/llm_calls.jsonl", limit=limit)
