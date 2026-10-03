from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Type, TypeVar

import yaml
from pydantic import BaseModel

from ..config import Settings
from ..llm.router import ModelRouter
from ..services.literature import LiteratureSearch
from ..services.web_search import WebSearch
from ..state.models import AgentResult, Paper, Project, TaskEnvelope
from ..state.research_state import ResearchState
from ..state.store import ProjectStore
from .prompts import system_prompt

T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentDeps:
    state: ResearchState
    router: ModelRouter
    settings: Settings
    literature: LiteratureSearch
    executor: Any
    web: WebSearch


class Agent:
    key = ""
    name = ""

    def __init__(self, deps: AgentDeps):
        self.deps = deps

    @property
    def state(self) -> ResearchState:
        return self.deps.state

    @property
    def store(self) -> ProjectStore:
        return self.deps.state.store

    @property
    def settings(self) -> Settings:
        return self.deps.settings

    @property
    def router(self) -> ModelRouter:
        return self.deps.router

    def ask(self, task_kind: str, user: str, schema: Type[T], context: Dict[str, Any] | None = None) -> T:
        return self.router.json(task_kind, system_prompt(self.key), user, schema, context=context)

    def ask_text(self, task_kind: str, user: str, context: Dict[str, Any] | None = None) -> str:
        return self.router.text(task_kind, system_prompt(self.key), user, context=context)

    def result(self, task: TaskEnvelope, **fields: Any) -> AgentResult:
        return AgentResult(task_id=task.task_id, agent=self.key, **fields)

    def run(self, task: TaskEnvelope) -> AgentResult:
        raise NotImplementedError

    # ------------------------------------------------------------ shared context

    def project(self) -> Project:
        return self.store.load_project()

    def project_block(self) -> str:
        project = self.project()
        proposal = self.state.proposal()
        lines = [f"Research topic: {project.topic}"]
        if project.question:
            lines.append(f"Research question: {project.question}")
        if project.constraints:
            lines.append("Constraints: " + "; ".join(project.constraints))
        if proposal:
            lines.append(f"Current proposal (v{proposal.version}):\n{clip(proposal.text, 2500)}")
        return "\n".join(lines)


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def as_yaml(data: Any) -> str:
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json", exclude_defaults=True)
    elif isinstance(data, list):
        data = [d.model_dump(mode="json", exclude_defaults=True) if isinstance(d, BaseModel) else d for d in data]
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120).strip()


def paper_digest(papers: Iterable[Paper], detail: bool = True, limit: int = 30) -> str:
    lines: List[str] = []
    for paper in list(papers)[:limit]:
        head = f"{paper.id} | {clip(paper.title, 140)} ({paper.year or 'n.d.'})"
        if paper.cluster:
            head += f" | cluster: {paper.cluster}"
        if not detail or not paper.extracted:
            lines.append(f"- {head}")
            continue
        parts = [
            f"method: {clip(paper.method, 200)}" if paper.method else "",
            f"results: {clip(paper.results, 200)}" if paper.results else "",
            "limitations: " + "; ".join(clip(x, 120) for x in paper.limitations[:3]) if paper.limitations else "",
            "assumptions: " + "; ".join(clip(x, 100) for x in paper.assumptions[:2]) if paper.assumptions else "",
            f"evidence quality: {paper.evidence_quality}",
        ]
        lines.append(f"- {head}\n    " + "\n    ".join(p for p in parts if p))
    return "\n".join(lines) or "(no papers)"


def ranked_papers(papers: Iterable[Paper]) -> List[Paper]:
    return sorted(papers, key=lambda p: (p.relevance_score, p.cited_by_count), reverse=True)


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = (item or "").strip()
        if key and key.lower() not in seen:
            seen.add(key.lower())
            out.append(key)
    return out


def match_by_id(returned: List[Any], ids: List[str]) -> List[Any]:
    """Align items a model returned with the ids it was asked about: by id, else by position."""
    by_id = {str(getattr(item, "id", "")).strip(): item for item in returned}
    if len(returned) == len(ids):
        return [by_id.get(i, item) for i, item in zip(ids, returned)]
    return [by_id.get(i) for i in ids]
