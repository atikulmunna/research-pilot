import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import views
from .config import Settings
from .orchestrator import create_project
from .runner import ProjectRunner
from .services.export import markdown_to_html
from .state.store import ProjectStore


class CreateProject(BaseModel):
    topic: str = Field(min_length=3, max_length=2000)
    question: str = Field("", max_length=4000)
    proposal: str = Field("", max_length=20000)
    seed_papers: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    title: str = Field("", max_length=200)
    autorun: bool = True
    max_steps: int | None = Field(None, ge=1, le=500)


class RunRequest(BaseModel):
    max_steps: int | None = Field(None, ge=1, le=500)


class IngestRequest(BaseModel):
    text: str = ""
    records: List[Dict[str, Any]] = Field(default_factory=list)


def create_app(settings: Settings | None = None, runner: ProjectRunner | None = None) -> FastAPI:
    settings = settings or Settings()
    runner = runner or ProjectRunner(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            runner.shutdown()

    app = FastAPI(title="Research Pilot API", version="0.2.0", lifespan=lifespan)
    rate_lock = Lock()
    rate_counters: Dict[str, Dict[str, int]] = {}

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)
        token = settings.api_auth_token.strip()
        provided = request.headers.get("x-api-key", "")
        if token and provided != token:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        per_minute = int(settings.api_rate_limit_per_minute or 0)
        if per_minute > 0:
            identifier = provided or (request.client.host if request.client else "anonymous")
            window = int(time.time() // 60)
            with rate_lock:
                bucket = rate_counters.get(identifier)
                if not bucket or bucket["window"] != window:
                    bucket = {"window": window, "count": 0}
                if bucket["count"] >= per_minute:
                    rate_counters[identifier] = bucket
                    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
                bucket["count"] += 1
                rate_counters[identifier] = bucket
        return await call_next(request)

    def open_store(project_id: str) -> ProjectStore:
        try:
            return ProjectStore.open(settings.workspace_dir, project_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Project not found")

    @app.post("/api/v1/projects", status_code=202)
    def create(payload: CreateProject):
        store = create_project(
            settings,
            topic=payload.topic,
            question=payload.question,
            proposal=payload.proposal,
            seed_papers=payload.seed_papers,
            constraints=payload.constraints,
            title=payload.title,
        )
        project_id = store.root.name
        started = runner.start(project_id, payload.max_steps) if payload.autorun else False
        return {"project_id": project_id, "status": "running" if started else "created"}

    @app.get("/api/v1/projects")
    def list_projects():
        running = {p.id for p in ProjectStore.list_projects(settings.workspace_dir) if runner.is_running(p.id)}
        return {
            "projects": [
                {**views.jsonable(p), "running": p.id in running}
                for p in ProjectStore.list_projects(settings.workspace_dir)
            ]
        }

    @app.get("/api/v1/projects/{project_id}")
    def get_project(project_id: str):
        store = open_store(project_id)
        running = runner.is_running(store.root.name)
        return {**views.overview(store, settings), "running": running}

    @app.post("/api/v1/projects/{project_id}/run", status_code=202)
    def run_project(project_id: str, payload: RunRequest | None = None):
        store = open_store(project_id)
        project = store.load_project()
        if project.status == "completed":
            raise HTTPException(status_code=409, detail="Project already completed")
        if not runner.start(project.id, payload.max_steps if payload else None):
            raise HTTPException(status_code=409, detail="Project is already running")
        return {"project_id": project.id, "status": "running"}

    @app.post("/api/v1/projects/{project_id}/cancel")
    def cancel_project(project_id: str):
        store = open_store(project_id)
        return {"project_id": store.root.name, "cancelled": runner.cancel(store.root.name)}

    @app.get("/api/v1/projects/{project_id}/state/{section}")
    def get_section(project_id: str, section: str):
        store = open_store(project_id)
        if section not in views.SECTIONS:
            raise HTTPException(status_code=404, detail=f"Unknown section. Choose from: {', '.join(views.SECTIONS)}")
        return {"section": section, "data": views.section(store, settings, section)}

    @app.get("/api/v1/projects/{project_id}/evidence")
    def get_evidence(project_id: str, graph: bool = False):
        return views.evidence_report(open_store(project_id), settings, include_graph=graph)

    @app.get("/api/v1/projects/{project_id}/activity")
    def get_activity(project_id: str, limit: int = 100):
        return {"events": open_store(project_id).activity(limit=max(1, min(limit, 1000)))}

    @app.get("/api/v1/projects/{project_id}/metrics")
    def get_metrics(project_id: str):
        return views.metrics_report(open_store(project_id))

    @app.get("/api/v1/projects/{project_id}/manuscript")
    def get_manuscript(project_id: str, format: str = "md"):
        store = open_store(project_id)
        markdown = store.manuscript()
        if not markdown:
            raise HTTPException(status_code=404, detail="No manuscript yet")
        if format == "html":
            return HTMLResponse(markdown_to_html(markdown, store.load_project().title))
        return PlainTextResponse(markdown, media_type="text/markdown")

    @app.post("/api/v1/projects/{project_id}/runs/{run_id}/results")
    def ingest(project_id: str, run_id: str, payload: IngestRequest):
        store = open_store(project_id)
        if runner.is_running(store.root.name):
            raise HTTPException(status_code=409, detail="Project is running")
        text = payload.text or "\n".join(json.dumps(r) for r in payload.records)
        try:
            return views.ingest_results(store, settings, run_id, text)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/v1/routing")
    def get_routing():
        return {"routes": views.routing_table(settings)}

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return HTMLResponse((Path(__file__).parent / "web" / "dashboard.html").read_text(encoding="utf-8"))

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/dashboard")

    app.state.runner = runner
    app.state.settings = settings
    return app


app = create_app()
