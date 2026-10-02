import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from . import views
from .config import Settings
from .orchestrator import Orchestrator, create_project
from .services.executor import SubprocessExecutor
from .services.export import markdown_to_html, markdown_to_pdf
from .state.research_state import ResearchState
from .state.store import ProjectStore

app = typer.Typer(help="Research agent swarm: from a research idea to an evidence-backed manuscript.", no_args_is_help=True)
experiments_app = typer.Typer(help="Inspect, execute and ingest experiment runs.", no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")

JSON_OPTION = typer.Option(False, "--json", help="Print JSON instead of text.")


def _settings() -> Settings:
    return Settings()


def _open(project: str, settings: Settings) -> ProjectStore:
    try:
        return ProjectStore.open(settings.workspace_dir, project)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


def _print(data: Any, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif isinstance(data, str):
        typer.echo(data)
    else:
        typer.echo(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110))


def _table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c) for c in columns}
    lines = ["  ".join(c.upper().ljust(widths[c]) for c in columns)]
    lines += ["  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows]
    return "\n".join(lines)


def _progress(event: Dict[str, Any]) -> None:
    kind = event["event"]
    if kind == "phase":
        typer.echo(f"\n== {event['phase'].replace('_', ' ').title()} ==")
    elif kind == "task_started":
        typer.echo(f"  > {views.AGENT_NAMES.get(event['agent'], event['agent'])}: {event['objective']}")
    elif kind == "task_finished":
        for finding in event.get("findings", [])[:2]:
            typer.echo(f"      {finding[:150]}")
        if event.get("violations"):
            typer.echo(f"      ! {event['violations']} guard violation(s) logged")
    elif kind == "action_retry":
        typer.echo(f"  ! retrying {event['action_id']}: {event['error'][:150]}")


def _run(store: ProjectStore, settings: Settings, max_steps: Optional[int], quiet: bool) -> None:
    orchestrator = Orchestrator(settings, store, on_event=None if quiet else _progress)
    project = orchestrator.run(max_steps)
    typer.echo("")
    typer.echo(f"Project {project.id}: {project.status}" + (f" ({project.outcome})" if project.outcome else ""))
    if project.stop_reason:
        typer.echo(f"Reason: {project.stop_reason}")
    if project.error:
        typer.echo(f"Error: {project.error}", err=True)
    usage = project.usage
    typer.echo(f"Steps {project.step}, LLM calls {usage.llm_calls}, tokens {usage.total_tokens}, cost ${usage.cost_usd:.4f}")
    for tier, row in sorted(usage.by_tier.items()):
        typer.echo(f"  {tier:<7} {int(row['calls']):>4} calls  {int(row['total_tokens']):>9} tokens  ${row['cost_usd']:.4f}")
    if project.status == "awaiting_experiments":
        typer.echo("Review and run the generated experiment, then ingest its results:")
        typer.echo(f"  research-pilot experiments list {project.id}")
    if store.manuscript():
        typer.echo(f"Manuscript: {store.path('paper/manuscript/manuscript.md')}")
    if project.status == "failed":
        raise typer.Exit(code=1)


@app.command()
def new(
    topic: str = typer.Argument(..., help="Research topic or idea."),
    question: str = typer.Option("", help="Research question."),
    proposal: str = typer.Option("", help="Proposal text, or @path to read it from a file."),
    seed_paper: List[str] = typer.Option([], "--seed-paper", help="DOI, arXiv id or title of a seed paper (repeatable)."),
    constraint: List[str] = typer.Option([], "--constraint", help="Constraint such as compute or deadline (repeatable)."),
    title: str = typer.Option("", help="Short project title."),
    run: bool = typer.Option(False, "--run", help="Start the swarm right away."),
    max_steps: Optional[int] = typer.Option(None, help="Override the total step budget."),
    quiet: bool = typer.Option(False, help="Hide live progress."),
):
    """Create a research project (and optionally run it)."""
    settings = _settings()
    if proposal.startswith("@"):
        proposal = Path(proposal[1:]).read_text(encoding="utf-8")
    store = create_project(settings, topic, question, proposal, seed_paper, constraint, title)
    typer.echo(f"Created project {store.root.name} in {store.root}")
    if run:
        _run(store, settings, max_steps, quiet)


@app.command()
def run(
    project: str = typer.Argument("latest", help="Project id (or prefix, or 'latest')."),
    max_steps: Optional[int] = typer.Option(None, help="Override the total step budget."),
    quiet: bool = typer.Option(False, help="Hide live progress."),
):
    """Run or resume a project."""
    settings = _settings()
    _run(_open(project, settings), settings, max_steps, quiet)


@app.command()
def projects(as_json: bool = JSON_OPTION):
    """List projects in the workspace."""
    settings = _settings()
    rows = [
        {"id": p.id, "status": p.status, "phase": p.phase, "outcome": p.outcome, "steps": p.step, "updated": p.updated_at}
        for p in ProjectStore.list_projects(settings.workspace_dir)
    ]
    if as_json:
        _print(rows, True)
    else:
        typer.echo(_table(rows, ["id", "status", "phase", "outcome", "steps", "updated"]) if rows else "No projects yet.")


@app.command()
def status(project: str = typer.Argument("latest"), as_json: bool = JSON_OPTION):
    """Show a project's status, counts, budget use and completion checks."""
    settings = _settings()
    data = views.overview(_open(project, settings), settings)
    if as_json:
        _print(data, True)
        return
    p = data["project"]
    typer.echo(f"{p['id']}: {p['title']}")
    typer.echo(f"Status {p['status']}  phase {p['phase']}  outcome {p['outcome'] or '-'}  step {p['step']}")
    typer.echo("Counts: " + ", ".join(f"{k} {v}" for k, v in data["counts"].items()))
    typer.echo(f"Evidence graph: {data['evidence']}")
    for tier, row in sorted(p["usage"]["by_tier"].items()):
        typer.echo(f"  {tier:<7} {int(row['calls']):>4} calls  {int(row['total_tokens']):>9} tokens  ${row['cost_usd']:.4f}")
    checks = p.get("completion", {}).get("checks", [])
    if checks:
        typer.echo("Completion checks:")
        for check in checks:
            typer.echo(f"  [{'x' if check['passed'] else ' '}] {check['name']} {check.get('detail', '')}")
    if data["next_actions"]:
        typer.echo("Next actions: " + ", ".join(f"{a['type']} {a['target']}".strip() for a in data["next_actions"]))


@app.command()
def show(
    project: str = typer.Argument(..., help="Project id or 'latest'."),
    section: str = typer.Argument(..., help=f"One of: {', '.join(views.SECTIONS)}"),
    as_json: bool = JSON_OPTION,
):
    """Print one section of the shared research state."""
    settings = _settings()
    try:
        _print(views.section(_open(project, settings), settings, section), as_json)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@app.command()
def evidence(project: str = typer.Argument("latest"), graph: bool = typer.Option(False, help="Include all nodes and edges."), as_json: bool = JSON_OPTION):
    """Query the evidence graph (single-support, contradicted, untested, unsupported statements, hotspots)."""
    settings = _settings()
    _print(views.evidence_report(_open(project, settings), settings, include_graph=graph), as_json)


@app.command()
def routing(as_json: bool = JSON_OPTION):
    """Show how each task kind is routed (difficulty, tier, model, reasoning effort)."""
    rows = views.routing_table(_settings())
    if as_json:
        _print(rows, True)
    else:
        typer.echo(_table(rows, ["task", "difficulty", "tier", "provider", "model", "reasoning_effort"]))


@app.command()
def metrics(project: str = typer.Argument("latest"), as_json: bool = JSON_OPTION):
    """Show LLM usage of a project by tier and by task."""
    settings = _settings()
    data = views.metrics_report(_open(project, settings))
    if as_json:
        _print(data, True)
        return
    typer.echo(_table(data["tasks"], ["task", "difficulty", "tier", "calls", "tokens", "cost_usd", "escalations", "failures"]))


@app.command()
def export(
    project: str = typer.Argument("latest"),
    to: str = typer.Option("html", help="md, html or pdf"),
    output: Optional[str] = typer.Option(None, help="Output path."),
):
    """Export the manuscript."""
    settings = _settings()
    store = _open(project, settings)
    markdown = store.manuscript()
    if not markdown:
        typer.echo("This project has no manuscript yet.", err=True)
        raise typer.Exit(code=1)
    fmt = to.lower().strip()
    if fmt not in {"md", "html", "pdf"}:
        typer.echo("--to must be md, html or pdf", err=True)
        raise typer.Exit(code=1)
    out = Path(output or store.path(f"paper/manuscript/manuscript.{fmt}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    title = store.load_project().title
    if fmt == "md":
        out.write_text(markdown, encoding="utf-8")
    elif fmt == "html":
        out.write_text(markdown_to_html(markdown, title), encoding="utf-8")
    else:
        markdown_to_pdf(out, markdown, title)
    typer.echo(f"Exported {fmt} to {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the API server and dashboard."""
    import uvicorn

    typer.echo(f"Dashboard: http://{host}:{port}/dashboard")
    uvicorn.run("research_pilot.api:app", host=host, port=port)


@experiments_app.command("list")
def experiments_list(project: str = typer.Argument("latest"), as_json: bool = JSON_OPTION):
    """List experiment runs with their status."""
    settings = _settings()
    store = _open(project, settings)
    runs = ResearchState(store, settings).experiments.runs()
    rows = [
        {"run": r.id, "experiment": r.spec_key, "status": r.status, "seeds": r.seeds, "code": r.code_version, "synthetic": r.synthetic, "dir": str(store.path(f"experiments/runs/{r.id}"))}
        for r in runs
    ]
    if as_json:
        _print(rows, True)
    else:
        typer.echo(_table(rows, ["run", "experiment", "status", "seeds", "code", "synthetic", "dir"]) if rows else "No runs yet.")


@experiments_app.command("ingest")
def experiments_ingest(project: str, run_id: str, results_file: Path):
    """Ingest results for a run that was executed manually."""
    settings = _settings()
    store = _open(project, settings)
    try:
        info = views.ingest_results(store, settings, run_id, results_file.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"Ingest failed: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{info['run_id']}: {info['records']} records ingested ({info['status']})")
    for problem in info["problems"]:
        typer.echo(f"  warning: {problem}")
    typer.echo(f"Resume with: research-pilot run {store.root.name}")


@experiments_app.command("execute")
def experiments_execute(project: str, run_id: str, timeout: int = typer.Option(900, help="Seconds per seed.")):
    """Run a generated experiment locally (you are responsible for reviewing its code first)."""
    settings = _settings()
    store = _open(project, settings)
    registry = ResearchState(store, settings).experiments
    run = registry.run(run_id)
    if run is None or run.status != "awaiting_execution":
        typer.echo(f"{run_id} is not awaiting execution.", err=True)
        raise typer.Exit(code=1)
    spec = registry.spec(run.spec_key)
    outcome = SubprocessExecutor(timeout_s=timeout, python=settings.experiment_python).execute(store.path(f"experiments/runs/{run.id}"), spec, run.seeds, run.id)
    if outcome.status != "completed":
        registry.record_attempt(run, "failed", error=outcome.error, duration_s=outcome.duration_s)
        typer.echo(f"Execution failed:\n{outcome.error}", err=True)
        raise typer.Exit(code=1)
    lines = "\n".join(r.model_dump_json() for r in outcome.records)
    info = views.ingest_results(store, settings, run_id, lines)
    typer.echo(f"{run_id} completed with {info['records']} records. Resume with: research-pilot run {store.root.name}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
