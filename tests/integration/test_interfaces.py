import json
import time

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from research_pilot.api import create_app
from research_pilot.cli import app as cli_app


def wait_until_idle(client, project_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/projects/{project_id}").json()
        if not body["running"]:
            return body
        time.sleep(0.2)
    raise AssertionError("project did not finish")


def test_api_project_lifecycle(mock_settings):
    settings = mock_settings()
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/v1/projects", json={"topic": "diffusion models for protein design", "max_steps": 25})
        assert created.status_code == 202
        project_id = created.json()["project_id"]
        body = wait_until_idle(client, project_id)
        assert body["project"]["status"] == "completed"
        assert body["counts"]["papers"] > 0 and body["has_manuscript"]

        assert client.get("/api/v1/projects").json()["projects"][0]["id"] == project_id
        assert client.get(f"/api/v1/projects/{project_id}/state/hypotheses").json()["data"]
        assert client.get(f"/api/v1/projects/{project_id}/state/nope").status_code == 404
        evidence = client.get(f"/api/v1/projects/{project_id}/evidence?graph=true").json()
        assert evidence["graph"]["nodes"] and "hotspots" in evidence["queries"]
        metrics = client.get(f"/api/v1/projects/{project_id}/metrics").json()
        assert {"lite", "strong", "coding"} <= set(metrics["usage"]["by_tier"])
        assert "<html>" in client.get(f"/api/v1/projects/{project_id}/manuscript?format=html").text
        assert client.get(f"/api/v1/projects/{project_id}/activity?limit=5").json()["events"]
        assert client.post(f"/api/v1/projects/{project_id}/run").status_code == 409
        assert len(client.get("/api/v1/routing").json()["routes"]) > 20
        assert client.get("/dashboard").status_code == 200
        assert client.get("/api/v1/projects/missing").status_code == 404


def test_api_ingests_manual_results(mock_settings):
    settings = mock_settings(experiment_executor="manual")
    with TestClient(create_app(settings)) as client:
        project_id = client.post("/api/v1/projects", json={"topic": "manual experiments via api"}).json()["project_id"]
        body = wait_until_idle(client, project_id)
        assert body["project"]["status"] == "awaiting_experiments"
        run = next(r for r in client.get(f"/api/v1/projects/{project_id}/state/experiments").json()["data"]["runs"] if r["status"] == "awaiting_execution")
        records = [{"arm": arm, "seed": s, "metrics": {"accuracy": 0.7}} for arm in ("proposed", "baseline", "strong_baseline") for s in run["seeds"]]
        ingested = client.post(f"/api/v1/projects/{project_id}/runs/{run['id']}/results", json={"records": records})
        assert ingested.status_code == 200 and ingested.json()["records"] == len(records)
        again = client.post(f"/api/v1/projects/{project_id}/runs/{run['id']}/results", json={"records": records})
        assert again.status_code == 400


def test_api_auth_and_rate_limit(mock_settings):
    settings = mock_settings(api_auth_token="secret", api_rate_limit_per_minute=2)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/routing").status_code == 401
        headers = {"X-API-Key": "secret"}
        assert client.get("/api/v1/routing", headers=headers).status_code == 200
        assert client.get("/api/v1/routing", headers=headers).status_code == 200
        assert client.get("/api/v1/routing", headers=headers).status_code == 429
        assert client.get("/dashboard").status_code == 200


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "LLM_PROVIDER": "mock",
        "LLM_MODEL": "mock",
        "LITERATURE_PROVIDERS": "mock",
        "EXPERIMENT_EXECUTOR": "simulated",
        "WORKSPACE_DIR": str(tmp_path / "ws"),
    }.items():
        monkeypatch.setenv(key, value)
    runner = CliRunner()

    routes = json.loads(runner.invoke(cli_app, ["routing", "--json"]).output)
    assert {r["tier"] for r in routes} == {"lite", "strong", "coding", "code"}

    created = runner.invoke(cli_app, ["new", "graph neural networks for chemistry", "--run", "--quiet", "--constraint", "one GPU"])
    assert created.exit_code == 0, created.output
    assert "completed" in created.output and "strong" in created.output

    status = json.loads(runner.invoke(cli_app, ["status", "latest", "--json"]).output)
    assert status["project"]["constraints"] == ["one GPU"]
    hypotheses = json.loads(runner.invoke(cli_app, ["show", "latest", "hypotheses", "--json"]).output)
    assert hypotheses and "evidence_state" in hypotheses[0]
    assert "Unknown section" in runner.invoke(cli_app, ["show", "latest", "bogus"]).output
    assert "uncertainty" in runner.invoke(cli_app, ["evidence", "latest"]).output
    assert "review.simulation" in runner.invoke(cli_app, ["metrics", "latest"]).output
    assert runner.invoke(cli_app, ["experiments", "list", "latest"]).exit_code == 0
    out = tmp_path / "paper.html"
    exported = runner.invoke(cli_app, ["export", "latest", "--to", "html", "--output", str(out)])
    assert exported.exit_code == 0 and out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "graph-neural" in runner.invoke(cli_app, ["projects"]).output
