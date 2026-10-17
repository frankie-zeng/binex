"""CLI, HTTP entrypoint, and the on-disk session file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from binex.flowise.cli import flowise_group
from binex.flowise.protocol import RunRequest, check_secret_names, redact_secrets, redact_tree
from binex.flowise.server import create_app
from binex.flowise.session import SessionStore, default_state_dir
from tests.unit.flowise.support import EXAMPLES, load_example, run_body


def test_protocol_redaction() -> None:
    with pytest.raises(ValueError):
        check_secret_names({"PATH": "x"})
    with pytest.raises(ValueError):
        check_secret_names({"not a name": "x"})
    event = {"text": "sk-live-secret", "nested": {"apiKey": "sk-live-secret"}}
    hidden = redact_secrets(event, {"OPENAI_API_KEY": "sk-live-secret"})
    assert hidden["text"] == "[redacted]"
    assert redact_secrets({"text": "short"}, {"TOKEN": "short"})["text"] == "short"
    assert redact_tree({"Authorization": "Bearer abc", "ok": 1})["Authorization"] == "[redacted]"
    request = RunRequest.model_validate(run_body(load_example("direct-reply.json"), vars={"n": 1}))
    assert request.variables == {"n": 1}
    assert request.session_key == "sess-1"


def test_session_store_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BINEX_FLOWISE_STATE_DIR", str(tmp_path / "state"))
    assert default_state_dir() == tmp_path / "state"
    store = SessionStore(tmp_path)
    store.save_a2a("flow/1", "sess", "node", {"contextId": "ctx", "taskId": "task"})
    assert store.get_a2a("flow/1", "sess", "node")["contextId"] == "ctx"
    store.save_runtime("flow/1", "sess", {"state": {"n": 1}})
    assert store.get_runtime("flow/1", "sess")["state"]["n"] == 1
    store.save_checkpoint("flow/1", "sess", {"status": "STOPPED"})
    assert store.get_checkpoint("flow/1", "sess")["status"] == "STOPPED"
    store.clear_checkpoint("flow/1", "sess")
    assert store.get_checkpoint("flow/1", "sess") is None
    path = tmp_path / "flow_1.json"
    path.write_text("{", encoding="utf-8")
    assert store.load("flow/1")["a2a"] == {}
    path.write_text("[]", encoding="utf-8")
    assert store.get_cao("flow/1", "missing") == {}


def test_cli_run_inspect_and_stdio(tmp_path) -> None:
    runner = CliRunner()
    flow = EXAMPLES / "direct-reply.json"
    result = runner.invoke(
        flowise_group,
        [
            "run",
            "--flow",
            str(flow),
            "--run-id",
            "run-1",
            "--chatflow-id",
            "flow-1",
            "--question",
            "Ada",
            "--state-dir",
            str(tmp_path),
            "--secret",
            "OPENAI_API_KEY=sk-test-value",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert lines[-1]["status"] == "FINISHED"
    assert lines[-1]["text"] == "Hello Ada"

    inspected = runner.invoke(
        flowise_group, ["inspect", "--flow", str(EXAMPLES / "human-approve.json")]
    )
    assert inspected.exit_code == 0
    assert "human://approve" in inspected.output

    tool_run = runner.invoke(
        flowise_group,
        [
            "run",
            "--flow",
            str(EXAMPLES / "tool.json"),
            "--run-id",
            "run-tool",
            "--chatflow-id",
            "flow-tool",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert tool_run.exit_code == 0, tool_run.output
    assert "FINISHED" in tool_run.output
    assert '"text": "4"' in tool_run.output or '"text":"4"' in tool_run.output

    unsupported = runner.invoke(
        flowise_group,
        [
            "run",
            "--flow",
            str(EXAMPLES / "unknown.json"),
            "--run-id",
            "run-x",
            "--chatflow-id",
            "flow-x",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert unsupported.exit_code == 1
    assert "Unknown Agentflow" in unsupported.output

    bad_secret = runner.invoke(
        flowise_group,
        [
            "run",
            "--flow",
            str(flow),
            "--run-id",
            "r",
            "--chatflow-id",
            "c",
            "--secret",
            "NOEQUALS",
        ],
    )
    assert bad_secret.exit_code != 0

    payload = run_body(load_example("direct-reply.json"), inputs={"question": "Zoe"})
    stdio = runner.invoke(
        flowise_group,
        ["stdio", "--state-dir", str(tmp_path)],
        input=json.dumps(payload),
    )
    assert stdio.exit_code == 0, stdio.output
    assert "Hello Zoe" in stdio.output
    broken = runner.invoke(flowise_group, ["stdio"], input="not-json")
    assert broken.exit_code != 0


def test_serve_command_uses_uvicorn(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run(app, host, port):  # type: ignore[no-untyped-def]
        captured["host"] = host
        captured["port"] = port
        captured["title"] = app.title

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = CliRunner().invoke(
        flowise_group,
        ["serve", "--host", "0.0.0.0", "--port", "8799", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert captured == {"host": "0.0.0.0", "port": 8799, "title": "Binex Flowise executor"}


def test_http_server_streams_ndjson(tmp_path, monkeypatch) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    health = client.get("/health")
    assert health.json() == {"status": "ok", "executor": "binex-flowise"}

    missing = client.post("/v1/runs", json={"flow": {}})
    assert missing.status_code == 400
    assert (
        client.post(
            "/v1/runs", content=b"not-json", headers={"content-type": "application/json"}
        ).status_code
        == 400
    )

    response = client.post(
        "/v1/runs", json=run_body(load_example("direct-reply.json"), inputs={"question": "Sam"})
    )
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[0]["type"] == "run.start"
    assert events[-1]["status"] == "FINISHED"
    assert events[-1]["text"] == "Hello Sam"

    failed = client.post("/v1/runs", json=run_body({"nodes": "nope"}))
    failed_events = [json.loads(line) for line in failed.text.splitlines() if line]
    assert failed_events[-1]["status"] == "ERROR"

    monkeypatch.setenv("BINEX_EXECUTOR_TOKEN", "executor-token")
    protected = create_app(Path(tmp_path) / "other")
    guarded = TestClient(protected)
    assert (
        guarded.post("/v1/runs", json=run_body(load_example("direct-reply.json"))).status_code
        == 401
    )
    allowed = guarded.post(
        "/v1/runs",
        json=run_body(load_example("direct-reply.json"), inputs={"question": "Pat"}),
        headers={"Authorization": "Bearer executor-token"},
    )
    assert allowed.status_code == 200
    assert "Hello Pat" in allowed.text
