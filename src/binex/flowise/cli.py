"""`binex flowise` commands invoked inside a workflow sandbox."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click

from binex.flowise.engine import execute_flow, inspect_flow
from binex.flowise.server import create_app


@click.group("flowise")
def flowise_group() -> None:
    """Run Flowise Agentflow V2 graphs inside a sandbox."""


@flowise_group.command("run")
@click.option("--flow", "flow_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--run-id", required=True)
@click.option("--chatflow-id", required=True)
@click.option("--session-id", default="")
@click.option("--workspace-id", default="")
@click.option("--question", default="")
@click.option("--secret", "secrets", multiple=True, help="KEY=VALUE for this run only.")
@click.option("--state-dir", default=None, type=click.Path(file_okay=False))
def run_cmd(
    flow_path: str,
    run_id: str,
    chatflow_id: str,
    session_id: str,
    workspace_id: str,
    question: str,
    secrets: tuple[str, ...],
    state_dir: str | None,
) -> None:
    """Execute a flow file and write NDJSON events to stdout."""
    request = _request(
        flow_path,
        run_id=run_id,
        chatflow_id=chatflow_id,
        session_id=session_id,
        workspace_id=workspace_id,
        question=question,
        secrets=secrets,
    )
    _exit_on_error(_run_to_stdout(request, state_dir))


@flowise_group.command("stdio")
@click.option("--state-dir", default=None, type=click.Path(file_okay=False))
def stdio_cmd(state_dir: str | None) -> None:
    """Read one JSON run request from stdin and write NDJSON events to stdout."""
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"stdin is not JSON: {exc}") from exc
    _exit_on_error(_run_to_stdout(request, state_dir))


@flowise_group.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8787, show_default=True, type=int)
@click.option("--state-dir", default=None, type=click.Path(file_okay=False))
def serve_cmd(host: str, port: int, state_dir: str | None) -> None:
    """Serve POST /v1/runs as an NDJSON stream."""
    import uvicorn

    root = Path(state_dir) if state_dir else None
    uvicorn.run(create_app(root), host=host, port=port)


@flowise_group.command("inspect")
@click.option("--flow", "flow_path", required=True, type=click.Path(exists=True, dir_okay=False))
def inspect_cmd(flow_path: str) -> None:
    """Print the Binex adapter binding for each node."""
    document = json.loads(Path(flow_path).read_text(encoding="utf-8"))
    for row in inspect_flow(document):
        status = "ok" if row["supported"] else "UNSUPPORTED"
        pattern = row["pattern"] or ""
        click.echo(f"{status}\t{row['nodeId']}\t{row['flowiseType']}\t{row['agent']}\t{pattern}")


def _request(
    flow_path: str,
    *,
    run_id: str,
    chatflow_id: str,
    session_id: str,
    workspace_id: str,
    question: str,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    flow = json.loads(Path(flow_path).read_text(encoding="utf-8"))
    secret_map: dict[str, str] = {}
    for item in secrets:
        if "=" not in item:
            raise click.ClickException(f"secret '{item}' must be KEY=VALUE")
        key, value = item.split("=", 1)
        secret_map[key] = value
    return {
        "runId": run_id,
        "chatflowId": chatflow_id,
        "sessionId": session_id,
        "workspaceId": workspace_id,
        "flow": flow,
        "inputs": {"question": question},
        "secrets": secret_map,
    }


async def _write_stdout(request: dict[str, Any] | Any, state_dir: str | None) -> str:
    from binex.flowise.session import SessionStore

    class _Stdout:
        async def emit(self, event: dict[str, Any]) -> None:
            click.echo(json.dumps(event, default=str))

    store = SessionStore(Path(state_dir)) if state_dir else None
    outcome = await execute_flow(request, _Stdout(), store=store)
    return outcome.status


def _run_to_stdout(request: dict[str, Any], state_dir: str | None) -> str:
    return asyncio.run(_write_stdout(request, state_dir))


def _exit_on_error(status: str) -> None:
    if status == "ERROR":
        raise SystemExit(1)


def main() -> None:
    flowise_group()
