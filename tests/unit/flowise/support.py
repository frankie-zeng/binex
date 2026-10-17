"""Builders for Agentflow JSON used by the sandbox executor tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "flowise"


def node(
    node_id: str,
    name: str,
    inputs: dict[str, Any] | None = None,
    *,
    parent: str | None = None,
    label: str | None = None,
    credential: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": node_id,
        "name": name,
        "label": label or node_id,
        "inputs": inputs or {},
    }
    if credential:
        data["credential"] = credential
    raw: dict[str, Any] = {"id": node_id, "type": "agentFlow", "data": data}
    if parent:
        raw["parentNode"] = parent
    return raw


def edge(source: str, target: str, output: int = 0) -> dict[str, str]:
    return {
        "source": source,
        "target": target,
        "sourceHandle": f"{source}-output-{output}",
        "targetHandle": f"{target}-input",
    }


def graph(nodes: list[dict[str, Any]], edges: list[dict[str, str]]) -> dict[str, Any]:
    return {"nodes": nodes, "edges": edges}


def run_body(flow: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "runId": "run-1",
        "chatflowId": "flow-1",
        "sessionId": "sess-1",
        "workspaceId": "ws-1",
        "orgId": "org-1",
        "flow": flow,
        "inputs": {"question": "hi"},
    }
    body.update(overrides)
    return body


def load_example(name: str) -> dict[str, Any]:
    loaded = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError(f"{name} is not an object")
    return loaded


def finish_ids(events: list[dict[str, Any]]) -> list[str]:
    return [str(event["nodeId"]) for event in events if event.get("type") == "node.finish"]
