"""Load a Flowise Agentflow V2 flowData document."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class FlowLoadError(ValueError):
    """Raised when a payload is not an Agentflow graph."""


@dataclass
class FlowNode:
    """One React Flow node from an Agentflow."""

    id: str
    name: str
    label: str
    inputs: dict[str, Any]
    parent_node: str | None = None
    credential: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowEdge:
    """One React Flow edge. source_handle selects a condition output."""

    source: str
    target: str
    source_handle: str
    target_handle: str


@dataclass
class FlowGraph:
    """Nodes and edges for one flow, including iteration children."""

    nodes: dict[str, FlowNode]
    edges: list[FlowEdge]

    def node(self, node_id: str) -> FlowNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise FlowLoadError(f"Flow references missing node '{node_id}'") from exc

    def children_of(self, node_id: str) -> list[FlowNode]:
        return [
            node
            for node in self.nodes.values()
            if node.parent_node == node_id and node.name != "stickyNoteAgentflow"
        ]

    def top_level_ids(self) -> list[str]:
        return [
            node.id
            for node in self.nodes.values()
            if not node.parent_node and node.name != "stickyNoteAgentflow"
        ]

    def edges_among(self, node_ids: set[str]) -> list[FlowEdge]:
        return [edge for edge in self.edges if edge.source in node_ids and edge.target in node_ids]


def _as_object(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FlowLoadError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FlowLoadError(f"{label} must be a JSON object")
    return value


def extract_flow_document(payload: Any) -> dict[str, Any]:
    """Accept a React Flow document or a chatflow row with stringified flowData."""
    document = _as_object(payload, "flow")
    if "flowData" in document:
        return _as_object(document["flowData"], "flowData")
    if "nodes" in document:
        return document
    raise FlowLoadError("flow JSON must contain 'nodes' or 'flowData'")


def load_flow(payload: Any) -> FlowGraph:
    """Parse Agentflow nodes and edges."""
    document = extract_flow_document(payload)
    raw_nodes = document.get("nodes") or []
    raw_edges = document.get("edges") or []
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise FlowLoadError("flow nodes and edges must be arrays")

    nodes: dict[str, FlowNode] = {}
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise FlowLoadError("each flow node must be an object")
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        node_id = str(raw.get("id") or data.get("id") or "")
        if not node_id:
            raise FlowLoadError("a flow node is missing an id")
        name = str(data.get("name") or raw.get("type") or "")
        if not name:
            raise FlowLoadError(f"node '{node_id}' is missing data.name")
        inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
        parent = raw.get("parentNode") or data.get("parentNode")
        credential = data.get("credential")
        nodes[node_id] = FlowNode(
            id=node_id,
            name=name,
            label=str(data.get("label") or node_id),
            inputs=dict(inputs),
            parent_node=str(parent) if parent else None,
            credential=str(credential) if credential else None,
            raw=raw,
        )

    edges: list[FlowEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, dict):
            raise FlowLoadError("each flow edge must be an object")
        source = str(raw.get("source") or "")
        target = str(raw.get("target") or "")
        if not source or not target:
            raise FlowLoadError("an edge is missing source or target")
        if source not in nodes or target not in nodes:
            raise FlowLoadError(f"edge {source} -> {target} references a missing node")
        edges.append(
            FlowEdge(
                source=source,
                target=target,
                source_handle=str(raw.get("sourceHandle") or f"{source}-output-0"),
                target_handle=str(raw.get("targetHandle") or ""),
            )
        )
    return FlowGraph(nodes=nodes, edges=edges)
