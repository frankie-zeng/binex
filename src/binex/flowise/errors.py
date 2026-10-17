"""Errors the sandbox executor returns to the Flowise server."""

from __future__ import annotations

from binex.flowise.graph import FlowNode
from binex.flowise.mapping import binding_for


class FlowiseExecError(RuntimeError):
    """A run cannot continue. The message is safe to show in the execution UI."""


class UnsupportedNodeError(FlowiseExecError):
    """One or more node types have no sandbox implementation."""

    def __init__(self, nodes: list[FlowNode]) -> None:
        self.nodes = nodes
        lines: list[str] = []
        for node in nodes:
            binding = binding_for(node.name)
            if binding is None:
                hint = (
                    f"Unknown Agentflow node type '{node.name}'. "
                    "The sandbox executor does not run it."
                )
            else:
                hint = binding.hint
            lines.append(f"{node.label} ({node.id}, {node.name}): {hint}")
        super().__init__(
            "This flow uses Agentflow nodes the sandbox executor does not run:\n"
            + "\n".join(lines)
        )


class NodeExecutionError(FlowiseExecError):
    """A supported node failed while it was running."""

    def __init__(self, node: FlowNode, message: str) -> None:
        self.node = node
        super().__init__(f"{node.label} ({node.id}): {message}")
