"""Map Flowise Agentflow node types onto Binex adapters and patterns.

The walker in ``engine`` is what actually runs a flow. Agentflow loops,
multi-branch conditions, and human pauses are not a static DAG, so this table
is the adapter binding each node uses rather than a second compiler.
"""

from __future__ import annotations

from dataclasses import dataclass

from binex.flowise.graph import FlowGraph, FlowNode


@dataclass(frozen=True)
class NodeBinding:
    """How one Agentflow node type is executed."""

    flowise_type: str
    label: str
    agent: str
    pattern: str | None
    supported: bool
    hint: str


# Agent URIs mirror binex.adapters prefixes. A2A and CAO use the Flowise fork's
# wire protocols (JSON-RPC message/send and cao-server run-step). Those are not
# the YAML `a2a://` `/execute` adapter or the YAML `cao://` handoff sequence.
_BINDINGS: dict[str, NodeBinding] = {
    "startAgentflow": NodeBinding(
        "startAgentflow",
        "Start",
        "local://start",
        None,
        True,
        "Emits the chat, form, webhook, or schedule input.",
    ),
    "llmAgentflow": NodeBinding(
        "llmAgentflow",
        "LLM",
        "llm://",
        None,
        True,
        "Chat completion through the same LiteLLM path as LLMAdapter.",
    ),
    "agentAgentflow": NodeBinding(
        "agentAgentflow",
        "Agent",
        "llm://",
        None,
        True,
        "LLMAdapter completion. Flowise tool catalogs are rejected with a pointer "
        "to HTTP, CustomFunction, or binexTools.",
    ),
    "conditionAgentflow": NodeBinding(
        "conditionAgentflow",
        "Condition",
        "local://condition",
        "when",
        True,
        "First matching branch, then the ELSE output. Same idea as a node `when`.",
    ),
    "conditionAgentAgentflow": NodeBinding(
        "conditionAgentAgentflow",
        "Condition Agent",
        "llm://",
        "when",
        True,
        "LLM picks a scenario, then the matching output handle runs.",
    ),
    "loopAgentflow": NodeBinding(
        "loopAgentflow",
        "Loop",
        "local://loop",
        "back_edge",
        True,
        "Re-enters an upstream node until maxLoopCount, then follows exit edges.",
    ),
    "iterationAgentflow": NodeBinding(
        "iterationAgentflow",
        "Iteration",
        "local://iteration",
        "foreach",
        True,
        "Runs the child subgraph once per array item, like foreach.",
    ),
    "humanInputAgentflow": NodeBinding(
        "humanInputAgentflow",
        "Human Input",
        "human://approve",
        None,
        True,
        "approve -> human://approve, input -> human://input, output -> human://output. "
        "The sandbox pauses instead of prompting on stdin.",
    ),
    "httpAgentflow": NodeBinding(
        "httpAgentflow",
        "HTTP",
        "local://http",
        None,
        True,
        "httpx request. Private addresses are blocked unless BINEX_ALLOW_PRIVATE_URLS=1.",
    ),
    "customFunctionAgentflow": NodeBinding(
        "customFunctionAgentflow",
        "Custom Function",
        "local://custom-function",
        None,
        True,
        "JavaScript body run with Node's vm. The sandbox image installs Node.",
    ),
    "directReplyAgentflow": NodeBinding(
        "directReplyAgentflow",
        "Direct Reply",
        "local://echo",
        None,
        True,
        "Returns the message as the node output.",
    ),
    "a2aAgentflow": NodeBinding(
        "a2aAgentflow",
        "A2A",
        "flowise-a2a://message/send",
        None,
        True,
        "JSON-RPC message/send. contextId and taskId persist in the sandbox.",
    ),
    "caoAgentflow": NodeBinding(
        "caoAgentflow",
        "CAO",
        "flowise-cao://run-step",
        None,
        True,
        "cao-server handoff, assign, or send_message. Session and terminal persist.",
    ),
    "stickyNoteAgentflow": NodeBinding(
        "stickyNoteAgentflow",
        "Sticky Note",
        "local://noop",
        None,
        True,
        "Canvas note. Skipped at runtime.",
    ),
    "toolAgentflow": NodeBinding(
        "toolAgentflow",
        "Tool",
        "",
        None,
        False,
        "The Flowise tool catalog is not installed in the sandbox. "
        "Call the service with an HTTP node or a Custom Function.",
    ),
    "retrieverAgentflow": NodeBinding(
        "retrieverAgentflow",
        "Retriever",
        "",
        None,
        False,
        "Document stores stay on the Flowise server. Pass retrieved text in as "
        "an input, or call a retrieval endpoint with an HTTP node.",
    ),
    "executeFlowAgentflow": NodeBinding(
        "executeFlowAgentflow",
        "Execute Flow",
        "",
        None,
        False,
        "Nested flows are not executed in the sandbox. Inline the child flow, "
        "or have the Flowise server expand it before dispatch.",
    ),
}

SUPPORTED = frozenset(name for name, binding in _BINDINGS.items() if binding.supported)
UNSUPPORTED_HINTS = {
    name: binding.hint for name, binding in _BINDINGS.items() if not binding.supported
}


def binding_for(node_name: str) -> NodeBinding | None:
    return _BINDINGS.get(node_name)


def human_agent_uri(mode: str | None) -> str:
    """Binex human adapter URI for a Human Input mode."""
    if mode == "input":
        return "human://input"
    if mode == "output":
        return "human://output"
    return "human://approve"


def describe_flow(graph: FlowGraph) -> list[dict[str, object]]:
    """Return one row per node describing the Binex binding."""
    rows: list[dict[str, object]] = []
    for node in graph.nodes.values():
        binding = binding_for(node.name)
        if binding is None:
            rows.append(
                {
                    "nodeId": node.id,
                    "label": node.label,
                    "flowiseType": node.name,
                    "agent": "",
                    "pattern": None,
                    "supported": False,
                    "hint": (
                        f"Unknown Agentflow node type '{node.name}'. "
                        "The sandbox executor does not run it."
                    ),
                }
            )
            continue
        agent = binding.agent
        if node.name == "humanInputAgentflow":
            agent = human_agent_uri(str(node.inputs.get("humanInputMode") or "approve"))
        rows.append(
            {
                "nodeId": node.id,
                "label": node.label,
                "flowiseType": node.name,
                "agent": agent,
                "pattern": binding.pattern,
                "supported": binding.supported,
                "hint": binding.hint,
            }
        )
    return rows


def unsupported_nodes(graph: FlowGraph) -> list[FlowNode]:
    """Nodes that cannot run. Sticky notes are supported as no-ops."""
    found: list[FlowNode] = []
    for node in graph.nodes.values():
        binding = binding_for(node.name)
        if binding is None or not binding.supported:
            found.append(node)
    return found
