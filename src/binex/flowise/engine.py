"""Walk an Agentflow graph and stream events for the Flowise server."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from binex.flowise.context import ExecContext
from binex.flowise.errors import FlowiseExecError, NodeExecutionError, UnsupportedNodeError
from binex.flowise.graph import FlowEdge, FlowGraph, FlowNode, load_flow
from binex.flowise.handlers import DECISION_NODES, HANDLERS, NodeResult, output_text
from binex.flowise.llm import LiteLLMCaller
from binex.flowise.mapping import binding_for, describe_flow, human_agent_uri, unsupported_nodes
from binex.flowise.protocol import (
    RunRequest,
    artifact_event,
    check_secret_names,
    conversation_event,
    json_text,
    redact_secrets,
    redact_tree,
)
from binex.flowise.session import SessionStore
from binex.flowise.variables import replace_output_placeholder, resolve_value

Emit = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ExecuteOutcome:
    status: str
    text: str
    execution_data: list[dict[str, Any]]
    state: dict[str, Any]
    error: str | None = None


@dataclass
class _QueueItem:
    node_id: str
    data: Any = None


@dataclass
class _Waiting:
    node_id: str
    expected: set[str] = field(default_factory=set)
    groups: dict[str, list[str]] = field(default_factory=dict)
    received: dict[str, Any] = field(default_factory=dict)

    def ready(self) -> bool:
        if any(source not in self.received for source in self.expected):
            return False
        return all(
            any(source in self.received for source in sources) for sources in self.groups.values()
        )


class _ListSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)


async def execute_flow(
    request: RunRequest | dict[str, Any],
    sink: Any | None = None,
    *,
    store: SessionStore | None = None,
    llm: Any | None = None,
    http: httpx.AsyncClient | None = None,
    js_runner: Any | None = None,
    skip_ssrf: bool = False,
    max_steps: int = 1000,
) -> ExecuteOutcome:
    """Run one flow. ``sink.emit`` receives each NDJSON-ready event."""
    parsed = request if isinstance(request, RunRequest) else RunRequest.model_validate(request)
    check_secret_names(parsed.secrets)
    target = sink or _ListSink()
    owns_http = http is None
    client = http or httpx.AsyncClient()
    try:
        with _apply_secrets(parsed.secrets):
            return await _execute(
                parsed,
                target,
                store or SessionStore(),
                llm or LiteLLMCaller(),
                client,
                js_runner,
                skip_ssrf,
                max_steps,
            )
    finally:
        if owns_http:
            await client.aclose()


async def collect_events(
    request: RunRequest | dict[str, Any],
    **kwargs: Any,
) -> tuple[ExecuteOutcome, list[dict[str, Any]]]:
    sink = _ListSink()
    outcome = await execute_flow(request, sink, **kwargs)
    return outcome, sink.events


async def _execute(
    request: RunRequest,
    sink: Any,
    store: SessionStore,
    llm: Any,
    http: httpx.AsyncClient,
    js_runner: Any,
    skip_ssrf: bool,
    max_steps: int,
) -> ExecuteOutcome:
    graph = load_flow(request.flow)
    ctx = ExecContext(
        request=request,
        graph=graph,
        store=store,
        llm=llm,
        http=http,
        max_steps=max_steps,
        skip_ssrf=skip_ssrf,
        js_runner=js_runner,
    )
    await _emit(
        sink,
        request,
        {
            "type": "run.start",
            "runId": request.run_id,
            "chatflowId": request.chatflow_id,
            "sessionId": request.session_key,
            "workspaceId": request.workspace_id,
            "status": "INPROGRESS",
        },
    )
    error: str | None = None
    try:
        missing = unsupported_nodes(graph)
        if missing:
            raise UnsupportedNodeError(missing)
        _load_inputs(ctx)
        queue = _initial_queue(ctx)
        top_ids = set(graph.top_level_ids())
        await _walk(ctx, sink, top_ids, graph.edges_among(top_ids), queue)
    except FlowiseExecError as exc:
        error = str(exc)
    status = "ERROR" if error else "STOPPED" if ctx.paused else "FINISHED"
    if ctx.paused and not error:
        _save_checkpoint(ctx)
    elif not error:
        if ctx.persist_state:
            ctx.store.save_runtime(
                request.chatflow_id,
                request.session_key,
                {"state": ctx.state, "form": ctx.form, "chatHistory": ctx.chat_history},
            )
        ctx.store.clear_checkpoint(request.chatflow_id, request.session_key)
    human = _human_pause_body(ctx) if ctx.paused else None
    await _emit(
        sink,
        request,
        {
            "type": "run.finish",
            "runId": request.run_id,
            "chatflowId": request.chatflow_id,
            "sessionId": request.session_key,
            "workspaceId": request.workspace_id,
            "status": status,
            "text": ctx.last_text,
            "state": redact_tree(ctx.state),
            "executionData": ctx.executed,
            "humanInput": human,
            "error": {"message": error} if error else None,
        },
    )
    return ExecuteOutcome(
        status=status,
        text=ctx.last_text,
        execution_data=ctx.executed,
        state=ctx.state,
        error=error,
    )


def _load_inputs(ctx: ExecContext) -> None:
    inputs = ctx.request.inputs
    question = inputs.get("question", inputs.get("input", ""))
    ctx.question = question if isinstance(question, str) else json_text(question)
    form = inputs.get("form") or {}
    webhook = inputs.get("webhook") or {}
    if question and isinstance(form, dict) and form:
        raise FlowiseExecError("Question and form cannot be provided at the same time")
    ctx.form = dict(form) if isinstance(form, dict) else {}
    ctx.webhook = dict(webhook) if isinstance(webhook, dict) else {}
    uploads = inputs.get("uploadsContent") or inputs.get("fileAttachment") or ""
    ctx.uploads = uploads if isinstance(uploads, str) else json_text(uploads)
    history = inputs.get("history") or []
    if isinstance(history, list):
        ctx.history = [item for item in history if isinstance(item, dict)]
    resume = ctx.request.resume or {}
    human = resume.get("humanInput") or inputs.get("humanInput")
    if isinstance(human, dict):
        _restore_checkpoint(ctx, human)
        return
    _check_trigger(ctx)
    start = _find_start(ctx.graph)
    ctx.persist_state = bool(start and start.inputs.get("startPersistState"))
    if ctx.persist_state and start is not None:
        runtime = ctx.store.get_runtime(ctx.request.chatflow_id, ctx.request.session_key)
        if isinstance(runtime.get("state"), dict):
            ctx.state = dict(runtime["state"])
        if isinstance(runtime.get("form"), dict) and not ctx.form:
            ctx.form = dict(runtime["form"])
    ctx.chat_history = [_history_message(item) for item in ctx.history]


def _restore_checkpoint(ctx: ExecContext, human: dict[str, Any]) -> None:
    checkpoint = ctx.store.get_checkpoint(ctx.request.chatflow_id, ctx.request.session_key)
    if checkpoint is None:
        raise FlowiseExecError(
            f"No paused run is stored for session '{ctx.request.session_key}'. "
            "Human input can only resume a run this sandbox stopped."
        )
    ctx.state = dict(checkpoint.get("state") or {})
    ctx.chat_history = list(checkpoint.get("chatHistory") or [])
    ctx.executed = list(checkpoint.get("executed") or [])
    ctx.loop_counts = {
        str(key): int(value) for key, value in dict(checkpoint.get("loopCounts") or {}).items()
    }
    ctx.question = str(checkpoint.get("question") or ctx.question)
    if not ctx.form:
        ctx.form = dict(checkpoint.get("form") or {})
    if not ctx.webhook:
        ctx.webhook = dict(checkpoint.get("webhook") or {})
    ctx.uploads = str(checkpoint.get("uploads") or ctx.uploads)
    ctx.persist_state = bool(checkpoint.get("persistState"))
    stopped = str(human.get("startNodeId") or checkpoint.get("stoppedNodeId") or "")
    if not stopped:
        raise FlowiseExecError("Resume payload is missing startNodeId")
    if stopped not in ctx.graph.nodes:
        raise FlowiseExecError(f"Cannot resume: node '{stopped}' is not in this flow")
    ctx.human_input = {**human, "startNodeId": stopped}
    ctx.pause_node_id = stopped
    ctx.executed = [
        item
        for item in ctx.executed
        if not (item.get("nodeId") == stopped and item.get("status") == "STOPPED")
    ]
    iteration = checkpoint.get("iterationResume")
    ctx.iteration_resume = dict(iteration) if isinstance(iteration, dict) else None


def _check_trigger(ctx: ExecContext) -> None:
    start = _find_start(ctx.graph)
    if start is None:
        raise FlowiseExecError("Start node not found")
    input_type = str(start.inputs.get("startInputType") or "chatInput")
    chat_type = str(ctx.request.inputs.get("chatType") or "INTERNAL")
    if input_type == "webhookTrigger" and chat_type != "WEBHOOK":
        method = str(start.inputs.get("webhookMethod") or "POST").upper()
        raise FlowiseExecError(
            "This flow is configured as a Webhook Trigger. "
            f"Call {method} /api/v1/webhook/{ctx.request.chatflow_id} "
            "instead of the prediction API."
        )
    if input_type == "scheduleInput" and chat_type != "SCHEDULED":
        raise FlowiseExecError(
            "This flow is configured as a Scheduled Trigger. It is fired by the scheduler "
            "and cannot be invoked via the prediction API."
        )


def _initial_queue(ctx: ExecContext) -> list[_QueueItem]:
    if ctx.human_input is not None:
        if ctx.iteration_resume:
            return [_QueueItem(str(ctx.iteration_resume["nodeId"]))]
        return [_QueueItem(str(ctx.human_input.get("startNodeId")))]
    top = set(ctx.graph.top_level_ids())
    starts = _start_ids(top, ctx.graph.edges_among(top))
    if len(starts) != 1:
        raise FlowiseExecError(
            "Agentflow must have exactly one start node. "
            f"Found {len(starts)} nodes with no inputs: {', '.join(starts) or '(none)'}."
        )
    return [_QueueItem(starts[0], ctx.question)]


def _find_start(graph: FlowGraph) -> FlowNode | None:
    for node in graph.nodes.values():
        if node.name == "startAgentflow" and not node.parent_node:
            return node
    return None


async def _walk(
    ctx: ExecContext,
    sink: Any,
    node_ids: set[str],
    edges: list[FlowEdge],
    queue: list[_QueueItem],
) -> str:
    waiting: dict[str, _Waiting] = {}
    while queue:
        if ctx.steps >= ctx.max_steps:
            raise FlowiseExecError("Maximum iteration limit reached")
        ctx.steps += 1
        item = queue.pop(0)
        if item.node_id not in node_ids:
            continue
        status = await _execute_node(
            ctx, sink, ctx.graph.node(item.node_id), item.data, queue, waiting, edges, node_ids
        )
        if status == "paused":
            return "paused"
    return "done"


async def _execute_node(
    ctx: ExecContext,
    sink: Any,
    node: FlowNode,
    incoming: Any,
    queue: list[_QueueItem],
    waiting: dict[str, _Waiting],
    edges: list[FlowEdge],
    node_ids: set[str],
) -> str:
    binding = binding_for(node.name)
    await _emit(
        sink,
        ctx.request,
        {
            "type": "node.start",
            "runId": ctx.request.run_id,
            "nodeId": node.id,
            "nodeLabel": node.label,
            "nodeName": node.name,
            "status": "INPROGRESS",
            "agent": _agent_uri(node),
            "pattern": binding.pattern if binding else None,
        },
    )
    inputs = resolve_value(node.inputs, ctx.templates())
    if not isinstance(inputs, dict):
        inputs = {}
    try:
        if node.name == "iterationAgentflow":
            result = await _run_iteration(ctx, sink, node, inputs)
        else:
            handler = HANDLERS.get(node.name)
            if handler is None:
                raise UnsupportedNodeError([node])
            result = await handler(ctx, node, inputs, incoming)
    except NodeExecutionError as exc:
        await _emit_failure(ctx, sink, node, exc)
        raise
    if result.status == "PAUSED":
        ctx.paused = True
        if ctx.pause_node_id is None:
            ctx.pause_node_id = node.id
        _record(ctx, node, result, "STOPPED")
        await _emit_node_done(ctx, sink, node, result, "STOPPED")
        return "paused"
    text = output_text(result.output)
    if result.chat_delta:
        ctx.chat_history.extend(result.chat_delta)
    if result.loop_target:
        count = ctx.loop_counts.get(node.id, 0) + 1
        ctx.loop_counts[node.id] = count
        limit = result.max_loop or 5
        result.output["loopCount"] = count
        if count < limit:
            ctx.human_input = None
            result.state = replace_output_placeholder(result.state, text)
            ctx.state = result.state
            _record(ctx, node, result, "FINISHED")
            await _emit_node_done(ctx, sink, node, result, "FINISHED")
            queue.append(_QueueItem(result.loop_target, _node_payload(node, result)))
            if text:
                ctx.last_text = text
            return "done"
        fallback = result.fallback_message or (
            f"Loop completed after reaching maximum iteration count of {limit}."
        )
        result.output["content"] = fallback
        text = fallback
        await _emit(
            sink,
            ctx.request,
            {"type": "log", "level": "info", "nodeId": node.id, "message": fallback},
        )
    status = "STOPPED" if result.status == "STOPPED" else "FINISHED"
    result.state = replace_output_placeholder(result.state, text)
    ctx.state = result.state
    _record(ctx, node, result, status)
    if text:
        ctx.last_text = text
    if node.name == "directReplyAgentflow" and text:
        await _emit(
            sink,
            ctx.request,
            {"type": "token", "runId": ctx.request.run_id, "nodeId": node.id, "token": text},
        )
    await _emit_node_done(ctx, sink, node, result, status)
    if status == "STOPPED":
        ctx.paused = True
        ctx.pause_node_id = node.id
        ctx.human_mode = str(result.output.get("humanInputMode") or "approve")
        await _emit(sink, ctx.request, _human_event(ctx, node, result))
        return "paused"
    _schedule(ctx, node, result, queue, waiting, edges, node_ids)
    return "done"


async def _run_iteration(
    ctx: ExecContext,
    sink: Any,
    node: FlowNode,
    inputs: dict[str, Any],
) -> NodeResult:
    resume = ctx.iteration_resume
    if isinstance(resume, dict) and resume.get("nodeId") == node.id:
        items = list(resume.get("items") or [])
        index = int(resume.get("index") or 0)
        results = [str(item) for item in resume.get("results") or []]
        sub_resume = resume.get("resumeNodeId")
        ctx.iteration_resume = None
    else:
        items = _parse_array(node, inputs.get("iterationInput"))
        index = 0
        results = []
        sub_resume = None
    children = ctx.graph.children_of(node.id)
    child_ids = {child.id for child in children}
    child_edges = ctx.graph.edges_among(child_ids)
    if not children:
        return NodeResult(
            input={"iterationInput": items},
            output={"content": "", "iterationResults": []},
            state=dict(ctx.state),
        )
    while index < len(items):
        ctx.iteration = {
            "index": index,
            "value": items[index],
            "isFirst": index == 0,
            "isLast": index == len(items) - 1,
        }
        if sub_resume:
            queue = [_QueueItem(str(sub_resume))]
            sub_resume = None
        else:
            queue = [_QueueItem(start_id) for start_id in _start_ids(child_ids, child_edges)]
        status = await _walk(ctx, sink, child_ids, child_edges, queue)
        if status == "paused":
            ctx.iteration_resume = {
                "nodeId": node.id,
                "items": items,
                "index": index,
                "results": results,
                "resumeNodeId": ctx.pause_node_id,
            }
            ctx.iteration = None
            return NodeResult(
                input={"iterationInput": items},
                output={"content": "\n".join(results), "iterationResults": list(results)},
                state=dict(ctx.state),
                status="PAUSED",
            )
        results.append(_latest_content(ctx, child_ids))
        index += 1
    ctx.iteration = None
    content = "\n".join(results)
    return NodeResult(
        input={"iterationInput": items},
        output={"content": content, "iterationResults": results},
        state=dict(ctx.state),
    )


def _parse_array(node: FlowNode, value: Any) -> list[Any]:
    if isinstance(value, str):
        if not value.strip():
            raise NodeExecutionError(node, "Iteration input array is empty")
        try:
            value = json_loads(value)
        except ValueError as exc:
            raise NodeExecutionError(node, "Iteration input is not a JSON array") from exc
    if not isinstance(value, list):
        raise NodeExecutionError(node, "Iteration input must be an array")
    return value


def json_loads(value: str) -> Any:
    import json

    return json.loads(value)


def _schedule(
    ctx: ExecContext,
    node: FlowNode,
    result: NodeResult,
    queue: list[_QueueItem],
    waiting: dict[str, _Waiting],
    edges: list[FlowEdge],
    node_ids: set[str],
) -> None:
    ignored = set(_ignored_targets(node, result, edges))
    ordered: list[str] = []
    for edge in edges:
        if edge.source != node.id or edge.target not in node_ids or edge.target in ignored:
            continue
        if edge.target not in ordered:
            ordered.append(edge.target)
    payload = _node_payload(node, result)
    for child_id in ordered:
        slot = waiting.get(child_id)
        if slot is None:
            slot = _make_waiting(child_id, edges, node_ids, ctx.graph)
            waiting[child_id] = slot
        slot.received[node.id] = payload
        if slot.ready():
            waiting.pop(child_id, None)
            queue.append(_QueueItem(child_id, _combine(slot.received)))


def _make_waiting(
    node_id: str,
    edges: list[FlowEdge],
    node_ids: set[str],
    graph: FlowGraph,
) -> _Waiting:
    slot = _Waiting(node_id=node_id)
    for edge in edges:
        if edge.target != node_id or edge.source not in node_ids:
            continue
        parent = _condition_parent(edge.source, edges, node_ids, graph)
        if parent:
            slot.groups.setdefault(parent, [])
            if edge.source not in slot.groups[parent]:
                slot.groups[parent].append(edge.source)
        else:
            slot.expected.add(edge.source)
    return slot


def _condition_parent(
    node_id: str,
    edges: list[FlowEdge],
    node_ids: set[str],
    graph: FlowGraph,
) -> str | None:
    current = node_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        node = graph.nodes.get(current)
        if node and node.name in DECISION_NODES:
            return current
        parent = next(
            (edge.source for edge in edges if edge.target == current and edge.source in node_ids),
            None,
        )
        if parent is None:
            return None
        current = parent
    return None


def _ignored_targets(node: FlowNode, result: NodeResult, edges: list[FlowEdge]) -> list[str]:
    if node.name not in DECISION_NODES:
        return []
    conditions = result.output.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        return []
    if not any(isinstance(item, dict) and item.get("isFulfilled") for item in conditions):
        last = conditions[-1]
        if isinstance(last, dict):
            last["isFulfilled"] = True
    blocked: set[str] = set()
    allowed: set[str] = set()
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            continue
        suffix = f"-output-{index}"
        bucket = allowed if condition.get("isFulfilled") else blocked
        for edge in edges:
            if edge.source != node.id:
                continue
            handle = edge.source_handle
            if handle == f"{node.id}{suffix}" or handle.endswith(suffix):
                bucket.add(edge.target)
    return [target for target in blocked if target not in allowed]


def _start_ids(node_ids: set[str], edges: list[FlowEdge]) -> list[str]:
    incoming = {edge.target for edge in edges if edge.source in node_ids}
    return [node_id for node_id in node_ids if node_id not in incoming]


def _combine(received: dict[str, Any]) -> Any:
    items = {key: value for key, value in received.items() if value is not None}
    if not items:
        return None
    if len(items) == 1:
        return next(iter(items.values()))
    combined: dict[str, Any] = {"json": {}}
    texts: list[str] = []
    for source_id, data in sorted(items.items()):
        if isinstance(data, dict) and data.get("json") is not None:
            combined["json"][source_id] = data["json"]
            if data.get("text"):
                texts.append(str(data["text"]))
            continue
        if isinstance(data, dict) and data.get("text"):
            texts.append(str(data["text"]))
            combined["json"][source_id] = data
            continue
        combined["json"][source_id] = data
        output = data.get("output") if isinstance(data, dict) else None
        if (
            isinstance(output, dict)
            and isinstance(output.get("content"), str)
            and output["content"]
        ):
            texts.append(output["content"])
    if texts:
        combined["text"] = "\n".join(texts)
    if not combined["json"] and combined.get("text"):
        combined["json"] = {"text": combined["text"]}
    return combined


def _node_payload(node: FlowNode, result: NodeResult) -> dict[str, Any]:
    return {
        "id": node.id,
        "name": node.name,
        "input": result.input,
        "output": result.output,
        "state": result.state,
    }


def _latest_content(ctx: ExecContext, node_ids: set[str]) -> str:
    for item in reversed(ctx.executed):
        if item.get("nodeId") not in node_ids:
            continue
        data = item.get("data")
        output = data.get("output") if isinstance(data, dict) else None
        if isinstance(output, dict) and isinstance(output.get("content"), str):
            return output["content"]
    return ""


def _record(ctx: ExecContext, node: FlowNode, result: NodeResult, status: str) -> None:
    payload = _node_payload(node, result)
    payload["input"] = redact_tree(payload["input"])
    payload["output"] = redact_tree(payload["output"])
    ctx.executed.append(
        {
            "nodeId": node.id,
            "nodeLabel": node.label,
            "status": status,
            "data": payload,
        }
    )


async def _emit_node_done(
    ctx: ExecContext,
    sink: Any,
    node: FlowNode,
    result: NodeResult,
    status: str,
) -> None:
    safe_input = redact_tree(result.input)
    safe_output = redact_tree(result.output)
    await _emit(
        sink,
        ctx.request,
        artifact_event(
            ctx.request,
            node_id=node.id,
            node_label=node.label,
            kind="input",
            file_name="input.json",
            body=json_text(safe_input),
        ),
    )
    await _emit(
        sink,
        ctx.request,
        artifact_event(
            ctx.request,
            node_id=node.id,
            node_label=node.label,
            kind="output",
            file_name="output.json",
            body=json_text(safe_output),
        ),
    )
    await _emit_protocol_artifacts(ctx, sink, node, safe_output)
    for turn in result.conversations:
        await _emit(sink, ctx.request, conversation_event(ctx.request, turn))
    await _emit(
        sink,
        ctx.request,
        {
            "type": "node.finish",
            "runId": ctx.request.run_id,
            "nodeId": node.id,
            "nodeLabel": node.label,
            "nodeName": node.name,
            "status": status,
            "input": safe_input,
            "output": safe_output,
            "state": result.state,
            "agent": _agent_uri(node),
        },
    )


async def _emit_protocol_artifacts(
    ctx: ExecContext,
    sink: Any,
    node: FlowNode,
    output: Any,
) -> None:
    if not isinstance(output, dict):
        return
    a2a = output.get("a2a") if isinstance(output.get("a2a"), dict) else None
    cao = output.get("cao") if isinstance(output.get("cao"), dict) else None
    pairs = []
    if a2a and a2a.get("request") is not None:
        pairs.append(("a2a-request", "a2a-request.json", a2a.get("request")))
    if a2a and a2a.get("response") is not None:
        pairs.append(("a2a-response", "a2a-response.json", a2a.get("response")))
    if cao and cao.get("request") is not None:
        pairs.append(("cao-request", "cao-request.json", cao.get("request")))
    if cao and cao.get("response") is not None:
        pairs.append(("cao-response", "cao-response.json", cao.get("response")))
    for kind, file_name, body in pairs:
        await _emit(
            sink,
            ctx.request,
            artifact_event(
                ctx.request,
                node_id=node.id,
                node_label=node.label,
                kind=kind,
                file_name=file_name,
                body=json_text(body),
            ),
        )
    files = output.get("files")
    if isinstance(files, list):
        for index, item in enumerate(files[:20]):
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                continue
            name = str(item.get("name") or f"file-{index}.txt")
            await _emit(
                sink,
                ctx.request,
                artifact_event(
                    ctx.request,
                    node_id=node.id,
                    node_label=node.label,
                    kind="file",
                    file_name=name,
                    body=item["content"],
                    mime_type=str(item.get("mimeType") or "text/plain"),
                ),
            )


async def _emit_failure(
    ctx: ExecContext, sink: Any, node: FlowNode, exc: NodeExecutionError
) -> None:
    cause = exc.__cause__
    turn = getattr(cause, "turn", None)
    if isinstance(turn, dict):
        await _emit(sink, ctx.request, conversation_event(ctx.request, turn))
    ctx.executed.append(
        {
            "nodeId": node.id,
            "nodeLabel": node.label,
            "status": "ERROR",
            "data": {"id": node.id, "name": node.name, "error": {"message": str(exc)}},
        }
    )
    await _emit(
        sink,
        ctx.request,
        {
            "type": "node.error",
            "runId": ctx.request.run_id,
            "nodeId": node.id,
            "nodeLabel": node.label,
            "nodeName": node.name,
            "status": "ERROR",
            "error": {"message": str(exc)},
        },
    )


def _human_event(ctx: ExecContext, node: FlowNode, result: NodeResult) -> dict[str, Any]:
    mode = str(result.output.get("humanInputMode") or "approve")
    return {
        "type": "human_input",
        "runId": ctx.request.run_id,
        "status": "STOPPED",
        "nodeId": node.id,
        "nodeLabel": node.label,
        "humanInput": _human_pause_body(
            ctx, node=node, description=output_text(result.output), mode=mode
        ),
    }


def _human_pause_body(
    ctx: ExecContext,
    *,
    node: FlowNode | None = None,
    description: str | None = None,
    mode: str | None = None,
) -> dict[str, Any] | None:
    stopped = node.id if node is not None else ctx.pause_node_id
    if not stopped:
        return None
    current = node or ctx.graph.nodes.get(stopped)
    resolved_mode = mode or ctx.human_mode or "approve"
    text = description if description is not None else ctx.last_text
    return {
        "mode": resolved_mode,
        "description": text,
        "startNodeId": stopped,
        "nodeLabel": current.label if current else stopped,
        "enableFeedback": resolved_mode in {"approve", "input"},
        "elements": _human_elements(resolved_mode),
    }


def _human_elements(mode: str) -> list[dict[str, Any]]:
    if mode == "input":
        return [{"type": "proceed", "label": "Submit", "input": True}]
    if mode == "output":
        return [{"type": "proceed", "label": "Acknowledge"}]
    return [{"type": "proceed", "label": "Proceed"}, {"type": "reject", "label": "Reject"}]


def _agent_uri(node: FlowNode) -> str:
    if node.name == "humanInputAgentflow":
        return human_agent_uri(str(node.inputs.get("humanInputMode") or "approve"))
    binding = binding_for(node.name)
    return binding.agent if binding else ""


def _save_checkpoint(ctx: ExecContext) -> None:
    ctx.store.save_checkpoint(
        ctx.request.chatflow_id,
        ctx.request.session_key,
        {
            "version": 1,
            "runId": ctx.request.run_id,
            "chatflowId": ctx.request.chatflow_id,
            "sessionId": ctx.request.session_key,
            "status": "STOPPED",
            "state": ctx.state,
            "chatHistory": ctx.chat_history,
            "executed": ctx.executed,
            "loopCounts": ctx.loop_counts,
            "question": ctx.question,
            "form": ctx.form,
            "webhook": ctx.webhook,
            "uploads": ctx.uploads,
            "persistState": ctx.persist_state,
            "stoppedNodeId": ctx.pause_node_id,
            "iterationResume": ctx.iteration_resume,
        },
    )


def _history_message(item: dict[str, Any]) -> dict[str, Any]:
    role = str(item.get("role") or "user")
    if role == "apiMessage":
        role = "assistant"
    elif role == "userMessage":
        role = "user"
    content = item.get("content")
    return {
        "role": role,
        "content": content if isinstance(content, str) else json_text(content),
    }


async def _emit(sink: Any, request: RunRequest, event: dict[str, Any]) -> None:
    await sink.emit(redact_secrets(event, request.secrets))


@contextmanager
def _apply_secrets(secrets: dict[str, str]) -> Any:
    previous: dict[str, str | None] = {}
    try:
        for key, value in secrets.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def inspect_flow(payload: Any) -> list[dict[str, object]]:
    """Describe adapter bindings without executing the flow."""
    return describe_flow(load_flow(payload))
