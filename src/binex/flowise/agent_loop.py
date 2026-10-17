"""Agent node tool loop: LiteLLM tool calls, local tools, and human approval."""

from __future__ import annotations

import json
from typing import Any

from binex.flowise.components import (
    ComponentError,
    function_schema,
    invoke_named_tool,
    local_tool_schema,
    parse_tool_arguments,
    retrieve_documents,
    safe_function_name,
)
from binex.flowise.context import ExecContext
from binex.flowise.errors import NodeExecutionError
from binex.flowise.graph import FlowNode
from binex.flowise.handlers import (
    NodeResult,
    _assemble_messages,
    _prepare_memory,
    apply_state_updates,
    incoming_text,
)
from binex.flowise.llm import parse_json_object, resolve_model, structured_schema

_MAX_ROUNDS = 8
_BUILTIN_KEYS = (
    ("agentToolsBuiltInOpenAI", "openai"),
    ("agentToolsBuiltInGemini", "gemini"),
    ("agentToolsBuiltInAnthropic", "anthropic"),
)


def flowise_tools_configured(inputs: dict[str, Any]) -> bool:
    keys = (
        "agentTools",
        "agentToolsBuiltInOpenAI",
        "agentToolsBuiltInGemini",
        "agentToolsBuiltInAnthropic",
        "agentKnowledgeDocumentStores",
        "agentKnowledgeVSEmbeddings",
    )
    return any(_configured(inputs.get(key)) for key in keys)


async def run_tool_agent(
    ctx: ExecContext,
    node: FlowNode,
    inputs: dict[str, Any],
    incoming: Any,
) -> NodeResult:
    """Call the model, run tool calls, and pause when a tool needs approval."""
    model = str(inputs.get("agentModel") or "")
    model_config = inputs.get("agentModelConfig")
    prompt = inputs.get("agentUserMessage")
    text = prompt if isinstance(prompt, str) and prompt.strip() else ""
    if not text:
        text = incoming_text(incoming) or ctx.question
    memory = bool(inputs.get("agentEnableMemory", True))
    history = list(ctx.chat_history) if memory else []
    history, memory_type = await _prepare_memory(
        ctx,
        node,
        model,
        model_config,
        history,
        str(inputs.get("agentMemoryType") or "allMessages"),
        inputs.get("agentMemoryWindowSize") or 20,
        inputs.get("agentMemoryMaxTokenLimit"),
    )
    messages = _assemble_messages(
        inputs.get("agentMessages") if isinstance(inputs.get("agentMessages"), list) else [],
        text,
        history,
        memory_type,
        20,
    )
    schemas, meta = _tool_catalog(inputs)
    fields = inputs.get("agentStructuredOutput")
    structured = []
    if isinstance(fields, list):
        structured = [item for item in fields if isinstance(item, dict)]
    resume = ctx.agent_resume if isinstance(ctx.agent_resume, dict) else None
    if resume and resume.get("nodeId") == node.id and isinstance(ctx.human_input, dict):
        messages = [dict(item) for item in resume.get("messages") or messages]
        steps_left = int(resume.get("stepsLeft") or _MAX_ROUNDS)
        pending = resume.get("pending") if isinstance(resume.get("pending"), dict) else None
        rest = [item for item in resume.get("rest") or [] if isinstance(item, dict)]
        decision = ctx.human_input
        ctx.human_input = None
        ctx.agent_resume = None
        if pending:
            approved = str(decision.get("type") or "proceed") != "reject"
            await _finish_call(
                ctx, node, messages, pending, meta, approved=approved, inputs=inputs
            )
        await _run_calls(ctx, node, messages, rest, meta, inputs, steps_left)
        if ctx.agent_resume:
            return _paused(ctx, messages, str(ctx.agent_resume.get("description") or ""))
    else:
        steps_left = _MAX_ROUNDS
    try:
        completion = await _rounds(
            ctx, node, model, model_config, messages, schemas, meta, inputs, steps_left
        )
    except _PauseError as paused:
        return _paused(ctx, messages, paused.description)
    output: dict[str, Any] = {"content": completion.content, "usedTools": _used(messages)}
    if completion.prompt_tokens is not None or completion.completion_tokens is not None:
        output["usageMetadata"] = {
            "input_tokens": completion.prompt_tokens or 0,
            "output_tokens": completion.completion_tokens or 0,
            "model": completion.model,
        }
    if structured:
        output = await _apply_structured(ctx, node, model, model_config, output, structured)
    state = apply_state_updates(ctx.state, inputs.get("agentUpdateState"))
    reply_as = str(inputs.get("agentReturnResponseAs") or "")
    role = "user" if reply_as == "userMessage" else "assistant"
    delta = []
    if text:
        delta.append({"role": "user", "content": text})
    delta.append({"role": role, "content": str(output.get("content") or "")})
    return NodeResult(input={"messages": messages}, output=output, state=state, chat_delta=delta)


async def _rounds(
    ctx: ExecContext,
    node: FlowNode,
    model: str,
    model_config: Any,
    messages: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
    steps_left: int,
) -> Any:
    while steps_left > 0:
        steps_left -= 1
        completion = await _complete(ctx, node, model, model_config, messages, schemas)
        if not completion.tool_calls:
            return completion
        messages.append(_assistant_message(completion))
        await _run_calls(
            ctx, node, messages, list(completion.tool_calls), meta, inputs, steps_left
        )
        if ctx.agent_resume:
            description = str(ctx.agent_resume.get("description") or "Approve this tool call?")
            raise _PauseError(description)
    raise NodeExecutionError(node, f"Agent stopped after {_MAX_ROUNDS} tool rounds")


async def _run_calls(
    ctx: ExecContext,
    node: FlowNode,
    messages: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
    steps_left: int,
) -> None:
    for index, call in enumerate(calls):
        info = meta.get(str(call.get("name") or ""))
        if info and info.get("requires_human"):
            description = (
                f"Approve tool call {call.get('name')} with "
                f"{json.dumps(parse_tool_arguments(call.get('arguments')), default=str)}?"
            )
            ctx.agent_resume = {
                "nodeId": node.id,
                "messages": messages,
                "pending": call,
                "rest": calls[index + 1 :],
                "stepsLeft": steps_left,
                "description": description,
            }
            return
        await _finish_call(ctx, node, messages, call, meta, approved=True, inputs=inputs)


def _paused(ctx: ExecContext, messages: list[dict[str, Any]], description: str) -> NodeResult:
    return NodeResult(
        input={"messages": messages},
        output={"content": description, "humanInputMode": "approve"},
        state=dict(ctx.state),
        status="STOPPED",
    )


async def _finish_call(
    ctx: ExecContext,
    node: FlowNode,
    messages: list[dict[str, Any]],
    call: dict[str, Any],
    meta: dict[str, dict[str, Any]],
    *,
    approved: bool,
    inputs: dict[str, Any],
) -> None:
    name = str(call.get("name") or "")
    if not approved:
        content = "Human rejected this tool call."
    else:
        try:
            content = await _execute_call(ctx, node, name, call, meta, inputs)
        except ComponentError as exc:
            raise NodeExecutionError(node, str(exc)) from exc
    messages.append(
        {
            "role": "tool",
            "tool_call_id": str(call.get("id") or name),
            "name": name,
            "content": content,
        }
    )


async def _execute_call(
    ctx: ExecContext,
    node: FlowNode,
    name: str,
    call: dict[str, Any],
    meta: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
) -> str:
    args = parse_tool_arguments(call.get("arguments"))
    info = meta.get(name) or {"kind": "tool", "component": name}
    credential = ctx.credential_for(node.id, node.credential)
    kind = str(info.get("kind") or "tool")
    if kind == "knowledge":
        docs = await retrieve_documents(
            ctx,
            store_id=str(info.get("store_id") or ""),
            query=str(args.get("query") or args.get("input") or ""),
            kind=str(info.get("retrieve_kind") or "retriever"),
            component=str(info.get("component") or ""),
            inputs=info.get("config") if isinstance(info.get("config"), dict) else {},
            credential=credential,
        )
        if info.get("return_sources"):
            return json.dumps(docs, default=str)
        return "\n".join(str(doc.get("pageContent") or "") for doc in docs)
    if kind == "builtin":
        result = await invoke_named_tool(
            ctx,
            name=name,
            args=args,
            credential=credential,
            config={"provider": info.get("provider")},
        )
        return result.content
    result = await invoke_named_tool(
        ctx,
        name=str(info.get("component") or name),
        args=args,
        credential=credential,
        config=info.get("config") if isinstance(info.get("config"), dict) else inputs,
    )
    return result.content


async def _complete(
    ctx: ExecContext,
    node: FlowNode,
    model: str,
    model_config: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    config = model_config if isinstance(model_config, dict) else {}
    try:
        resolved = resolve_model(model, config)
    except RuntimeError as exc:
        raise NodeExecutionError(node, str(exc)) from exc
    try:
        return await ctx.llm.complete(
            model=resolved.model,
            messages=messages,
            api_key=resolved.api_key,
            api_base=resolved.api_base,
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
            tools=tools or None,
        )
    except Exception as exc:
        raise NodeExecutionError(node, f"LLM call failed: {exc}") from exc


async def _apply_structured(
    ctx: ExecContext,
    node: FlowNode,
    model: str,
    model_config: Any,
    output: dict[str, Any],
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    schema = structured_schema(fields)
    messages = [
        {
            "role": "system",
            "content": "Respond with JSON only, matching this schema: " + json.dumps(schema),
        },
        {"role": "user", "content": str(output.get("content") or "")},
    ]
    completion = await _complete(ctx, node, model, model_config, messages, [])
    try:
        parsed = parse_json_object(completion.content)
    except (ValueError, json.JSONDecodeError) as exc:
        raise NodeExecutionError(
            node, f"Structured output was not valid JSON: {completion.content}"
        ) from exc
    merged = dict(output)
    merged["content"] = completion.content
    merged.update(parsed)
    return merged


def _assistant_message(completion: Any) -> dict[str, Any]:
    calls = []
    for item in completion.tool_calls:
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, default=str)
        calls.append(
            {
                "id": str(item.get("id") or ""),
                "type": "function",
                "function": {"name": str(item.get("name") or ""), "arguments": arguments},
            }
        )
    return {"role": "assistant", "content": completion.content or "", "tool_calls": calls}


def _tool_catalog(
    inputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    schemas: list[dict[str, Any]] = []
    meta: dict[str, dict[str, Any]] = {}
    tools = inputs.get("agentTools")
    if isinstance(tools, list):
        for item in tools:
            if isinstance(item, str):
                _add_function(schemas, meta, item, item, {}, False)
            elif isinstance(item, dict):
                name = str(item.get("agentSelectedTool") or item.get("name") or "")
                if not name:
                    continue
                config = item.get("agentSelectedToolConfig")
                _add_function(
                    schemas,
                    meta,
                    name,
                    name,
                    config if isinstance(config, dict) else {},
                    bool(item.get("agentSelectedToolRequiresHumanInput")),
                )
    _add_knowledge(schemas, meta, inputs.get("agentKnowledgeDocumentStores"), "retriever")
    _add_vectors(schemas, meta, inputs.get("agentKnowledgeVSEmbeddings"))
    for key, provider in _BUILTIN_KEYS:
        for name in _builtin_names(inputs.get(key)):
            schemas.append({"type": name})
            meta[name] = {"kind": "builtin", "provider": provider, "component": name}
    return schemas, meta


def _add_function(
    schemas: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    name: str,
    component: str,
    config: dict[str, Any],
    requires_human: bool,
) -> None:
    schema = local_tool_schema(name) or function_schema(
        name,
        str(config.get("description") or f"Flowise tool {name}"),
        None,
    )
    schema = dict(schema)
    schema["function"] = dict(schema["function"])
    schema["function"]["name"] = name
    schemas.append(schema)
    mcp = component in {"customMCP", "mcpTool"} or bool(config.get("mcpServerUrl"))
    meta[name] = {
        "kind": "mcp" if mcp else "tool",
        "component": component,
        "config": config,
        "requires_human": requires_human,
    }


def _add_knowledge(
    schemas: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    raw: Any,
    kind: str,
) -> None:
    if not isinstance(raw, list):
        return
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("documentStore") or "")
        store_id = label.split(":", 1)[0] if label else ""
        if not store_id:
            continue
        description = str(item.get("docStoreDescription") or f"Search document store {store_id}")
        name = safe_function_name("search", store_id)
        schemas.append(
            function_schema(
                name,
                description,
                {"query": {"type": "string", "description": "Search text"}},
            )
        )
        meta[name] = {
            "kind": "knowledge",
            "store_id": store_id,
            "retrieve_kind": kind,
            "return_sources": bool(item.get("returnSourceDocuments")),
            "config": item,
        }


def _add_vectors(schemas: list[dict[str, Any]], meta: dict[str, dict[str, Any]], raw: Any) -> None:
    if not isinstance(raw, list):
        return
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("knowledgeName") or item.get("vectorStore") or "knowledge")
        description = str(item.get("knowledgeDescription") or f"Search {label}")
        name = safe_function_name("search", label)
        schemas.append(function_schema(name, description, {"query": {"type": "string"}}))
        meta[name] = {
            "kind": "knowledge",
            "store_id": str(item.get("vectorStore") or label),
            "retrieve_kind": "vectorstore",
            "component": str(item.get("vectorStore") or ""),
            "return_sources": bool(item.get("returnSourceDocuments")),
            "config": item,
        }


def _builtin_names(raw: Any) -> list[str]:
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, dict):
            name = item.get("name") or item.get("type") or item.get("tool")
            if name:
                names.append(str(name))
    return names


def _used(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in messages:
        if item.get("role") == "tool" and item.get("name"):
            names.append(str(item.get("name")))
    return names


def _configured(value: Any) -> bool:
    return value not in (None, "", [], {})


class _PauseError(Exception):
    def __init__(self, description: str) -> None:
        super().__init__(description)
        self.description = description
