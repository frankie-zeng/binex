"""Leaf Agentflow nodes. Iteration and the graph walk live in the engine."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import httpx

from binex.flowise.a2a import A2AProtocolError, send_a2a
from binex.flowise.cao import CaoProtocolError, execute_cao, normalize_cao_config
from binex.flowise.components import (
    ComponentError,
    files_from_artifacts,
    invoke_named_tool,
    parse_input_args,
    retrieve_documents,
)
from binex.flowise.context import ExecContext
from binex.flowise.custom_function import run_javascript
from binex.flowise.errors import NodeExecutionError
from binex.flowise.graph import FlowNode
from binex.flowise.llm import (
    LiteLLMCaller,
    complete_with_binex_tools,
    parse_json_object,
    resolve_model,
    structured_schema,
)
from binex.tools.ssrf import BlockedURLError, validate_url

_MARKDOWN = re.compile(r"[*_`]+")
DECISION_NODES = frozenset(
    {"conditionAgentflow", "conditionAgentAgentflow", "humanInputAgentflow"}
)


@dataclass
class NodeResult:
    input: Any
    output: dict[str, Any]
    state: dict[str, Any]
    status: str = "FINISHED"
    chat_delta: list[dict[str, Any]] = field(default_factory=list)
    conversations: list[dict[str, Any]] = field(default_factory=list)
    loop_target: str | None = None
    max_loop: int | None = None
    fallback_message: str | None = None


def incoming_text(data: Any) -> str:
    """Text an upstream node left for a node with an empty message field."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        output = data.get("output")
        if isinstance(output, dict) and output.get("content") is not None:
            content = output["content"]
            return content if isinstance(content, str) else json.dumps(content, default=str)
        if isinstance(data.get("content"), str):
            return data["content"]
        if isinstance(data.get("question"), str):
            return data["question"]
    return ""


def output_text(output: dict[str, Any]) -> str:
    content = output.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, default=str)


def apply_state_updates(state: dict[str, Any], updates: Any) -> dict[str, Any]:
    merged = dict(state)
    if isinstance(updates, list):
        for item in updates:
            if isinstance(item, dict) and item.get("key"):
                merged[str(item["key"])] = item.get("value")
    return merged


def _plain(value: Any) -> str:
    text = "" if value is None else str(value)
    return _MARKDOWN.sub("", text).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _configured(value: Any) -> bool:
    return value not in (None, "", [], {})


async def handle_start(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], _: Any
) -> NodeResult:
    input_type = str(inputs.get("startInputType") or "chatInput")
    flow_state = _parse_state_list(node, inputs.get("startState"))
    if inputs.get("startPersistState") and ctx.state:
        flow_state.update(ctx.state)
    input_data: dict[str, Any] = {}
    output: dict[str, Any] = {}
    if input_type == "chatInput":
        input_data["question"] = ctx.question
        output["question"] = ctx.question
    elif input_type == "formInput":
        input_data["form"] = {
            "title": inputs.get("formTitle"),
            "description": inputs.get("formDescription"),
            "inputs": inputs.get("formInputTypes"),
        }
        output["form"] = ctx.form or {}
    elif input_type == "webhookTrigger":
        output["webhook"] = ctx.webhook
        input_data["webhook"] = ctx.webhook
        mode = str(inputs.get("webhookInputMode") or "text")
        if mode == "none":
            output["question"] = " "
            input_data["question"] = " "
        elif mode == "text":
            output["question"] = ctx.question or " "
            input_data["question"] = ctx.question or " "
    elif input_type == "scheduleInput":
        mode = str(inputs.get("scheduleInputMode") or "text")
        if mode == "form":
            output["form"] = ctx.form or {}
        elif mode == "none":
            output["question"] = " "
        else:
            text = ctx.question or str(inputs.get("scheduleDefaultInput") or "")
            output["question"] = text
            input_data["question"] = text
    else:
        raise NodeExecutionError(node, f"Unknown start input type '{input_type}'")
    if inputs.get("startEphemeralMemory"):
        output["ephemeralMemory"] = True
    if inputs.get("startPersistState"):
        output["persistState"] = True
        ctx.persist_state = True
    return NodeResult(input=input_data, output=output, state=flow_state)


def _parse_state_list(node: FlowNode, value: Any) -> dict[str, Any]:
    items = value
    if isinstance(value, str) and value.strip():
        try:
            items = json.loads(value)
        except json.JSONDecodeError as exc:
            raise NodeExecutionError(node, "Invalid Flow State") from exc
    state: dict[str, Any] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("key"):
                state[str(item["key"])] = item.get("value")
    return state


async def handle_direct_reply(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], _: Any
) -> NodeResult:
    message = inputs.get("directReplyMessage")
    text = message if isinstance(message, str) else "" if message is None else str(message)
    return NodeResult(input={}, output={"content": text}, state=dict(ctx.state))


async def handle_condition(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], _: Any
) -> NodeResult:
    raw = inputs.get("conditions")
    conditions = _condition_list(node, raw)
    original = json.loads(json.dumps(conditions, default=str))
    for condition in conditions:
        if _condition_matches(condition):
            condition["isFulfilled"] = True
            break
    else_row = {"type": "string", "value1": "", "operation": "equal", "value2": ""}
    if any(item.get("isFulfilled") for item in conditions):
        conditions.append({**else_row, "isFulfilled": False})
    else:
        conditions.append({**else_row, "isFulfilled": True})
    return NodeResult(
        input={"conditions": original},
        output={"conditions": conditions},
        state=dict(ctx.state),
    )


def _condition_list(node: FlowNode, raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NodeExecutionError(node, "Conditions must be a JSON array") from exc
    if not isinstance(raw, list) or not raw:
        raise NodeExecutionError(node, "Condition node requires at least one condition")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise NodeExecutionError(node, "Each condition must be an object")
        rows.append(dict(item))
    return rows


def _condition_matches(condition: dict[str, Any]) -> bool:
    operation = str(condition.get("operation") or "")
    kind = str(condition.get("type") or "string")
    left = condition.get("value1")
    right = condition.get("value2")
    if kind == "number":
        left_num = _as_float(left)
        right_num = _as_float(right)
        return _compare_number(operation, left_num, right_num, left)
    if kind == "boolean":
        return _compare_bool(operation, _as_bool(left), _as_bool(right))
    return _compare_text(operation, _plain(left), _plain(right))


def _compare_text(operation: str, left: str, right: str) -> bool:
    if operation == "contains":
        return right in left
    if operation == "notContains":
        return right not in left
    if operation == "startsWith":
        return left.startswith(right)
    if operation == "endsWith":
        return left.endswith(right)
    if operation == "equal":
        return left == right
    if operation == "notEqual":
        return left != right
    if operation == "regex":
        try:
            return re.search(right, left) is not None
        except re.error:
            return False
    if operation == "isEmpty":
        return left == ""
    if operation == "notEmpty":
        return left != ""
    return False


def _compare_number(operation: str, left: float, right: float, raw_left: Any) -> bool:
    # Flowise checks isEmpty on the original value, so numeric 0 is not empty.
    if operation == "isEmpty":
        return raw_left is None or raw_left == ""
    if operation == "notEmpty":
        return raw_left is not None and raw_left != ""
    if operation == "smaller":
        return left < right
    if operation == "smallerEqual":
        return left <= right
    if operation == "equal":
        return left == right
    if operation == "notEqual":
        return left != right
    if operation == "larger":
        return left > right
    if operation == "largerEqual":
        return left >= right
    return False


def _compare_bool(operation: str, left: bool, right: bool) -> bool:
    if operation == "equal":
        return left is right
    if operation == "notEqual":
        return left is not right
    return False


async def handle_loop(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], _: Any
) -> NodeResult:
    spec = str(inputs.get("loopBackToNode") or "")
    target = _loop_target(node, spec, ctx)
    try:
        max_loop = int(inputs.get("maxLoopCount") or 5)
    except (TypeError, ValueError):
        max_loop = 5
    if max_loop < 1:
        max_loop = 1
    fallback = inputs.get("fallbackMessage")
    fallback_text = fallback if isinstance(fallback, str) else ""
    target_node = ctx.graph.node(target)
    state = apply_state_updates(ctx.state, inputs.get("loopUpdateState"))
    content = f"Loop back to {target_node.label} ({target})"
    return NodeResult(
        input={"nodeID": target, "maxLoopCount": max_loop},
        output={
            "content": content,
            "nodeID": target,
            "maxLoopCount": max_loop,
            "fallbackMessage": fallback_text,
        },
        state=state,
        loop_target=target,
        max_loop=max_loop,
        fallback_message=fallback_text or None,
    )


def _loop_target(node: FlowNode, spec: str, ctx: ExecContext) -> str:
    if not spec:
        raise NodeExecutionError(node, "Loop node requires loopBackToNode")
    if spec in ctx.graph.nodes:
        return spec
    head = spec.split("-", 1)[0]
    if head in ctx.graph.nodes:
        return head
    raise NodeExecutionError(node, f"Loop target '{spec}' is not a node in this flow")


async def handle_human(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], _: Any
) -> NodeResult:
    mode = str(inputs.get("humanInputMode") or "approve")
    if mode not in {"approve", "input", "output"}:
        mode = "approve"
    provided = _take_human(ctx, node)
    if provided is None:
        description = await _human_description(ctx, node, inputs)
        return NodeResult(
            input={"humanInputMode": mode},
            output={"content": description, "humanInputMode": mode},
            state=dict(ctx.state),
            status="STOPPED",
            chat_delta=[{"role": "assistant", "content": description}] if description else [],
        )
    kind = str(provided.get("type") or "proceed")
    feedback = provided.get("feedback")
    feedback_text = feedback if isinstance(feedback, str) else ""
    proceed = {
        "type": "proceed",
        "startNodeId": node.id,
        "isFulfilled": kind == "proceed",
        "feedback": feedback_text or None,
    }
    reject = {
        "type": "reject",
        "startNodeId": node.id,
        "isFulfilled": kind == "reject",
        "feedback": feedback_text or None,
    }
    content = feedback_text or kind
    return NodeResult(
        input={**provided, "humanInputMode": mode},
        output={"content": content, "conditions": [proceed, reject], "humanInputMode": mode},
        state=dict(ctx.state),
        chat_delta=[{"role": "user", "content": content}],
    )


def _take_human(ctx: ExecContext, node: FlowNode) -> dict[str, Any] | None:
    human = ctx.human_input
    if not isinstance(human, dict):
        return None
    target = str(human.get("startNodeId") or ctx.pause_node_id or node.id)
    if target != node.id:
        return None
    ctx.human_input = None
    return human


async def _human_description(ctx: ExecContext, node: FlowNode, inputs: dict[str, Any]) -> str:
    kind = str(inputs.get("humanInputDescriptionType") or "fixed")
    if kind == "dynamic" and inputs.get("humanInputModel"):
        prompt = str(
            inputs.get("humanInputModelPrompt") or "Describe what the human should decide."
        )
        messages = [*_history_messages(ctx.chat_history), {"role": "user", "content": prompt}]
        completion = await _complete(
            ctx,
            node,
            str(inputs.get("humanInputModel")),
            inputs.get("humanInputModelConfig"),
            messages,
        )
        return completion.content
    description = str(inputs.get("humanInputDescription") or "Do you want to proceed?")
    if ctx.chat_history:
        last = ctx.chat_history[-1].get("content") or ""
        if isinstance(last, str) and last.strip():
            return f"{last}\n\n{description}"
    return description


def _history_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in history:
        content = item.get("content")
        messages.append(
            {
                "role": str(item.get("role") or "user"),
                "content": content
                if isinstance(content, str)
                else json.dumps(content, default=str),
            }
        )
    return messages


async def handle_llm(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    return await _chat(
        ctx,
        node,
        incoming,
        model=str(inputs.get("llmModel") or ""),
        model_config=inputs.get("llmModelConfig"),
        messages=inputs.get("llmMessages"),
        user_message=inputs.get("llmUserMessage"),
        memory=bool(inputs.get("llmEnableMemory", True)),
        memory_type=str(inputs.get("llmMemoryType") or "allMessages"),
        memory_window=inputs.get("llmMemoryWindowSize") or 20,
        memory_max_tokens=inputs.get("llmMemoryMaxTokenLimit"),
        structured=inputs.get("llmStructuredOutput"),
        state_updates=inputs.get("llmUpdateState"),
        binex_tools=inputs.get("binexTools"),
        return_as=str(inputs.get("llmReturnResponseAs") or "assistantMessage"),
    )


async def handle_agent(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    from binex.flowise.agent_loop import flowise_tools_configured, run_tool_agent

    if flowise_tools_configured(inputs):
        return await run_tool_agent(ctx, node, inputs, incoming)
    return await _chat(
        ctx,
        node,
        incoming,
        model=str(inputs.get("agentModel") or ""),
        model_config=inputs.get("agentModelConfig"),
        messages=inputs.get("agentMessages"),
        user_message=inputs.get("agentUserMessage"),
        memory=bool(inputs.get("agentEnableMemory", True)),
        memory_type=str(inputs.get("agentMemoryType") or "allMessages"),
        memory_window=inputs.get("agentMemoryWindowSize") or 20,
        memory_max_tokens=inputs.get("agentMemoryMaxTokenLimit"),
        structured=inputs.get("agentStructuredOutput"),
        state_updates=inputs.get("agentUpdateState"),
        binex_tools=inputs.get("binexTools"),
        return_as=str(inputs.get("agentReturnResponseAs") or "assistantMessage"),
    )


async def handle_condition_agent(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    scenarios = inputs.get("conditionAgentScenarios")
    if isinstance(scenarios, str):
        try:
            scenarios = json.loads(scenarios)
        except json.JSONDecodeError as exc:
            raise NodeExecutionError(node, "Scenarios must be a JSON array") from exc
    if not isinstance(scenarios, list) or not scenarios:
        raise NodeExecutionError(node, "Condition Agent requires scenarios")
    names = [str(item.get("scenario") if isinstance(item, dict) else item) for item in scenarios]
    question = inputs.get("conditionAgentInput") or incoming_text(incoming) or ctx.question
    instructions = str(inputs.get("conditionAgentInstructions") or "")
    system = (
        str(inputs.get("conditionAgentSystemPrompt"))
        if inputs.get("conditionAgentOverrideSystemPrompt")
        and inputs.get("conditionAgentSystemPrompt")
        else (
            "Classify the input into exactly one scenario. "
            'Reply with JSON {"output": "<scenario>"} and nothing else.'
        )
    )
    user = json.dumps(
        {"input": question, "scenarios": names, "instruction": instructions},
        default=str,
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    completion = await _complete(
        ctx,
        node,
        str(inputs.get("conditionAgentModel") or ""),
        inputs.get("conditionAgentModelConfig"),
        messages,
    )
    try:
        parsed = parse_json_object(completion.content)
        chosen = parsed.get("output")
        if not isinstance(chosen, str):
            raise ValueError("missing output")
    except (ValueError, json.JSONDecodeError) as exc:
        raise NodeExecutionError(
            node,
            f"Condition Agent did not return a JSON scenario. Raw response: {completion.content}",
        ) from exc
    index = _scenario_index(names, chosen)
    if index is None:
        raise NodeExecutionError(
            node,
            f"Condition Agent returned '{chosen}', which does not match any scenario",
        )
    conditions = [
        {"output": name, "isFulfilled": position == index} for position, name in enumerate(names)
    ]
    output: dict[str, Any] = {"conditions": conditions, "content": completion.content}
    _attach_usage(output, completion)
    return NodeResult(
        input={"messages": messages},
        output=output,
        state=dict(ctx.state),
    )


def _scenario_index(names: list[str], chosen: str) -> int | None:
    folded = chosen.strip().casefold()
    for index, name in enumerate(names):
        if name == chosen or name.strip().casefold() == folded:
            return index
    for index, name in enumerate(names):
        if folded and folded in name.casefold():
            return index
    return None


async def _chat(
    ctx: ExecContext,
    node: FlowNode,
    incoming: Any,
    *,
    model: str,
    model_config: Any,
    messages: Any,
    user_message: Any,
    memory: bool,
    memory_type: str,
    memory_window: Any,
    structured: Any,
    state_updates: Any,
    binex_tools: Any,
    memory_max_tokens: Any = None,
    return_as: str = "assistantMessage",
) -> NodeResult:
    config = model_config if isinstance(model_config, dict) else {}
    prompt = user_message if isinstance(user_message, str) and user_message.strip() else ""
    if not prompt:
        prompt = incoming_text(incoming) or ctx.question
    try:
        window = int(memory_window)
    except (TypeError, ValueError):
        window = 20
    history = list(ctx.chat_history) if memory else []
    history, memory_type = await _prepare_memory(
        ctx,
        node,
        model,
        config,
        history,
        memory_type,
        window,
        memory_max_tokens,
    )
    transcript = _assemble_messages(
        messages if isinstance(messages, list) else [],
        prompt,
        history,
        memory_type,
        window,
    )
    fields = structured if isinstance(structured, list) else []
    if fields:
        schema = structured_schema([item for item in fields if isinstance(item, dict)])
        transcript = [
            *transcript,
            {
                "role": "system",
                "content": "Respond with JSON only, matching this schema: " + json.dumps(schema),
            },
        ]
    tools = [str(item) for item in binex_tools] if isinstance(binex_tools, list) else []
    if tools:
        resolved = resolve_model(model, config)
        completion = await complete_with_binex_tools(
            resolved,
            transcript,
            tools,
            run_id=ctx.request.run_id,
            node_id=node.id,
        )
    else:
        completion = await _complete(ctx, node, model, config, transcript)
    output: dict[str, Any] = {"content": completion.content}
    _attach_usage(output, completion)
    if fields:
        try:
            parsed = parse_json_object(completion.content)
        except (ValueError, json.JSONDecodeError) as exc:
            raise NodeExecutionError(
                node,
                f"Structured output was not valid JSON: {completion.content}",
            ) from exc
        output.update(parsed)
    state = apply_state_updates(ctx.state, state_updates)
    delta = []
    if prompt:
        delta.append({"role": "user", "content": prompt})
    reply_role = "user" if return_as == "userMessage" else "assistant"
    delta.append({"role": reply_role, "content": completion.content})
    return NodeResult(
        input={"messages": transcript},
        output=output,
        state=state,
        chat_delta=delta,
    )


async def _prepare_memory(
    ctx: ExecContext,
    node: FlowNode,
    model: str,
    model_config: Any,
    history: list[dict[str, Any]],
    memory_type: str,
    window: int,
    max_tokens: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Window, summary, and summary-buffer memory. Does not edit ``ctx.chat_history``."""
    if not history:
        return history, "allMessages" if memory_type != "windowSize" else memory_type
    if memory_type == "windowSize":
        return history[-(max(window, 1) * 2) :], "allMessages"
    if memory_type == "conversationSummary":
        summary = await _summarize(ctx, node, model, model_config, history)
        note = f"Summary of the conversation so far:\n{summary}"
        remembered = [{"role": "system", "content": note}]
        return remembered, "allMessages"
    over_budget = _estimate_tokens(history) > _token_limit(max_tokens)
    if memory_type == "conversationSummaryBuffer" and over_budget:
        older = history[:-2] if len(history) > 2 else history
        recent = history[-2:] if len(history) > 2 else []
        summary = await _summarize(ctx, node, model, model_config, older)
        remembered = [
            {"role": "system", "content": f"Summary of the conversation so far:\n{summary}"},
            *recent,
        ]
        return remembered, "allMessages"
    return history, "allMessages" if memory_type != "windowSize" else memory_type


def _token_limit(value: Any) -> int:
    try:
        return int(value) if value not in (None, "") else 2000
    except (TypeError, ValueError):
        return 2000


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for item in messages:
        total += max(1, len(str(item.get("content") or "")) // 4)
    return total


async def _summarize(
    ctx: ExecContext,
    node: FlowNode,
    model: str,
    model_config: Any,
    history: list[dict[str, Any]],
) -> str:
    transcript = json.dumps(_history_messages(history), default=str)
    completion = await _complete(
        ctx,
        node,
        model,
        model_config,
        [
            {
                "role": "user",
                "content": "Summarize this conversation in a few sentences:\n" + transcript,
            }
        ],
    )
    return completion.content


def _assemble_messages(
    configured: list[Any],
    user_message: str,
    history: list[dict[str, Any]],
    memory_type: str,
    window: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in configured:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        messages.append({"role": str(item.get("role") or "user"), "content": str(item["content"])})
    remembered = history
    if memory_type == "windowSize":
        remembered = history[-(max(window, 1) * 2) :]
    elif memory_type in {"conversationSummary", "conversationSummaryBuffer"}:
        remembered = history[-20:]
    if remembered:
        system = [item for item in messages if item["role"] == "system"]
        rest = [item for item in messages if item["role"] != "system"]
        messages = [*system, *_history_messages(remembered), *rest]
    if user_message:
        messages.append({"role": "user", "content": user_message})
    if not messages:
        messages.append({"role": "user", "content": user_message})
    return messages


async def _complete(
    ctx: ExecContext,
    node: FlowNode,
    component: str,
    model_config: Any,
    messages: list[dict[str, Any]],
) -> Any:
    config = model_config if isinstance(model_config, dict) else {}
    try:
        resolved = resolve_model(component, config)
    except RuntimeError as exc:
        raise NodeExecutionError(node, str(exc)) from exc
    caller = ctx.llm or LiteLLMCaller()
    try:
        return await caller.complete(
            model=resolved.model,
            messages=messages,
            api_key=resolved.api_key,
            api_base=resolved.api_base,
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
        )
    except Exception as exc:
        raise NodeExecutionError(node, f"LLM call failed: {exc}") from exc


def _attach_usage(output: dict[str, Any], completion: Any) -> None:
    if completion.prompt_tokens is None and completion.completion_tokens is None:
        return
    output["usageMetadata"] = {
        "input_tokens": completion.prompt_tokens or 0,
        "output_tokens": completion.completion_tokens or 0,
        "model": completion.model,
    }


async def handle_http(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], _: Any
) -> NodeResult:
    method = str(inputs.get("method") or "GET").upper()
    url = str(inputs.get("url") or "")
    if not url:
        raise NodeExecutionError(node, "HTTP node requires a URL")
    if not ctx.skip_ssrf:
        try:
            validate_url(url)
        except BlockedURLError as exc:
            raise NodeExecutionError(node, str(exc)) from exc
    headers = _pairs(inputs.get("headers"))
    query = _pairs(inputs.get("queryParams"))
    if query:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}{urlencode(query)}"
    _apply_http_credential(headers, ctx.credential_for(node.id, node.credential))
    body_type = str(inputs.get("bodyType") or "json")
    response_type = str(inputs.get("responseType") or "json")
    request_body = inputs.get("body")
    data: Any = None
    json_body: Any = None
    if method != "GET" and request_body not in (None, ""):
        json_body, data, headers = _http_body(body_type, request_body, headers)
    try:
        response = await ctx.http.request(
            method,
            url,
            headers=headers,
            json=json_body,
            data=data,
            timeout=60,
        )
    except httpx.HTTPError as exc:
        raise NodeExecutionError(node, f"HTTP request failed: {exc}") from exc
    if response.status_code >= 400:
        raise NodeExecutionError(
            node,
            f"HTTP {response.status_code} from {url}: {response.text[:500]}",
        )
    payload = _http_payload(response, response_type)
    return NodeResult(
        input={
            "http": {
                "method": method,
                "url": url,
                "headers": headers,
                "bodyType": body_type,
                "responseType": response_type,
            }
        },
        output={
            "http": {
                "data": payload,
                "status": response.status_code,
                "statusText": response.reason_phrase,
            }
        },
        state=dict(ctx.state),
    )


def _pairs(value: Any) -> dict[str, str]:
    pairs: dict[str, str] = {}
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("key") and item.get("value") is not None:
                pairs[str(item["key"])] = str(item["value"])
    elif isinstance(value, dict):
        pairs = {str(key): str(item) for key, item in value.items()}
    return pairs


def _apply_http_credential(headers: dict[str, str], credential: dict[str, Any]) -> None:
    username = credential.get("basicAuthUsername")
    password = credential.get("basicAuthPassword")
    if username or password:
        token = base64.b64encode(f"{username or ''}:{password or ''}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
        return
    bearer = credential.get("token") or credential.get("bearerToken")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
        return
    key = credential.get("key")
    value = credential.get("value")
    if key and value:
        headers[str(key)] = str(value)


def _http_body(
    body_type: str, body: Any, headers: dict[str, str]
) -> tuple[Any, Any, dict[str, str]]:
    if body_type == "raw":
        return None, body if isinstance(body, str) else json.dumps(body), headers
    if body_type == "formData":
        return None, _pairs(body), headers
    if body_type == "xWwwFormUrlencoded":
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        payload = body
        if isinstance(body, str):
            payload = json.loads(body)
        return None, payload if isinstance(payload, dict) else _pairs(payload), headers
    headers["Content-Type"] = "application/json"
    if isinstance(body, str):
        try:
            return json.loads(body), None, headers
        except json.JSONDecodeError:
            return body, None, headers
    return body, None, headers


def _http_payload(response: httpx.Response, response_type: str) -> Any:
    if response_type == "text":
        return response.text
    if response_type in {"base64", "arraybuffer"}:
        return base64.b64encode(response.content).decode("ascii")
    try:
        return response.json()
    except ValueError:
        return response.text


async def handle_custom_function(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    code = inputs.get("customFunctionJavascriptFunction")
    if not isinstance(code, str) or not code.strip():
        raise NodeExecutionError(node, "Custom Function requires JavaScript")
    extra: dict[str, Any] = {}
    variables = inputs.get("customFunctionInputVariables")
    if isinstance(variables, list):
        for item in variables:
            if isinstance(item, dict) and item.get("variableName"):
                extra[f"${item['variableName']}"] = item.get("variableValue")
    flow = {
        "input": incoming,
        "state": ctx.state,
        "chatflowId": ctx.request.chatflow_id,
        "sessionId": ctx.request.session_key,
        "chatId": str(ctx.request.inputs.get("chatId") or ctx.request.session_key),
        "chatHistory": ctx.chat_history,
    }
    runner = ctx.js_runner or run_javascript
    try:
        result = await runner(
            code,
            flow_input=incoming,
            flow=flow,
            variables=ctx.request.variables,
            extra=extra,
        )
    except Exception as exc:
        raise NodeExecutionError(node, str(exc)) from exc
    content = result if isinstance(result, str) else json.dumps(result, default=str)
    state = apply_state_updates(ctx.state, inputs.get("customFunctionUpdateState"))
    return NodeResult(
        input={"code": code},
        output={"content": content},
        state=state,
    )


async def handle_a2a(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    message = inputs.get("a2aMessage")
    text = (
        message.strip()
        if isinstance(message, str) and message.strip()
        else incoming_text(incoming)
    )
    if not text.strip():
        raise NodeExecutionError(node, "A2A node requires a message")
    url = str(inputs.get("a2aAgentUrl") or "")
    stored = ctx.store.get_a2a(ctx.request.chatflow_id, ctx.request.session_key, node.id)
    state_session = ctx.state.get(f"a2a.{node.id}")
    session = dict(stored)
    if isinstance(state_session, dict):
        session = {**session, **state_session}
    bearer = inputs.get("a2aBearerToken")
    bearer_text = bearer if isinstance(bearer, str) and bearer else None
    try:
        timeout_ms = int(inputs.get("a2aTimeoutMs") or 60000)
    except (TypeError, ValueError):
        timeout_ms = 60000
    try:
        result = await send_a2a(
            agent_url=url,
            message=text,
            session=session,
            bearer=bearer_text,
            timeout_ms=timeout_ms,
            client=ctx.http,
        )
    except A2AProtocolError as exc:
        turn = _turn(
            node.id,
            "a2a",
            "error",
            text,
            str(exc),
            session.get("contextId"),
            session.get("taskId"),
            exc.request,
            exc.response,
        )
        raise NodeExecutionError(node, str(exc)) from _with_turn(exc, turn)
    next_session = {"contextId": result.get("contextId"), "taskId": result.get("taskId")}
    ctx.store.save_a2a(ctx.request.chatflow_id, ctx.request.session_key, node.id, next_session)
    state = dict(ctx.state)
    state[f"a2a.{node.id}"] = next_session
    turn = _turn(
        node.id,
        "a2a",
        "completed",
        text,
        str(result.get("text") or ""),
        next_session.get("contextId"),
        next_session.get("taskId"),
        result.get("request"),
        result.get("response"),
    )
    return NodeResult(
        input={"agentUrl": url, "message": text, "session": session},
        output={
            "content": result.get("text") or "",
            "a2a": {
                "request": result.get("request"),
                "response": result.get("response"),
                "contextId": next_session.get("contextId"),
                "taskId": next_session.get("taskId"),
                "turns": [
                    {"role": "user", "text": text},
                    {"role": "agent", "text": result.get("text") or ""},
                ],
            },
        },
        state=state,
        conversations=[turn],
    )


async def handle_cao(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    task = inputs.get("caoTask")
    if not (isinstance(task, str) and task.strip()):
        inputs = dict(inputs)
        inputs["caoTask"] = incoming_text(incoming)
    credential = ctx.credential_for(node.id, node.credential)
    try:
        config = normalize_cao_config(inputs, env=dict(os.environ), credential=credential)
    except CaoProtocolError as exc:
        raise NodeExecutionError(node, str(exc)) from exc
    persisted = ctx.store.get_cao(ctx.request.chatflow_id, node.id)
    try:
        result = await execute_cao(
            config,
            persisted,
            chatflow_id=ctx.request.chatflow_id,
            node_id=node.id,
            client=ctx.http,
        )
    except CaoProtocolError as exc:
        turn = _turn(
            node.id,
            "cao",
            "error",
            config.task,
            str(exc),
            exc.session_name,
            exc.terminal_id,
            exc.request,
            exc.response,
        )
        raise NodeExecutionError(node, str(exc)) from _with_turn(exc, turn)
    saved = {
        "caoSessionName": result.get("sessionName"),
        "terminalId": result.get("terminalId"),
        "provider": config.provider,
        "agentProfile": config.agent_profile,
        "workspaceId": ctx.request.workspace_id,
    }
    ctx.store.save_cao(ctx.request.chatflow_id, node.id, saved)
    content = str(result.get("content") or "")
    turn = _turn(
        node.id,
        "cao",
        str(result.get("status") or ""),
        config.task,
        content,
        result.get("sessionName"),
        result.get("terminalId"),
        result.get("request"),
        result.get("response"),
    )
    return NodeResult(
        input={
            "provider": config.provider,
            "agentProfile": config.agent_profile,
            "mode": config.mode,
            "task": config.task,
        },
        output={
            "content": content,
            "cao": {
                "provider": config.provider,
                "agentProfile": config.agent_profile,
                "mode": config.mode,
                "status": result.get("status"),
                "sessionName": result.get("sessionName"),
                "terminalId": result.get("terminalId"),
                "request": result.get("request"),
                "response": result.get("response"),
                "turns": [
                    {"role": "user", "text": config.task},
                    {"role": "agent", "text": content},
                ],
            },
        },
        state=dict(ctx.state),
        conversations=[turn],
    )


def _turn(
    node_id: str,
    kind: str,
    status: str,
    message: str,
    output: str,
    external_session: Any,
    external_task: Any,
    raw_request: Any,
    raw_response: Any,
) -> dict[str, Any]:
    return {
        "nodeId": node_id,
        "kind": kind,
        "status": status,
        "message": message,
        "output": output,
        "externalSession": external_session,
        "externalTask": external_task,
        "rawRequest": raw_request,
        "rawResponse": raw_response,
    }


def _with_turn(exc: Exception, turn: dict[str, Any]) -> Exception:
    exc.turn = turn  # type: ignore[attr-defined]
    return exc


async def handle_tool(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    selected = inputs.get("toolAgentflowSelectedTool") or inputs.get("selectedTool")
    name = str(selected or "").strip()
    if not name:
        raise NodeExecutionError(node, "Tool not selected")
    config = inputs.get("toolAgentflowSelectedToolConfig") or inputs.get("selectedToolConfig")
    config = config or {}
    if not isinstance(config, dict):
        config = {}
    args = parse_input_args(inputs.get("toolInputArgs"))
    if not args:
        text = incoming_text(incoming) or ctx.question
        if text:
            args = {"input": text, "expression": text}
    try:
        result = await invoke_named_tool(
            ctx,
            name=name,
            args=args,
            credential=ctx.credential_for(node.id, node.credential),
            config=config,
        )
    except ComponentError as exc:
        raise NodeExecutionError(node, str(exc)) from exc
    output: dict[str, Any] = {"content": result.content}
    if result.source_documents:
        output["sourceDocuments"] = result.source_documents
    files = files_from_artifacts(result.artifacts)
    if files:
        output["files"] = files
        output["artifacts"] = result.artifacts
    state = apply_state_updates(ctx.state, inputs.get("toolUpdateState"))
    return NodeResult(input={"tool": name, "args": args}, output=output, state=state)


async def handle_retriever(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    query = inputs.get("retrieverQuery")
    text = query.strip() if isinstance(query, str) and query.strip() else ""
    if not text:
        text = incoming_text(incoming) or ctx.question
    stores = inputs.get("retrieverKnowledgeDocumentStores") or inputs.get("documentStores") or []
    vectors = inputs.get("retrieverVectorStores") or []
    if not isinstance(stores, list):
        stores = []
    if not isinstance(vectors, list):
        vectors = []
    if not stores and not vectors:
        raise NodeExecutionError(node, "Retriever requires a document store or vector store")
    documents: list[dict[str, Any]] = []
    try:
        for item in stores:
            store_id = _store_id(item)
            if not store_id:
                continue
            documents.extend(
                await retrieve_documents(
                    ctx,
                    store_id=store_id,
                    query=text,
                    kind="retriever",
                    credential=ctx.credential_for(node.id, node.credential),
                    inputs=item if isinstance(item, dict) else {},
                )
            )
        for item in vectors:
            if not isinstance(item, dict):
                continue
            store_id = str(item.get("vectorStore") or item.get("id") or "")
            documents.extend(
                await retrieve_documents(
                    ctx,
                    store_id=store_id,
                    query=text,
                    kind="vectorstore",
                    component=str(item.get("vectorStore") or ""),
                    credential=ctx.credential_for(node.id, node.credential),
                    inputs=item,
                )
            )
    except ComponentError as exc:
        raise NodeExecutionError(node, str(exc)) from exc
    style = str(inputs.get("outputFormat") or inputs.get("retrieverOutputFormat") or "text")
    if style == "textWithMetadata":
        content = json.dumps(documents, default=str)
    else:
        content = "\n".join(str(doc.get("pageContent") or "") for doc in documents)
    state = apply_state_updates(ctx.state, inputs.get("retrieverUpdateState"))
    return NodeResult(
        input={"query": text},
        output={"content": content, "sourceDocuments": documents},
        state=state,
    )


def _store_id(item: Any) -> str:
    if isinstance(item, str):
        return item.split(":", 1)[0]
    if isinstance(item, dict):
        raw = str(item.get("documentStore") or item.get("id") or "")
        return raw.split(":", 1)[0]
    return ""


async def handle_execute_flow(
    ctx: ExecContext, node: FlowNode, inputs: dict[str, Any], incoming: Any
) -> NodeResult:
    from binex.flowise.components import prediction_base
    from binex.flowise.engine import execute_flow
    from binex.flowise.protocol import RunRequest

    selected = str(inputs.get("executeFlowSelectedFlow") or "").strip()
    if not selected:
        raise NodeExecutionError(node, "Execute Flow requires executeFlowSelectedFlow")
    if selected == ctx.request.chatflow_id:
        raise NodeExecutionError(node, "Cannot call the same agentflow")
    if ctx.flow_depth >= 5:
        raise NodeExecutionError(node, "Execute Flow nesting is deeper than 5")
    question = inputs.get("executeFlowInput")
    text = question.strip() if isinstance(question, str) and question.strip() else ""
    if not text:
        text = incoming_text(incoming) or ctx.question
    override = _override_config(node, inputs.get("executeFlowOverrideConfig"))
    role = (
        "user"
        if str(inputs.get("executeFlowReturnResponseAs") or "userMessage") == "userMessage"
        else "assistant"
    )
    child_flow = ctx.request.flows.get(selected) if isinstance(ctx.request.flows, dict) else None
    if child_flow is not None:
        outcome = await _run_nested_flow(
            ctx, node, execute_flow, RunRequest, selected, text, override, child_flow
        )
    else:
        outcome_text, extra = await _run_prediction(
            ctx,
            node,
            prediction_base,
            str(inputs.get("executeFlowBaseURL") or ""),
            selected,
            text,
            override,
        )
        output = {"content": outcome_text, **extra}
        state = apply_state_updates(ctx.state, inputs.get("executeFlowUpdateState"))
        return NodeResult(
            input={"flowId": selected, "question": text},
            output=output,
            state=state,
            chat_delta=[{"role": role, "content": outcome_text}] if outcome_text else [],
        )
    if outcome.status == "ERROR":
        raise NodeExecutionError(node, outcome.error or "Execute Flow failed")
    if outcome.status == "STOPPED":
        ctx.nested_resume = {
            "nodeId": node.id,
            "chatflowId": selected,
            "startNodeId": outcome.pause_node_id,
        }
        return NodeResult(
            input={"flowId": selected, "question": text},
            output={"content": outcome.text, "humanInputMode": "approve"},
            state=dict(ctx.state),
            status="STOPPED",
        )
    state = apply_state_updates(ctx.state, inputs.get("executeFlowUpdateState"))
    return NodeResult(
        input={"flowId": selected, "question": text},
        output={"content": outcome.text, "nestedExecution": outcome.execution_data},
        state=state,
        chat_delta=[{"role": role, "content": outcome.text}] if outcome.text else [],
    )


def _override_config(node: FlowNode, raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NodeExecutionError(node, "executeFlowOverrideConfig must be JSON") from exc
        if isinstance(loaded, dict):
            return loaded
    return {}


async def _run_nested_flow(
    ctx: ExecContext,
    node: FlowNode,
    execute_flow: Any,
    run_request: Any,
    selected: str,
    text: str,
    override: dict[str, Any],
    child_flow: Any,
) -> Any:
    variables = dict(ctx.request.variables)
    extra_vars = override.get("vars")
    if isinstance(extra_vars, dict):
        variables.update(extra_vars)
    child_inputs: dict[str, Any] = {
        "question": text,
        "history": list(ctx.chat_history),
        "chatType": "INTERNAL",
    }
    for key in ("question", "history", "form", "webhook"):
        if key in override:
            child_inputs[key] = override[key]
    resume = None
    nested = ctx.nested_resume if isinstance(ctx.nested_resume, dict) else None
    if nested and nested.get("nodeId") == node.id and isinstance(ctx.human_input, dict):
        human = dict(ctx.human_input)
        human["startNodeId"] = nested.get("startNodeId")
        ctx.human_input = None
        ctx.nested_resume = None
        resume = {"humanInput": human}
    child = run_request.model_validate(
        {
            "runId": f"{ctx.request.run_id}:{selected}",
            "chatflowId": selected,
            "sessionId": ctx.request.session_id,
            "workspaceId": ctx.request.workspace_id,
            "orgId": ctx.request.org_id,
            "flow": child_flow,
            "inputs": child_inputs,
            "secrets": dict(ctx.request.secrets),
            "credentials": dict(ctx.request.credentials),
            "vars": variables,
            "flows": dict(ctx.request.flows),
            "knowledge": dict(ctx.request.knowledge),
            "componentUrl": ctx.request.component_url,
            "callbackToken": ctx.request.callback_token,
            "resume": resume,
        }
    )
    return await execute_flow(
        child,
        ctx.sink,
        store=ctx.store,
        llm=ctx.llm,
        http=ctx.http,
        js_runner=ctx.js_runner,
        skip_ssrf=ctx.skip_ssrf,
        max_steps=max(1, ctx.max_steps - ctx.steps),
        nested=True,
        depth=ctx.flow_depth + 1,
    )


async def _run_prediction(
    ctx: ExecContext,
    node: FlowNode,
    prediction_base: Any,
    node_base: str,
    selected: str,
    text: str,
    override: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    from binex.flowise.components import PREDICTION_PATH

    base, check = prediction_base(node_base, ctx.request)
    if not base:
        raise NodeExecutionError(
            node,
            "Execute Flow child is not in the run payload and no Flowise base URL is set. "
            "Pass flows[chatflowId] or FLOWISE_BASE_URL / executeFlowBaseURL.",
        )
    url = f"{base}{PREDICTION_PATH}/{selected}"
    if check and not ctx.skip_ssrf:
        try:
            validate_url(url)
        except BlockedURLError as exc:
            raise NodeExecutionError(node, str(exc)) from exc
    credential = ctx.credential_for(node.id, node.credential)
    headers = {"Content-Type": "application/json", "flowise-tool": "true"}
    token = credential.get("chatflowApiKey") or credential.get("apiKey")
    if isinstance(token, str) and token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "question": text,
        "chatId": str(ctx.request.inputs.get("chatId") or ctx.request.session_key),
        "overrideConfig": override,
    }
    try:
        response = await ctx.http.post(url, json=body, headers=headers, timeout=120)
    except Exception as exc:
        raise NodeExecutionError(node, f"Execute Flow request failed: {exc}") from exc
    if response.status_code >= 400:
        raise NodeExecutionError(
            node, f"Execute Flow returned {response.status_code}: {response.text[:300]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise NodeExecutionError(node, "Execute Flow response was not JSON") from exc
    if not isinstance(data, dict):
        return str(data), {}
    content = data.get("text")
    if not isinstance(content, str) or not content:
        if data.get("json") is not None:
            content = "```json\n" + json.dumps(data["json"], default=str) + "\n```"
        else:
            content = json.dumps(data, default=str)
    extra: dict[str, Any] = {}
    for key in ("sourceDocuments", "usedTools", "fileAnnotations"):
        if key in data:
            extra[key] = data[key]
    artifacts = data.get("artifacts")
    if isinstance(artifacts, list):
        extra["artifacts"] = artifacts
        files = files_from_artifacts([item for item in artifacts if isinstance(item, dict)])
        if files:
            extra["files"] = files
    return content, extra


HANDLERS = {
    "startAgentflow": handle_start,
    "llmAgentflow": handle_llm,
    "agentAgentflow": handle_agent,
    "conditionAgentflow": handle_condition,
    "conditionAgentAgentflow": handle_condition_agent,
    "loopAgentflow": handle_loop,
    "humanInputAgentflow": handle_human,
    "httpAgentflow": handle_http,
    "customFunctionAgentflow": handle_custom_function,
    "directReplyAgentflow": handle_direct_reply,
    "a2aAgentflow": handle_a2a,
    "caoAgentflow": handle_cao,
    "toolAgentflow": handle_tool,
    "retrieverAgentflow": handle_retriever,
    "executeFlowAgentflow": handle_execute_flow,
}

__all__ = ["DECISION_NODES", "HANDLERS", "NodeResult", "incoming_text", "output_text"]
