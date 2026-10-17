"""Resolve Flowise ``{{ ... }}`` templates against the current run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_TEMPLATE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
_LEADING_HTML = re.compile(
    r"^\s*<(?:p|div|h[1-6]|ul|ol|blockquote|pre|span|table)\b",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")


@dataclass
class TemplateContext:
    """Values a template can read. Rebuilt as the run advances."""

    question: str = ""
    form: dict[str, Any] = field(default_factory=dict)
    webhook: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    chatflow_id: str = ""
    session_id: str = ""
    chat_id: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    executed: list[dict[str, Any]] = field(default_factory=list)
    loop_counts: dict[str, int] = field(default_factory=dict)
    iteration: dict[str, Any] | None = None
    uploads: str = ""
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    runtime_messages_length: int = 0


def get_path(obj: Any, path: str) -> Any:
    """Read a dotted path. Missing keys return None."""
    current = obj
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _flow_view(ctx: TemplateContext) -> dict[str, Any]:
    question = ctx.question
    flow_input: Any = question
    if not question and ctx.form:
        flow_input = ctx.form
    elif not question and ctx.webhook:
        flow_input = ctx.webhook
    return {
        "input": flow_input,
        "state": ctx.state,
        "chatflowId": ctx.chatflow_id,
        "chatflowid": ctx.chatflow_id,
        "sessionId": ctx.session_id,
        "chatId": ctx.chat_id,
        "chatHistory": ctx.chat_history,
        "runtimeChatHistoryLength": ctx.runtime_messages_length,
    }


def _latest_executed(ctx: TemplateContext, node_id: str) -> dict[str, Any] | None:
    for item in reversed(ctx.executed):
        if item.get("nodeId") == node_id:
            data = item.get("data")
            return data if isinstance(data, dict) else None
    return None


def _loop_count(ctx: TemplateContext) -> int:
    for item in reversed(ctx.executed):
        data = item.get("data")
        if isinstance(data, dict) and data.get("name") == "loopAgentflow":
            node_id = str(item.get("nodeId") or "")
            return int(ctx.loop_counts.get(node_id, 0))
    return 0


def _chat_history_text(ctx: TemplateContext) -> str:
    lines: list[str] = []
    for message in ctx.chat_history:
        role = str(message.get("role") or "user")
        content = message.get("content")
        lines.append(f"{role}: {content if isinstance(content, str) else json.dumps(content)}")
    return "\n".join(lines)


def lookup(reference: str, ctx: TemplateContext) -> Any:
    """Resolve one template body. None means leave the template in place."""
    if reference == "question":
        if ctx.uploads:
            return f"{ctx.uploads}\n\n{ctx.question}" if ctx.question else ctx.uploads
        return ctx.question
    if reference == "file_attachment":
        return ctx.uploads
    if reference == "chat_history":
        return _chat_history_text(ctx)
    if reference == "runtime_messages_length":
        return ctx.runtime_messages_length
    if reference == "loop_count":
        return _loop_count(ctx)
    if reference == "current_date_time":
        return datetime.now(UTC).isoformat()
    if reference.startswith("$form."):
        return get_path(ctx.form, reference.removeprefix("$form."))
    if reference.startswith("$webhook."):
        return get_path(ctx.webhook, reference.removeprefix("$webhook."))
    if reference.startswith("$vars."):
        return get_path(ctx.variables, reference.removeprefix("$vars."))
    if reference.startswith("$flow."):
        return get_path(_flow_view(ctx), reference.removeprefix("$flow."))
    if reference == "$iteration" or reference.startswith("$iteration."):
        if not ctx.iteration:
            return None
        value = ctx.iteration.get("value")
        if reference == "$iteration":
            return value
        path = reference.removeprefix("$iteration.")
        if isinstance(value, dict) and path in value:
            return get_path(value, path)
        if path in ctx.iteration and path != "value":
            return ctx.iteration.get(path)
        if isinstance(value, dict):
            return get_path(value, path)
        return value if isinstance(value, str) else None

    output_match = re.fullmatch(r"(.*?)\.output\.(.+)", reference)
    if output_match:
        node_id = output_match.group(1).replace("\\", "")
        data = _latest_executed(ctx, node_id)
        if data is None:
            return None
        output = data.get("output")
        if not isinstance(output, dict):
            return None
        return get_path(output, output_match.group(2).strip())

    node_id = reference.replace("\\", "")
    data = _latest_executed(ctx, node_id)
    if data is None:
        return None
    output = data.get("output")
    if not isinstance(output, dict):
        return None
    if "content" in output:
        return output.get("content")
    http = output.get("http")
    if isinstance(http, dict) and "data" in http:
        return http.get("data")
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, default=str)


def _strip_editor_html(value: str) -> str:
    if not _LEADING_HTML.match(value):
        return value
    return _TAG.sub("", value)


def resolve_value(value: Any, ctx: TemplateContext) -> Any:
    """Resolve templates in strings, and walk lists and dicts."""
    if isinstance(value, list):
        return [resolve_value(item, ctx) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item, ctx) for key, item in value.items()}
    if not isinstance(value, str) or "{{" not in value:
        return value

    text = _strip_editor_html(value)
    match = _TEMPLATE.fullmatch(text.strip())
    if match:
        found = lookup(match.group(1).strip(), ctx)
        if found is None:
            return text
        return found

    resolved = text
    for item in _TEMPLATE.finditer(text):
        found = lookup(item.group(1).strip(), ctx)
        if found is None:
            continue
        resolved = resolved.replace(item.group(0), _stringify(found), 1)
    return resolved


def replace_output_placeholder(state: dict[str, Any], output_text: str) -> dict[str, Any]:
    """Replace ``{{ output }}`` left in flow-state updates."""
    updated: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, str) and "{{ output }}" in value:
            updated[key] = value.replace("{{ output }}", output_text)
        else:
            updated[key] = value
    return updated
