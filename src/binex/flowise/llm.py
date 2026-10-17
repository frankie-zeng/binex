"""LiteLLM calls for Flowise LLM, Agent, and Condition Agent nodes.

Completion kwargs match ``LLMAdapter`` (model, api_base, api_key, temperature,
max_tokens). Messages stay as the Agentflow transcript instead of being folded
into a single user artifact. When an Agent node sets ``binexTools``, the call
goes through ``LLMAdapter`` so Binex tool URIs actually run.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import litellm

from binex.adapters.llm import LLMAdapter
from binex.models.artifact import Artifact, Lineage
from binex.models.task import TaskNode

_PROVIDER_PREFIX = {
    "chatOpenAI": "openai",
    "azureChatOpenAI": "azure",
    "chatAnthropic": "anthropic",
    "chatGoogleGenerativeAI": "gemini",
    "chatGoogle": "gemini",
    "chatMistralAI": "mistral",
    "chatGroq": "groq",
    "chatOllama": "ollama",
    "ollama": "ollama",
    "chatDeepseek": "deepseek",
    "chatTogetherAI": "together_ai",
    "chatOpenRouter": "openrouter",
}


@dataclass
class LLMCompletion:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_model: str = ""


class LLMCaller(Protocol):
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        api_key: str | None,
        api_base: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMCompletion: ...


@dataclass
class ResolvedModel:
    model: str
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    component: str = ""


def resolve_model(component: str | None, config: dict[str, Any] | None) -> ResolvedModel:
    """Turn a Flowise model component plus modelConfig into a LiteLLM id."""
    settings = dict(config or {})
    component_name = str(component or "")
    model_name = settings.get("modelName") or settings.get("model") or settings.get("model_name")
    model_name = str(model_name) if model_name else ""

    if "/" in component_name or _looks_like_model(component_name):
        model = component_name
    elif model_name and "/" in model_name:
        model = model_name
    elif model_name:
        prefix = _PROVIDER_PREFIX.get(component_name)
        model = f"{prefix}/{model_name}" if prefix else model_name
    else:
        raise RuntimeError(
            "LLM node is missing a model. Set llmModel to a LiteLLM id "
            "(for example openai/gpt-4o-mini) or set modelName on the model config. "
            "The Flowise server should pass that config with the run, not bake a key "
            "into the sandbox image."
        )

    temperature = _optional_float(settings.get("temperature"))
    max_tokens = _optional_int(settings.get("maxTokens") or settings.get("max_tokens"))
    api_key = _first_text(
        settings.get("apiKey"),
        settings.get("openAIApiKey"),
        settings.get("anthropicApiKey"),
    )
    api_base = _first_text(
        settings.get("baseUrl"),
        settings.get("baseURL"),
        settings.get("apiBase"),
        settings.get("api_base"),
    )
    return ResolvedModel(
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        component=component_name,
    )


def _looks_like_model(value: str) -> bool:
    if not value or value.endswith("Agentflow"):
        return False
    prefixes = ("gpt-", "claude", "gemini", "ollama/", "o1", "o3", "o4")
    return value.startswith(prefixes)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def completion_kwargs(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Same optional keys LLMAdapter forwards to litellm.acompletion."""
    kwargs: dict[str, Any] = {"model": resolved.model, "messages": messages}
    optional = {
        "api_base": resolved.api_base,
        "api_key": resolved.api_key,
        "temperature": resolved.temperature,
        "max_tokens": resolved.max_tokens,
    }
    for key, value in optional.items():
        if value is not None:
            kwargs[key] = value
    return kwargs


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


class LiteLLMCaller:
    """Default caller. Tests replace this with a fake."""

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        api_key: str | None,
        api_base: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMCompletion:
        resolved = ResolvedModel(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await litellm.acompletion(**completion_kwargs(resolved, messages))
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return LLMCompletion(
            content=_message_text(getattr(message, "content", "")),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_model=str(getattr(response, "model", "") or model),
        )


async def complete_with_binex_tools(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    tool_uris: list[str],
    *,
    run_id: str,
    node_id: str,
) -> LLMCompletion:
    """Run an Agent node that listed ``binexTools`` through LLMAdapter."""
    system_parts = [str(item["content"]) for item in messages if item.get("role") == "system"]
    user_parts = [
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in messages
        if item.get("role") != "system"
    ]
    adapter = LLMAdapter(
        resolved.model,
        api_base=resolved.api_base,
        api_key=resolved.api_key,
        temperature=resolved.temperature,
        max_tokens=resolved.max_tokens,
    )
    task = TaskNode(
        id=f"task_{uuid.uuid4().hex[:12]}",
        run_id=run_id,
        node_id=node_id,
        agent=f"llm://{resolved.model}",
        system_prompt="\n".join(system_parts) or None,
        inputs={"prompt": "\n".join(user_parts)},
        tools=list(tool_uris),
    )
    artifact = Artifact(
        id=f"art_{uuid.uuid4().hex[:12]}",
        run_id=run_id,
        type="prompt",
        content="\n".join(user_parts),
        lineage=Lineage(produced_by=node_id),
    )
    result = await adapter.execute(task, [artifact], run_id)
    content = ""
    if result.artifacts:
        raw = result.artifacts[0].content
        content = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    return LLMCompletion(content=content, model=resolved.model)


def structured_schema(fields: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON schema for an Agentflow structured-output field list."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for item in fields:
        key = str(item.get("key") or "")
        if not key:
            continue
        kind = str(item.get("type") or "string")
        schema: dict[str, Any]
        if kind == "number":
            schema = {"type": "number"}
        elif kind == "boolean":
            schema = {"type": "boolean"}
        elif kind == "stringArray":
            schema = {"type": "array", "items": {"type": "string"}}
        elif kind == "enum":
            values = [part.strip() for part in str(item.get("enumValues") or "").split(",")]
            schema = {"type": "string", "enum": [part for part in values if part]}
        elif kind == "jsonArray":
            schema = {"type": "array"}
            raw_schema = item.get("jsonSchema")
            if isinstance(raw_schema, str) and raw_schema.strip():
                try:
                    schema = json.loads(raw_schema)
                except json.JSONDecodeError:
                    schema = {"type": "array"}
        else:
            schema = {"type": "string"}
        if item.get("description"):
            schema["description"] = str(item["description"])
        properties[key] = schema
        required.append(key)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response, including fenced blocks."""
    candidate = text.strip()
    fence = re_fence(candidate)
    if fence is not None:
        candidate = fence
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response did not contain a JSON object")
    loaded = json.loads(candidate[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("JSON response was not an object")
    return loaded


def re_fence(text: str) -> str | None:
    marker = "```"
    if marker not in text:
        return None
    start = text.find(marker)
    rest = text[start + len(marker) :]
    if rest.startswith("json"):
        rest = rest[4:]
    end = rest.find(marker)
    if end == -1:
        return None
    return rest[:end].strip()


@dataclass
class ChatRequest:
    """Shared inputs for LLM and Agent nodes."""

    model_component: str
    model_config: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    user_message: str = ""
    memory: bool = True
    memory_type: str = "allMessages"
    memory_window: int = 20
    structured: list[dict[str, Any]] = field(default_factory=list)
    binex_tools: list[str] = field(default_factory=list)
    state_updates: list[dict[str, Any]] = field(default_factory=list)
