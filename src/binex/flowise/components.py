"""Run a Flowise component without shipping the Flowise catalog.

Resolution order for a tool, retriever, vector store, or MCP server:

1. A small local Python equivalent (calculator, HTTP GET/POST, current time,
   files under ``/workspace``, custom-tool JavaScript).
2. Data already on the run: ``knowledge`` chunks, or an MCP HTTP URL.
3. ``POST {componentUrl}/api/v1/internal/components/execute``. That route is
   not in Flowise today. The server design note in ``docs/flowise-sandbox.md``
   describes it. Chat models do not use this path; they go through LiteLLM.

The component callback is the operator-configured Flowise server, so it is not
passed through the SSRF check. MCP URLs and ``requestsGet`` / ``requestsPost``
are flow-authored and are checked.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from binex.flowise.context import ExecContext
from binex.tools.builtins import calculator
from binex.tools.ssrf import BlockedURLError, validate_url

COMPONENT_PATH = "/api/v1/internal/components/execute"
PREDICTION_PATH = "/api/v1/prediction"
_WORKSPACE = Path("/workspace").resolve()
_TOKEN = re.compile(r"\w+")

# Flowise tool ``name`` values that have a local implementation.
LOCAL_TOOLS = frozenset(
    {
        "calculator",
        "calculatorTool",
        "requestsGet",
        "requestsPost",
        "requestGet",
        "requestPost",
        "currentDateTime",
        "getCurrentDateTime",
        "getDateTime",
        "currentDate",
        "readFile",
        "writeFile",
        "customTool",
    }
)


class ComponentError(RuntimeError):
    """A Flowise component could not be executed."""


@dataclass
class ComponentResult:
    content: str
    source_documents: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


def component_base(request: Any) -> str:
    configured = getattr(request, "component_url", "") or ""
    env_url = os.environ.get("FLOWISE_COMPONENT_URL") or os.environ.get("FLOWISE_BASE_URL")
    base = configured or env_url
    return str(base or "").strip().rstrip("/")


def callback_token(request: Any) -> str:
    configured = getattr(request, "callback_token", "") or ""
    return str(configured or os.environ.get("FLOWISE_CALLBACK_TOKEN") or "").strip()


def prediction_base(node_base: str, request: Any) -> tuple[str, bool]:
    """Return ``(base, check_ssrf)``. Operator URLs skip the SSRF check."""
    if node_base.strip():
        return node_base.strip().rstrip("/"), True
    env = (os.environ.get("FLOWISE_BASE_URL") or "").strip().rstrip("/")
    if env:
        return env, False
    return component_base(request), False


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}
        if isinstance(loaded, dict):
            return loaded
        return {"input": loaded}
    return {}


def parse_input_args(raw: Any) -> dict[str, Any]:
    """Flowise ``[{inputArgName, inputArgValue}]`` rows, JSON values included."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}
    if isinstance(raw, dict):
        return dict(raw)
    args: dict[str, Any] = {}
    if not isinstance(raw, list):
        return args
    for item in raw:
        if not isinstance(item, dict) or not item.get("inputArgName"):
            continue
        args[str(item["inputArgName"])] = _coerce_arg(item.get("inputArgValue"))
    return args


def function_schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = properties or {"input": {"type": "string"}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": fields,
                "required": list(fields),
            },
        },
    }


def local_tool_schema(name: str) -> dict[str, Any] | None:
    folded = name.strip()
    if folded in {"calculator", "calculatorTool"}:
        return function_schema(
            folded,
            "Evaluate a mathematical expression.",
            {"expression": {"type": "string"}},
        )
    if folded in {"requestsGet", "requestGet"}:
        return function_schema(folded, "HTTP GET a public URL.", {"url": {"type": "string"}})
    if folded in {"requestsPost", "requestPost"}:
        return function_schema(
            folded,
            "HTTP POST JSON to a public URL.",
            {"url": {"type": "string"}, "body": {"type": "string"}},
        )
    if folded in {"currentDateTime", "getCurrentDateTime", "getDateTime", "currentDate"}:
        return function_schema(folded, "Current UTC time in ISO 8601.", {})
    if folded == "readFile":
        return function_schema(
            folded, "Read a UTF-8 file under /workspace.", {"path": {"type": "string"}}
        )
    if folded == "writeFile":
        return function_schema(
            folded,
            "Write a UTF-8 file under /workspace.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
        )
    if folded == "customTool":
        return function_schema(
            folded, "Run the custom tool JavaScript.", {"input": {"type": "string"}}
        )
    return None


def safe_function_name(prefix: str, raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw).strip("_") or "store"
    return f"{prefix}_{cleaned}"[:64]


def knowledge_chunks(request: Any, store_id: str) -> list[dict[str, Any]] | None:
    bucket = getattr(request, "knowledge", None) or {}
    if not isinstance(bucket, dict) or store_id not in bucket:
        return None
    raw = bucket[store_id]
    if isinstance(raw, dict):
        raw = raw.get("documents") or raw.get("docs") or raw.get("chunks") or []
    if not isinstance(raw, list):
        return []
    docs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            docs.append({"pageContent": item, "metadata": {}})
        elif isinstance(item, dict):
            text = item.get("pageContent")
            if text is None:
                text = item.get("content") or ""
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            docs.append({"pageContent": str(text), "metadata": metadata})
    return docs


def lexical_search(docs: list[dict[str, Any]], query: str, limit: int = 4) -> list[dict[str, Any]]:
    """Rank chunks by overlapping words. Empty overlap returns nothing."""
    tokens = set(_TOKEN.findall(query.lower()))
    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in docs:
        words = set(_TOKEN.findall(str(doc.get("pageContent") or "").lower()))
        score = len(tokens & words)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in scored[:limit]]


async def invoke_named_tool(
    ctx: ExecContext,
    *,
    name: str,
    args: dict[str, Any],
    credential: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ComponentResult:
    """Run one Flowise tool component."""
    settings = dict(config or {})
    mcp_url = _first_text(
        settings.get("mcpServerUrl"),
        settings.get("mcpUrl"),
        settings.get("serverUrl"),
    )
    if name in {"customMCP", "mcpTool"} or mcp_url:
        tool_name = str(settings.get("toolName") or args.get("name") or name)
        if name in {"customMCP", "mcpTool"} and tool_name in {"customMCP", "mcpTool"}:
            tool_name = str(args.get("tool") or args.get("name") or name)
        return await invoke_mcp(
            ctx,
            url=mcp_url or "",
            tool_name=tool_name,
            args=args,
            credential=credential,
        )
    if name in LOCAL_TOOLS or name in {"calculatorTool", "requestGet", "requestPost"}:
        return await _local_tool(ctx, name, args, settings)
    return await call_component(
        ctx,
        kind="tool",
        component=name,
        inputs=settings,
        credential=credential or {},
        tool_args=args,
        query=str(args.get("query") or args.get("input") or ""),
    )


async def retrieve_documents(
    ctx: ExecContext,
    *,
    store_id: str,
    query: str,
    kind: str = "retriever",
    component: str = "",
    inputs: dict[str, Any] | None = None,
    credential: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Document-store or vector-store retrieval."""
    inline = knowledge_chunks(ctx.request, store_id)
    if inline is not None:
        return lexical_search(inline, query)
    result = await call_component(
        ctx,
        kind=kind,
        component=component or store_id,
        inputs={**(inputs or {}), "storeId": store_id},
        credential=credential or {},
        tool_args={},
        query=query,
    )
    if result.source_documents:
        return result.source_documents
    if result.content:
        return [{"pageContent": result.content, "metadata": {"storeId": store_id}}]
    return []


async def invoke_mcp(
    ctx: ExecContext,
    *,
    url: str,
    tool_name: str,
    args: dict[str, Any],
    credential: dict[str, Any] | None = None,
) -> ComponentResult:
    if not url:
        return await call_component(
            ctx,
            kind="mcp",
            component=tool_name,
            inputs={},
            credential=credential or {},
            tool_args=args,
            query="",
        )
    _check_ssrf(ctx, url)
    headers = {"Content-Type": "application/json"}
    token = _first_text((credential or {}).get("token"), (credential or {}).get("bearerToken"))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
    }
    try:
        response = await ctx.http.post(url, json=body, headers=headers, timeout=60)
    except Exception as exc:
        raise ComponentError(f"MCP request failed: {exc}") from exc
    if response.status_code >= 400:
        raise ComponentError(f"MCP {response.status_code} from {url}: {response.text[:300]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ComponentError("MCP response was not JSON") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise ComponentError(f"MCP error: {payload['error']}")
    result = payload.get("result") if isinstance(payload, dict) else payload
    return ComponentResult(content=_mcp_text(result))


async def call_component(
    ctx: ExecContext,
    *,
    kind: str,
    component: str,
    inputs: dict[str, Any],
    credential: dict[str, Any],
    tool_args: dict[str, Any],
    query: str,
) -> ComponentResult:
    base = component_base(ctx.request)
    if not base:
        raise ComponentError(
            f"No local implementation for Flowise {kind} '{component}'. "
            "Set componentUrl, FLOWISE_COMPONENT_URL, or FLOWISE_BASE_URL so the sandbox "
            f"can POST {COMPONENT_PATH}. The Flowise server must implement that route."
        )
    url = f"{base}{COMPONENT_PATH}"
    headers = {"Content-Type": "application/json"}
    token = callback_token(ctx.request)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "kind": kind,
        "component": component,
        "inputs": inputs,
        "credential": credential,
        "toolArgs": tool_args,
        "query": query,
        "chatflowId": ctx.request.chatflow_id,
        "sessionId": ctx.request.session_key,
    }
    try:
        response = await ctx.http.post(url, json=body, headers=headers, timeout=120)
    except Exception as exc:
        raise ComponentError(f"Component callback failed: {exc}") from exc
    if response.status_code >= 400:
        raise ComponentError(
            f"Component endpoint returned {response.status_code}: {response.text[:300]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ComponentError("Component endpoint did not return JSON") from exc
    if not isinstance(data, dict):
        return ComponentResult(content=str(data))
    content = data.get("content")
    if content is None:
        content = data.get("text")
    if not isinstance(content, str):
        content = json.dumps(content if content is not None else data, default=str)
    documents = data.get("sourceDocuments")
    artifacts = data.get("artifacts")
    return ComponentResult(
        content=content,
        source_documents=[item for item in documents if isinstance(item, dict)]
        if isinstance(documents, list)
        else [],
        artifacts=[item for item in artifacts if isinstance(item, dict)]
        if isinstance(artifacts, list)
        else [],
    )


def files_from_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for item in artifacts:
        content = item.get("content")
        if isinstance(content, str):
            files.append(
                {
                    "name": str(item.get("name") or item.get("fileName") or "artifact.txt"),
                    "mimeType": str(item.get("mimeType") or "text/plain"),
                    "content": content,
                }
            )
    return files


async def _local_tool(
    ctx: ExecContext,
    name: str,
    args: dict[str, Any],
    config: dict[str, Any],
) -> ComponentResult:
    if name in {"calculator", "calculatorTool"}:
        expression = str(
            args.get("expression")
            or args.get("formula")
            or args.get("input")
            or args.get("query")
            or ""
        )
        if not expression:
            raise ComponentError("calculator requires an expression")
        return ComponentResult(content=str(calculator(expression)))
    if name in {"requestsGet", "requestGet", "requestsPost", "requestPost"}:
        return ComponentResult(content=await _requests_tool(ctx, name, args))
    if name in {"currentDateTime", "getCurrentDateTime", "getDateTime", "currentDate"}:
        return ComponentResult(content=datetime.now(UTC).isoformat())
    if name == "readFile":
        path = _workspace_file(str(args.get("path") or args.get("file") or ""))
        if not path.is_file():
            raise ComponentError(f"File not found: {path}")
        return ComponentResult(content=path.read_text(encoding="utf-8"))
    if name == "writeFile":
        path = _workspace_file(str(args.get("path") or args.get("file") or ""))
        content = str(args.get("content") or args.get("text") or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ComponentResult(content=f"wrote {len(content.encode('utf-8'))} bytes")
    if name == "customTool":
        code = config.get("javascriptFunction") or config.get("customToolFunc") or args.get("code")
        code = str(code or "")
        if not code.strip():
            raise ComponentError("customTool requires javascriptFunction on the tool config")
        return ComponentResult(content=await _custom_tool_js(ctx, code, args))
    raise ComponentError(f"No local implementation for '{name}'")


async def _requests_tool(ctx: ExecContext, name: str, args: dict[str, Any]) -> str:
    url = str(args.get("url") or args.get("uri") or "")
    if not url:
        raise ComponentError(f"{name} requires url")
    _check_ssrf(ctx, url)
    try:
        if name in {"requestsPost", "requestPost"}:
            body = args.get("body") if "body" in args else args.get("data")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    pass
            response = await ctx.http.post(url, json=body, timeout=60)
        else:
            response = await ctx.http.get(url, timeout=60)
    except Exception as exc:
        raise ComponentError(f"{name} failed: {exc}") from exc
    if response.status_code >= 400:
        raise ComponentError(f"{name} returned {response.status_code}: {response.text[:300]}")
    return response.text


async def _custom_tool_js(ctx: ExecContext, code: str, args: dict[str, Any]) -> str:
    from binex.flowise.custom_function import run_javascript

    runner = ctx.js_runner or run_javascript
    result = await runner(
        code,
        flow_input=args,
        flow={"state": ctx.state, "chatflowId": ctx.request.chatflow_id},
        variables=ctx.request.variables,
        extra={},
    )
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _workspace_file(raw: str) -> Path:
    if not raw.strip():
        raise ComponentError("A path under /workspace is required")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _WORKSPACE / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(_WORKSPACE):
        raise ComponentError(f"Path '{raw}' is outside /workspace")
    return resolved


def _check_ssrf(ctx: ExecContext, url: str) -> None:
    if ctx.skip_ssrf:
        return
    try:
        validate_url(url)
    except BlockedURLError as exc:
        raise ComponentError(str(exc)) from exc


def _coerce_arg(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped[:1] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _mcp_text(result: Any) -> str:
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            ]
            return "\n".join(part for part in parts if part)
        if isinstance(content, str):
            return content
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)
