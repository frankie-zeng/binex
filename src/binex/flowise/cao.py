"""Flowise CAO node client.

Speaks the fork's cao-server contract (session create, ``/terminals/run-step``,
assign via terminal input, inbox send_message). Binex's YAML ``cao://`` adapter
uses a different handoff sequence and is not called here, so stored turns match
what the Flowise CAO node records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

import httpx


class CaoProtocolError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        request: list[dict[str, Any]] | None = None,
        response: Any = None,
        session_name: str | None = None,
        terminal_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.request = request or []
        self.response = response
        self.session_name = session_name
        self.terminal_id = terminal_id


CAO_PROVIDERS = frozenset(
    {
        "claude_code",
        "codex",
        "cursor_cli",
        "q_cli",
        "kiro_cli",
        "copilot_cli",
        "opencode_cli",
        "kimi_cli",
        "hermes",
        "antigravity_cli",
    }
)
CAO_MODES = frozenset({"handoff", "assign", "send_message"})


@dataclass
class CaoConfig:
    server_url: str
    bearer_token: str | None
    provider: str
    agent_profile: str
    mode: str
    task: str
    explicit_session_name: str | None = None
    reuse_terminal: bool = True
    teardown: bool = False
    timeout_seconds: int = 600
    wait: bool = True
    working_directory: str | None = None
    caller_id: str | None = None
    receiver_id: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    use_worktree: bool = False
    memory_manager: bool = False
    env_vars: dict[str, str] | None = None
    claim_id: str | None = None
    group: Any = None
    metadata: Any = None
    shape_prompt: bool = True
    inject_sender_id: bool = False
    job_id: str | None = None
    poll_interval_seconds: float = 2


def assert_cao_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CaoProtocolError("CAO server URL must be an http(s) URL")
    return value.rstrip("/")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any, fallback: bool) -> bool:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return fallback


def _parse_object(value: Any, label: str) -> dict[str, str] | None:
    if isinstance(value, dict):
        return {str(key): "" if item is None else str(item) for key, item in value.items()}
    text = _text(value)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CaoProtocolError(f"{label} must be JSON") from exc
    if not isinstance(parsed, dict):
        raise CaoProtocolError(f"{label} must be a JSON object")
    return {str(key): "" if item is None else str(item) for key, item in parsed.items()}


def _parse_json(value: Any, label: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = _text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CaoProtocolError(f"{label} must be JSON") from exc


def _to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        return value - 0x100000000
    return value


def session_name_for_flow(chatflow_id: str, node_id: str) -> str:
    """Same 32-bit hash the Flowise CAO node uses for a stable session name."""
    raw = f"{chatflow_id}:{node_id}"
    hash_value = 0
    for char in raw:
        hash_value = _to_int32(_to_int32(31) * _to_int32(hash_value) + ord(char))
    hex_value = format(hash_value & 0xFFFFFFFF, "x")
    node = "".join(char for char in node_id if char.isalnum())[:8]
    return f"flow{hex_value}{node}"[:32]


def sanitize_session_name(value: str) -> str:
    cleaned = "".join(char for char in value if char.isalnum() or char in "_-")[:32]
    if not cleaned:
        raise CaoProtocolError("CAO session name must contain letters or numbers")
    return cleaned


def normalize_cao_config(
    inputs: dict[str, Any],
    *,
    env: dict[str, str],
    credential: dict[str, Any],
) -> CaoConfig:
    server_url = assert_cao_url(
        _text(inputs.get("caoServerUrl"))
        or _text(credential.get("serverUrl"))
        or _text(env.get("CAO_SERVER_URL"))
        or _text(credential.get("caoServerUrl"))
    )
    provider = _text(inputs.get("caoProvider")) or "claude_code"
    if provider not in CAO_PROVIDERS:
        raise CaoProtocolError(f"Unsupported CAO provider: {provider}")
    mode = _text(inputs.get("caoMode")) or "handoff"
    if mode not in CAO_MODES:
        raise CaoProtocolError(f"Unsupported CAO mode: {mode}")
    task = _text(inputs.get("caoTask"))
    if not task:
        raise CaoProtocolError("CAO node requires a task")
    behavior = _text(inputs.get("caoWaitBehavior")) or "default"
    if behavior == "wait":
        wait = True
    elif behavior == "return":
        wait = False
    else:
        wait = mode == "handoff"
    timeout_seconds = _positive_int(
        inputs.get("caoTimeoutSeconds") or env.get("CAO_TIMEOUT_SECONDS"), 600
    )
    poll_interval = _positive_float(inputs.get("caoPollIntervalSeconds"), 2)
    claim_id = _text(inputs.get("caoClaimId"))
    if claim_id and (
        len(claim_id) != 32 or any(char not in "0123456789abcdef" for char in claim_id)
    ):
        raise CaoProtocolError("CAO claim id must be 32 lowercase hex characters")
    job_id = _text(inputs.get("caoJobId"))
    if job_id and (len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id)):
        raise CaoProtocolError("CAO job id must be 32 lowercase hex characters")
    tools = [
        item.strip() for item in _text(inputs.get("caoAllowedTools")).split(",") if item.strip()
    ]
    return CaoConfig(
        server_url=server_url,
        bearer_token=_text(inputs.get("caoBearerToken"))
        or _text(credential.get("bearerToken"))
        or _text(env.get("CAO_BEARER_TOKEN"))
        or None,
        provider=provider,
        agent_profile=_text(inputs.get("caoAgentProfile"))
        or _text(env.get("CAO_AGENT_PROFILE"))
        or "developer",
        mode=mode,
        task=task,
        explicit_session_name=_text(inputs.get("caoSessionName")) or None,
        reuse_terminal=_as_bool(inputs.get("caoReuseTerminal"), True),
        teardown=_as_bool(inputs.get("caoTeardown"), False),
        timeout_seconds=timeout_seconds,
        wait=wait,
        working_directory=_text(inputs.get("caoWorkingDirectory")) or None,
        caller_id=_text(inputs.get("caoCallerId")) or None,
        receiver_id=_text(inputs.get("caoReceiverId")) or None,
        allowed_tools=tools,
        use_worktree=_as_bool(inputs.get("caoUseWorktree"), False),
        memory_manager=_as_bool(inputs.get("caoMemoryManager"), False),
        env_vars=_parse_object(inputs.get("caoEnvVars"), "Env vars"),
        claim_id=claim_id or None,
        group=_parse_json(inputs.get("caoGroup"), "Group"),
        metadata=_parse_json(inputs.get("caoMetadata"), "Metadata"),
        shape_prompt=_as_bool(inputs.get("caoShapePrompt"), True),
        inject_sender_id=_as_bool(inputs.get("caoInjectSenderId"), False),
        job_id=job_id or None,
        poll_interval_seconds=poll_interval,
    )


def _positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def resolve_session_name(
    config: CaoConfig, persisted: dict[str, Any], chatflow_id: str, node_id: str
) -> str:
    if config.explicit_session_name:
        return sanitize_session_name(config.explicit_session_name)
    saved = _text(persisted.get("caoSessionName"))
    if saved:
        return saved
    return session_name_for_flow(chatflow_id, node_id)


def build_cao_prompt(config: CaoConfig, terminal_id: str | None) -> str:
    text = config.task
    if config.shape_prompt and config.provider == "codex":
        text = f"CAO_TERMINAL_ID={terminal_id or 'pending'}\n{text}"
    if config.inject_sender_id and config.caller_id:
        text = f"{text}\n\n(sender: {config.caller_id})"
    return text


def _detail(data: Any, fallback: str) -> str:
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, dict) and detail.get("message"):
            return str(detail["message"])
        if data.get("message"):
            return str(data["message"])
    return fallback


def _terminal_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in ("terminal_id", "terminalId", "id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _query(pairs: dict[str, str | None]) -> dict[str, str] | None:
    query = {key: value for key, value in pairs.items() if value}
    return query or None


async def execute_cao(
    config: CaoConfig,
    persisted: dict[str, Any],
    *,
    chatflow_id: str,
    node_id: str,
    client: httpx.AsyncClient,
    sleep: Any | None = None,
) -> dict[str, Any]:
    """Run one CAO mode. ``sleep`` is injectable so tests do not wait."""
    import asyncio

    pause = sleep or asyncio.sleep
    session_name = resolve_session_name(config, persisted, chatflow_id, node_id)
    calls: list[dict[str, Any]] = []
    timeout = config.timeout_seconds + 30

    async def call(
        method: str,
        path: str,
        query: dict[str, str] | None = None,
        body: Any = None,
    ) -> httpx.Response:
        recorded: Any = body
        if isinstance(body, dict) and "env_vars" in body:
            recorded = dict(body)
            recorded["env_vars"] = "[redacted]"
        calls.append({"method": method, "path": path, "query": query, "body": recorded})
        headers = {"accept": "application/json", "content-type": "application/json"}
        if config.bearer_token:
            headers["authorization"] = f"Bearer {config.bearer_token}"
        response = await client.request(
            method,
            f"{config.server_url}{path}",
            params=query,
            json=body,
            headers=headers,
            timeout=timeout,
        )
        return response

    async def create_session() -> None:
        query = _query(
            {
                "agent_profile": config.agent_profile,
                "provider": config.provider,
                "session_name": session_name,
                "working_directory": config.working_directory,
                "allowed_tools": ",".join(config.allowed_tools) if config.allowed_tools else None,
                "memory_manager": "true" if config.memory_manager else None,
            }
        )
        body: dict[str, Any] = {}
        if config.env_vars:
            body["env_vars"] = config.env_vars
        if config.group is not None:
            body["group"] = config.group
        if config.metadata is not None:
            body["metadata"] = config.metadata
        response = await call("POST", "/sessions", query, body or None)
        if response.status_code not in {200, 201, 409}:
            payload = _safe_json(response)
            raise CaoProtocolError(
                f"CAO session create failed: {_detail(payload, f'HTTP {response.status_code}')}",
                request=calls,
                response=payload,
                session_name=session_name,
            )

    if config.mode == "handoff" and config.wait:
        await create_session()
        reuse = None
        if config.reuse_terminal and persisted.get("terminalId") and not config.claim_id:
            reuse = str(persisted["terminalId"])
        prompt = build_cao_prompt(config, reuse)
        body = {
            "provider": config.provider,
            "agent": config.agent_profile,
            "prompt": prompt,
            "session_name": session_name,
            "teardown": config.teardown,
            "timeout": config.timeout_seconds,
        }
        if reuse:
            body["reuse_terminal_id"] = reuse
        if config.working_directory:
            body["working_directory"] = config.working_directory
        if config.caller_id:
            body["caller_id"] = config.caller_id
        if config.allowed_tools:
            body["allowed_tools"] = config.allowed_tools
        if config.job_id:
            body["job_id"] = config.job_id
        if config.use_worktree:
            body["use_worktree"] = True
        if config.claim_id and not reuse:
            body["claim_id"] = config.claim_id
        response = await call("POST", "/terminals/run-step", None, body)
        if response.status_code == 404 and reuse:
            body.pop("reuse_terminal_id", None)
            response = await call("POST", "/terminals/run-step", None, body)
        payload = _safe_json(response)
        if response.status_code < 200 or response.status_code >= 300:
            kind = payload.get("detail", {}).get("kind") if isinstance(payload, dict) else None
            message = _detail(payload, f"HTTP {response.status_code}")
            prefix = "CAO handoff timed out" if kind == "timeout" else "CAO handoff failed"
            terminal = None
            if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
                terminal = payload["detail"].get("terminal_id") or reuse
            raise CaoProtocolError(
                f"{prefix}: {message}",
                request=calls,
                response=payload,
                session_name=session_name,
                terminal_id=terminal if isinstance(terminal, str) else reuse,
            )
        return {
            "status": str(payload.get("status") or "completed")
            if isinstance(payload, dict)
            else "completed",
            "content": str(payload.get("last_message") or "") if isinstance(payload, dict) else "",
            "sessionName": session_name,
            "terminalId": (payload.get("terminal_id") if isinstance(payload, dict) else None)
            or reuse,
            "request": calls,
            "response": payload,
        }

    await create_session()
    terminal_id = (
        str(persisted.get("terminalId"))
        if config.reuse_terminal and persisted.get("terminalId")
        else None
    )
    if config.mode == "send_message":
        terminal_id = config.receiver_id or terminal_id
    if not terminal_id:
        terminal_id = await _create_terminal(config, persisted, session_name, call)
    prompt = build_cao_prompt(config, terminal_id)

    if config.mode == "send_message":
        response = await call(
            "POST",
            f"/terminals/{quote(terminal_id, safe='')}/inbox/messages",
            None,
            {"sender_id": config.caller_id or "flowise", "message": prompt},
        )
        payload = _safe_json(response)
        if response.status_code < 200 or response.status_code >= 300:
            raise CaoProtocolError(
                f"CAO send_message failed: {_detail(payload, f'HTTP {response.status_code}')}",
                request=calls,
                response=payload,
                session_name=session_name,
                terminal_id=terminal_id,
            )
        if not config.wait:
            return _accepted(session_name, terminal_id, calls, payload)
        waited = await _poll(config, terminal_id, call, pause)
        return {**waited, "sessionName": session_name, "terminalId": terminal_id, "request": calls}

    response = await call(
        "POST",
        f"/terminals/{quote(terminal_id, safe='')}/input",
        None,
        {"message": prompt},
    )
    payload = _safe_json(response)
    if response.status_code < 200 or response.status_code >= 300:
        raise CaoProtocolError(
            f"CAO input failed: {_detail(payload, f'HTTP {response.status_code}')}",
            request=calls,
            response=payload,
            session_name=session_name,
            terminal_id=terminal_id,
        )
    if not config.wait:
        return _accepted(session_name, terminal_id, calls, payload)
    waited = await _poll(config, terminal_id, call, pause)
    return {**waited, "sessionName": session_name, "terminalId": terminal_id, "request": calls}


async def _create_terminal(
    config: CaoConfig,
    persisted: dict[str, Any],
    session_name: str,
    call: Any,
) -> str:
    claim = None if config.reuse_terminal and persisted.get("terminalId") else config.claim_id
    query = _query(
        {
            "provider": config.provider,
            "agent_profile": config.agent_profile,
            "working_directory": config.working_directory,
            "allowed_tools": ",".join(config.allowed_tools) if config.allowed_tools else None,
            "caller_id": config.caller_id,
            "claim_id": claim,
            "use_worktree": "true" if config.use_worktree else None,
        }
    )
    response = await call(
        "POST",
        f"/sessions/{quote(session_name, safe='')}/terminals",
        query,
        None,
    )
    payload = _safe_json(response)
    if response.status_code < 200 or response.status_code >= 300:
        raise CaoProtocolError(
            f"CAO terminal create failed: {_detail(payload, f'HTTP {response.status_code}')}",
            response=payload,
            session_name=session_name,
        )
    terminal_id = _terminal_id(payload)
    if not terminal_id:
        raise CaoProtocolError(
            "CAO terminal create did not return a terminal id", response=payload
        )
    return terminal_id


async def _poll(config: CaoConfig, terminal_id: str, call: Any, pause: Any) -> dict[str, Any]:
    import time

    deadline = time.monotonic() + config.timeout_seconds
    last: Any = {}
    status = "running"
    while time.monotonic() <= deadline:
        response = await call("GET", f"/terminals/{quote(terminal_id, safe='')}", None, None)
        last = _safe_json(response)
        status = str(last.get("status") or "").lower() if isinstance(last, dict) else ""
        if status in {"completed", "error"}:
            break
        await pause(config.poll_interval_seconds)
        if time.monotonic() > deadline:
            status = "timeout"
            break
    if status not in {"completed", "error"}:
        status = "timeout"
    output = await call(
        "GET",
        f"/terminals/{quote(terminal_id, safe='')}/output",
        {"mode": "full"},
        None,
    )
    output_body = _safe_json(output)
    content = ""
    if isinstance(output_body, dict) and output_body.get("output"):
        content = str(output_body["output"])
    elif isinstance(last, dict):
        content = str(last.get("last_message") or last.get("output") or "")
    return {
        "status": status,
        "content": content,
        "response": {"terminal": last, "output": output_body},
    }


def _accepted(
    session_name: str,
    terminal_id: str,
    calls: list[dict[str, Any]],
    payload: Any,
) -> dict[str, Any]:
    return {
        "status": "accepted",
        "content": "",
        "sessionName": session_name,
        "terminalId": terminal_id,
        "request": calls,
        "response": payload,
    }


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}
