"""Flowise A2A node client: JSON-RPC ``message/send``.

This is the fork's protocol, not Binex's ``a2a://`` adapter which posts to
``/execute``. contextId and taskId are persisted by the caller.
"""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urlparse

import httpx


class A2AProtocolError(RuntimeError):
    """The remote agent rejected the call or the URL is not http(s)."""

    def __init__(self, message: str, request: dict[str, Any], response: Any) -> None:
        super().__init__(message)
        self.request = request
        self.response = response


def assert_http_url(value: str) -> str:
    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise A2AProtocolError("A2A agent URL must be an absolute http(s) URL", {}, None) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise A2AProtocolError("A2A agent URL must be an absolute http(s) URL", {}, None)
    return value.rstrip("/")


def build_message_send_request(
    message: str,
    session: dict[str, Any],
    message_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": message_id,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": message_id,
                "parts": [{"kind": "text", "text": message}],
            }
        },
    }
    body = payload["params"]["message"]
    if session.get("contextId"):
        body["contextId"] = session["contextId"]
    if session.get("taskId"):
        body["taskId"] = session["taskId"]
    return payload


def _collect_text(value: Any, found: list[str]) -> None:
    if not value:
        return
    if isinstance(value, str):
        found.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text(item, found)
        return
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            found.append(value["text"])
        for key in ("parts", "message", "artifacts", "status"):
            if key in value:
                _collect_text(value[key], found)


def parse_a2a_response(body: Any) -> dict[str, Any]:
    result = body.get("result", body) if isinstance(body, dict) else body
    texts: list[str] = []
    _collect_text(result, texts)
    context_id = None
    task_id = None
    if isinstance(result, dict):
        context_id = result.get("contextId") or _nested(result, "message", "contextId")
        task_id = result.get("taskId") or result.get("id") or _nested(result, "task", "id")
    if isinstance(body, dict) and not context_id:
        nested = body.get("result")
        if isinstance(nested, dict):
            context_id = nested.get("contextId")
    unique: list[str] = []
    for text in texts:
        cleaned = text.strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return {
        "text": "\n".join(unique),
        "contextId": context_id if isinstance(context_id, str) else None,
        "taskId": task_id if isinstance(task_id, str) else None,
        "raw": body,
    }


def _nested(value: dict[str, Any], key: str, child: str) -> Any:
    inner = value.get(key)
    if isinstance(inner, dict):
        return inner.get(child)
    return None


async def send_a2a(
    *,
    agent_url: str,
    message: str,
    session: dict[str, Any],
    bearer: str | None,
    timeout_ms: int,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """POST message/send. The bearer header is not part of the recorded request."""
    url = assert_http_url(agent_url)
    message_id = str(uuid.uuid4())
    request_body = build_message_send_request(message, session, message_id)
    headers = {"content-type": "application/json", "accept": "application/json"}
    if bearer:
        headers["authorization"] = f"Bearer {bearer}"
    response = await client.post(
        url, json=request_body, headers=headers, timeout=timeout_ms / 1000
    )
    try:
        parsed_body = response.json()
    except ValueError:
        parsed_body = {"text": response.text}
    if response.status_code < 200 or response.status_code >= 300:
        raise A2AProtocolError(
            f"A2A request failed with HTTP {response.status_code}",
            request_body,
            parsed_body,
        )
    if isinstance(parsed_body, dict) and parsed_body.get("error"):
        error = parsed_body["error"]
        detail = error.get("message") if isinstance(error, dict) else str(error)
        raise A2AProtocolError(f"A2A agent error: {detail}", request_body, parsed_body)
    parsed = parse_a2a_response(parsed_body)
    return {
        "request": request_body,
        "response": parsed["raw"],
        "text": parsed["text"],
        "contextId": parsed["contextId"] or session.get("contextId"),
        "taskId": parsed["taskId"] or session.get("taskId"),
    }
