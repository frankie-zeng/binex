"""Run request and the NDJSON events the Flowise server stores."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# "token" must not redact usage fields such as input_tokens. A following letter
# means the word continues (tokens). bearerToken and apiKey still match.
_SECRET_KEY = re.compile(
    r"(token|password|secret|api[-_]?key|authorization)(?![a-z])",
    re.IGNORECASE,
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# A request must not be able to redirect the sandbox's own files or loader.
_BLOCKED_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "PYTHONPATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "NODE_OPTIONS",
        "BINEX_FLOWISE_STATE_DIR",
    }
)


class RunRequest(BaseModel):
    """One sandbox invocation. Field names match the Flowise server payload."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    run_id: str = Field(alias="runId")
    chatflow_id: str = Field(alias="chatflowId")
    session_id: str = Field(default="", alias="sessionId")
    workspace_id: str = Field(default="", alias="workspaceId")
    org_id: str = Field(default="", alias="orgId")
    flow: Any
    inputs: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    credentials: dict[str, dict[str, Any]] = Field(default_factory=dict)
    variables: dict[str, Any] = Field(default_factory=dict, alias="vars")
    resume: dict[str, Any] | None = None
    flows: dict[str, Any] = Field(default_factory=dict)
    knowledge: dict[str, Any] = Field(default_factory=dict)
    component_url: str = Field(default="", alias="componentUrl")
    callback_token: str = Field(default="", alias="callbackToken")

    @property
    def session_key(self) -> str:
        return self.session_id or "default"


def check_secret_names(secrets: dict[str, str]) -> None:
    """Reject secret names that are not environment variables."""
    for name in secrets:
        if name in _BLOCKED_ENV or not _ENV_NAME.match(name):
            raise ValueError(
                f"Refusing secret name '{name}'. Send provider keys as environment "
                "variable names such as OPENAI_API_KEY."
            )


def redact_tree(value: Any) -> Any:
    """Replace credential-like dict keys before an artifact is recorded."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = redact_tree(item)
        return redacted
    if isinstance(value, list):
        return [redact_tree(item) for item in value]
    return value


def json_text(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def redact_secrets(event: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    """Strip secret values from an event before it leaves the sandbox."""
    values = [item for item in secrets.values() if isinstance(item, str) and len(item) >= 8]
    if not values:
        return event
    encoded = json_text(event)
    for secret in values:
        encoded = encoded.replace(secret, "[redacted]")
    loaded = json.loads(encoded)
    return loaded if isinstance(loaded, dict) else event


def artifact_event(
    request: RunRequest,
    *,
    node_id: str,
    node_label: str,
    kind: str,
    file_name: str,
    body: str,
    mime_type: str = "application/json",
) -> dict[str, Any]:
    """Shape matches Flowise ``execution_artifact`` plus inline content to store."""
    encoded = body.encode("utf-8")
    return {
        "type": "artifact",
        "executionId": request.run_id,
        "agentflowId": request.chatflow_id,
        "sessionId": request.session_key,
        "workspaceId": request.workspace_id,
        "orgId": request.org_id,
        "nodeId": node_id,
        "nodeLabel": node_label,
        "kind": kind,
        "fileName": file_name,
        "mimeType": mime_type,
        "size": len(encoded),
        "storagePath": [
            request.org_id or "org",
            request.chatflow_id,
            "executions",
            request.run_id,
            node_id,
        ],
        "content": body,
    }


def conversation_event(request: RunRequest, turn: dict[str, Any]) -> dict[str, Any]:
    """Shape matches Flowise ``agent_conversation_turn`` / ConversationTurnInput."""
    return {
        "type": "conversation",
        "workspaceId": request.workspace_id,
        "chatflowId": request.chatflow_id,
        "sessionId": request.session_key,
        "executionId": request.run_id,
        "nodeId": turn.get("nodeId"),
        "kind": turn.get("kind"),
        "status": turn.get("status"),
        "message": turn.get("message"),
        "output": turn.get("output"),
        "externalSession": turn.get("externalSession"),
        "externalTask": turn.get("externalTask"),
        "rawRequest": turn.get("rawRequest"),
        "rawResponse": turn.get("rawResponse"),
    }
