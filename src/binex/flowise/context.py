"""Mutable state for one sandbox run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from binex.flowise.graph import FlowGraph
from binex.flowise.llm import LLMCaller
from binex.flowise.protocol import RunRequest
from binex.flowise.session import SessionStore
from binex.flowise.variables import TemplateContext


@dataclass
class ExecContext:
    request: RunRequest
    graph: FlowGraph
    store: SessionStore
    llm: LLMCaller
    http: httpx.AsyncClient
    state: dict[str, Any] = field(default_factory=dict)
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    loop_counts: dict[str, int] = field(default_factory=dict)
    question: str = ""
    form: dict[str, Any] = field(default_factory=dict)
    webhook: dict[str, Any] = field(default_factory=dict)
    uploads: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    iteration: dict[str, Any] | None = None
    human_input: dict[str, Any] | None = None
    iteration_resume: dict[str, Any] | None = None
    paused: bool = False
    pause_node_id: str | None = None
    human_mode: str | None = None
    steps: int = 0
    max_steps: int = 1000
    last_text: str = ""
    persist_state: bool = False
    skip_ssrf: bool = False
    js_runner: Any = None

    def templates(self) -> TemplateContext:
        return TemplateContext(
            question=self.question,
            form=self.form,
            webhook=self.webhook,
            state=self.state,
            chatflow_id=self.request.chatflow_id,
            session_id=self.request.session_key,
            chat_id=str(self.request.inputs.get("chatId") or self.request.session_key),
            variables=self.request.variables,
            executed=self.executed,
            loop_counts=self.loop_counts,
            iteration=self.iteration,
            uploads=self.uploads,
            chat_history=self.chat_history,
            runtime_messages_length=max(0, len(self.chat_history) - 1),
        )

    def credential_for(self, node_id: str, credential_id: str | None) -> dict[str, Any]:
        found = self.request.credentials.get(node_id)
        if isinstance(found, dict):
            return found
        if credential_id:
            named = self.request.credentials.get(credential_id)
            if isinstance(named, dict):
                return named
        return {}
