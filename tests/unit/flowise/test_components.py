"""Tool, Retriever, Execute Flow, agent tools, and memory."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from binex.flowise.components import ComponentError
from binex.flowise.engine import collect_events
from binex.flowise.llm import LLMCompletion
from binex.flowise.session import SessionStore
from tests.unit.flowise.support import edge, graph, load_example, node, run_body
from tests.unit.flowise.test_clients import FakeLLM, RecordingTransport

_EXEC = ("store", "llm", "http", "js_runner", "skip_ssrf", "max_steps")


def _start() -> dict[str, Any]:
    return node("start", "startAgentflow", {"startInputType": "chatInput"})


async def _run(flow: dict[str, Any], tmp_path, **overrides: Any):
    exec_kwargs = {key: overrides.pop(key) for key in _EXEC if key in overrides}
    store = exec_kwargs.pop("store", None) or SessionStore(tmp_path)
    outcome, events = await collect_events(run_body(flow, **overrides), store=store, **exec_kwargs)
    return outcome, events, store


async def test_retriever_inline_knowledge_and_metadata(tmp_path) -> None:
    knowledge = {
        "docs": [
            {"pageContent": "cats are great hunters", "metadata": {"id": 1}},
            {"pageContent": "dogs fetch sticks", "metadata": {"id": 2}},
        ]
    }
    text, _events, _store = await _run(
        load_example("retriever.json"),
        tmp_path,
        inputs={"question": "cats"},
        knowledge=knowledge,
    )
    assert text.status == "FINISHED"
    assert text.text == "cats are great hunters"

    detailed = graph(
        [
            _start(),
            node(
                "retriever",
                "retrieverAgentflow",
                {
                    "retrieverKnowledgeDocumentStores": [{"documentStore": "docs:Articles"}],
                    "retrieverQuery": "cats",
                    "outputFormat": "textWithMetadata",
                },
            ),
        ],
        [edge("start", "retriever")],
    )
    rich, _events, _store = await _run(
        detailed, tmp_path, knowledge=knowledge, sessionId="meta", runId="run-m"
    )
    assert '"id": 1' in rich.text


async def test_execute_flow_inline_prediction_and_self_call(tmp_path) -> None:
    child = graph(
        [
            _start(),
            node("out", "directReplyAgentflow", {"directReplyMessage": "child-says"}),
        ],
        [edge("start", "out")],
    )
    inline, events, _store = await _run(
        load_example("execute-flow.json"),
        tmp_path,
        flows={"child-flow": child},
        inputs={"question": "go"},
    )
    assert inline.status == "FINISHED"
    assert inline.text == "child-says"
    assert any(event.get("chatflowId") == "child-flow" for event in events)

    same = graph(
        [
            _start(),
            node(
                "nested",
                "executeFlowAgentflow",
                {"executeFlowSelectedFlow": "flow-1", "executeFlowInput": "x"},
            ),
        ],
        [edge("start", "nested")],
    )
    denied, _events, _store = await _run(same, tmp_path, sessionId="self")
    assert "same agentflow" in (denied.error or "")

    missing, _events, _store = await _run(
        load_example("execute-flow.json"), tmp_path, sessionId="missing", runId="run-missing"
    )
    assert "not in the run payload" in (missing.error or "")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/prediction/child-flow"
        assert request.headers["authorization"] == "Bearer cf-key"
        assert request.headers["flowise-tool"] == "true"
        return httpx.Response(200, json={"text": "from-server", "sourceDocuments": []})

    transport = RecordingTransport(handler)
    remote = graph(
        [
            _start(),
            node(
                "nested",
                "executeFlowAgentflow",
                {
                    "executeFlowSelectedFlow": "child-flow",
                    "executeFlowInput": "ping",
                    "executeFlowBaseURL": "https://example.com",
                    "executeFlowReturnResponseAs": "userMessage",
                },
                credential="chatflowApi",
            ),
        ],
        [edge("start", "nested")],
    )
    async with httpx.AsyncClient(transport=transport.transport()) as client:
        called, _events, _store = await _run(
            remote,
            tmp_path,
            http=client,
            skip_ssrf=True,
            sessionId="remote",
            runId="run-remote",
            credentials={"chatflowApi": {"chatflowApiKey": "cf-key"}},
        )
    assert called.text == "from-server"
    assert transport.calls


async def test_tool_callback_and_missing_endpoint(tmp_path) -> None:
    flow = graph(
        [
            _start(),
            node(
                "tool",
                "toolAgentflow",
                {
                    "toolAgentflowSelectedTool": "serpAPI",
                    "toolInputArgs": [{"inputArgName": "query", "inputArgValue": "cats"}],
                },
            ),
        ],
        [edge("start", "tool")],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/internal/components/execute"
        assert request.headers["authorization"] == "Bearer call-token"
        body = request.read()
        assert b"serpAPI" in body
        assert b'"kind": "tool"' in body or b'"kind":"tool"' in body
        return httpx.Response(200, json={"content": "results", "artifacts": []})

    transport = RecordingTransport(handler)
    async with httpx.AsyncClient(transport=transport.transport()) as client:
        ran, _events, _store = await _run(
            flow,
            tmp_path,
            http=client,
            componentUrl="http://flowise.test",
            callbackToken="call-token",
        )
    assert ran.text == "results"

    missing, _events, _store = await _run(flow, tmp_path, sessionId="none", runId="run-none")
    assert missing.status == "ERROR"
    assert "serpAPI" in (missing.error or "")
    assert "/api/v1/internal/components/execute" in (missing.error or "")


async def test_agent_human_gated_tool_resume(tmp_path) -> None:
    class Scripted:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def complete(self, **kwargs: Any) -> LLMCompletion:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return LLMCompletion(
                    content="",
                    model=kwargs["model"],
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        }
                    ],
                )
            return LLMCompletion(content="4", model=kwargs["model"])

    flow = graph(
        [
            _start(),
            node(
                "agent",
                "agentAgentflow",
                {
                    "agentModel": "openai/gpt-4o-mini",
                    "agentUserMessage": "math",
                    "agentTools": [
                        {
                            "agentSelectedTool": "calculator",
                            "agentSelectedToolRequiresHumanInput": True,
                        }
                    ],
                },
            ),
        ],
        [edge("start", "agent")],
    )
    llm = Scripted()
    store = SessionStore(tmp_path)
    paused, events, _store = await _run(flow, tmp_path, llm=llm, store=store)
    assert paused.status == "STOPPED"
    assert "calculator" in paused.text
    checkpoint = store.get_checkpoint("flow-1", "sess-1")
    assert checkpoint is not None
    assert checkpoint["agentResume"]["pending"]["name"] == "calculator"
    assert any(event["type"] == "human_input" for event in events)

    finished, _events, _store = await _run(
        flow,
        tmp_path,
        llm=llm,
        store=store,
        runId="run-2",
        resume={"humanInput": {"type": "proceed", "startNodeId": "agent"}},
    )
    assert finished.status == "FINISHED"
    assert finished.text == "4"
    assert store.get_checkpoint("flow-1", "sess-1") is None


async def test_memory_summary_and_return_role(tmp_path) -> None:
    def reply(kwargs: dict[str, Any]) -> str:
        contents = [str(item.get("content")) for item in kwargs["messages"]]
        if any("Summarize this conversation" in item for item in contents):
            return "short summary"
        return "answer"

    llm = FakeLLM(reply)
    flow = graph(
        [
            _start(),
            node(
                "llm",
                "llmAgentflow",
                {
                    "llmModel": "openai/gpt-4o-mini",
                    "llmUserMessage": "next",
                    "llmEnableMemory": True,
                    "llmMemoryType": "conversationSummary",
                    "llmReturnResponseAs": "userMessage",
                },
            ),
        ],
        [edge("start", "llm")],
    )
    outcome, _events, _store = await _run(
        flow,
        tmp_path,
        llm=llm,
        inputs={
            "question": "next",
            "history": [
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
            ],
        },
    )
    assert outcome.text == "answer"
    assert len(llm.calls) == 2
    sent = llm.calls[1]["messages"]
    assert any("short summary" in str(item.get("content")) for item in sent)


async def test_http_put_form_and_basic_auth(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.headers["authorization"].startswith("Basic ")
        assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
        assert b"a=1" in request.content
        return httpx.Response(200, json={"ok": True})

    transport = RecordingTransport(handler)
    flow = graph(
        [
            _start(),
            node(
                "http",
                "httpAgentflow",
                {
                    "method": "PUT",
                    "url": "https://example.com/items",
                    "bodyType": "xWwwFormUrlencoded",
                    "body": [{"key": "a", "value": "1"}],
                    "responseType": "json",
                },
                credential="basic",
            ),
        ],
        [edge("start", "http")],
    )
    async with httpx.AsyncClient(transport=transport.transport()) as client:
        outcome, _events, _store = await _run(
            flow,
            tmp_path,
            http=client,
            skip_ssrf=True,
            credentials={"basic": {"basicAuthUsername": "ada", "basicAuthPassword": "secret"}},
        )
    assert outcome.status == "FINISHED"
    http_row = next(item for item in outcome.execution_data if item["nodeId"] == "http")
    assert http_row["data"]["output"]["http"]["data"]["ok"] is True


async def test_custom_function_variables(tmp_path) -> None:
    seen: dict[str, Any] = {}

    async def runner(code: str, **kwargs: Any) -> str:
        seen["code"] = code
        seen["extra"] = kwargs["extra"]
        return f"hi {kwargs['extra']['$name']}"

    flow = load_example("custom-function.json")
    outcome, _events, _store = await _run(
        flow, tmp_path, js_runner=runner, inputs={"question": "Lovelace"}
    )
    assert outcome.text == "hi Ada"
    assert seen["extra"]["$name"] == "Ada"


async def test_local_tool_defaults() -> None:
    from types import SimpleNamespace

    from binex.flowise.components import invoke_named_tool

    target = "/workspace/.binex-flowise-tool-test.txt"

    class Ctx:
        skip_ssrf = True
        state: dict[str, Any] = {}
        js_runner = None
        request = SimpleNamespace(
            variables={},
            chatflow_id="c",
            component_url="",
            callback_token="",
            session_key="s",
            knowledge={},
        )

        def __init__(self, client: httpx.AsyncClient) -> None:
            self.http = client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ping"
        return httpx.Response(200, text="pong")

    transport = RecordingTransport(handler)
    async with httpx.AsyncClient(transport=transport.transport()) as client:
        ctx = Ctx(client)
        clock = await invoke_named_tool(ctx, name="currentDateTime", args={})
        assert "T" in clock.content
        fetched = await invoke_named_tool(
            ctx, name="requestsGet", args={"url": "https://example.com/ping"}
        )
        assert fetched.content == "pong"
        written = await invoke_named_tool(
            ctx, name="writeFile", args={"path": target, "content": "hello"}
        )
        assert "bytes" in written.content
        read = await invoke_named_tool(ctx, name="readFile", args={"path": target})
        assert read.content == "hello"
    from pathlib import Path

    Path(target).unlink(missing_ok=True)


async def test_workspace_path_is_confined() -> None:
    with pytest.raises(ComponentError):
        from binex.flowise.components import _workspace_file

        _workspace_file("../etc/passwd")


def test_provider_builtins_are_passed_through() -> None:
    from binex.flowise.agent_loop import _tool_catalog

    schemas, meta = _tool_catalog(
        {
            "agentToolsBuiltInOpenAI": ["web_search_preview"],
            "agentKnowledgeDocumentStores": [
                {"documentStore": "docs:Articles", "docStoreDescription": "Search articles"}
            ],
        }
    )
    assert {"type": "web_search_preview"} in schemas
    assert meta["web_search_preview"]["kind"] == "builtin"
    assert any(item.get("function", {}).get("name", "").startswith("search_") for item in schemas)
