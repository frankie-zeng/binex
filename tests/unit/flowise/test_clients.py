"""LLM, HTTP, Custom Function, A2A, and CAO nodes."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
import litellm
import pytest

from binex.flowise.a2a import assert_http_url, parse_a2a_response, send_a2a
from binex.flowise.cao import (
    CaoProtocolError,
    build_cao_prompt,
    execute_cao,
    normalize_cao_config,
    session_name_for_flow,
)
from binex.flowise.custom_function import CustomFunctionError, run_javascript, wrap_user_code
from binex.flowise.engine import ExecuteOutcome, _emit_protocol_artifacts, collect_events
from binex.flowise.graph import FlowNode
from binex.flowise.llm import (
    LiteLLMCaller,
    LLMCompletion,
    complete_with_binex_tools,
    completion_kwargs,
    parse_json_object,
    resolve_model,
    structured_schema,
)
from binex.flowise.protocol import RunRequest
from binex.flowise.session import SessionStore
from tests.unit.flowise.support import edge, graph, load_example, node, run_body


class FakeLLM:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMCompletion:
        self.calls.append(kwargs)
        text = self.content(kwargs) if callable(self.content) else self.content
        return LLMCompletion(
            content=text,
            model=kwargs["model"],
            prompt_tokens=3,
            completion_tokens=4,
        )


class RecordingTransport:
    def __init__(self, handler) -> None:
        self.calls: list[httpx.Request] = []
        self._handler = handler

    def transport(self) -> httpx.MockTransport:
        def _handle(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            return self._handler(request)

        return httpx.MockTransport(_handle)


def _start() -> dict[str, Any]:
    return node("start", "startAgentflow", {"startInputType": "chatInput"})


async def _collect(flow: dict[str, Any], tmp_path, **kwargs: Any):
    store = kwargs.pop("store", None) or SessionStore(tmp_path)
    body_keys = {
        "inputs",
        "secrets",
        "credentials",
        "vars",
        "resume",
        "runId",
        "sessionId",
        "chatflowId",
    }
    body_over = {key: kwargs.pop(key) for key in list(kwargs) if key in body_keys}
    outcome, events = await collect_events(run_body(flow, **body_over), store=store, **kwargs)
    return outcome, events, store


def test_model_resolution_and_schema() -> None:
    resolved = resolve_model("chatOpenAI", {"modelName": "gpt-4o-mini", "temperature": "0.2"})
    assert resolved.model == "openai/gpt-4o-mini"
    assert resolved.temperature == 0.2
    assert "max_tokens" not in completion_kwargs(resolved, [{"role": "user", "content": "hi"}])
    assert resolve_model("chatAnthropic", {"modelName": "claude-3-5-haiku"}).model.startswith(
        "anthropic/"
    )
    assert resolve_model("openai/gpt-4o-mini", {}).model == "openai/gpt-4o-mini"
    assert resolve_model("chatGroq", {"modelName": "llama-3.1-8b-instant"}).model == (
        "groq/llama-3.1-8b-instant"
    )
    assert resolve_model("chatLitellm", {"modelName": "openai/gpt-4o-mini"}).model == (
        "openai/gpt-4o-mini"
    )
    custom = resolve_model(
        "chatLocal",
        {"modelName": "local-model", "baseUrl": "http://localhost:8080", "groqApiKey": "g-key"},
    )
    assert custom.model == "openai/local-model"
    assert custom.api_base == "http://localhost:8080"
    assert custom.api_key == "g-key"
    with pytest.raises(RuntimeError):
        resolve_model("chatOpenAI", {})
    schema = structured_schema(
        [
            {"key": "name", "type": "string", "description": "who"},
            {"key": "n", "type": "number"},
            {"key": "ok", "type": "boolean"},
            {"key": "tags", "type": "stringArray"},
            {"key": "color", "type": "enum", "enumValues": "red, blue"},
            {"key": "rows", "type": "jsonArray", "jsonSchema": '{"type": "array"}'},
            {"key": "bad", "type": "jsonArray", "jsonSchema": "{"},
        ]
    )
    assert schema["properties"]["color"]["enum"] == ["red", "blue"]
    assert parse_json_object('```json\n{"output": "yes"}\n```') == {"output": "yes"}
    with pytest.raises(ValueError):
        parse_json_object("nope")


async def test_litellm_caller_and_binex_tools(monkeypatch) -> None:
    class Message:
        content = [{"text": "hello"}]

    class Choice:
        message = Message()

    class Usage:
        prompt_tokens = 1
        completion_tokens = 2

    class Response:
        choices = [Choice()]
        usage = Usage()
        model = "openai/gpt-4o-mini"

    async def fake_completion(**_kwargs: Any) -> Response:
        return Response()

    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    result = await LiteLLMCaller().complete(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        api_key=None,
        api_base=None,
        temperature=None,
        max_tokens=None,
    )
    assert result.content == "hello"
    assert result.prompt_tokens == 1

    class Adapter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.tools: list[str] = []

        async def execute(self, task: Any, _artifacts: Any, _run_id: str) -> Any:
            self.tools = list(task.tools)

            class Artifact:
                content = "from-tool"

            class Done:
                artifacts = [Artifact()]

            return Done()

    monkeypatch.setattr("binex.flowise.llm.LLMAdapter", Adapter)
    tooled = await complete_with_binex_tools(
        resolve_model("openai/gpt-4o-mini", {}),
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
        ["local://echo"],
        run_id="run-1",
        node_id="agent",
    )
    assert tooled.content == "from-tool"


async def test_llm_example_and_agent_guards(tmp_path) -> None:
    llm = FakeLLM("model-says")
    outcome, events, _store = await _collect(
        load_example("llm.json"),
        tmp_path,
        llm=llm,
        inputs={"question": "hi", "history": [{"role": "userMessage", "content": "earlier"}]},
    )
    assert outcome.text == "model-says"
    assert outcome.state["last"] == "model-says"
    assert llm.calls[0]["model"] == "openai/gpt-4o-mini"
    assert any(item["content"] == "earlier" for item in llm.calls[0]["messages"])
    usage = next(
        event
        for event in events
        if event.get("nodeId") == "llmAgentflow_0" and event["type"] == "node.finish"
    )
    assert usage["output"]["usageMetadata"]["input_tokens"] == 3

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

    scripted = Scripted()
    tooled = graph(
        [
            _start(),
            node(
                "agent",
                "agentAgentflow",
                {
                    "agentModel": "openai/gpt-4o-mini",
                    "agentUserMessage": "hi",
                    "agentTools": [{"agentSelectedTool": "calculator"}],
                },
            ),
        ],
        [edge("start", "agent")],
    )
    ran_tool, _events, _store = await _collect(tooled, tmp_path, llm=scripted, sessionId="agent")
    assert ran_tool.status == "FINISHED"
    assert ran_tool.text == "4"
    assert any(item.get("role") == "tool" for item in scripted.calls[1]["messages"])

    allowed = graph(
        [
            _start(),
            node(
                "agent",
                "agentAgentflow",
                {
                    "agentModel": "openai/gpt-4o-mini",
                    "agentUserMessage": "hi",
                    "binexTools": ["local://echo"],
                },
            ),
        ],
        [edge("start", "agent")],
    )

    class Adapter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def execute(self, task: Any, _artifacts: Any, _run_id: str) -> Any:
            class Artifact:
                content = "tool-ok"

            class Done:
                artifacts = [Artifact()]

            return Done()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("binex.flowise.llm.LLMAdapter", Adapter)
    try:
        ran, _events, _store = await _collect(allowed, tmp_path, sessionId="tools")
    finally:
        monkeypatch.undo()
    assert ran.text == "tool-ok"


async def test_condition_agent_and_bad_json(tmp_path) -> None:
    flow = graph(
        [
            _start(),
            node(
                "route",
                "conditionAgentAgentflow",
                {
                    "conditionAgentModel": "openai/gpt-4o-mini",
                    "conditionAgentScenarios": [{"scenario": "yes"}, {"scenario": "no"}],
                },
            ),
            node("yes", "directReplyAgentflow", {"directReplyMessage": "YES"}),
            node("no", "directReplyAgentflow", {"directReplyMessage": "NO"}),
        ],
        [edge("start", "route"), edge("route", "yes", 0), edge("route", "no", 1)],
    )
    ok, _events, _store = await _collect(flow, tmp_path, llm=FakeLLM('{"output": "no"}'))
    assert ok.text == "NO"
    bad, _events, _store = await _collect(
        flow, tmp_path, llm=FakeLLM("maybe"), sessionId="bad", runId="run-b"
    )
    assert "JSON scenario" in (bad.error or "")


async def test_http_success_ssrf_and_redaction(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("BINEX_ALLOW_PRIVATE_URLS", raising=False)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/fail":
            return httpx.Response(500, text="nope")
        if request.url.path == "/text":
            return httpx.Response(200, text="plain")
        if request.url.path == "/bin":
            return httpx.Response(200, content=b"\x00\x01")
        return httpx.Response(200, json={"id": 7})

    flow = graph(
        [
            _start(),
            node(
                "http",
                "httpAgentflow",
                {
                    "method": "POST",
                    "url": "http://example.test/items",
                    "queryParams": [{"key": "q", "value": "1"}],
                    "headers": [{"key": "X-Test", "value": "yes"}],
                    "bodyType": "json",
                    "body": '{"name": "ada"}',
                },
                credential="basic",
            ),
        ],
        [edge("start", "http")],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome, events, _store = await _collect(
            flow,
            tmp_path,
            http=client,
            skip_ssrf=True,
            credentials={
                "basic": {"basicAuthUsername": "user", "basicAuthPassword": "super-secret-token"}
            },
        )
    assert outcome.status == "FINISHED"
    assert seen[0].url.params["q"] == "1"
    assert json.loads(seen[0].content) == {"name": "ada"}
    artifact = next(
        event
        for event in events
        if event.get("type") == "artifact"
        and event["nodeId"] == "http"
        and event["kind"] == "input"
    )
    assert "super-secret-token" not in artifact["content"]
    assert "[redacted]" in artifact["content"]

    blocked = graph(
        [_start(), node("http", "httpAgentflow", {"url": "http://127.0.0.1/secret"})],
        [edge("start", "http")],
    )
    failed, _events, _store = await _collect(blocked, tmp_path, sessionId="ssrf")
    assert "blocked" in (failed.error or "")

    failing = graph(
        [_start(), node("http", "httpAgentflow", {"url": "http://example.test/fail"})],
        [edge("start", "http")],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        errored, _events, _store = await _collect(
            failing, tmp_path, http=client, skip_ssrf=True, sessionId="fail"
        )
    assert "HTTP 500" in (errored.error or "")

    text_flow = graph(
        [
            _start(),
            node(
                "http",
                "httpAgentflow",
                {
                    "url": "http://example.test/text",
                    "responseType": "text",
                    "bodyType": "raw",
                    "method": "POST",
                    "body": "raw",
                },
            ),
        ],
        [edge("start", "http")],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text_out, _events, _store = await _collect(
            text_flow, tmp_path, http=client, skip_ssrf=True, sessionId="text"
        )
    assert text_out.status == "FINISHED"


def test_a2a_parse_and_url() -> None:
    with pytest.raises(Exception):
        assert_http_url("ftp://agent.example/rpc")
    parsed = parse_a2a_response(
        {
            "result": {
                "contextId": "ctx-1",
                "id": "task-1",
                "status": {"message": {"parts": [{"kind": "text", "text": "pong"}]}},
            }
        }
    )
    assert parsed["text"] == "pong"
    assert parsed["contextId"] == "ctx-1"
    assert parsed["taskId"] == "task-1"


async def test_a2a_persists_context_and_hides_bearer(tmp_path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "result": {
                        "contextId": "ctx-1",
                        "id": "task-1",
                        "status": {"message": {"parts": [{"text": "pong"}]}},
                    }
                },
            )
        return httpx.Response(200, json={"error": {"message": "no"}})

    flow = load_example("a2a.json")
    flow["nodes"][1]["data"]["inputs"]["a2aBearerToken"] = "bearer-token-value-99"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first, events, store = await _collect(flow, tmp_path, http=client)
        assert first.status == "FINISHED"
        assert first.text == "pong"
        conversation = next(event for event in events if event["type"] == "conversation")
        assert conversation["kind"] == "a2a"
        assert conversation["externalSession"] == "ctx-1"
        assert conversation["externalTask"] == "task-1"
        kinds = {event["kind"] for event in events if event["type"] == "artifact"}
        assert {"a2a-request", "a2a-response", "input", "output"} <= kinds
        dumped = json.dumps(events)
        assert "bearer-token-value-99" not in dumped
        assert "bearer-token-value-99" not in (tmp_path / "flow-1.json").read_text()

        second, second_events, _store = await _collect(
            flow, tmp_path, http=client, store=store, runId="run-2"
        )
    assert requests[1]["params"]["message"]["contextId"] == "ctx-1"
    assert requests[1]["params"]["message"]["taskId"] == "task-1"
    assert second.status == "ERROR"
    failed = next(event for event in second_events if event["type"] == "conversation")
    assert failed["status"] == "error"
    assert failed["kind"] == "a2a"


async def test_send_a2a_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="nope")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(Exception, match="HTTP 401"):
            await send_a2a(
                agent_url="http://agent.test/rpc",
                message="hi",
                session={},
                bearer=None,
                timeout_ms=1000,
                client=client,
            )


def test_cao_session_name_and_prompt() -> None:
    assert session_name_for_flow("flow-1", "caoAgentflow_0") == "flowf4bdc13bcaoAgent"
    from binex.flowise.cao import CaoConfig

    config = CaoConfig(
        server_url="http://127.0.0.1:9889",
        bearer_token=None,
        provider="codex",
        agent_profile="developer",
        mode="handoff",
        task="fix the test",
        explicit_session_name=None,
        reuse_terminal=True,
        teardown=False,
        timeout_seconds=30,
        wait=True,
        working_directory=None,
        caller_id="flowise",
        receiver_id=None,
        allowed_tools=[],
        use_worktree=False,
        memory_manager=False,
        env_vars=None,
        claim_id=None,
        group=None,
        metadata=None,
        shape_prompt=True,
        inject_sender_id=True,
        job_id=None,
        poll_interval_seconds=0.01,
    )
    prompt = build_cao_prompt(config, None)
    assert prompt.startswith("CAO_TERMINAL_ID=pending")
    assert prompt.endswith("(sender: flowise)")
    with pytest.raises(CaoProtocolError):
        normalize_cao_config(
            {"caoProvider": "nope", "caoTask": "x", "caoServerUrl": "http://127.0.0.1:9"},
            env={},
            credential={},
        )


def _cao_handler(store: dict[str, Any]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        store["calls"].append((request.method, request.url.path, body, dict(request.url.params)))
        path = request.url.path
        if path == "/sessions" and request.method == "POST":
            return httpx.Response(201, json={"ok": True})
        if path == "/terminals/run-step":
            if isinstance(body, dict) and body.get("reuse_terminal_id") == "gone":
                return httpx.Response(404, json={"detail": "missing terminal"})
            if isinstance(body, dict) and body.get("prompt") == "explode":
                return httpx.Response(
                    500,
                    json={
                        "detail": {
                            "kind": "timeout",
                            "message": "too slow",
                            "terminal_id": "term-x",
                        }
                    },
                )
            return httpx.Response(
                200,
                json={"status": "completed", "last_message": "patched", "terminal_id": "term-9"},
            )
        if path.endswith("/terminals") and request.method == "POST":
            return httpx.Response(201, json={"terminal_id": "term-assign"})
        if path.endswith("/inbox/messages"):
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/input"):
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/output"):
            return httpx.Response(200, json={"output": "polled"})
        if request.method == "GET":
            return httpx.Response(200, json={"status": "completed", "last_message": "polled"})
        return httpx.Response(500, json={"detail": path})

    return handler


async def test_cao_handoff_reuses_terminal(tmp_path) -> None:
    captured: dict[str, Any] = {"calls": []}
    flow = load_example("cao.json")
    flow["nodes"][1]["data"]["inputs"]["caoEnvVars"] = {"SECRET": "cao-env-secret-value"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(_cao_handler(captured))) as client:
        first, events, store = await _collect(
            flow, tmp_path, http=client, inputs={"question": "fix it"}
        )
        assert first.status == "FINISHED"
        assert first.text == "patched"
        turn = next(event for event in events if event["type"] == "conversation")
        assert turn["kind"] == "cao"
        assert turn["externalSession"] == "flowf4bdc13bcaoAgent"
        assert turn["externalTask"] == "term-9"
        dumped = json.dumps(turn["rawRequest"])
        assert "cao-env-secret-value" not in dumped
        assert "[redacted]" in dumped
        saved = store.get_cao("flow-1", "caoAgentflow_0")
        assert saved["terminalId"] == "term-9"

        _second, _events, _store = await _collect(
            flow, tmp_path, http=client, store=store, runId="run-2", inputs={"question": "again"}
        )
    run_steps = [call for call in captured["calls"] if call[1] == "/terminals/run-step"]
    sessions = [call for call in captured["calls"] if call[1] == "/sessions"]
    assert sessions[0][3]["session_name"] == "flowf4bdc13bcaoAgent"
    assert run_steps[1][2]["reuse_terminal_id"] == "term-9"


async def test_cao_404_retry_assign_and_failure(tmp_path) -> None:
    captured: dict[str, Any] = {"calls": []}
    store = SessionStore(tmp_path)
    store.save_cao(
        "flow-1",
        "caoAgentflow_0",
        {"caoSessionName": "flowf4bdc13bcaoAgent", "terminalId": "gone"},
    )
    flow = load_example("cao.json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(_cao_handler(captured))) as client:
        outcome, _events, _store = await _collect(flow, tmp_path, http=client, store=store)
    assert outcome.status == "FINISHED"
    run_steps = [call for call in captured["calls"] if call[1] == "/terminals/run-step"]
    assert "reuse_terminal_id" not in run_steps[1][2]

    assign = graph(
        [
            _start(),
            node(
                "cao",
                "caoAgentflow",
                {
                    "caoServerUrl": "http://127.0.0.1:9889",
                    "caoProvider": "cursor_cli",
                    "caoMode": "assign",
                    "caoTask": "look",
                    "caoWaitBehavior": "wait",
                    "caoPollIntervalSeconds": 0.01,
                    "caoTimeoutSeconds": 5,
                },
            ),
        ],
        [edge("start", "cao")],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_cao_handler(captured))) as client:
        waited, _events, _store = await _collect(assign, tmp_path, http=client, sessionId="assign")
    assert waited.text == "polled"

    inbox = graph(
        [
            _start(),
            node(
                "cao",
                "caoAgentflow",
                {
                    "caoServerUrl": "http://127.0.0.1:9889",
                    "caoProvider": "codex",
                    "caoMode": "send_message",
                    "caoReceiverId": "term-r",
                    "caoTask": "ping",
                    "caoShapePrompt": True,
                },
            ),
        ],
        [edge("start", "cao")],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_cao_handler(captured))) as client:
        sent, _events, _store = await _collect(inbox, tmp_path, http=client, sessionId="inbox")
    assert sent.status == "FINISHED"
    assert any(call[1].endswith("/inbox/messages") for call in captured["calls"])

    boom = load_example("cao.json")
    boom["nodes"][1]["data"]["inputs"]["caoTask"] = "explode"
    async with httpx.AsyncClient(transport=httpx.MockTransport(_cao_handler(captured))) as client:
        failed, events, _store = await _collect(
            boom, tmp_path, http=client, sessionId="boom", chatflowId="flow-boom"
        )
    assert failed.status == "ERROR"
    assert "timed out" in (failed.error or "")
    assert any(
        event.get("type") == "conversation" and event.get("status") == "error" for event in events
    )


async def test_execute_cao_direct_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "bad session"})

    with pytest.raises(CaoProtocolError, match="claim id"):
        normalize_cao_config(
            {
                "caoServerUrl": "http://127.0.0.1:9889",
                "caoTask": "x",
                "caoClaimId": "not-hex",
            },
            env={},
            credential={},
        )
    with pytest.raises(CaoProtocolError, match="job id"):
        normalize_cao_config(
            {"caoServerUrl": "http://127.0.0.1:9889", "caoTask": "x", "caoJobId": "zz"},
            env={},
            credential={},
        )
    with pytest.raises(CaoProtocolError):
        normalize_cao_config({"caoTask": "x"}, env={}, credential={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        good = normalize_cao_config(
            {"caoServerUrl": "http://127.0.0.1:9889", "caoTask": "x", "caoMode": "handoff"},
            env={},
            credential={},
        )
        with pytest.raises(CaoProtocolError, match="session create"):
            await execute_cao(
                good, {}, chatflow_id="flow-1", node_id="caoAgentflow_0", client=client
            )


async def test_custom_function_runner(monkeypatch, tmp_path) -> None:
    assert "return" in wrap_user_code("return 1")
    assert wrap_user_code("1 + 2").count("return") == 1
    with pytest.raises(CustomFunctionError):
        wrap_user_code("   ")
    monkeypatch.setattr("binex.flowise.custom_function.shutil.which", lambda _name: None)
    with pytest.raises(CustomFunctionError, match="Node.js"):
        await run_javascript("return 1", flow_input=None, flow={}, variables={})

    class Proc:
        def __init__(self, stdout: bytes, code: int = 0) -> None:
            self.returncode = code
            self._stdout = stdout

        async def communicate(self, _payload: bytes) -> tuple[bytes, bytes]:
            return self._stdout, b""

        def kill(self) -> None:
            return None

    async def fake_exec(*_args: Any, **_kwargs: Any) -> Proc:
        return Proc(b'{"ok": true, "result": 3}')

    monkeypatch.setattr(
        "binex.flowise.custom_function.shutil.which", lambda _name: "/usr/bin/node"
    )
    monkeypatch.setattr("binex.flowise.custom_function.asyncio.create_subprocess_exec", fake_exec)
    assert await run_javascript("return 1", flow_input={"n": 1}, flow={}, variables={}) == 3

    async def bad_exec(*_args: Any, **_kwargs: Any) -> Proc:
        return Proc(b'{"ok": false, "error": "boom"}', code=1)

    monkeypatch.setattr("binex.flowise.custom_function.asyncio.create_subprocess_exec", bad_exec)
    with pytest.raises(CustomFunctionError, match="boom"):
        await run_javascript("return 1", flow_input=None, flow={}, variables={})

    async def slow(*_args: Any, **_kwargs: Any) -> Proc:
        return Proc(b"{}")

    async def timeout(awaitable: Any, *_args: Any, **_kwargs: Any) -> Any:
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise TimeoutError

    monkeypatch.setattr("binex.flowise.custom_function.asyncio.create_subprocess_exec", slow)
    monkeypatch.setattr("binex.flowise.custom_function.asyncio.wait_for", timeout)
    with pytest.raises(CustomFunctionError, match="timed out"):
        await run_javascript("return 1", flow_input=None, flow={}, variables={}, timeout_ms=10)

    flow = graph(
        [
            _start(),
            node(
                "fn",
                "customFunctionAgentflow",
                {
                    "customFunctionJavascriptFunction": "return $input.output.question",
                    "customFunctionInputVariables": [
                        {"variableName": "suffix", "variableValue": "!"}
                    ],
                },
            ),
        ],
        [edge("start", "fn")],
    )
    if os.environ.get("BINEX_SKIP_NODE") == "1":
        return
    import shutil

    if shutil.which("node") is None:
        return
    monkeypatch.undo()
    outcome, _events, _store = await _collect(flow, tmp_path, inputs={"question": "hi"})
    assert outcome.text == "hi"


async def test_secrets_are_scoped_to_the_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "original-key")
    seen: dict[str, str | None] = {}

    class Spy(FakeLLM):
        async def complete(self, **kwargs: Any) -> LLMCompletion:
            seen["during"] = os.environ.get("OPENAI_API_KEY")
            return await super().complete(**kwargs)

    flow = load_example("llm.json")
    outcome, events, _store = await _collect(
        flow,
        tmp_path,
        llm=Spy("sk-from-server-99"),
        secrets={"OPENAI_API_KEY": "sk-from-server-99"},
        inputs={"question": "sk-from-server-99"},
    )
    assert seen["during"] == "sk-from-server-99"
    assert os.environ["OPENAI_API_KEY"] == "original-key"
    assert outcome.text == "sk-from-server-99"
    assert "sk-from-server-99" not in json.dumps(events)
    with pytest.raises(ValueError):
        await collect_events(run_body(flow, secrets={"PATH": "/tmp"}))


async def test_protocol_artifacts_include_files() -> None:
    events: list[dict[str, Any]] = []

    class Sink:
        async def emit(self, event: dict[str, Any]) -> None:
            events.append(event)

    request = RunRequest.model_validate(run_body(load_example("direct-reply.json")))
    node_spec = FlowNode(id="n", name="directReplyAgentflow", label="N", inputs={})
    await _emit_protocol_artifacts(
        type("Ctx", (), {"request": request})(),
        Sink(),
        node_spec,
        {
            "files": [{"name": "note.txt", "mimeType": "text/plain", "content": "hello"}],
            "cao": {"request": [{"path": "/sessions"}], "response": {"ok": True}},
        },
    )
    kinds = [event["kind"] for event in events]
    assert kinds == ["cao-request", "cao-response", "file"]
    assert events[-1]["fileName"] == "note.txt"


def test_outcome_defaults() -> None:
    outcome = ExecuteOutcome(status="FINISHED", text="", execution_data=[], state={})
    assert outcome.error is None


def test_unused_async_marker() -> None:
    assert asyncio.iscoroutinefunction(_collect)
