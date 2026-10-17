"""Graph walk: conditions, loops, iteration, human pause, and templates."""

from __future__ import annotations

import json
from typing import Any

import pytest

from binex.flowise.engine import collect_events
from binex.flowise.graph import FlowLoadError, load_flow
from binex.flowise.mapping import SUPPORTED, describe_flow
from binex.flowise.session import SessionStore
from binex.flowise.variables import (
    TemplateContext,
    get_path,
    lookup,
    replace_output_placeholder,
    resolve_value,
)
from tests.unit.flowise.support import (
    EXAMPLES,
    edge,
    finish_ids,
    graph,
    load_example,
    node,
    run_body,
)


def _start() -> dict[str, Any]:
    return node("start", "startAgentflow", {"startInputType": "chatInput"}, label="Start")


_EXEC_KEYS = ("store", "llm", "http", "js_runner", "skip_ssrf", "max_steps")


async def _run(flow: dict[str, Any], tmp_path, **overrides: Any):
    exec_kwargs = {key: overrides.pop(key) for key in _EXEC_KEYS if key in overrides}
    store = exec_kwargs.pop("store", None) or SessionStore(tmp_path)
    outcome, events = await collect_events(run_body(flow, **overrides), store=store, **exec_kwargs)
    return (outcome, events), store


def test_load_flow_data_string_and_bindings() -> None:
    document = load_example("direct-reply.json")
    wrapped = {"flowData": json.dumps(document)}
    loaded = load_flow(wrapped)
    rows = describe_flow(loaded)
    assert [row["flowiseType"] for row in rows] == ["startAgentflow", "directReplyAgentflow"]
    assert rows[0]["agent"] == "local://start"
    assert rows[1]["agent"] == "local://echo"
    with pytest.raises(FlowLoadError):
        load_flow({"flowData": "{"})
    with pytest.raises(FlowLoadError):
        load_flow({"nodes": [{"id": "x", "data": {}}]})


def test_every_supported_node_has_a_sample() -> None:
    seen: set[str] = set()
    for path in sorted(EXAMPLES.glob("*.json")):
        for row in describe_flow(load_flow(load_example(path.name))):
            if row["supported"]:
                seen.add(str(row["flowiseType"]))
    assert SUPPORTED <= seen


async def test_examples_are_supported() -> None:
    for name in (
        "direct-reply.json",
        "condition.json",
        "human-approve.json",
        "loop.json",
        "iteration.json",
        "llm.json",
        "a2a.json",
        "cao.json",
        "tool.json",
        "retriever.json",
        "execute-flow.json",
        "agent.json",
        "http.json",
        "custom-function.json",
        "condition-agent.json",
    ):
        rows = describe_flow(load_flow(load_example(name)))
        assert all(row["supported"] for row in rows), name


async def test_direct_reply_example(tmp_path) -> None:
    (outcome, events), _store = await _run(load_example("direct-reply.json"), tmp_path)
    assert outcome.status == "FINISHED"
    assert outcome.text == "Hello hi"
    kinds = [event["type"] for event in events]
    assert kinds[0] == "run.start"
    assert kinds[-1] == "run.finish"
    assert "token" in kinds
    artifact = next(
        event
        for event in events
        if event.get("kind") == "output" and event.get("nodeId") == "directReplyAgentflow_0"
    )
    assert artifact["executionId"] == "run-1"
    assert artifact["agentflowId"] == "flow-1"
    assert artifact["nodeLabel"] == "Reply"
    assert json.loads(artifact["content"])["content"] == "Hello hi"


async def test_condition_example_routes_else(tmp_path) -> None:
    yes, _ = await _run(
        load_example("condition.json"),
        tmp_path,
        inputs={"question": "yes"},
        sessionId="yes",
    )
    no, _ = await _run(
        load_example("condition.json"),
        tmp_path,
        inputs={"question": "no"},
        sessionId="no",
        runId="run-2",
    )
    assert yes[0].text == "approved"
    assert "directReplyAgentflow_1" not in finish_ids(yes[1])
    assert no[0].text == "rejected"
    assert "directReplyAgentflow_0" not in finish_ids(no[1])


@pytest.mark.parametrize(
    ("kind", "operation", "left", "right", "matched"),
    [
        ("string", "contains", "hello", "ell", True),
        ("string", "notContains", "hello", "x", True),
        ("string", "startsWith", "hello", "he", True),
        ("string", "endsWith", "hello", "lo", True),
        ("string", "equal", "**hi**", "hi", True),
        ("string", "notEqual", "a", "b", True),
        ("string", "isEmpty", "", "", True),
        ("string", "notEmpty", "a", "", True),
        ("string", "regex", "abc", "b.", True),
        ("string", "regex", "abc", "(", False),
        ("number", "smaller", 1, 2, True),
        ("number", "smallerEqual", 2, 2, True),
        ("number", "larger", 3, 2, True),
        ("number", "largerEqual", 2, 2, True),
        ("number", "equal", "2", 2, True),
        ("number", "notEqual", 2, 3, True),
        ("number", "isEmpty", 0, 0, False),
        ("number", "isEmpty", "", "", True),
        ("number", "notEmpty", 0, 0, True),
        ("number", "notEmpty", 1, 0, True),
        ("boolean", "equal", True, True, True),
        ("boolean", "notEqual", True, False, True),
        ("boolean", "equal", "yes", False, False),
        ("string", "contains", "hello", "zzz", False),
    ],
)
async def test_condition_operations(tmp_path, kind, operation, left, right, matched) -> None:
    flow = graph(
        [
            _start(),
            node(
                "cond",
                "conditionAgentflow",
                {
                    "conditions": [
                        {
                            "type": kind,
                            "value1": left,
                            "operation": operation,
                            "value2": right,
                        }
                    ]
                },
            ),
            node("yes", "directReplyAgentflow", {"directReplyMessage": "yes"}),
            node("no", "directReplyAgentflow", {"directReplyMessage": "no"}),
        ],
        [edge("start", "cond"), edge("cond", "yes", 0), edge("cond", "no", 1)],
    )
    (outcome, _events), _store = await _run(flow, tmp_path)
    assert outcome.text == ("yes" if matched else "no")


async def test_condition_fan_in_runs_once(tmp_path) -> None:
    flow = graph(
        [
            _start(),
            node(
                "cond",
                "conditionAgentflow",
                {
                    "conditions": [
                        {"type": "string", "value1": "a", "operation": "equal", "value2": "a"}
                    ]
                },
            ),
            node("left", "directReplyAgentflow", {"directReplyMessage": "L"}),
            node("right", "directReplyAgentflow", {"directReplyMessage": "R"}),
            node("join", "directReplyAgentflow", {"directReplyMessage": "joined"}),
        ],
        [
            edge("start", "cond"),
            edge("cond", "left", 0),
            edge("cond", "right", 1),
            edge("left", "join"),
            edge("right", "join"),
        ],
    )
    (outcome, events), _store = await _run(flow, tmp_path)
    assert outcome.text == "joined"
    assert finish_ids(events).count("join") == 1
    assert "right" not in finish_ids(events)


async def test_two_parents_wait_for_both(tmp_path) -> None:
    seen: dict[str, Any] = {}

    async def runner(_code: str, **kwargs: Any) -> str:
        seen["input"] = kwargs["flow_input"]
        return "joined"

    flow = graph(
        [
            _start(),
            node("left", "directReplyAgentflow", {"directReplyMessage": "L"}),
            node("right", "directReplyAgentflow", {"directReplyMessage": "R"}),
            node(
                "join", "customFunctionAgentflow", {"customFunctionJavascriptFunction": "return 1"}
            ),
        ],
        [
            edge("start", "left"),
            edge("start", "right"),
            edge("left", "join"),
            edge("right", "join"),
        ],
    )
    (outcome, events), _store = await _run(flow, tmp_path, js_runner=runner)
    assert outcome.status == "FINISHED"
    assert finish_ids(events).index("left") < finish_ids(events).index("join")
    assert finish_ids(events).index("right") < finish_ids(events).index("join")
    assert set(seen["input"]["json"]) == {"left", "right"}


async def test_loop_example_exits_after_max(tmp_path) -> None:
    (outcome, events), _store = await _run(load_example("loop.json"), tmp_path)
    assert outcome.text == "done"
    assert finish_ids(events) == [
        "startAgentflow_0",
        "body",
        "loopAgentflow_0",
        "body",
        "loopAgentflow_0",
        "directReplyAgentflow_1",
    ]
    loop_outputs = [
        event["output"]["content"]
        for event in events
        if event.get("type") == "node.finish" and event["nodeId"] == "loopAgentflow_0"
    ]
    assert loop_outputs[1] == "loop finished"


async def test_iteration_example_joins_items(tmp_path) -> None:
    (outcome, events), _store = await _run(load_example("iteration.json"), tmp_path)
    assert outcome.text == "a\nb"
    assert finish_ids(events).count("directReplyAgentflow_0") == 2
    iteration = next(
        event
        for event in events
        if event.get("type") == "node.finish" and event["nodeId"] == "iterationAgentflow_0"
    )
    assert iteration["output"]["iterationResults"] == ["a", "b"]


async def test_human_pause_and_resume(tmp_path) -> None:
    store = SessionStore(tmp_path)
    flow = load_example("human-approve.json")
    (paused, events), _store = await _run(flow, tmp_path, store=store)
    assert paused.status == "STOPPED"
    human = next(event for event in events if event["type"] == "human_input")
    assert human["humanInput"]["startNodeId"] == "humanInputAgentflow_0"
    assert human["humanInput"]["enableFeedback"] is True
    finish = events[-1]
    assert finish["status"] == "STOPPED"
    assert finish["humanInput"]["mode"] == "approve"
    assert store.get_checkpoint("flow-1", "sess-1") is not None

    proceeded, _ = await _run(
        flow,
        tmp_path,
        store=store,
        runId="run-2",
        resume={
            "humanInput": {
                "type": "proceed",
                "feedback": "ok",
                "startNodeId": "humanInputAgentflow_0",
            }
        },
    )
    assert proceeded[0].status == "FINISHED"
    assert proceeded[0].text == "proceeded"
    assert store.get_checkpoint("flow-1", "sess-1") is None

    await _run(flow, tmp_path, store=store, runId="run-3", sessionId="other")
    rejected, _ = await _run(
        flow,
        tmp_path,
        store=store,
        runId="run-4",
        sessionId="other",
        resume={"humanInput": {"type": "reject", "startNodeId": "humanInputAgentflow_0"}},
    )
    assert rejected[0].text == "rejected"


async def test_human_inside_iteration_resumes(tmp_path) -> None:
    store = SessionStore(tmp_path)
    flow = graph(
        [
            _start(),
            node("iter", "iterationAgentflow", {"iterationInput": '["a", "b"]'}),
            node(
                "ask",
                "humanInputAgentflow",
                {"humanInputMode": "input", "humanInputDescription": "note"},
                parent="iter",
            ),
            node(
                "item",
                "directReplyAgentflow",
                {"directReplyMessage": "{{ $iteration }}"},
                parent="iter",
            ),
        ],
        [edge("start", "iter"), edge("ask", "item")],
    )
    (first, _events), _store = await _run(flow, tmp_path, store=store)
    assert first.status == "STOPPED"
    checkpoint = store.get_checkpoint("flow-1", "sess-1")
    assert checkpoint is not None
    assert checkpoint["iterationResume"]["nodeId"] == "iter"
    assert checkpoint["iterationResume"]["index"] == 0

    (second, _events), _store = await _run(
        flow,
        tmp_path,
        store=store,
        runId="run-2",
        resume={"humanInput": {"type": "proceed", "feedback": "a", "startNodeId": "ask"}},
    )
    assert second.status == "STOPPED"
    checkpoint = store.get_checkpoint("flow-1", "sess-1")
    assert checkpoint["iterationResume"]["index"] == 1
    assert checkpoint["iterationResume"]["results"] == ["a"]

    (third, _events), _store = await _run(
        flow,
        tmp_path,
        store=store,
        runId="run-3",
        resume={"humanInput": {"type": "proceed", "feedback": "b", "startNodeId": "ask"}},
    )
    assert third.status == "FINISHED"
    assert third.text == "a\nb"


async def test_human_modes_and_unknown_choice(tmp_path) -> None:
    flow = graph(
        [
            _start(),
            node(
                "ask",
                "humanInputAgentflow",
                {"humanInputMode": "output", "humanInputDescription": "look"},
            ),
            node("done", "directReplyAgentflow", {"directReplyMessage": "seen"}),
        ],
        [edge("start", "ask"), edge("ask", "done", 0), edge("ask", "done", 1)],
    )
    (paused, events), store = await _run(flow, tmp_path, sessionId="out")
    body = events[-1]["humanInput"]
    assert body["mode"] == "output"
    assert body["elements"] == [{"type": "proceed", "label": "Acknowledge"}]
    (done, _events), _store = await _run(
        flow,
        tmp_path,
        store=store,
        sessionId="out",
        runId="run-2",
        resume={"humanInput": {"type": "proceed", "startNodeId": "ask"}},
    )
    assert done.text == "seen"

    weird = graph(
        [
            _start(),
            node("ask", "humanInputAgentflow", {"humanInputMode": "approve"}),
            node("yes", "directReplyAgentflow", {"directReplyMessage": "yes"}),
            node("no", "directReplyAgentflow", {"directReplyMessage": "no"}),
        ],
        [edge("start", "ask"), edge("ask", "yes", 0), edge("ask", "no", 1)],
    )
    _paused, store = await _run(weird, tmp_path, sessionId="weird")
    (routed, _events), _store = await _run(
        weird,
        tmp_path,
        store=store,
        sessionId="weird",
        runId="run-9",
        resume={"humanInput": {"type": "nope", "startNodeId": "ask"}},
    )
    assert routed.text == "no"


async def test_dynamic_human_description(tmp_path) -> None:
    from binex.flowise.llm import LLMCompletion

    class Fake:
        async def complete(self, **_kwargs: Any) -> LLMCompletion:
            return LLMCompletion(content="please confirm", model="openai/gpt-4o-mini")

    flow = graph(
        [
            _start(),
            node(
                "ask",
                "humanInputAgentflow",
                {
                    "humanInputMode": "approve",
                    "humanInputDescriptionType": "dynamic",
                    "humanInputModel": "chatOpenAI",
                    "humanInputModelConfig": {"modelName": "gpt-4o-mini"},
                    "humanInputModelPrompt": "Ask.",
                },
            ),
        ],
        [edge("start", "ask")],
    )
    (outcome, _events), _store = await _run(flow, tmp_path, llm=Fake())
    assert outcome.status == "STOPPED"
    assert outcome.text == "please confirm"


async def test_persist_state_roundtrip(tmp_path) -> None:
    store = SessionStore(tmp_path)

    async def runner(_code: str, **_kwargs: Any) -> str:
        return "hello"

    flow = graph(
        [
            node(
                "start",
                "startAgentflow",
                {
                    "startInputType": "chatInput",
                    "startPersistState": True,
                    "startState": [{"key": "n", "value": 1}],
                },
            ),
            node(
                "fn",
                "customFunctionAgentflow",
                {
                    "customFunctionJavascriptFunction": "return 1",
                    "customFunctionUpdateState": [{"key": "seen", "value": "{{ output }}"}],
                },
            ),
        ],
        [edge("start", "fn")],
    )
    (first, _events), _store = await _run(flow, tmp_path, store=store, js_runner=runner)
    assert first.state["seen"] == "hello"
    (second, events), _store = await _run(
        flow, tmp_path, store=store, runId="run-2", js_runner=runner
    )
    start_record = next(
        event
        for event in events
        if event.get("nodeId") == "start" and event["type"] == "node.finish"
    )
    assert start_record["state"]["seen"] == "hello"
    assert second.state["n"] == 1


async def test_webhook_form_and_schedule(tmp_path) -> None:
    webhook = graph(
        [
            node("start", "startAgentflow", {"startInputType": "webhookTrigger"}),
            node(
                "out",
                "directReplyAgentflow",
                {"directReplyMessage": "flag {{ $webhook.body.ok }}"},
            ),
        ],
        [edge("start", "out")],
    )
    (blocked, _events), _store = await _run(webhook, tmp_path, inputs={"question": ""})
    assert blocked.status == "ERROR"
    assert "Webhook Trigger" in (blocked.error or "")
    (ok, _events), _store = await _run(
        webhook,
        tmp_path,
        runId="run-w",
        sessionId="hook",
        inputs={"question": "", "chatType": "WEBHOOK", "webhook": {"body": {"ok": True}}},
    )
    assert ok.text == "flag true"

    both = graph(
        [_start(), node("out", "directReplyAgentflow", {"directReplyMessage": "x"})],
        [edge("start", "out")],
    )
    (bad, _events), _store = await _run(
        both, tmp_path, inputs={"question": "q", "form": {"name": "a"}}, sessionId="form"
    )
    assert "cannot be provided" in (bad.error or "")

    scheduled = graph(
        [
            node(
                "start",
                "startAgentflow",
                {"startInputType": "scheduleInput", "scheduleDefaultInput": "tick"},
            ),
            node(
                "out",
                "directReplyAgentflow",
                {"directReplyMessage": "{{ start.output.question }}"},
            ),
        ],
        [edge("start", "out")],
    )
    (denied, _events), _store = await _run(
        scheduled, tmp_path, inputs={"question": ""}, sessionId="sched"
    )
    assert "Scheduled Trigger" in (denied.error or "")
    (fired, _events), _store = await _run(
        scheduled,
        tmp_path,
        runId="run-s",
        sessionId="sched2",
        inputs={"question": "", "chatType": "SCHEDULED"},
    )
    assert fired.text == "tick"


async def test_unsupported_nodes_and_structural_errors(tmp_path) -> None:
    (tool, events), _store = await _run(load_example("tool.json"), tmp_path)
    assert tool.status == "FINISHED"
    assert tool.text == "4"
    assert tool.state["last"] == "4"
    assert events[0]["type"] == "run.start"
    assert events[-1]["type"] == "run.finish"

    (unknown, _events), _store = await _run(
        load_example("unknown.json"), tmp_path, sessionId="mystery"
    )
    assert unknown.status == "ERROR"
    assert "Unknown Agentflow" in (unknown.error or "")
    assert "notANode" in (unknown.error or "")

    two = graph(
        [_start(), node("other", "directReplyAgentflow", {"directReplyMessage": "x"})],
        [],
    )
    (extra, _events), _store = await _run(two, tmp_path, sessionId="two")
    assert "exactly one start" in (extra.error or "")

    sticky = graph(
        [
            _start(),
            node("note", "stickyNoteAgentflow", {"note": "hi"}),
            node("out", "directReplyAgentflow", {"directReplyMessage": "ok"}),
        ],
        [edge("start", "out")],
    )
    (fine, _events), _store = await _run(sticky, tmp_path, sessionId="sticky")
    assert fine.text == "ok"


async def test_max_steps_and_resume_errors(tmp_path) -> None:
    flow = graph(
        [
            _start(),
            node("body", "directReplyAgentflow", {"directReplyMessage": "x"}),
            node("loop", "loopAgentflow", {"loopBackToNode": "body-output-0", "maxLoopCount": 50}),
        ],
        [edge("start", "body"), edge("body", "loop")],
    )
    (outcome, _events), store = await _run(flow, tmp_path, max_steps=4)
    assert "Maximum iteration limit" in (outcome.error or "")
    missing, _ = await _run(
        load_example("human-approve.json"),
        tmp_path,
        store=store,
        sessionId="none",
        resume={"humanInput": {"type": "proceed", "startNodeId": "humanInputAgentflow_0"}},
    )
    assert "No paused run" in (missing[0].error or "")


def test_templates_and_paths() -> None:
    ctx = TemplateContext(
        question="q",
        uploads="file.txt",
        form={"name": "ada"},
        webhook={"body": {"ok": True}},
        state={"n": 2},
        chatflow_id="flow-1",
        session_id="sess",
        chat_id="chat",
        variables={"suffix": "!"},
        executed=[
            {
                "nodeId": "loop1",
                "data": {"name": "loopAgentflow", "output": {"content": "back"}},
            },
            {
                "nodeId": "http1",
                "data": {"name": "httpAgentflow", "output": {"http": {"data": {"id": 7}}}},
            },
            {
                "nodeId": "llm1",
                "data": {
                    "name": "llmAgentflow",
                    "output": {"content": "answer", "usageMetadata": {"model": "m"}},
                },
            },
        ],
        loop_counts={"loop1": 2},
        iteration={"index": 1, "value": {"city": "Paris"}, "isFirst": False, "isLast": True},
        chat_history=[{"role": "user", "content": "hi"}],
        runtime_messages_length=4,
    )
    assert lookup("question", ctx) == "file.txt\n\nq"
    assert lookup("file_attachment", ctx) == "file.txt"
    assert "user: hi" in str(lookup("chat_history", ctx))
    assert lookup("runtime_messages_length", ctx) == 4
    assert lookup("loop_count", ctx) == 2
    assert "T" in str(lookup("current_date_time", ctx))
    assert lookup("$form.name", ctx) == "ada"
    assert lookup("$webhook.body.ok", ctx) is True
    assert lookup("$vars.suffix", ctx) == "!"
    assert lookup("$flow.state.n", ctx) == 2
    assert lookup("$flow.chatflowid", ctx) == "flow-1"
    assert lookup("$iteration.city", ctx) == "Paris"
    assert lookup("$iteration.index", ctx) == 1
    assert lookup("$iteration.isLast", ctx) is True
    assert lookup("llm1", ctx) == "answer"
    assert lookup("llm1.output.usageMetadata.model", ctx) == "m"
    assert lookup("http1", ctx) == {"id": 7}
    assert lookup("missing", ctx) is None
    assert get_path({"items": [{"n": 1}]}, "items.0.n") == 1
    assert get_path({"items": []}, "items.3") is None
    rendered = resolve_value("<p>Hello {{ question }}</p>", ctx)
    assert rendered == "Hello file.txt\n\nq"
    whole = resolve_value("{{ $flow.state }}", ctx)
    assert whole == {"n": 2}
    assert resolve_value("flag {{ $webhook.body.ok }}", ctx) == "flag true"
    assert replace_output_placeholder({"last": "see {{ output }}"}, "ANSWER") == {
        "last": "see ANSWER"
    }
