# Flowise sandbox executor

Binex runs a Flowise Agentflow V2 graph inside that workflow's sandbox. The
Flowise server keeps the editor, the database, and the execution UI. It does
not walk nodes. One sandbox belongs to one workflow: create it on the first
run, and destroy it only when the workflow is deleted or explicitly released.

The sandbox image installs Python, Binex, Node, tmux, and
[cli-agent-orchestrator](https://pypi.org/project/cli-agent-orchestrator/)
(`cao-server`). CLI coding agents are optional build args. Flowise's own
engine is not installed.

## What runs

The walker follows the same queue rules as Agentflow V2: conditional branches,
fan-in, loops, iteration children (`parentNode`), and a human pause. Each node
type is bound to a Binex adapter or pattern. `binex flowise inspect --flow
flow.json` prints the binding.

| Agentflow node | Binding | Pattern |
|---|---|---|
| Start | `local://start` | |
| LLM | `llm://` | |
| Agent | `llm://` | |
| Condition | `local://condition` | `when` |
| Condition Agent | `llm://` | `when` |
| Loop | `local://loop` | `back_edge` |
| Iteration | `local://iteration` | `foreach` |
| Human Input | `human://approve`, `human://input`, or `human://output` | |
| HTTP | `local://http` | |
| Custom Function | `local://custom-function` | |
| Direct Reply | `local://echo` | |
| A2A | `flowise-a2a://message/send` | |
| CAO | `flowise-cao://run-step` | |
| Sticky Note | `local://noop` (skipped) | |

Tool, Retriever, and Execute Flow stop the run with an error that says what to
do instead. An unknown `data.name` does the same. Sticky notes are ignored.

LLM, Agent, and Condition Agent call LiteLLM with the same optional arguments
as `LLMAdapter` (`api_base`, `api_key`, `temperature`, `max_tokens`). Map a
Flowise component plus `modelName` (`chatOpenAI` + `gpt-4o-mini` becomes
`openai/gpt-4o-mini`) or pass a LiteLLM id directly. An Agent node that lists
Flowise tool catalogs or document stores fails. Set `inputs.binexTools` to
Binex tool URIs, and clear the Flowise tool list, to run those tools through
`LLMAdapter`.

A2A speaks the fork's JSON-RPC `message/send` contract. CAO speaks the fork's
`cao-server` contract (`POST /sessions`, `POST /terminals/run-step`, terminal
input, inbox). Neither path uses Binex's YAML `a2a://` or `cao://` adapters,
so the stored turns match what Flowise already writes.

Custom Function runs the JavaScript body with Node's `vm`. The context receives
`$input`, `$flow`, `$vars`, and named `$variables`. It does not receive
`process.env`.

HTTP requests go through the existing SSRF check. Private and loopback
addresses are rejected unless `BINEX_ALLOW_PRIVATE_URLS=1`. A2A and CAO are
not passed through that check, so `cao-server` on `127.0.0.1:9889` works.

### Loop exit

`loopBackToNode` is the back-edge. Edges drawn out of the Loop node are the
exit. While the loop count is below `maxLoopCount`, only the back-edge runs.
When the count reaches the max, the node emits `fallbackMessage` (or a default
sentence) and only the exit edges run. Flowise's current walker requeues the
back-edge and also follows the exit on every iteration; the product behavior
is to leave the loop after the max. This executor follows the product behavior.

## Run protocol

Invoke either command inside the sandbox:

```bash
binex flowise stdio < request.json
binex flowise serve --host 0.0.0.0 --port 8787
```

`POST /v1/runs` with `Content-Type: application/json` returns
`application/x-ndjson`. `GET /health` returns
`{"status": "ok", "executor": "binex-flowise"}`. If `BINEX_EXECUTOR_TOKEN` is
set, the request must send `Authorization: Bearer <token>`.

A request is one JSON object:

```json
{
  "runId": "execution-id",
  "chatflowId": "chatflow-id",
  "sessionId": "session-id",
  "workspaceId": "workspace-id",
  "orgId": "org-id",
  "flow": {"nodes": [], "edges": []},
  "inputs": {
    "question": "hello",
    "form": {},
    "webhook": {},
    "uploadsContent": "",
    "history": [],
    "chatType": "INTERNAL"
  },
  "secrets": {"OPENAI_API_KEY": "sk-from-the-server"},
  "credentials": {"node-or-credential-id": {"token": "sent-per-run"}},
  "vars": {},
  "resume": {"humanInput": {"type": "proceed", "feedback": "", "startNodeId": "humanInputAgentflow_0"}}
}
```

`flow` may be the React Flow document or a chatflow row whose `flowData` is a
JSON string. `secrets` are environment variable names. They are set for that
run and restored afterwards. `PATH`, `HOME`, `PYTHONPATH`, `LD_PRELOAD`,
`LD_LIBRARY_PATH`, `NODE_OPTIONS`, and `BINEX_FLOWISE_STATE_DIR` are rejected.
Secret values of 8 characters or more are stripped from events. Dict keys that
look like tokens, passwords, or API keys are replaced with `[redacted]` in
artifacts.

### Events

Each line is one JSON object.

| `type` | When |
|---|---|
| `run.start` | Run accepted. `status` is `INPROGRESS`. |
| `node.start` | Node begins. `status` is `INPROGRESS`. |
| `node.finish` | Node ended. `status` is `FINISHED` or `STOPPED`. Includes `input`, `output`, `state`. |
| `node.error` | Node failed. The run then finishes with `ERROR`. |
| `log` | Loop fallback and similar notices. |
| `token` | Direct Reply text. |
| `artifact` | One file the server should store. |
| `conversation` | One A2A or CAO turn. |
| `human_input` | Human node stopped. Same pause shape as `run.finish`. |
| `run.finish` | `FINISHED`, `STOPPED`, or `ERROR`. |

`run.finish` carries `text`, `state`, `executionData`, and, when stopped,
`humanInput` (`mode`, `description`, `startNodeId`, `nodeLabel`,
`enableFeedback`, `elements`). `ERROR` carries `error.message`. A bad flow
document or a refused secret name can emit `run.finish` without `run.start`.

### Artifacts

`type: "artifact"` matches `execution_artifact` plus inline `content` so the
server can write the blob:

`executionId`, `agentflowId`, `sessionId`, `workspaceId`, `orgId`, `nodeId`,
`nodeLabel`, `kind`, `fileName`, `mimeType`, `size`, `storagePath`, `content`.

`kind` is `input`, `output`, `a2a-request`, `a2a-response`, `cao-request`,
`cao-response`, or `file` (from `output.files[]` with `name`, `mimeType`,
`content`). `storagePath` is
`[orgId, chatflowId, "executions", runId, nodeId]`.

### Conversation turns

`type: "conversation"` matches `ConversationTurnInput` /
`agent_conversation_turn`:

`workspaceId`, `chatflowId`, `sessionId`, `executionId`, `nodeId`, `kind`
(`a2a` or `cao`), `status`, `message`, `output`, `externalSession`,
`externalTask`, `rawRequest`, `rawResponse`.

For A2A, `externalSession` is `contextId` and `externalTask` is `taskId`. The
bearer header is not part of `rawRequest`. For CAO, `externalSession` is the
session name and `externalTask` is the terminal id. `env_vars` in the recorded
request are `[redacted]`.

## Pause, resume, and session state

A Human Input node does not prompt on stdin. The run finishes with status
`STOPPED` and a checkpoint under `BINEX_FLOWISE_STATE_DIR` (default
`~/.binex/flowise/<chatflowId>.json`). Resume by sending the same flow with
`resume.humanInput`. `type` `proceed` takes output 0 and `reject` takes output
1. The checkpoint is removed when the run finishes. A resume with no
checkpoint is an error.

The same file stores, across runs of that sandbox:

- A2A `contextId` / `taskId`, keyed by session id and node id
- CAO session name and terminal id, keyed by node id (shared by every chat
  session of that workflow, matching the fork)
- Flow state when the Start node sets `startPersistState`

A human pause inside an iteration is checkpointed one level deep. Resume
re-enters the iteration node and continues at the human node. Nested
iterations that pause are not given a separate resume frame.

## Sandbox image

```bash
docker build -f sandbox/Dockerfile -t binex-flowise-executor .
# optional CLI agents
docker build -f sandbox/Dockerfile \
  --build-arg INSTALL_CLAUDE_CODE=1 \
  --build-arg INSTALL_CODEX=1 \
  --build-arg INSTALL_CURSOR_CLI=1 \
  -t binex-flowise-executor:agents .
```

`e2b.toml` at the repo root is the E2B-compatible template (`template_name =
binex-flowise-executor`). Aliyun ACS Agent Sandbox uses that SDK. The
entrypoint starts `cao-server` on `127.0.0.1:9889` when it is not already up,
then `binex flowise serve` on `0.0.0.0:8787`. Pass another command to skip
that (`docker run --rm binex-flowise-executor binex flowise inspect --flow
/workspace/flow.json`).

State is `/var/lib/binex/flowise`. Do not commit or bake API keys. The image
does not set `BINEX_ALLOW_PRIVATE_URLS`.

## What the Flowise server must change

This repository does not modify Flowise. The server work, against
`frankie-zeng/Flowise` `main` and the CAO/A2A records in PR #2, is:

1. **Dispatch, don't walk.** When `WORKFLOW_SANDBOX_ENABLED` is on, skip the
   node loop in `executeAgentFlowInner` / `buildAgentflow`. Keep the editor,
   credential storage, and execution UI. On the first run of a chatflow,
   create one ACS/E2B sandbox from `binex-flowise-executor` and remember its
   id on the workflow. Later runs reuse it. Do not kill it on finish, error,
   or human pause. Kill it only when the workflow is deleted or the user
   calls the existing sandbox release.

2. **POST the run into the sandbox.** Resolve credentials and provider keys
   on the server for that request. Send them in `secrets` and `credentials`.
   Send the saved flow JSON, the prediction inputs (`question`, `form`,
   `webhook`, `history`, `uploadsContent`, `chatType`), and the execution id
   as `runId`. Call `binex flowise stdio` or `POST /v1/runs` on port 8787.
   The sandbox is the source of truth for A2A `contextId`/`taskId` and for
   the CAO session and tmux terminal. The server may mirror those from
   events. It must not open a second CAO session on the host.

3. **Ingest the NDJSON stream.** `node.start`, `node.finish`, and `node.error`
   update the execution UI. `artifact` inserts an `execution_artifact` row
   and writes `content` to `storagePath`. `conversation` inserts an
   `agent_conversation_turn`. `token` is the Direct Reply stream. `log` can
   surface as an execution log line.

4. **Human resume.** On `human_input` or `run.finish` with `status: STOPPED`,
   store the execution as `STOPPED` and render the existing human-input
   action from `humanInput`. The follow-up request uses the same sandbox and
   may reuse the execution id. Put the UI payload
   `{type, feedback, startNodeId}` in `resume.humanInput`. Do not run the
   human node on the server.

5. **Leave unsupported nodes as a publish-time error.** If inspect or the
   run reports Tool, Retriever, Execute Flow, or an unknown type, show that
   message in the UI. Do not fall back to the in-process walker for those
   nodes while the sandbox flag is on.

Sample graphs are in `examples/flowise/`.
