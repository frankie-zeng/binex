"""Small HTTP server the Flowise process calls inside the sandbox."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from binex.flowise.engine import execute_flow
from binex.flowise.graph import FlowLoadError
from binex.flowise.protocol import RunRequest
from binex.flowise.session import SessionStore, default_state_dir


def create_app(state_dir: Path | None = None) -> FastAPI:
    """NDJSON run API. Secrets in a request are applied only for that run."""
    app = FastAPI(title="Binex Flowise executor")
    store = SessionStore(state_dir or default_state_dir())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "executor": "binex-flowise"}

    @app.post("/v1/runs", response_model=None)
    async def start_run(request: Request) -> StreamingResponse | JSONResponse:
        denied = _unauthorized(request)
        if denied is not None:
            return denied
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)
        try:
            parsed = RunRequest.model_validate(payload)
        except ValidationError as exc:
            return JSONResponse({"error": exc.errors()}, status_code=400)
        return StreamingResponse(
            _stream(parsed, store),
            media_type="application/x-ndjson",
        )

    return app


def _unauthorized(request: Request) -> JSONResponse | None:
    expected = os.environ.get("BINEX_EXECUTOR_TOKEN")
    if not expected:
        return None
    header = request.headers.get("authorization", "")
    if header == f"Bearer {expected}":
        return None
    return JSONResponse({"error": "unauthorized"}, status_code=401)


async def _stream(parsed: RunRequest, store: SessionStore) -> AsyncIterator[str]:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    class _Sink:
        async def emit(self, event: dict[str, Any]) -> None:
            await queue.put(event)

    async def _runner() -> None:
        try:
            await execute_flow(parsed, _Sink(), store=store)
        except (FlowLoadError, ValueError, OSError) as exc:
            await queue.put(
                {
                    "type": "run.finish",
                    "runId": parsed.run_id,
                    "status": "ERROR",
                    "error": {"message": str(exc)},
                }
            )
        finally:
            await queue.put(None)

    task = asyncio.create_task(_runner())
    while True:
        event = await queue.get()
        if event is None:
            break
        yield json.dumps(event, default=str) + "\n"
    await task
