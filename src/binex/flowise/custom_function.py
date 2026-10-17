"""Run an Agentflow Custom Function JavaScript body with Node's vm.

The workflow sandbox is the isolation boundary. The vm context receives the
flow input and state, not process environment secrets.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

_RUNNER = Path(__file__).with_name("custom_function.js")
_RETURN = re.compile(r"\breturn\b")


class CustomFunctionError(RuntimeError):
    """The function failed or Node.js is not available."""


def wrap_user_code(code: str) -> str:
    """Treat the editor body as an async function so ``return`` works."""
    body = code.strip()
    if not body:
        raise CustomFunctionError("Custom Function is empty")
    if _RETURN.search(body):
        wrapped = body
    else:
        wrapped = f"return (\n{body}\n);"
    return f"(async () => {{\n{wrapped}\n}})()"


async def run_javascript(
    code: str,
    *,
    flow_input: Any,
    flow: dict[str, Any],
    variables: dict[str, Any],
    extra: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
) -> Any:
    node = shutil.which("node")
    if node is None:
        raise CustomFunctionError(
            "Custom Function requires Node.js. The Binex sandbox image installs it. "
            "Install node on PATH, or replace this node with an HTTP node."
        )
    limit = timeout_ms or int(os.environ.get("BINEX_CUSTOM_FUNCTION_TIMEOUT_MS", "30000"))
    payload = json.dumps(
        {
            "code": wrap_user_code(code),
            "input": flow_input,
            "flow": flow,
            "vars": variables,
            "extra": extra or {},
            "timeoutMs": min(limit, 5000),
        }
    ).encode("utf-8")
    process = await asyncio.create_subprocess_exec(
        node,
        str(_RUNNER),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=limit / 1000)
    except TimeoutError as exc:
        process.kill()
        raise CustomFunctionError(f"Custom Function timed out after {limit}ms") from exc
    text = stdout.decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError as exc:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise CustomFunctionError(detail or "Custom Function returned invalid JSON") from exc
    if process.returncode != 0 or not parsed.get("ok", False):
        detail = str(parsed.get("error") or stderr.decode("utf-8", errors="replace")).strip()
        raise CustomFunctionError(detail or "Custom Function failed")
    return parsed.get("result")
