"""Per-flow session file kept inside the sandbox across runs.

A2A contextId/taskId and the CAO session/terminal outlive a single node. The
Flowise server reads them back from the event stream; this file is what the
next run of the same flow loads.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any


def default_state_dir() -> Path:
    configured = os.environ.get("BINEX_FLOWISE_STATE_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".binex" / "flowise"


class SessionStore:
    """JSON file per chatflow. Writes are atomic and process-local locked."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_state_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, chatflow_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", chatflow_id) or "flow"
        return self.root / f"{safe}.json"

    def load(self, chatflow_id: str) -> dict[str, Any]:
        path = self._path(chatflow_id)
        if not path.is_file():
            return _empty()
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty()
        if not isinstance(loaded, dict):
            return _empty()
        base = _empty()
        base.update(loaded)
        return base

    def update(self, chatflow_id: str, mutator: Any) -> dict[str, Any]:
        with self._lock:
            data = self.load(chatflow_id)
            mutator(data)
            path = self._path(chatflow_id)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            return data

    def get_a2a(self, chatflow_id: str, session_id: str, node_id: str) -> dict[str, Any]:
        bucket = self.load(chatflow_id).get("a2a") or {}
        found = bucket.get(f"{session_id}:{node_id}") if isinstance(bucket, dict) else None
        return dict(found) if isinstance(found, dict) else {}

    def save_a2a(
        self,
        chatflow_id: str,
        session_id: str,
        node_id: str,
        session: dict[str, Any],
    ) -> None:
        key = f"{session_id}:{node_id}"

        def mutate(data: dict[str, Any]) -> None:
            data.setdefault("a2a", {})[key] = {
                "contextId": session.get("contextId"),
                "taskId": session.get("taskId"),
            }

        self.update(chatflow_id, mutate)

    def get_cao(self, chatflow_id: str, node_id: str) -> dict[str, Any]:
        bucket = self.load(chatflow_id).get("cao") or {}
        found = bucket.get(node_id) if isinstance(bucket, dict) else None
        return dict(found) if isinstance(found, dict) else {}

    def save_cao(self, chatflow_id: str, node_id: str, session: dict[str, Any]) -> None:
        def mutate(data: dict[str, Any]) -> None:
            data.setdefault("cao", {})[node_id] = session

        self.update(chatflow_id, mutate)

    def get_runtime(self, chatflow_id: str, session_id: str) -> dict[str, Any]:
        bucket = self.load(chatflow_id).get("runtime") or {}
        found = bucket.get(session_id) if isinstance(bucket, dict) else None
        return dict(found) if isinstance(found, dict) else {}

    def save_runtime(self, chatflow_id: str, session_id: str, runtime: dict[str, Any]) -> None:
        def mutate(data: dict[str, Any]) -> None:
            data.setdefault("runtime", {})[session_id] = runtime

        self.update(chatflow_id, mutate)

    def get_checkpoint(self, chatflow_id: str, session_id: str) -> dict[str, Any] | None:
        bucket = self.load(chatflow_id).get("checkpoint") or {}
        found = bucket.get(session_id) if isinstance(bucket, dict) else None
        return dict(found) if isinstance(found, dict) else None

    def save_checkpoint(
        self, chatflow_id: str, session_id: str, checkpoint: dict[str, Any]
    ) -> None:
        def mutate(data: dict[str, Any]) -> None:
            data.setdefault("checkpoint", {})[session_id] = checkpoint

        self.update(chatflow_id, mutate)

    def clear_checkpoint(self, chatflow_id: str, session_id: str) -> None:
        def mutate(data: dict[str, Any]) -> None:
            checkpoint = data.setdefault("checkpoint", {})
            if isinstance(checkpoint, dict):
                checkpoint.pop(session_id, None)

        self.update(chatflow_id, mutate)


def _empty() -> dict[str, Any]:
    return {"a2a": {}, "cao": {}, "runtime": {}, "checkpoint": {}}
