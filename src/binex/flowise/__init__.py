"""Execute Flowise Agentflow V2 graphs inside a workflow sandbox.

The Flowise server keeps editing and viewing. This package is the runtime that
runs a flow JSON inside the sandbox and streams events the server can store.
"""

from binex.flowise.engine import ExecuteOutcome, execute_flow
from binex.flowise.graph import load_flow
from binex.flowise.mapping import describe_flow

__all__ = ["ExecuteOutcome", "describe_flow", "execute_flow", "load_flow"]
