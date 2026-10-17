# Flowise workflow sandbox. Build from the repository root:
#   docker build -f sandbox/Dockerfile -t binex-flowise-executor .
#
# The image runs Agentflow graphs with Binex. It does not contain LLM API keys.
# The Flowise server sends those on each run. CLI coding agents are optional;
# pass build args or run /usr/local/bin/install-cli-agents.sh later.

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BINEX_FLOWISE_STATE_DIR=/var/lib/binex/flowise \
    CAO_SERVER_URL=http://127.0.0.1:9889 \
    CAO_API_HOST=127.0.0.1 \
    CAO_API_PORT=9889 \
    BINEX_FLOWISE_PORT=8787

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/binex
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir "cli-agent-orchestrator==2.5.3"

COPY sandbox/entrypoint.sh /usr/local/bin/binex-flowise-entrypoint
COPY sandbox/install-cli-agents.sh /usr/local/bin/install-cli-agents
RUN chmod 0755 /usr/local/bin/binex-flowise-entrypoint /usr/local/bin/install-cli-agents \
    && mkdir -p /var/lib/binex/flowise /workspace

ARG INSTALL_CLAUDE_CODE=0
ARG INSTALL_CODEX=0
ARG INSTALL_CURSOR_CLI=0
RUN INSTALL_CLAUDE_CODE="$INSTALL_CLAUDE_CODE" \
    INSTALL_CODEX="$INSTALL_CODEX" \
    INSTALL_CURSOR_CLI="$INSTALL_CURSOR_CLI" \
    /usr/local/bin/install-cli-agents.sh

WORKDIR /workspace
EXPOSE 8787
ENTRYPOINT ["/usr/local/bin/binex-flowise-entrypoint"]
