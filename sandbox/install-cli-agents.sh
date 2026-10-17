#!/bin/sh
# Optional CLI coding agents. API keys stay out of the image; the Flowise
# server injects them per run. Debian's Node is enough for Custom Function.
# Claude Code, Codex, and Cursor may need a newer Node than bookworm ships.
set -eu

if [ "${INSTALL_CLAUDE_CODE:-0}" = "1" ]; then
  npm install -g @anthropic-ai/claude-code
fi

if [ "${INSTALL_CODEX:-0}" = "1" ]; then
  npm install -g @openai/codex
fi

if [ "${INSTALL_CURSOR_CLI:-0}" = "1" ]; then
  curl -fsS https://cursor.com/install | bash
fi

echo "CLI agent install flags: claude=${INSTALL_CLAUDE_CODE:-0} codex=${INSTALL_CODEX:-0} cursor=${INSTALL_CURSOR_CLI:-0}"
