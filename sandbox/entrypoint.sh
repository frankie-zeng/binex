#!/bin/sh
# Start cao-server beside the Flowise executor. Secrets are not read here.
set -eu

STATE_DIR="${BINEX_FLOWISE_STATE_DIR:-/var/lib/binex/flowise}"
CAO_HOST="${CAO_API_HOST:-127.0.0.1}"
CAO_PORT="${CAO_API_PORT:-9889}"
PORT="${BINEX_FLOWISE_PORT:-8787}"

mkdir -p "$STATE_DIR"

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

if command -v cao-server >/dev/null 2>&1; then
  if ! curl -fsS "http://${CAO_HOST}:${CAO_PORT}/health" >/dev/null 2>&1; then
    cao-server --host "$CAO_HOST" --port "$CAO_PORT" --terminal tmux \
      >/tmp/cao-server.log 2>&1 &
    i=0
    while [ "$i" -lt 50 ]; do
      if curl -fsS "http://${CAO_HOST}:${CAO_PORT}/health" >/dev/null 2>&1; then
        break
      fi
      i=$((i + 1))
      sleep 0.2
    done
  fi
fi

exec binex flowise serve --host 0.0.0.0 --port "$PORT"
