#!/usr/bin/env bash
# Run Fiddlehead's three processes together — the training engine, the API and the
# web app — with each line of output tagged by its source. Ctrl-C stops all three.
#
#   ./scripts/dev.sh                    # http://localhost:5173
#   PYTHON=.venv/bin/python ./scripts/dev.sh
#   API_PORT=8010 WEB_PORT=5199 ./scripts/dev.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
export VITE_API_URL="${VITE_API_URL:-http://localhost:$API_PORT}"

# stop every process this script started when it exits for any reason
trap 'trap - INT TERM EXIT; kill -- -$$ 2>/dev/null || kill 0' INT TERM EXIT

tag() {  # tag <label> <ansi colour>: prefix each line, flushing as it goes
  awk -v p="$(printf '\033[%sm%-6s\033[0m│ ' "$2" "$1")" '{ print p $0; fflush() }'
}

(cd "$ROOT/backend" && exec "$PY" engine.py) 2>&1 | tag engine 32 &
(cd "$ROOT/backend" && exec "$PY" -m uvicorn api.app:app --port "$API_PORT") 2>&1 | tag api 36 &
(cd "$ROOT/frontend" && exec npm run dev -- --port "$WEB_PORT" --strictPort) 2>&1 | tag web 35 &

printf '\n  \033[1mFiddlehead\033[0m  →  http://localhost:%s   (engine ws://localhost:8765 · API http://localhost:%s)\n\n' "$WEB_PORT" "$API_PORT"
wait
