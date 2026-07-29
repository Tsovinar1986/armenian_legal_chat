#!/bin/bash
# One-command way to bring up the whole app (backend + merged frontend) on
# macOS. Handles the three things that actually caused "backend unreachable"
# in practice: Ollama not running yet, frontend/dist missing, and a stale
# uvicorn from a previous run still (or no longer) holding port 8000.
#
# Usage: ./start.sh [--rebuild-frontend]
set -euo pipefail
cd "$(dirname "$0")"

REBUILD_FRONTEND=0
if [[ "${1:-}" == "--rebuild-frontend" ]]; then
    REBUILD_FRONTEND=1
fi

# --- 1. Ollama --------------------------------------------------------
# `open -a Ollama` starts the app (and its bundled `ollama serve`) if it
# isn't running, and does nothing if it already is -- safer than invoking
# `ollama serve` directly, which would start a second, unmanaged instance
# alongside the app's own.
if ! curl -s -o /dev/null http://localhost:11434/api/tags; then
    echo "Starting Ollama..."
    open -a Ollama
    for i in $(seq 1 30); do
        curl -s -o /dev/null http://localhost:11434/api/tags && break
        sleep 1
    done
fi
if curl -s -o /dev/null http://localhost:11434/api/tags; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama still isn't responding after 30s -- the backend will start" \
         "anyway, but chat responses will fail until Ollama is up. Check that" \
         "Ollama.app is installed: https://ollama.com/"
fi

# --- 2. Frontend --------------------------------------------------------
if [[ ! -d frontend/dist || $REBUILD_FRONTEND -eq 1 ]]; then
    echo "Building frontend..."
    (cd frontend && npm install --silent && npm run build)
fi
echo "✅ Frontend build present (frontend/dist)"

# --- 3. Backend ---------------------------------------------------------
# Free port 8000 if a previous run is still (or again) holding it, so this
# script is always safe to re-run instead of failing with EADDRINUSE.
EXISTING_PID=$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "Stopping previous backend (PID $EXISTING_PID)..."
    kill "$EXISTING_PID"
    sleep 2
fi

echo "Starting backend on http://localhost:8000 ..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
