#!/bin/bash
# One-command way to bring up the whole app (backend + merged frontend) on
# macOS or Linux. Handles the things that actually cause "it doesn't work on
# my machine": no virtualenv / missing Python deps, Ollama not running yet,
# frontend/dist missing, and a stale server still (or again) holding port
# 8000.
#
# Usage: ./start.sh [--rebuild-frontend|--sync-only]
#   --rebuild-frontend  force a fresh frontend build even if dist/ exists
#   --sync-only         install/refresh Python + npm deps, then exit --
#                        doesn't start Ollama, build the frontend, or run
#                        the backend (useful after `git pull`)
# Windows: use .\start.ps1 instead (see that file).
set -euo pipefail
cd "$(dirname "$0")"

REBUILD_FRONTEND=0
SYNC_ONLY=0
case "${1:-}" in
    --rebuild-frontend) REBUILD_FRONTEND=1 ;;
    --sync-only) SYNC_ONLY=1 ;;
esac

OS_NAME="$(uname -s 2>/dev/null || echo Unknown)"
VENV_DIR=".venv"

# --- 0. Python virtual environment + dependencies ------------------------
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "❌ No python3/python found on PATH. Install Python 3.10+ from https://www.python.org/downloads/ and try again." >&2
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment ($VENV_DIR)..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PIP="$VENV_DIR/bin/pip"
VENV_UVICORN="$VENV_DIR/bin/uvicorn"

# Only reinstall when requirements.txt changed (or first run) so this stays
# fast on every subsequent ./start.sh -- unless --sync-only explicitly asked
# for a refresh.
if [[ $SYNC_ONLY -eq 1 || ! -f "$VENV_DIR/.deps-installed" || requirements.txt -nt "$VENV_DIR/.deps-installed" ]]; then
    echo "Installing Python dependencies (this can take a few minutes the first time)..."
    "$VENV_PIP" install --upgrade pip --quiet
    if ! "$VENV_PIP" install -r requirements.txt; then
        echo "⚠️  Some Python dependencies failed to install. A common cause is" \
             "PyAudio needing PortAudio's system headers first:" \
             "macOS: brew install portaudio | Debian/Ubuntu: sudo apt install portaudio19-dev" \
             "Fix that, then re-run ./start.sh." >&2
        exit 1
    fi
    # Required separately (--no-deps) for the two-agent crew responses -- see
    # START_HERE.md. Non-fatal: the app still runs without it, just with a
    # simpler non-crew chat response.
    "$VENV_PIP" install --no-deps crewai==1.15.2 --quiet || \
        echo "⚠️  crewai install failed -- chat still works, just without the two-agent crew (falls back automatically)."
    touch "$VENV_DIR/.deps-installed"
fi
echo "✅ Python dependencies installed ($VENV_DIR)"

if [[ $SYNC_ONLY -eq 1 ]]; then
    if command -v npm >/dev/null 2>&1; then
        echo "Syncing frontend dependencies..."
        (cd frontend && npm install --silent)
        echo "✅ Frontend dependencies installed (frontend/node_modules)"
    else
        echo "⚠️  npm not found on PATH -- skipped frontend dependency sync. Install Node.js (https://nodejs.org/) if you need it."
    fi
    echo "✅ Sync complete."
    exit 0
fi

# --- 1. Ollama -------------------------------------------------------------
ollama_up() { curl -s -o /dev/null http://localhost:11434/api/tags; }

if ! ollama_up; then
    echo "Starting Ollama..."
    if [[ "$OS_NAME" == "Darwin" && -d "/Applications/Ollama.app" ]]; then
        # Starts the app (and its bundled `ollama serve`) if it isn't
        # running, does nothing if it already is -- safer than invoking
        # `ollama serve` directly, which would start a second, unmanaged
        # instance alongside the app's own.
        open -a Ollama
    elif command -v ollama >/dev/null 2>&1; then
        # Linux (and macOS without the Ollama.app installed): if Ollama's own
        # background service already owns port 11434 this just fails
        # immediately with "address already in use" -- harmless, the loop
        # below only cares whether the API is actually responding.
        nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
        disown 2>/dev/null || true
    else
        echo "⚠️  Ollama isn't installed -- get it from https://ollama.com/. The" \
             "backend will still start, but chat responses will fail until it's running."
    fi
    for i in $(seq 1 30); do
        ollama_up && break
        sleep 1
    done
fi
if ollama_up; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama still isn't responding after 30s -- the backend will start" \
         "anyway, but chat responses will fail until Ollama is up."
fi

# --- 2. Frontend -------------------------------------------------------------
if [[ ! -d frontend/dist || $REBUILD_FRONTEND -eq 1 ]]; then
    if ! command -v npm >/dev/null 2>&1; then
        echo "❌ npm not found on PATH. Install Node.js (https://nodejs.org/) and re-run ./start.sh." >&2
        exit 1
    fi
    echo "Building frontend..."
    (cd frontend && npm install --silent && npm run build)
fi
echo "✅ Frontend build present (frontend/dist)"

# --- 3. Backend --------------------------------------------------------------
# Free port 8000 if a previous run is still (or again) holding it, so this
# script is always safe to re-run instead of failing with EADDRINUSE. Not
# every minimal Linux install has lsof, so fall back to fuser.
EXISTING_PID=""
if command -v lsof >/dev/null 2>&1; then
    EXISTING_PID=$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null || true)
elif command -v fuser >/dev/null 2>&1; then
    EXISTING_PID=$(fuser 8000/tcp 2>/dev/null | tr -d ' ' || true)
fi
if [[ -n "$EXISTING_PID" ]]; then
    echo "Stopping previous backend (PID $EXISTING_PID)..."
    kill $EXISTING_PID 2>/dev/null || true
    sleep 2
fi

echo "Starting backend on http://localhost:8000 ..."
exec "$VENV_UVICORN" api:app --host 0.0.0.0 --port 8000
