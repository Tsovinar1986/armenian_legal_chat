#!/usr/bin/env bash
# Clean-start wrapper for the desktop CLI (src/main.py).
#
# Why this exists: pressing Ctrl-Z to suspend src/main.py (instead of quitting
# with 'q') leaves the process alive in the background, still attached to the
# terminal. Starting a second `python src/main.py` in the same terminal after
# that can make keyboard input (e.g. a file path typed at the [u]pload prompt)
# land on the wrong process and get corrupted — the file "not found" even
# though it genuinely exists. This kills any leftover instance first so every
# run starts from a clean terminal, then launches a fresh one.
set -euo pipefail
cd "$(dirname "$0")/.."

# `ps` shows the command as invoked (usually the relative "src/main.py", not
# an absolute path), so this matches loosely on that. Tradeoff: on a machine
# with another unrelated project also launched as "python .../src/main.py",
# this would kill that too — acceptable here since this is a personal dev
# convenience script for this one repo, not something run on a shared host.
pkill -9 -f "python.*src/main\.py" 2>/dev/null && echo "🧹 Cleared a leftover src/main.py process." || true

exec python src/main.py
