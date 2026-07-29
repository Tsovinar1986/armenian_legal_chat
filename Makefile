# One-command way to bring up the whole app (backend + merged frontend) --
# works on macOS, Linux, AND native Windows (if `make` is installed there,
# e.g. `choco install make` or `winget install GnuWin32.Make`; also works
# unmodified from WSL or Git Bash). Thin wrapper around start.sh / start.ps1,
# which do the actual work: creating a venv, installing Python + npm deps,
# starting Ollama, building the frontend if needed, freeing port 8000 if a
# previous run is still holding it, then starting uvicorn.
#
# Usage:
#   make            # same as `make all`
#   make all        # install deps, build frontend if needed, run the backend
#   make rebuild    # same, but force a fresh frontend build
#   make sync       # install/refresh Python + npm deps only, don't run anything
#   make stop       # free port 8000 without starting anything
#   make clean      # remove .venv, frontend/dist, frontend/node_modules

.DEFAULT_GOAL := all
.PHONY: all rebuild sync stop clean help

ifeq ($(OS),Windows_NT)

SHELL := cmd.exe
.SHELLFLAGS := /C

all:
	powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1

rebuild:
	powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 -RebuildFrontend

sync:
	powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 -SyncOnly

stop:
	powershell -NoProfile -Command "$$e = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if ($$e) { Stop-Process -Id ($$e | Select-Object -First 1 -ExpandProperty OwningProcess) -Force -ErrorAction SilentlyContinue; Write-Host 'Stopped process on port 8000.' } else { Write-Host 'Nothing listening on port 8000.' }"

clean:
	powershell -NoProfile -Command "Remove-Item -Recurse -Force .venv, frontend\dist, frontend\node_modules -ErrorAction SilentlyContinue; Write-Host 'Cleaned.'"

else

SHELL := /bin/bash

all:
	@./start.sh

rebuild:
	@./start.sh --rebuild-frontend

sync:
	@./start.sh --sync-only

stop:
	@PID=$$( (command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null) || \
	         (command -v fuser >/dev/null 2>&1 && fuser 8000/tcp 2>/dev/null | tr -d ' ') || true ); \
	if [ -n "$$PID" ]; then \
		echo "Stopping process on port 8000 (PID $$PID)..."; \
		kill $$PID 2>/dev/null || true; \
	else \
		echo "Nothing listening on port 8000."; \
	fi

clean:
	rm -rf .venv frontend/dist frontend/node_modules

endif

help:
	@echo "Targets:"
	@echo "  all      (default) install deps, build frontend if needed, run backend"
	@echo "  rebuild  same as all, but force a fresh frontend build"
	@echo "  sync     install/refresh Python + npm deps only, don't run anything"
	@echo "  stop     free port 8000 without starting anything"
	@echo "  clean    remove .venv, frontend/dist, frontend/node_modules"
