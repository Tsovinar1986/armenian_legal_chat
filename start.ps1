# One-command way to bring up the whole app (backend + merged frontend) on
# Windows -- the PowerShell equivalent of start.sh (macOS/Linux). Handles
# the things that actually cause "backend unreachable" in practice: no
# virtualenv / missing Python deps, Ollama not running yet, frontend/dist
# missing, and a stale server still (or again) holding port 8000.
#
# Don't double-click THIS file in Explorer -- Windows opens .ps1 files in a
# text editor on double-click instead of running them, which silently does
# nothing and then looks exactly like "backend unreachable" once you try
# http://localhost:8000. Double-click start.bat instead (same directory),
# or run this file from an actual PowerShell prompt as shown below.
#
# Usage (from the repo root, in PowerShell):
#   .\start.ps1
#   .\start.ps1 -RebuildFrontend
#   .\start.ps1 -SyncOnly     # install/refresh Python + npm deps, then exit
#                              # (doesn't start Ollama/build/run -- useful
#                              # after `git pull`)
#
# If PowerShell blocks running local scripts, run this once as
# Administrator: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param(
    [switch]$RebuildFrontend,
    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Test-Ollama {
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# --- 0. Python virtual environment + dependencies ------------------------
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) { $pythonCmd = Get-Command py -ErrorAction SilentlyContinue }
if (-not $pythonCmd) {
    Write-Host "No 'python' or 'py' found on PATH. Install Python 3.10+ from https://www.python.org/downloads/ (check 'Add python.exe to PATH' during install) and try again." -ForegroundColor Red
    exit 1
}
$pythonExe = $pythonCmd.Source

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    & $pythonExe -m venv .venv
}

$venvPip = ".venv\Scripts\pip.exe"
$venvUvicorn = ".venv\Scripts\uvicorn.exe"
$depsMarker = ".venv\.deps-installed"

$needsInstall = $SyncOnly -or (-not (Test-Path $depsMarker)) -or
    ((Get-Item "requirements.txt").LastWriteTime -gt (Get-Item $depsMarker).LastWriteTime)
if ($needsInstall) {
    Write-Host "Installing Python dependencies (this can take a few minutes the first time)..."
    & $venvPip install --upgrade pip --quiet
    & $venvPip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Some Python dependencies failed to install. A common cause on Windows is PyAudio, which needs a prebuilt wheel -- see RUNNING_LOCALLY.md / README.md for the workaround. Fix that, then re-run .\start.ps1." -ForegroundColor Red
        exit 1
    }
    # Required separately (--no-deps) for the two-agent crew responses -- see
    # START_HERE.md. Non-fatal: the app still runs without it, just with a
    # simpler non-crew chat response.
    & $venvPip install --no-deps crewai==1.15.2 --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "crewai install failed -- chat still works, just without the two-agent crew (falls back automatically)." -ForegroundColor Yellow
    }
    New-Item -ItemType File -Path $depsMarker -Force | Out-Null
}
Write-Host "Python dependencies installed (.venv)" -ForegroundColor Green

if ($SyncOnly) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "Syncing frontend dependencies..."
        Push-Location frontend
        npm install --silent
        Pop-Location
        Write-Host "Frontend dependencies installed (frontend\node_modules)" -ForegroundColor Green
    } else {
        Write-Host "npm not found on PATH -- skipped frontend dependency sync. Install Node.js (https://nodejs.org/) if you need it." -ForegroundColor Yellow
    }
    Write-Host "Sync complete." -ForegroundColor Green
    exit 0
}

# --- 1. Ollama --------------------------------------------------------
if (-not (Test-Ollama)) {
    Write-Host "Starting Ollama..."
    # `ollama serve` starts the server directly. If Ollama's own background
    # app/tray process already owns port 11434, this just fails immediately
    # with "address already in use" -- harmless, the loop below only cares
    # whether the API is actually responding, not which process served it.
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 30; $i++) {
        if (Test-Ollama) { break }
        Start-Sleep -Seconds 1
    }
}
if (Test-Ollama) {
    Write-Host "Ollama is running" -ForegroundColor Green
} else {
    Write-Host "Ollama still isn't responding after 30s -- the backend will start" -ForegroundColor Yellow
    Write-Host "anyway, but chat responses will fail until Ollama is up. Install it" -ForegroundColor Yellow
    Write-Host "from https://ollama.com/download/windows if you haven't." -ForegroundColor Yellow
}

# --- 2. Frontend --------------------------------------------------------
if ((-not (Test-Path "frontend\dist")) -or $RebuildFrontend) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Host "npm not found on PATH. Install Node.js (https://nodejs.org/) and re-run .\start.ps1." -ForegroundColor Red
        exit 1
    }
    Write-Host "Building frontend..."
    Push-Location frontend
    npm install --silent
    npm run build
    Pop-Location
}
Write-Host "Frontend build present (frontend\dist)" -ForegroundColor Green

# --- 3. Backend ---------------------------------------------------------
# Free port 8000 if a previous run is still holding it, so this script is
# always safe to re-run instead of failing with "address already in use".
$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $existingPid = ($existing | Select-Object -First 1 -ExpandProperty OwningProcess)
    Write-Host "Stopping previous backend (PID $existingPid)..."
    Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "Starting backend on http://localhost:8000 ..." -ForegroundColor Green
& $venvUvicorn api:app --host 0.0.0.0 --port 8000
