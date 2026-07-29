# One-command way to bring up the whole app (backend + merged frontend) on
# Windows -- the PowerShell equivalent of start.sh (macOS/Linux). Handles
# the same three things that actually cause "backend unreachable" in
# practice: Ollama not running yet, frontend/dist missing, and a stale
# server still (or again) holding port 8000.
#
# Usage (from the repo root, in PowerShell):
#   .\start.ps1
#   .\start.ps1 -RebuildFrontend
#
# If PowerShell blocks running local scripts, run this once as
# Administrator: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param(
    [switch]$RebuildFrontend
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
uvicorn api:app --host 0.0.0.0 --port 8000
