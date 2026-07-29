@echo off
rem Double-click-friendly launcher for start.ps1.
rem
rem Why this file exists: Windows does NOT run a .ps1 file when you
rem double-click it in File Explorer -- it opens in a text editor instead
rem (or does nothing), which looks exactly like "the backend is
rem unreachable" once you then try http://localhost:8000 in a browser and
rem nothing is actually running. Double-clicking THIS file (.bat) does run,
rem and it launches start.ps1 with the one-time execution-policy restriction
rem bypassed for just this process -- no need to change any system setting.
rem
rem The final `pause` keeps the window open after start.ps1 exits (success
rem or failure) so a double-click launch doesn't flash an error and close
rem before anyone can read it.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
echo.
echo (window kept open so you can read the output above -- press any key to close)
pause >nul
