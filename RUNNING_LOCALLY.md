# Running the Legal AI backend locally (for teammates)

This is the short version, just for getting the backend — chat API, auth,
bookings, payments, video-call signaling — running locally. For the full
setup (payments, bookings, deployment, API contract, etc.) see
[README.md](README.md) and [START_HERE.md](START_HERE.md).

**Shortcut — skip straight to step 5:**

- **macOS/Linux:** `./start.sh`, or `make` (same as `make all`).
- **Windows:** **double-click `start.bat`** — this is the reliable way to
  hand this to a teammate who isn't comfortable in a terminal. Don't
  double-click `start.ps1` directly: Windows Explorer doesn't run `.ps1`
  files on double-click (it just opens the file in a text editor, does
  nothing, and then `http://localhost:8000` predictably shows "backend
  unreachable" because nothing ever started). If you're already in a
  PowerShell prompt, `.\start.ps1` works too.

Any of these do everything in steps 2-4 for you every time — creates
`.venv` and installs Python deps if missing, starts Ollama if it isn't
running, builds the frontend if `frontend/dist` is missing, frees port 8000
if a previous run is still holding it, then starts the backend. Safe to
re-run any time (e.g. after "backend unreachable" in the browser — just run
it again). Pass `--rebuild-frontend` (`-RebuildFrontend` on `start.ps1`, or
`make rebuild`) to force a fresh frontend build, or `--sync-only`
(`-SyncOnly` / `make sync`) to just install/refresh dependencies without
starting anything (handy right after `git pull`). You'll see it print
`Uvicorn running on http://0.0.0.0:8000` once it's up — that line is what
actually matters, not which of these you used to get there.

Only follow steps 2-4 by hand if you want more control (a specific Python
version, editing frontend code with hot reload, etc.) — otherwise the
scripts above do it all.

**"Backend unreachable" checklist (Windows, especially on a machine that
isn't your own):**
1. Did you double-click `start.bat` (not `start.ps1`)? That's the one
   difference that most often explains "I ran it and nothing happened."
2. Is a window still open showing `Uvicorn running on http://0.0.0.0:8000`?
   If that window was closed (or never got that far), the server isn't
   running — `start.bat` now keeps the window open with `pause` specifically
   so any error is readable instead of flashing shut.
3. If it stopped on a Python dependency error, read the message — it's
   almost always PyAudio needing a prebuilt wheel, or an old Python version.
   See the PortAudio/ffmpeg notes below.
4. Firewall/antivirus prompts for "Python" or "uvicorn" the first time —
   click **Allow**. This only matters for reaching the server from another
   device on the network; the machine's own browser hitting
   `http://localhost:8000` works regardless.

If `.\start.ps1` refuses to run at all from PowerShell ("running scripts is
disabled on this system"), run this once as Administrator, then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. `start.bat` avoids
needing this — it bypasses the policy for just that one run.

## 1. Get the code

```bash
git clone https://github.com/Tsovinar1986/armenian_legal_chat.git
cd armenian_legal_chat
```

If someone sent you a `.zip` instead of this being a real `git clone`, get a
real clone instead if you can — a zip snapshot can't be updated with `git pull`
later, so you'll miss fixes.

## 2. One-time setup

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

You'll also need:
- **[Node.js](https://nodejs.org/) 20.19+ or 22.12+** for step 3 (building/running the frontend) — an older Node (18.x and earlier included `node:util`, but without the `styleText` export Vite's bundler now imports) fails with a `SyntaxError` from deep inside `node_modules/rolldown` that doesn't mention Node at all. `node --version` to check what you have.
- **[Ollama](https://ollama.com/)** running, with the `nomic-embed-text` and `armenia-lawyer-router` models pulled — ask whoever set up your machine if these aren't already there.
- **ffmpeg on your PATH** — only needed for speech-to-text. macOS: `brew install ffmpeg`. Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) or your distro's package manager. Windows: download a build from [gyan.dev's ffmpeg builds](https://www.gyan.dev/ffmpeg/builds/), extract it somewhere permanent, and add its `bin` folder to your `PATH`.
- **PortAudio**, only needed for `pip install PyAudio` to succeed (voice I/O). macOS: `brew install portaudio`. Linux: `sudo apt install portaudio19-dev` (Debian/Ubuntu) or your distro's equivalent. Windows: `pip install PyAudio` normally just works from a prebuilt wheel; if it doesn't, grab a matching wheel from [Christoph Gohlke's unofficial builds](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

## 3. Build the frontend (one-time, only when its code changes)

```bash
cd frontend
npm install
npm run build
cd ..
```

This writes `frontend/dist/` — a plain static build of the chat console.
`api.py` serves it directly at `/` (see step 5), so **the backend alone is
the entire app**: no separate frontend server needed for normal use. Only
run this again after pulling changes to `frontend/src/`; skip it entirely
if `frontend/dist/` already exists and you're not touching frontend code.

(If you *are* actively editing frontend code, `npm run dev` inside
`frontend/` instead runs Vite's own dev server on port 5173 with hot
reload, proxying `/api` and `/health` to the backend on :8000 — see
`frontend/vite.config.js`. Use that instead of rebuilding on every change.)

## 4. Run it

Make sure your virtual environment from step 2 is active in this terminal
first (you'll see `(.venv)` at the start of the prompt — if not, re-run
`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on
Windows). Then, from the repo root:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

`uvicorn` is the server that runs `api.py` (the FastAPI backend) and listens
on port 8000; `--reload` restarts it automatically if you edit the Python
code. You should see it print something like:

```
✅ Classifier: Indexed 3000 historical cases.
✅ Loaded 2073 court cases from CSV
🔄 Initializing Ollama LLM with model: armenia-lawyer-router
✅ LLM initialized successfully with model: armenia-lawyer-router
✅ Legal AI chat backend ready
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If it instead prints something like `Ollama LLM failed to initialize`, Ollama
itself isn't running or is missing the `armenia-lawyer-router` model — start
Ollama and confirm with `ollama list` before trying again.

### Running this from PyCharm on Windows

Same command as above — PyCharm just gives you a place to run it without
leaving the IDE.

1. Open the integrated terminal: **View → Tool Windows → Terminal** (or `Alt+F12`). This opens a PowerShell tab at the project root, with your `.venv` usually already active automatically if PyCharm's Python interpreter is set to it (look for `(.venv)` in the prompt — if it's not there, run `.venv\Scripts\activate` yourself).
2. Start the backend:
   ```powershell
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

## 5. Open it in your browser

**http://localhost:8000**

This is the real Armenian Legal AI chat console (`frontend/`, built in step
3) — everything you do there makes a real request to the backend, nothing
is a mockup. The original hand-rolled registration/dashboard/booking/
video-call demo page still exists too, moved to
**http://localhost:8000/legacy-demo**.

See [START_HERE.md](START_HERE.md) for the full API contract if you're
integrating against the endpoints directly instead.

## If something's not working

- **"backend unreachable" in the browser (chat console's status pill / any chat message)** — this means the page's `fetch("/health")` couldn't reach a server at all, i.e. `uvicorn` isn't actually running (it's unrelated to Ollama or model setup — `/health` doesn't touch either). On Windows, first check you launched it via `start.bat` (double-click) or `.\start.ps1` from an actual PowerShell prompt, not by double-clicking `start.ps1` itself — see the checklist above. On any OS, re-run `./start.sh` / `start.bat` / `.\start.ps1` and read its output for the actual error.
- **Port already in use** (`Address already in use` / `EADDRINUSE`) — something else is already running on 8000. See "Port already in use" in [README.md](README.md) for how to find and stop it (macOS/Linux: `lsof`/`kill`, or `make stop`; Windows: `netstat`/`taskkill`, or `.\start.ps1` handles this automatically).
- **`/` returns a raw JSON 404 instead of the chat console** — `frontend/dist/` doesn't exist yet; go back to step 3 and run `npm run build`. The backend still runs fine without it (all `/api/*` endpoints work), it just has nothing to serve at `/`.
- **`npm install`/`npm run dev` fails with `SyntaxError: ... does not provide an export named 'styleText'`** — your Node.js is too old (see the Node prerequisite above). `frontend/package.json`'s `engines` field plus `frontend/.npmrc`'s `engine-strict=true` now make `npm install` itself refuse to proceed on an unsupported Node instead of installing and failing later here — if you're still hitting this, `npm --version` and `node --version` to confirm you're actually picking up the upgraded Node (a second, older Node install earlier on `PATH` is a common cause).
- **`npm install`/`npm run dev` fails with `Error: Cannot find native binding` / `Cannot find module '@rolldown/binding-<platform>'`** — a known npm bug with platform-specific optional dependencies (npm/cli#4828), not a problem with this repo's lockfile. Usually caused by an `npm install` that partially completed against the wrong Node version (see above) leaving `node_modules` in a bad state. Fix: delete both `frontend/node_modules` and `frontend/package-lock.json`, run `npm cache clean --force`, then `npm install` again.
- **Speech-to-text fails** — almost always missing `ffmpeg` on the backend's PATH, or the client denied microphone permission.
- **Upload says "Unsupported file type"** — only `.txt`/`.xlsx` (documents) and `.mp4`/`.mov`/`.avi`/`.mkv` (video) are handled right now.
- Still stuck — check the backend terminal's output first; it prints the real error (missing Ollama model, etc.).
