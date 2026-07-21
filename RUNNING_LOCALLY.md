# Running the Legal AI console locally (for teammates)

This is the short version, just for seeing the chat console — language buttons,
typed/voice questions, document/video upload — running in your own browser at
`http://localhost:5173`. For the full setup (payments, bookings, deployment,
API contract, etc.) see [README.md](README.md) and [START_HERE.md](START_HERE.md).

You need **two things running at once**: the backend (answers questions,
Python) and the frontend (the web page you look at, Node/Vite). Two terminal
windows, one for each, both left open the whole time.

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
- **[Ollama](https://ollama.com/)** running, with the `nomic-embed-text` and `armenia-lawyer-router` models pulled — ask whoever set up your machine if these aren't already there.
- **Node.js + npm** — see below.
- **ffmpeg on your PATH** — only needed for the mic button. macOS: `brew install ffmpeg`. Windows: see the "ffmpeg on PATH" steps in [README.md](README.md)'s Windows section — it's a manual download + PATH setup there, no Homebrew equivalent.

### Installing Node.js + npm

`npm` isn't installed separately — it comes bundled with Node.js, so installing Node.js is the only step. `frontend/` needs it to install and run the Vite dev server (`npm install`, `npm run dev`).

**Check first** — you may already have it:
```bash
node -v
npm -v
```
If both print a version number, skip ahead to "Run it" below. Anything Node 18+ works.

**macOS:**
```bash
brew install node
```
(No Homebrew? Install it from [brew.sh](https://brew.sh/) first, or download the macOS installer from [nodejs.org](https://nodejs.org/) instead — pick the **LTS** version.)

**Windows:**
Download the **LTS** installer from [nodejs.org](https://nodejs.org/) and run it (defaults are fine — it adds `node`/`npm` to your `PATH` automatically). Or, in PowerShell, if you have `winget`:
```powershell
winget install OpenJS.NodeJS.LTS
```
Open a **new** PowerShell window afterward (PATH changes don't apply to already-open ones), then confirm with `node -v` and `npm -v`.

**Linux:**
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```
(Debian/Ubuntu shown; use your distro's package manager otherwise — the key point is get an **LTS** build, not whatever ancient `nodejs` version your distro ships by default.)

## 3. Run it — two terminals

**Terminal 1 — backend:**

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

Wait for that last line before moving on to terminal 2 — the first startup
loads the classifier/LLM and takes a bit. If it instead prints something
like `Ollama LLM failed to initialize`, Ollama itself isn't running or is
missing the `armenia-lawyer-router` model — start Ollama and confirm with
`ollama list` before trying again.

**Terminal 2 — frontend:**

```bash
cd frontend
npm install
npm run dev
```

- `npm install` reads `frontend/package.json` and downloads the frontend's dependencies (Vite, etc.) into `frontend/node_modules/` — only needed once, or again later if `package.json` changes. Takes a few seconds to a minute depending on your connection; it's normal to see some warnings, only actual errors matter.
- `npm run dev` starts the Vite dev server. You'll know it's ready when you see something like:
  ```
  VITE ready in 300 ms
  ➜  Local:   http://localhost:5173/
  ```
  Leave this running — closing the terminal (or `Ctrl+C`) stops the frontend.

> **Forgot to `cd frontend`?** Running `npm install`/`npm run dev` straight
> from the repo root used to fail with `npm ERR! code ENOENT ... no such
> file or directory, open '.../package.json'` — there's no `package.json`
> at the root, only in `frontend/`. That's fixed now: the repo root has its
> own thin `package.json` that forwards to `frontend/` automatically, so
> `npm install` / `npm run dev` work correctly from *either* the repo root
> or `frontend/`. If you still see that exact ENOENT error, you're on an
> older copy of the repo (or a `.zip` snapshot) — `git pull`, or re-download
> if it's a zip, to pick up the fix.

### Running this from PyCharm on Windows

Same two commands as above — PyCharm just gives you a place to run them
without leaving the IDE. **Backend first, always** — the frontend expects it
to already be listening on port 8000.

**Two terminals (recommended — matches the two-terminal setup above exactly):**

1. Open the integrated terminal: **View → Tool Windows → Terminal** (or `Alt+F12`). This opens a PowerShell tab at the project root, with your `.venv` usually already active automatically if PyCharm's Python interpreter is set to it (look for `(.venv)` in the prompt — if it's not there, run `.venv\Scripts\activate` yourself).
2. In that first tab, start the backend:
   ```powershell
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```
3. Click the **+** at the top of the Terminal tool window to open a *second*, independent tab (the first keeps running the backend in the background — don't close it or press `Ctrl+C` in it).
4. In that second tab:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
5. Open **http://localhost:5173** in your browser — same as any other setup.

**One terminal instead (if you'd rather not juggle two tabs):** run the backend as a background job in a single PowerShell terminal, so the same tab is free to run the frontend afterward:
```powershell
Start-Job -Name backend -ScriptBlock { uvicorn api:app --host 0.0.0.0 --port 8000 }
cd frontend
npm install
npm run dev
```
`Start-Job` launches the backend in the background and immediately returns control of the terminal — check on it anytime with `Receive-Job backend -Keep` (prints its output so far) or `Get-Job` (shows if it's still running). Stop it with `Stop-Job backend` when you're done (or just close the terminal — background jobs die with their parent session). The tradeoff versus two tabs: you don't see the backend's live log output by default, only on request via `Receive-Job`, which makes spotting a crash less obvious — worth it only if you specifically want to avoid a second tab.

## 4. Open it in your browser

**http://localhost:5173**

You should see the console: a green **● backend online** pill top-right means
both processes are talking to each other correctly. If it says
**"backend unreachable"** instead, terminal 1 either isn't running or hasn't
finished starting up yet — check it before anything else.

## What you can actually do there

- **Pick a language** first — 🇦🇲 / 🇬🇧 / 🇷🇺 buttons below the chat. This sets both the answer language and the mic's speech-recognition language.
- **Type a question** (click the box, or press `t`) and hit Send / Enter.
- **Ask by voice** — click 🎤, speak, click again to stop; it transcribes and sends automatically.
- **Upload a document or video** — 📎 button. `.txt`/`.xlsx` get embedded into the case database; `.mp4`/`.mov`/`.avi`/`.mkv` get analyzed for detected actions and emotion (the first video you upload is slow — it's loading the vision models).

Everything you do here makes a real request to the backend — nothing is a mockup.

## If something's not working

- **Port already in use** (`Address already in use` / `EADDRINUSE`) — something else is already running on 8000 or 5173. See "Port already in use" in [README.md](README.md) for how to find and stop it (macOS/Linux: `lsof`/`kill`; Windows: `netstat`/`taskkill`).
- **Mic button fails** — almost always missing `ffmpeg` on the backend's PATH, or the browser denied microphone permission.
- **Upload says "Unsupported file type"** — only `.txt`/`.xlsx` (documents) and `.mp4`/`.mov`/`.avi`/`.mkv` (video) are handled right now.
- Still stuck — check the backend terminal's output first; it prints the real error (missing Ollama model, etc.) that the browser only summarizes.
