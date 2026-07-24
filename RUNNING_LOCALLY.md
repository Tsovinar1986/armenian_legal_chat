# Running the Legal AI backend locally (for teammates)

This is the short version, just for getting the backend — chat API, auth,
bookings, payments, video-call signaling — running locally. For the full
setup (payments, bookings, deployment, API contract, etc.) see
[README.md](README.md) and [START_HERE.md](START_HERE.md).

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
- **ffmpeg on your PATH** — only needed for speech-to-text. macOS: `brew install ffmpeg`. Windows: download a build from [gyan.dev's ffmpeg builds](https://www.gyan.dev/ffmpeg/builds/), extract it somewhere permanent, and add its `bin` folder to your `PATH`.

## 3. Run it

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

## 4. Open it in your browser

**http://localhost:8000**

You should see the demo UI, including the Armenian Legal AI chat widget —
everything you do there makes a real request to the backend, nothing is a
mockup. See [START_HERE.md](START_HERE.md) for the full API contract if
you're integrating against the endpoints directly instead.

## If something's not working

- **Port already in use** (`Address already in use` / `EADDRINUSE`) — something else is already running on 8000. See "Port already in use" in [README.md](README.md) for how to find and stop it (macOS/Linux: `lsof`/`kill`; Windows: `netstat`/`taskkill`).
- **Speech-to-text fails** — almost always missing `ffmpeg` on the backend's PATH, or the client denied microphone permission.
- **Upload says "Unsupported file type"** — only `.txt`/`.xlsx` (documents) and `.mp4`/`.mov`/`.avi`/`.mkv` (video) are handled right now.
- Still stuck — check the backend terminal's output first; it prints the real error (missing Ollama model, etc.).
