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
- **[Node.js](https://nodejs.org/)** (LTS) — check with `node -v`.
- **ffmpeg on your PATH** — only needed for the mic button. macOS: `brew install ffmpeg`. Windows: see the "ffmpeg on PATH" steps in [README.md](README.md)'s Windows section — it's a manual download + PATH setup there, no Homebrew equivalent.

## 3. Run it — two terminals

**Terminal 1 — backend:**

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Wait for `Uvicorn running on http://0.0.0.0:8000` before moving on — the
first startup loads the classifier/LLM and takes a bit.

**Terminal 2 — frontend:**

```bash
cd frontend
npm install
npm run dev
```

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
