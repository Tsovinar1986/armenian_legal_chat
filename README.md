# Armenian Legal AI

Project documentation is maintained in [DOCUMENTATION.md](DOCUMENTATION.md).

This repository is an Armenian-language legal assistance prototype that combines retrieval-augmented generation (RAG) over court-style text, live webcam interpretation of simple body-language cues, and voice input and output in Eastern Armenian. It is intended for experimentation and research workflows around Armenian legal text, not as a substitute for a licensed attorney.

All rights reserved — see [LICENSE](LICENSE). This is not open-source software.

## Remaining work
- Add object detection support for action and scene understanding.
- Therapist-matching (mirroring the lawyer top-lawyer/similar-cases ranking, using `get_top_lawyer_for_query`-style logic) is not built — `MentalHealthQAClassifier` currently does supportive Q&A retrieval only, not matching to a specific human therapist.
- Payments are wired end-to-end against Stripe's API but need real `STRIPE_SECRET_KEY`/`STRIPE_PUBLISHABLE_KEY`/`STRIPE_WEBHOOK_SECRET` values and Apple Pay domain verification (both in the Stripe Dashboard) before they'll work outside of tests.
- `/mood-tracking` and `/therapist` are backend-ready placeholder pages (same status as the auth/booking demo UI) — a B2B partner's real frontend still needs to be built against the underlying APIs.
- `chat_sessions` and `therapist_chat_sessions` in `main.py` are still in-memory only and reset on server restart; only users/bookings/payments are persisted in SQLite.
- Booking a therapist session currently only wires through to payment (`/pay?consultation_type=therapist`); it doesn't yet also create a calendar `booking` record automatically.
- Availability (`/api/bookings/availability`) uses a fixed default business-hours window (09:00–18:00) for every provider — there's no per-provider schedule/working-hours configuration yet, and no concept of days off/holidays.
- The legal/therapist crews (`src/agents/legal_crew.py`, `src/agents/therapist_crew.py`) run two sequential LLM calls (researcher then writer) instead of one, so RAG-fallback and therapist-chat responses are slower than before; if `crewai` isn't installed or a call fails, both paths fall back to their pre-crew behavior (a template message / direct QA retrieval) rather than erroring.

## Completed vision updates
- Shared classifier module implemented in `src/services/vision_classifier.py` for action and emotion inference.
- Action detection support implemented in `src/services/vision.py` for Armenian body-language and activity cues, including ձեռքեր գոտկատեղին, ցուցադրում, ծռված դիրք, քայլում և վազում.
- Emotion detection support implemented in `src/services/vision.py` using face analysis and heuristic facial expression inference.

## What it does

- Interactive app (`src/main.py`): runs a desktop loop with a webcam preview and on-screen Armenian hints from pose and object detection, optional microphone questions via Google Speech Recognition (`hy-AM`), and answers from a local Ollama model grounded in a Chroma vector store. Press the documented keys to upload `.txt` case dumps or `.xlsx` rows and extend the knowledge base.

- Data preparation under `src/`: scripts such as `Extraction_text.py` turn raw `caseList`-style exports into tabular data. `analysis.py` reads a CSV of legal cases and writes a JSON summary (lengths, missing fields, frequent tokens).

- Notebooks in `notebook/`: exploratory analysis on full Armenian document text, labeling helpers, and embedding plus classical ML experiments (`Modeling.ipynb`, `Modeling1.ipynb`) using sentence-transformers, scikit-learn, and XGBoost.

## Repository layout

| Path | Role |
|------|------|
| `src/main.py` | Desktop CLI entry point: Ollama + Chroma, vision, voice, hotkeys (see the file-header comment for how this differs from `main.py`) |
| `src/services/` | `vision.py`, `voice.py`, `ingestion.py`, `classifier.py` (case matching + zero-shot mental-health risk + `MentalHealthQAClassifier`), `crisis_detection.py` (keyword crisis check) |
| `src/agents/legal_agent.py` | Orchestrates the legal RAG pipeline: classifier match → vector search → `legal_crew.py`, multi-turn conversation memory |
| `src/agents/legal_crew.py` | CrewAI researcher+writer crew that drafts the Armenian RAG-fallback answer |
| `src/agents/therapist_crew.py` | CrewAI researcher+writer crew that drafts the therapist supportive-chat reply |
| `src/db/repository.py` | Thin repo wrapper around the vector store (`CompanyLegalRepo`) |
| `src/db/vector_store.py` | `ChromaVectorStore` — talks to `chromadb` directly (not `langchain-chroma`; see "Vector search & CrewAI" below) |
| `src/db/portal_store.py` | SQLite persistence for the web portal (users, bookings, password resets, payments); hashed passwords |
| `src/data/` | Case lists, CSVs, exports (large files may be gitignored) |
| `main.py` | FastAPI web portal: auth, bookings/availability, payments, WebRTC signaling, `/api/chat` and `/api/therapist-chat` (see the file-header comment for how this differs from `src/main.py`) |
| `notebook/` | EDA, labeling, and modeling experiments |

## Prerequisites

1. Python 3.10 or newer (3.12 used in development).
2. [Ollama](https://ollama.com/) on your PATH. The app expects `nomic-embed-text` for embeddings and `armenia_lawyer_router:latest` (or a compatible tag) for answers. Pull with `ollama pull nomic-embed-text` and `ollama pull armenia_lawyer_router`, or change the names in code.
3. Webcam and microphone for the full interactive experience.
4. Network access for Google Web Speech API when using the voice service.
5. At least 8GB RAM works but is tight — Ollama keeps the LLM (~1.7GB) and embedding model (~274MB) resident while serving, on top of everything else running on your machine. Close other memory-heavy apps before running the CLI or portal if you notice heavy swapping/slowdowns. 16GB is recommended if you'll also use the webcam features. The zero-shot mental-health risk model (~280MB, `transformers` + `torch`) is loaded lazily on the first message of each process's lifetime — expect a one-time delay and memory bump the first time `get_advice()` is called.

## Setup

```bash
git clone https://github.com/Tsovinar1986/armenian_legal_chat.git
cd "armenian_legal_chat"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

All packages referenced in the README and the app are listed in `requirements.txt`. If `pip install PyAudio` fails, install PortAudio first (for example on macOS: `brew install portaudio`) and retry.

**Second, separate command required for multi-agent chat responses:**

```bash
pip install --no-deps crewai==1.15.2
```

See "Vector search & multi-agent orchestration (CrewAI)" below for why this has to be a second, `--no-deps` command rather than a normal `requirements.txt` line — running a plain `pip install crewai` in this venv will silently downgrade `chromadb` and break the vector database. If this step is skipped, `/api/chat` and `/api/therapist-chat` still work — they just fall back to a template/direct-retrieval response instead of a crew-drafted one.

### Vector search & multi-agent orchestration (CrewAI)

Legal RAG-fallback answers (`main.py`'s `/api/chat`, and `src/main.py`'s CLI) and therapist supportive-chat answers (`/api/therapist-chat`) are drafted by a two-agent CrewAI crew — a **Researcher** agent that organizes already-retrieved context (case precedents / a similar past Q&A pair), and a **Writer** agent that drafts the final response from what the Researcher produced. See `src/agents/legal_crew.py` and `src/agents/therapist_crew.py`.

Two implementation details worth knowing if you touch this code:

- **The researcher agents are never given a callable tool.** The local Ollama model in use (`armenia-lawyer-router`) does not support tool/function calling — giving a crewai `Agent` any `tools=[...]` against this model fails with `Error code: 400 ... does not support tools`. Retrieval therefore stays exactly where it already was (plain Python, before the crew runs) and is handed to the researcher as task input instead.
- **`crewai` is intentionally not a normal line in `requirements.txt`.** It hard-pins `chromadb~=1.1.0` (true across every version from 1.0.0 to 1.15.2, the newest at time of writing). That's a problem for two reasons: (1) it conflicts with the `chromadb>=1.5.9` this project actually needs, and (2) chromadb's on-disk format isn't backward compatible — `chromadb<1.2` can't even open a `./chroma_legal_data` directory written by `chromadb>=1.5` (it crashes with a Rust panic on open, not a clean error). Vector search was moved off `langchain-chroma` onto a small direct-`chromadb` wrapper (`src/db/vector_store.py`, `ChromaVectorStore`) specifically to remove that pin conflict from the *chromadb version itself*; `crewai` is still installed separately with `--no-deps` so pip never tries to resolve its chromadb requirement at all. crewai's `Agent`/`Task`/`Crew`/`LLM` classes work fine this way — only crewai's own optional memory/knowledge features need its pinned chromadb version, and this project doesn't use them.

The typed question path now normalizes Unicode input exactly like microphone input. When a typed or spoken question matches a known case, the answer automatically includes the recommended lawyer for that case plus the lawyer with the strongest approved-case track record among similar cases in the database. Both the CLI (`src/main.py`) and the web chat (`main.py`) keep real multi-turn conversation history: follow-up questions are folded into the search query, and the RAG fallback path passes the conversation into the LLM prompt, so context carries across turns instead of treating every message as a fresh, unrelated question.

### Crisis/safety and mental-health risk screening

Every call to `LegalAgent.get_advice()` (CLI and `/api/chat`) is screened for self-harm/suicide risk language before any legal-advice logic runs, using two layered signals:

1. **Keyword check** (`src/services/crisis_detection.py`) — fast, deterministic substring matching against a fixed Armenian/English phrase list.
2. **Zero-shot classification** (`LegalCaseClassifier.classify_mental_health_risk()` in `src/services/classifier.py`) — a second, heavier signal using a multilingual NLI model (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, loaded lazily via HuggingFace `transformers` on first use) to catch phrasings the fixed keyword list misses. It classifies the message against `["suicide or self-harm risk", "acute emotional or mental health crisis", "general legal question", "casual conversation"]` and only adds a positive signal above a confidence threshold — it never overrides or weakens the keyword check.

If either signal fires, `get_advice()` returns `CRISIS_RESPONSE_HY` immediately — real emergency numbers and a recommendation to contact a trusted person or a licensed therapist — without touching the classifier, vector DB, or LLM. This is a heuristic safety net, not a clinical assessment, and must never be presented as one.

### Language support (Armenian, English, others)

`POST /api/chat` and `POST /api/therapist-chat` accept an optional `language` field (short code, e.g. `"hy"`, `"en"`) in the request body:

- The crisis response (see above) is translated via `get_crisis_response(language)` in `src/services/crisis_detection.py` — currently `hy` and `en` have real translations (`CRISIS_RESPONSE_HY`, `CRISIS_RESPONSE_EN`); any other code falls back to English rather than defaulting to Armenian-only, since more people are likely to read English than Armenian.
- The LLM-drafted answer (legal RAG-fallback via `run_legal_crew`, therapist chat via `run_therapist_crew`) is instructed to respond in the requested language — codes without a display-name mapping (`LANGUAGE_NAMES` in `src/agents/legal_crew.py` / `therapist_crew.py`) are passed through as-is to the LLM; actual fluency beyond Armenian/English from the local `armenia-lawyer-router` model is unverified.
- `/api/chat` defaults to `"hy"` (unchanged behavior for existing callers); `/api/therapist-chat` defaults to `"en"` (the counseling dataset is English).
- The classifier-match and vector-match template responses in `LegalAgent.get_advice()` (Steps 2-3) are **not** localized — those remain fixed Armenian text regardless of `language`; only the free-text LLM-drafted answer and the crisis response are actually multi-language right now.

### Webcam mental-health concern nudge

`LegalVisionService` (`src/services/vision.py`) tracks a rolling window of the last 30 per-frame emotions detected by `VisionClassifier.detect_emotion()`. If at least 60% of that window is sad/angry, it draws a soft on-screen suggestion to talk to a therapist and sets `SystemState.get_mental_health_concern()` / `get_mental_health_suggestion()` so other code (e.g. a future web/mobile client) can read the flag. This clears itself once the mood pattern is no longer sustained. It is a coarse heuristic on facial expression, not a diagnosis, and only ever suggests talking to a therapist.

### Therapist chat (supportive Q&A, not legal advice)

`POST /api/therapist-chat` / `GET /api/therapist-chat/{session_id}` (and the demo page at `/therapist`) provide a separate, non-legal conversation surface backed by `MentalHealthQAClassifier` (`src/services/classifier.py`), a TF-IDF retrieval matcher over `src/data/student_mh_counseling_100k_with_label_column.csv` (question/answer/label — labels like `depression`, `stress`, `seeking help`, `suicidal thoughts`). It mirrors `LegalCaseClassifier`'s similarity approach but is lazily loaded (indexing ~100k rows is comparatively expensive) and only built on first use.

Every message still goes through the same crisis check as legal chat first (keyword, then zero-shot if the legal agent's classifier is available) — a matched Q&A answer is never returned in place of `CRISIS_RESPONSE_HY`, even though the dataset itself contains "suicidal thoughts"-labeled rows. Non-crisis responses are always labeled as a supportive-conversation demo, not a licensed therapist, with a link to `/therapist` to book a real session.

### Payments (Apple Pay / Google Pay / card) via Stripe

`POST /api/payments/create-intent`, `GET /api/payments/{payment_id}`, and `POST /api/payments/webhook` in `main.py` handle payment for both lawyer and therapist consultations through [Stripe](https://stripe.com). Stripe's Payment Element (used on the `/pay` demo page) shows card, Apple Pay, and Google Pay automatically for eligible browsers/devices from one integration — no separate code per payment method.

Requires these environment variables (never commit real keys):
- `STRIPE_SECRET_KEY` — server-side key, required for `create-intent` to work.
- `STRIPE_PUBLISHABLE_KEY` — sent to the client to initialize Stripe.js.
- `STRIPE_WEBHOOK_SECRET` — required for `/api/payments/webhook` to verify Stripe's signature.

Apple Pay additionally requires verifying your domain in the Stripe Dashboard — that step happens outside this codebase. Payments are recorded in the `payments` table in `portal.db` via `src/db/portal_store.py` (`create_payment`, `update_payment_status`, `get_payment`).

### Therapist role and bookings

Registration/login already accept `role: "therapist"` alongside `"individual"`/`"lawyer"` (the field was never restricted to just two values). `POST /api/bookings` now also accepts an optional `provider_type` (`"lawyer"` or `"therapist"`, defaults to `"lawyer"` for backward compatibility with existing callers) so the same booking flow covers both consultation types; `lawyer_name` continues to hold whichever provider's name applies. `/mood-tracking` is a web fallback landing page for mood tracking when not on the mobile app, linking to `/therapist`.

### Booking calendar: free/busy times in local time and UTC

`GET /api/bookings/availability?provider_name=...&date=YYYY-MM-DD&timezone=Area/City` returns a day's slots for a provider, each with both the local time (in the given IANA timezone) and the UTC equivalent, plus `is_free`. Business hours are a fixed default (09:00–18:00 in the given timezone, 60-minute slots by default — override with `slot_minutes`, `start_hour`, `end_hour`) since there's no per-provider schedule configuration yet.

`POST /api/bookings` now also accepts an optional `timezone` (IANA name, defaults to `"UTC"`): if `start_time` has no UTC offset, it's interpreted as local time in that timezone. Internally this is normalized to `start_time_utc` (via `src/db/portal_store.start_time_to_utc_iso`, using the stdlib `zoneinfo` — no new dependency) so availability can reliably compare bookings made in different timezones. `start_time` itself is stored unchanged for backward compatibility with the existing API contract.

The demo page's "Calendar booking" card includes a "Check free times" panel (auto-detects the browser's timezone via `Intl.DateTimeFormat`) — click a free slot to fill in the booking form.

Ultralytics downloads `yolov8n.pt` on first use. NLTK-based notebooks may need `nltk.download("punkt")` once.

## Run the main app

From the project root so `src` imports resolve:

```bash
python src/main.py
```

When prompted, choose a camera source:
- Enter `0` to use the MacBook or notebook built-in webcam.
- Enter a network camera URL such as `http://192.168.1.10:8080/video` or `rtsp://<ip>:554/stream` to use a mobile/IP stream.

Controls match the on-screen help: **m** speak, **t** type a question, **u** upload a document, **q** quit and close the video window.

Vector data is stored under `./chroma_legal_data` by default.

The vision stack (PyTorch + YOLOv8 + MediaPipe Pose/FaceMesh, `src/services/vision_classifier.py`) is loaded lazily on first actual use — a webcam frame or an uploaded video (`u`) — not at startup. If you answer `n` to the webcam prompt and never upload a video, that whole stack (and its memory footprint, ~270MB+) is never loaded, which matters on memory-constrained machines (e.g. 8GB RAM).

## Run the web portal (chat API, auth, bookings, video calls)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for a demo UI including an Armenian Legal AI chat widget, or integrate directly against `POST /api/chat` and `GET /api/chat/{session_id}` — see [START_HERE.md](START_HERE.md) for the full API contract. Users and bookings are persisted in a local SQLite database (`portal.db`, gitignored) with salted/hashed passwords instead of the earlier in-memory, plaintext-password version. Chat history is still in-memory per server process (resets on restart).

### Port already in use

If `uvicorn` fails with `[Errno 48] Address already in use`, something is already listening on that port — usually a previous `uvicorn` process that was never stopped (e.g. the terminal was closed with `Ctrl+C` skipped, or it was left running in the background).

**Option A — use a different port** (no need to touch the other process):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

Then open http://localhost:8080 instead.

**Option B — find and stop whatever is holding the port:**

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

This prints the PID (process ID) of whatever is listening on port 8000. Then stop it:

```bash
kill <PID>
```

If it doesn't stop (rare), force it with `kill -9 <PID>`. Only kill a process you recognize — `lsof` also shows the command name (`Python`, `node`, etc.) so you can confirm it's safe before killing it. Re-run `lsof -nP -iTCP:8000 -sTCP:LISTEN` afterward with no output to confirm the port is free, then start `uvicorn` again on port 8000.

## Notebooks

```bash
jupyter notebook
# or
jupyter lab
```

Open files under `notebook/`. Paths inside notebooks may still point to the author’s machine; update paths to match your clone.

## 