# Armenian Legal AI

Project documentation is maintained in [DOCUMENTATION.md](DOCUMENTATION.md).

This repository is an Armenian-language legal assistance prototype that combines retrieval-augmented generation (RAG) over court-style text, live webcam interpretation of simple body-language cues, and voice input and output in Eastern Armenian. It is intended for experimentation and research workflows around Armenian legal text, not as a substitute for a licensed attorney.

## Remaining work
- Add object detection support for action and scene understanding.

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
| `src/main.py` | Entry point: Ollama + Chroma, vision, voice, hotkeys |
| `src/services/` | `vision.py`, `voice.py`, `ingestion.py`, `classifier.py` (case matching + zero-shot mental-health risk), `crisis_detection.py` (keyword crisis check) |
| `src/agents/legal_agent.py` | Ollama LLM prompts in Armenian + RAG context, multi-turn conversation memory |
| `src/db/repository.py` | Chroma access and PHP-style case list parsing |
| `src/db/portal_store.py` | SQLite persistence for the web portal (users, bookings, password resets); hashed passwords |
| `src/data/` | Case lists, CSVs, exports (large files may be gitignored) |
| `main.py` | FastAPI web portal: auth, bookings, WebRTC signaling, and the `/api/chat` Legal AI chat API |
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

The typed question path now normalizes Unicode input exactly like microphone input. When a typed or spoken question matches a known case, the answer automatically includes the recommended lawyer for that case plus the lawyer with the strongest approved-case track record among similar cases in the database. Both the CLI (`src/main.py`) and the web chat (`main.py`) keep real multi-turn conversation history: follow-up questions are folded into the search query, and the RAG fallback path passes the conversation into the LLM prompt, so context carries across turns instead of treating every message as a fresh, unrelated question.

### Crisis/safety and mental-health risk screening

Every call to `LegalAgent.get_advice()` (CLI and `/api/chat`) is screened for self-harm/suicide risk language before any legal-advice logic runs, using two layered signals:

1. **Keyword check** (`src/services/crisis_detection.py`) — fast, deterministic substring matching against a fixed Armenian/English phrase list.
2. **Zero-shot classification** (`LegalCaseClassifier.classify_mental_health_risk()` in `src/services/classifier.py`) — a second, heavier signal using a multilingual NLI model (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, loaded lazily via HuggingFace `transformers` on first use) to catch phrasings the fixed keyword list misses. It classifies the message against `["suicide or self-harm risk", "acute emotional or mental health crisis", "general legal question", "casual conversation"]` and only adds a positive signal above a confidence threshold — it never overrides or weakens the keyword check.

If either signal fires, `get_advice()` returns `CRISIS_RESPONSE_HY` immediately — real emergency numbers and a recommendation to contact a trusted person or a licensed therapist — without touching the classifier, vector DB, or LLM. This is a heuristic safety net, not a clinical assessment, and must never be presented as one.

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