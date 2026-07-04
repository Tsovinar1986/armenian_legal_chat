# Armenian Legal AI

Project documentation is maintained in [DOCUMENTATION.md](DOCUMENTATION.md).

This repository is an Armenian-language legal assistance prototype that combines retrieval-augmented generation (RAG) over court-style text, live webcam interpretation of simple body-language cues, and voice input and output in Eastern Armenian. It is intended for experimentation and research workflows around Armenian legal text, not as a substitute for a licensed attorney.

## What it does

- Interactive app (`src/main.py`): runs a desktop loop with a webcam preview and on-screen Armenian hints from pose and object detection, optional microphone questions via Google Speech Recognition (`hy-AM`), and answers from a local Ollama model grounded in a Chroma vector store. Press the documented keys to upload `.txt` case dumps or `.xlsx` rows and extend the knowledge base.

- Data preparation under `src/`: scripts such as `Extraction_text.py` turn raw `caseList`-style exports into tabular data. `analysis.py` reads a CSV of legal cases and writes a JSON summary (lengths, missing fields, frequent tokens).

- Notebooks in `notebook/`: exploratory analysis on full Armenian document text, labeling helpers, and embedding plus classical ML experiments (`Modeling.ipynb`, `Modeling1.ipynb`) using sentence-transformers, scikit-learn, and XGBoost.

## Repository layout

| Path | Role |
|------|------|
| `src/main.py` | Entry point: Ollama + Chroma, vision, voice, hotkeys |
| `src/services/` | `vision.py`, `voice.py`, `ingestion.py` |
| `src/agents/legal_agent.py` | Ollama LLM prompts in Armenian + RAG context |
| `src/db/` | Chroma access and PHP-style case list parsing |
| `src/data/` | Case lists, CSVs, exports (large files may be gitignored) |
| `notebook/` | EDA, labeling, and modeling experiments |

## Prerequisites

1. Python 3.10 or newer (3.12 used in development).
2. [Ollama](https://ollama.com/) on your PATH. The app expects `nomic-embed-text` for embeddings and `armenia_lawyer_router:latest` (or a compatible tag) for answers. Pull with `ollama pull nomic-embed-text` and `ollama pull armenia_lawyer_router`, or change the names in code.
3. Webcam and microphone for the full interactive experience.
4. Network access for Google Web Speech API when using the voice service.

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

The typed question path now normalizes Unicode input exactly like microphone input, and after a typed question you can optionally enter a lawyer's name to search that lawyer's cases from the database.

Ultralytics downloads `yolov8n.pt` on first use. NLTK-based notebooks may need `nltk.download("punkt")` once.

## Run the main app

From the project root so `src` imports resolve:

```bash
python src/main.py
```

Controls match the on-screen help: **m** speak, **t** type a question, **u** upload a document, **q** quit and close the video window.

Vector data is stored under `./chroma_legal_data` by default.

## Notebooks

```bash
jupyter notebook
# or
jupyter lab
```

Open files under `notebook/`. Paths inside notebooks may still point to the author’s machine; update paths to match your clone.

## 