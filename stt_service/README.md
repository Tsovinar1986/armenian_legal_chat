# Speech-to-Text Service

Standalone from the rest of this repo — a single Python service, its own
endpoint, no dependency on `api.py`/the main app.

`POST /api/speech-to-text` a video or audio file, get back:

```json
{"success": true, "text": "...what was said...", "language": "hy"}
```

Uses [OpenAI Whisper](https://github.com/openai/whisper) — an existing,
pretrained multilingual model (no training/fine-tuning). Whisper's training
data includes Armenian, so `"hy"` works the same as any other supported
language; the language is auto-detected unless you pass one explicitly.

## Setup

```bash
cd stt_service
pip install -r requirements.txt
```

Needs `ffmpeg` on your `PATH` (macOS: `brew install ffmpeg`) — Whisper shells
out to it to decode both audio and video files, pulling the audio track out
of a video automatically.

## Run

```bash
python app.py
```

Then open **http://localhost:8090** — a simple page with a file picker and a
"Transcribe" button, no terminal interaction needed after that. Or call the
API directly:

```bash
curl -X POST http://localhost:8090/api/speech-to-text -F "file=@clip.mp4"
```

## Model size

Defaults to `medium` — noticeably more accurate than `small`/`base` on a
lower-resource language like Armenian, at the cost of a slower first
transcribe (bigger model to download once and load into memory). Override
with the `WHISPER_MODEL` env var, e.g.:

```bash
WHISPER_MODEL=small python app.py    # faster, less accurate
WHISPER_MODEL=large-v3 python app.py # slower, most accurate
```

The model downloads automatically on first use and is cached locally after
that.
