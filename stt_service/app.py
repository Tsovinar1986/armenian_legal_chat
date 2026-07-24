"""Standalone speech-to-text microservice — independent of api.py / the main
app. POST a video or audio file to /api/speech-to-text and get back
{"text": "..."} with whatever was said in it, in whichever language.

Uses OpenAI's Whisper (openai-whisper), a pretrained multilingual model —
no training or fine-tuning. Whisper's training data includes Armenian, so
"hy" works out of the box like any other supported language; the language
is auto-detected unless the caller passes one explicitly.

Run: pip install -r requirements.txt, then `python app.py`
(needs ffmpeg on PATH — Whisper shells out to it to decode both audio and
video containers, pulling the audio track automatically).
"""
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import whisper
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# "medium" gives noticeably better accuracy than "small"/"base" on
# lower-resource languages like Armenian, at the cost of a slower first
# transcribe (bigger model to load/run). Override with WHISPER_MODEL if
# you want to trade quality for speed (e.g. "small") or push for the best
# possible quality ("large-v3").
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Speech-to-Text Service")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_model = None


def get_model():
    """Lazy-loaded so the server starts instantly; the model itself (plus
    a one-time download on first-ever run) only loads on the first real
    request."""
    global _model
    if _model is None:
        _model = whisper.load_model(MODEL_SIZE)
    return _model


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "model_loaded": _model is not None}


def _join_transcript(result: dict) -> str:
    """result["text"] is Whisper's own concatenation of its segments, which
    doesn't reliably insert a space at every segment boundary — on longer
    clips (more than one segment) that shows up as two words from adjacent
    segments getting glued together (e.g. "...manwalked out..."). Rebuilding
    the text by explicitly joining each segment's (stripped) text with a
    single space guarantees a word boundary at every seam regardless of what
    Whisper itself included."""
    segments = result.get("segments") or []
    if segments:
        text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
    else:
        text = (result.get("text") or "").strip()
    return re.sub(r"\s+", " ", text).strip()


@app.post("/api/speech-to-text")
async def speech_to_text(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        model = get_model()
        result = model.transcribe(
            tmp_path,
            language=language or None,
            # Defaults (no_speech_threshold=0.6, condition_on_previous_text=True)
            # are tuned for English and can make Whisper misjudge a quieter or
            # lower-resource-language (e.g. Armenian) segment as silence and
            # skip it, or drift into a repetition loop that eats the rest of
            # the transcript — both show up as "only got part of the sentence"
            # even though the whole thing was said clearly.
            no_speech_threshold=0.3,
            condition_on_previous_text=False,
            fp16=False,
        )
        return {
            "success": True,
            "text": _join_transcript(result),
            "language": result.get("language"),
        }
    except Exception as exc:
        return {"success": False, "text": "", "message": str(exc)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8090)))
