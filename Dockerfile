# Containerizes the FastAPI backend (api.py) — auth, bookings/availability,
# payments, /api/chat, /api/therapist-chat, WebRTC signaling.
#
# Deliberately does NOT bundle Ollama into this image: Ollama is a large,
# independent model-serving runtime with its own multi-GB model downloads
# (armenia-lawyer-router, nomic-embed-text) — it runs as its own service (see
# docker-compose.yml), the same way a database wouldn't be bundled into an
# application image either.
#
# NOT covered by this image: src/main.py (the desktop CLI app — needs a local
# webcam/microphone, doesn't make sense containerized).
FROM python:3.12-slim

WORKDIR /app

# System libraries needed by opencv-python / mediapipe / PyAudio, which are in
# requirements.txt even though the web backend itself doesn't exercise the
# webcam path — kept as one requirements.txt for the whole project (see
# README.md), so the image installs all of it for now.
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps crewai==1.15.2

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
