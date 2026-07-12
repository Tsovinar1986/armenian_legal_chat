# Project Documentation

## Overview
This repository contains an Armenian legal chat application, a real-time vision analysis pipeline, and supporting documentation. It now supports live webcam and network camera streams for real-time Armenian action detection and emotion inference.

## Project Summary
- Added shared vision classifier support in `src/services/vision_classifier.py`.
- Added Armenian body-language and activity detection support in `src/services/vision.py`.
- Added emotion detection support using face landmarks and heuristic inference.
- Added live camera source selection in `src/main.py` for built-in webcams and IP/mobile streams.
- Documented usage for `0`, `http://...`, and `rtsp://...` camera sources.
- Maintained the configured GitHub remote `https://github.com/Tsovinar1986/armenian_legal_chat.git`.

## Implemented updates
- Real-time webcam and network camera streaming support.
- Armenian-language action detection for gestures like hands-on-hips, pointing, walking, running, and more.
- Emotion detection support for facial expression inference.
- Documentation updates to mirror the new live stream and vision capabilities.

## Recent Changes
- Added camera source prompt and IP stream fallbacks in `src/main.py`.
- Extended action detection heuristics in `src/services/vision.py`.
- Updated `README.md`, `DOCUMENTATION.md`, and `DOCUMENTATION_hy.md` with the new runtime instructions.
- Replaced the standalone `[l]awyer` CLI control with automatic top-lawyer ranking: `LegalCaseClassifier.get_top_lawyer_for_query()` ranks lawyers by approved cases among similar matches, surfaced inline in `LegalAgent.get_advice()` responses.
- Fixed a bug where `src/main.py` initialized `LegalCaseClassifier` against an empty auto-created `data/` folder instead of `src/data/`, silently returning zero cases for every search.
- Added a browser-based chat API (`POST /api/chat`, `GET /api/chat/{session_id}`) to the FastAPI portal (`main.py`), wired to the same `LegalAgent` used by the CLI.
- Added real multi-turn conversation memory: `LegalAgent.get_advice()` now accepts a `history` list, folds recent user turns into the search query so follow-up questions retrieve relevant cases, and passes the conversation into the LLM prompt for the RAG fallback path. Wired into both the CLI (`src/main.py`) and the web chat (`main.py`).
- Replaced the web portal's in-memory `users_db`/`bookings_db` (lost on restart, plaintext passwords) with SQLite persistence in `src/db/portal_store.py`, storing salted PBKDF2 password hashes instead of plaintext.
- Removed the standalone `[s]imilar` and `[a]pproved` CLI controls. That functionality now surfaces automatically inside `LegalAgent.get_advice()` answers: a "📚 Նմանատիպ գործեր" block lists up to 3 other similar cases with lawyer name and a ✅ approved marker each, in both the classifier-match path and the RAG fallback path. `LegalCaseClassifier.find_multiple_similar_cases()` now tags each case with `is_approved`.
- `LegalVisionService` (`src/services/vision.py`) now lazy-loads `VisionClassifier` (PyTorch + YOLOv8 + MediaPipe Pose/FaceMesh) via a `classifier` property, only on the first real frame or uploaded video, instead of unconditionally at construction. Text/voice-only sessions — including every web chat request — no longer pay that import/memory cost, which matters on memory-constrained machines (measured ~268MB saved on an 8GB MacBook already running near its swap limit).
- Added crisis/safety detection: `src/services/crisis_detection.py` provides `detect_crisis_signal()` (fast Armenian/English keyword check) and `CRISIS_RESPONSE_HY` (real emergency-resource response). Wired as Step 0 in `LegalAgent.get_advice()`, ahead of every other check, so it applies to both the CLI and `/api/chat`.
- Added zero-shot mental-health risk classification as a second, heavier signal alongside the keyword check: `LegalCaseClassifier.classify_mental_health_risk()` (`src/services/classifier.py`) uses a lazily-loaded HuggingFace `transformers` zero-shot pipeline (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) to classify a message against `["suicide or self-harm risk", "acute emotional or mental health crisis", "general legal question", "casual conversation"]`. It only adds coverage (catches phrasings the fixed keyword list misses) — it never overrides a keyword hit, and a low-confidence or failed classification silently falls through to normal legal-advice handling. The pipeline loader is split into a standalone `_load_zero_shot_pipeline()` function for testability, and the classifier itself is lazy-loaded via a `zero_shot_classifier` property so constructing `LegalCaseClassifier` stays fast and the model (~280MB) is only downloaded/loaded on first actual use.
- `LegalAgent.get_advice()` now has both crisis checks (Step 0: keyword, Step 0b: zero-shot) ahead of the short-query clarification check and all legal-advice logic — a mocked end-to-end test confirms a zero-shot-only risk flag (no keyword hit) still short-circuits before touching `classifier.find_similar_case`, the vector DB, or the LLM.
- Added `transformers>=4.40` and `torch>=2.2` to `requirements.txt` for the zero-shot classifier.

## Repository Notes
- Target remote: https://github.com/Tsovinar1986/armenian_legal_chat.git
- Commit and push documentation changes after verifying the working tree.

## Next steps
- Test the live camera stream with both built-in webcam and IP/mobile sources.
- Enhance object detection and scene understanding support.
- Continue refining Armenian vision feedback and documentation.
