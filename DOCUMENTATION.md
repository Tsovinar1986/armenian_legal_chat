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

## Repository Notes
- Target remote: https://github.com/Tsovinar1986/armenian_legal_chat.git
- Commit and push documentation changes after verifying the working tree.

## Next steps
- Test the live camera stream with both built-in webcam and IP/mobile sources.
- Enhance object detection and scene understanding support.
- Continue refining Armenian vision feedback and documentation.
