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
- Added a soft mental-health concern nudge to the webcam pipeline: `LegalVisionService` (`src/services/vision.py`) tracks a rolling 30-frame window of detected emotions; if ≥60% are sad/angry, it draws an on-screen therapist suggestion and sets `SystemState.mental_health_concern`/`mental_health_suggestion` (new fields + `update_mental_health_concern()`/`get_mental_health_concern()`/`get_mental_health_suggestion()` in `src/core/state.py`), clearing itself once the pattern isn't sustained. Heuristic only — not a diagnosis.
- Added `MentalHealthQAClassifier` (`src/services/classifier.py`) — a TF-IDF retrieval matcher over the newly provided `src/data/student_mh_counseling_100k_with_label_column.csv` (question/answer/label, ~100k rows; labels include depression, stress, seeking help, suicidal thoughts). Lazily loaded/indexed only on first use (`find_similar_answer()`), same pattern as the zero-shot risk model. This is the therapist-side analog of `LegalCaseClassifier`, but does supportive Q&A retrieval, not matching to a specific human therapist (that part is still not built).
- Added `POST /api/therapist-chat` and `GET /api/therapist-chat/{session_id}` to `main.py` — a separate conversation surface from `/api/chat`, backed by `MentalHealthQAClassifier`. Runs the same crisis checks (keyword, then zero-shot via the legal agent's classifier if available) before ever calling the QA classifier, since the dataset itself contains "suicidal thoughts"-labeled rows that must never substitute for `CRISIS_RESPONSE_HY`.
- Added Stripe-backed payments (card, Apple Pay, Google Pay) for both lawyer and therapist consultations: `payments` table + `create_payment`/`update_payment_status`/`get_payment`/`get_payment_by_intent` in `src/db/portal_store.py`; `POST /api/payments/create-intent`, `GET /api/payments/{payment_id}`, `POST /api/payments/webhook` in `main.py`; configured via `STRIPE_SECRET_KEY`/`STRIPE_PUBLISHABLE_KEY`/`STRIPE_WEBHOOK_SECRET` env vars (never hardcoded). Added `stripe>=9.0` to `requirements.txt`.
- Added backend-ready placeholder pages: `/pay` (Stripe Payment Element demo), `/mood-tracking` (web fallback for mood tracking when not on the mobile app), `/therapist` (supportive-chat demo + link to book/pay for a real session) — same "demo, not final UI" status as the existing auth/booking page, since the real frontend is a B2B partner's responsibility.
- Extended bookings to cover therapist consultations without breaking the existing API contract: `bookings.provider_type` (migrated in, defaults to `'lawyer'` for existing rows/callers) — `POST /api/bookings` accepts an optional `provider_type: "lawyer" | "therapist"`. Registration/login already accepted any `role` string, so `role: "therapist"` needed no backend change; the demo page's role/provider-type dropdowns were extended to expose it.

## Repository Notes
- Target remote: https://github.com/Tsovinar1986/armenian_legal_chat.git
- Commit and push documentation changes after verifying the working tree.

## Next steps
- Test the live camera stream with both built-in webcam and IP/mobile sources.
- Enhance object detection and scene understanding support.
- Continue refining Armenian vision feedback and documentation.
