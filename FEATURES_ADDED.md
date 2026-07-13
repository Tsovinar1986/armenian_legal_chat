# New Features Added to Armenian Legal AI

## Overview
Added functionality to find similar legal cases and display approved cases with lawyer information. Cases are automatically exported to text files for easy review.

---

## New Features

### 1. **Similar Cases and Approved-Case Info, Surfaced Automatically in Chat Answers**
The standalone `[s]imilar` and `[a]pproved` CLI controls have been removed. That functionality is now folded directly into the chat answer (`LegalAgent.get_advice()`), the same way the lawyer-search control (`l`) was replaced earlier — no separate lookup step required.

**Features:**
- Every classifier-match answer now includes a "📚 Նմանատիպ գործեր" (similar cases) block: up to 3 other similar cases (excluding the primary match), each showing case number, classification, lawyer name, and a ✅ "Հաստատված" (approved) marker when applicable
- The RAG fallback path (Step 4, when no classifier/vector match is found) appends the same similar-cases block after the generated answer, when relevant cases exist
- Approval detection is unchanged: cases are marked "approved" if they contain keywords like հաստատել, հաստատվել, հաճախել, մեղադրանք, դատել (Armenian) or approved, success, successful, granted, upheld, affirm, confirm (English)
- `LegalCaseClassifier.find_multiple_similar_cases()` now tags each returned case with `is_approved`, so callers don't need to re-run the approval check themselves

**How to use:**
1. Press `t` and type your legal question (or `m` to speak it), or send it via `POST /api/chat`
2. The response includes the recommended lawyer, the top lawyer by approved cases among similar matches, and a short list of other similar cases with their lawyers and approval status — all inline

---

### 2. **Automatic Top Lawyer for Typed/Spoken Questions**
When a typed (`t`) or spoken (`m`) question matches a known case via the classifier, the answer now automatically includes the lawyer with the strongest approved-case track record among similar cases — no separate lookup step required.

**Features:**
- Ranks lawyers among cases similar to the query, not just the single closest match
- Shows approved-case count out of total similar cases considered
- Appears inline in the classifier-match response, alongside the originally matched case's recommended lawyer

**How to use:**
1. Press `t` and type your legal question (or `m` to speak it)
2. If a classifier match is found, the response includes a "🏆 Ամենահաջողակ փաստաբանը" (most successful lawyer) block

The standalone "search cases by lawyer" control (previously `l`) has been removed since this information now surfaces automatically as part of the answer.

---

### 3. **Real Multi-Turn Conversation Memory**
Follow-up questions now carry context from earlier turns in the same conversation, instead of every message being answered in isolation.

**Features:**
- `LegalAgent.get_advice(user_query, history)` folds the last few user turns into the search query, so a follow-up like "what about the property?" after a divorce question retrieves property-related cases instead of returning nothing or an unrelated match
- The RAG fallback path (Step 4, when no classifier/vector match is found) passes the full recent conversation into the LLM prompt so generated answers stay consistent with what was already discussed
- Wired into both interfaces:
  - CLI (`src/main.py`): `LegalAIController` keeps `self.conversation_history` across `t`/`m` turns for the life of the process
  - Web chat (`main.py`): `/api/chat` keeps history per `session_id` (client must pass the `session_id` returned from the first response on every subsequent call)

**How to use:**
1. Ask a question, get an answer, then ask a follow-up in the same session (same CLI process, or same `session_id` for the web API)
2. The assistant resolves the follow-up using the earlier turns

---

### 4. **Browser-Based Legal AI Chat API**
The FastAPI portal (`main.py`) now exposes the same legal Q&A used by the CLI as a web API, for a B2B partner's frontend (or the built-in demo chat widget) to call directly.

**Endpoints:**
- `POST /api/chat` — body `{message, session_id}` (session_id optional on the first call), returns `{success, session_id, response}`
- `GET /api/chat/{session_id}` — returns the full message history for that session

See [START_HERE.md](START_HERE.md) for the full API contract.

---

### 5. **Persistent, Hashed-Password Storage for the Web Portal**
The portal's `users_db`/`bookings_db` were previously plain in-memory Python lists — all data (including plaintext passwords) was lost on every server restart.

**Features:**
- `src/db/portal_store.py` persists users and bookings in a local SQLite database (`portal.db`, gitignored)
- Passwords are hashed with salted PBKDF2 (`hashlib.pbkdf2_hmac`, 390,000 iterations) — never stored or returned in plaintext
- Registration, login, forgot/reset-password, bookings, and the dashboard all read/write through this store instead of in-memory lists

---

### 6. **Lazy-Loaded Vision Models (Memory Fix for 8GB Machines)**
`LegalVisionService` used to construct `VisionClassifier` (PyTorch + YOLOv8 + MediaPipe Pose/FaceMesh) unconditionally at startup — even when the webcam was disabled and no video was uploaded, which is the case for every text/voice-only chat session and every web chat request.

**Features:**
- `src/services/vision.py`'s `classifier` is now a lazy property: the import and model construction only happen on the first real webcam frame (`process_frame`) or uploaded video (`process_video`)
- Measured ~268MB saved (plus load time) for sessions that never touch vision — verified constructing `LegalVisionService` loads zero `torch`/`mediapipe`/`ultralytics` modules until `.classifier` is actually accessed
- Matters most on memory-constrained machines (e.g. 8GB RAM), where the OS was already swapping before this app even started

**How to use:**
No action needed — answering `n` to the webcam prompt (or never pressing `u` to upload a video) now genuinely skips loading the vision stack, instead of loading it anyway.

---

### 7. **Crisis/Safety Detection + Zero-Shot Mental-Health Risk Screening**
Every message passed to `LegalAgent.get_advice()` (CLI and `/api/chat`) is now screened for self-harm/suicide risk language before any legal-advice logic runs, using two layered, complementary signals.

**Features:**
- **Step 0 — keyword check** (`src/services/crisis_detection.py`): fast, deterministic substring matching against a fixed list of Armenian and English self-harm/suicide phrases (`detect_crisis_signal()`)
- **Step 0b — zero-shot classification** (`LegalCaseClassifier.classify_mental_health_risk()` in `src/services/classifier.py`): a second, heavier signal using a multilingual NLI model (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, via HuggingFace `transformers`) to catch phrasings the fixed keyword list misses — no labeled training data needed, since zero-shot classification matches the message against candidate labels (`"suicide or self-harm risk"`, `"acute emotional or mental health crisis"`, `"general legal question"`, `"casual conversation"`) at inference time
- The zero-shot signal only *adds* coverage — it never overrides or weakens the keyword check, and only flags risk when its top label is a risk label above a confidence threshold (default 0.55)
- Both the model pipeline (`zero_shot_classifier` property) and the vision stack use the same lazy-loading pattern: `transformers`/`torch` are only imported and the ~280MB model only downloaded on the first actual message that reaches this check, not at classifier construction time
- If either signal fires, `get_advice()` returns `CRISIS_RESPONSE_HY` immediately — real Armenia emergency numbers (911/102/103) plus a recommendation to contact a trusted person or licensed therapist — without touching the case classifier, vector DB, or LLM
- This is a heuristic safety net, not a clinical assessment tool, and is documented as such in code comments; it can both miss risk and over-trigger on unrelated text

**How to use:**
No action needed — this runs automatically and transparently ahead of every answer, for both the CLI (`t`/`m`) and `POST /api/chat`.

---

### 8. **Therapist Chat, Webcam Mental-Health Nudge, and Stripe Payments**
Mirrors the lawyer-side chat/booking features for a therapist/mental-health track, plus payment processing for both.

**Features:**
- **Webcam mental-health concern nudge**: `LegalVisionService` (`src/services/vision.py`) tracks a rolling 30-frame window of detected emotions; if ≥60% are sad/angry, it draws an on-screen suggestion to talk to a therapist and exposes the flag via `SystemState.get_mental_health_concern()`/`get_mental_health_suggestion()`. Clears itself once the pattern isn't sustained. Heuristic on facial expression only — not a diagnosis.
- **`MentalHealthQAClassifier`** (`src/services/classifier.py`): TF-IDF retrieval matcher over `src/data/student_mh_counseling_100k_with_label_column.csv` (~100k question/answer/label rows — labels include depression, stress, seeking help, suicidal thoughts). Lazily loaded/indexed on first use, mirroring the zero-shot risk model's pattern.
- **`POST /api/therapist-chat`, `GET /api/therapist-chat/{session_id}`**: a separate conversation surface from `/api/chat`, backed by `MentalHealthQAClassifier`. Always runs the keyword + zero-shot crisis checks first — a retrieved Q&A answer never substitutes for `CRISIS_RESPONSE_HY`, even though the dataset itself contains "suicidal thoughts"-labeled rows.
- **Stripe payments** (card, Apple Pay, Google Pay) for both lawyer and therapist consultations: `POST /api/payments/create-intent`, `GET /api/payments/{payment_id}`, `POST /api/payments/webhook`, backed by a new `payments` table in `src/db/portal_store.py`. Configured via `STRIPE_SECRET_KEY`/`STRIPE_PUBLISHABLE_KEY`/`STRIPE_WEBHOOK_SECRET` env vars; Apple Pay also needs domain verification in the Stripe Dashboard.
- **Backend-ready placeholder pages**: `/therapist` (chat demo + booking link), `/mood-tracking` (web fallback landing page), `/pay` (Stripe Payment Element demo) — same "demo, not final UI" status as the existing auth/booking page.
- **Bookings extended for therapists**: `bookings.provider_type` (`"lawyer"` or `"therapist"`, defaults to `"lawyer"`) added via an idempotent migration so existing rows/callers are unaffected; registration/login already accepted any `role` string, so `role: "therapist"` needed no backend change.

**How to use:**
1. Set `STRIPE_SECRET_KEY`/`STRIPE_PUBLISHABLE_KEY`/`STRIPE_WEBHOOK_SECRET` if you want payments to work (optional otherwise)
2. Visit `/therapist` for the supportive-chat demo, or call `POST /api/therapist-chat` directly
3. Visit `/pay?consultation_type=lawyer` or `/pay?consultation_type=therapist` to test a payment
4. Therapist-matching to a *specific* human therapist (mirroring the lawyer top-lawyer ranking) is not built yet — this is Q&A retrieval, not provider matching

---

### 9. **Booking Calendar: Free/Busy Times in Local Time and UTC**
`GET /api/bookings/availability` computes free/busy slots for a provider on a given calendar date, shown in both a chosen timezone and UTC — no separate timezone-conversion work needed by the caller.

**Features:**
- `bookings.timezone` and `bookings.start_time_utc` columns (idempotent migration; `start_time` itself is unchanged for backward compatibility) — `portal_store.start_time_to_utc_iso()` normalizes any booking to a UTC instant using the stdlib `zoneinfo` (no new dependency), trusting an explicit offset if present or interpreting a naive value as local time in the booking's `timezone`
- `POST /api/bookings` accepts an optional `timezone` (IANA name, default `"UTC"`)
- `GET /api/bookings/availability?provider_name=...&date=YYYY-MM-DD&timezone=Area/City` returns fixed business-hours (09:00–18:00 by default, configurable via `slot_minutes`/`start_hour`/`end_hour`) slots, each with `local_start`/`local_end`/`utc_start`/`utc_end`/`is_free` — busy slots come from existing bookings for that provider, correctly compared across timezones (a booking's local calendar date can differ from its UTC date near midnight; the query window is widened to still catch it)
- Demo page's "Calendar booking" card gained a "Check free times" panel that auto-detects the browser's timezone and lets you click a free slot to fill the booking form

**How to use:**
1. `GET /api/bookings/availability?provider_name=Bob+Lawyer&date=2026-08-03&timezone=Asia/Yerevan`
2. Or on `/`, enter a provider name + date, click "Check free times", then click a green (free) slot

---

## File Structure

### New Files Created:
1. **`src/services/case_export.py`** - CaseExportService class
   - `export_similar_cases()` - Export similar cases to text
   - `export_approved_cases()` - Export approved cases to text
   - `get_export_directory()` - Get exports folder path
   - `list_exports()` - List all exported files

2. **`src/services/crisis_detection.py`** - Keyword-based crisis/safety detection
   - `detect_crisis_signal(text)` - Armenian/English self-harm/suicide phrase check
   - `CRISIS_RESPONSE_HY` - Response text with real emergency contact numbers

3. **`src/db/portal_store.py`** - SQLite persistence for the web portal
   - `init_db()`, `hash_password()` / `verify_password()` (salted PBKDF2)
   - `create_user()`, `find_user()`, `authenticate_user()`, `update_password()`
   - `set_password_reset_otp()`, `get_password_reset()`, `clear_password_reset()`
   - `create_booking()`, `list_bookings()`, `recent_bookings()`, `count_users()`, `count_bookings()`, `distinct_roles()`

### Modified Files:
1. **`src/services/classifier.py`** - Enhanced LegalCaseClassifier
   - `find_multiple_similar_cases(text, limit)` - Get multiple similar cases
   - `_is_approved_case(case)` - Detect approved cases
   - `find_approved_cases(limit)` - Get all approved cases
   - `find_cases_by_lawyer(name, limit)` - Get cases by lawyer
   - `get_top_lawyers_by_cases(limit)` - Get top lawyers ranking
   - `get_top_lawyer_for_query(text, search_limit)` - Rank lawyers by approved cases among cases similar to a query
   - `classify_mental_health_risk(text, risk_threshold)` - Zero-shot mental-health risk classification (lazy-loaded `transformers` pipeline)
   - `zero_shot_classifier` - Lazy property holding the HuggingFace zero-shot pipeline
   - `MentalHealthQAClassifier` (new class in the same file) - `find_similar_answer(text, min_similarity)`, lazy-loaded TF-IDF retrieval over the counseling Q&A dataset

2. **`src/agents/legal_agent.py`** - Enhanced LegalAgent
   - Added `CaseExportService` initialization
   - `get_similar_cases(query, limit)` - Find and export similar cases (no longer wired to a CLI control, but still callable directly)
   - `get_approved_cases_with_lawyers(limit)` - Get approved cases with lawyer stats (same as above)
   - `get_lawyer_cases(name, limit)` - Find cases by lawyer (same as above)
   - `format_similar_cases_response(cases)` - Format similar cases for display
   - `format_approved_cases_response(result)` - Format approved cases for display
   - `get_advice(user_query, history=None)` - Classifier-match and RAG-fallback responses now include the top lawyer, an inline similar-cases block, and real multi-turn memory via `history`; also runs the Step 0/0b crisis and mental-health risk checks before any of the above
   - `_build_search_query()`, `_format_history_for_prompt()` - Fold conversation history into search queries and the LLM prompt
   - `_build_similar_cases_block()` - Render similar cases with lawyer name + approved marker inline

3. **`src/main.py`** - Updated LegalAIController
   - Removed `handle_similar_cases()`, `handle_approved_cases()`, and `handle_search_lawyer()` — that information now surfaces automatically in `get_advice()` answers instead of separate `s`/`a`/`l` controls
   - Keyboard controls reduced to `[m]ic, [t]ype, [u]pload, [q]uit`
   - `self.conversation_history` tracked across `t`/`m` turns and passed into `get_advice()`
   - Fixed `LegalCaseClassifier(data_folder=...)` to point at `src/data` instead of an empty auto-created `data/` folder

4. **`main.py`** - FastAPI portal
   - `get_legal_agent()` - Lazily initializes the shared `LegalAgent` (Chroma + classifier + Ollama LLM) for the web process
   - `POST /api/chat`, `GET /api/chat/{session_id}` - Browser/partner-integration chat API with per-session history
   - Auth, booking, and dashboard endpoints now read/write through `src/db/portal_store.py` instead of in-memory lists

5. **`src/services/vision.py`** - LegalVisionService
   - `classifier` is now a lazy property instead of an eager `__init__` attribute, deferring the `VisionClassifier` (PyTorch + YOLOv8 + MediaPipe) import and construction until first real use
   - `_check_mental_health_concern(emotion, negative_labels)` - Rolling-window sad/angry detection that sets `SystemState.mental_health_concern`

6. **`src/core/state.py`** - SystemState
   - `mental_health_concern`, `mental_health_suggestion` fields + `update_mental_health_concern()`, `get_mental_health_concern()`, `get_mental_health_suggestion()`

7. **`src/db/portal_store.py`** - Added `payments` table + `create_payment()`, `update_payment_status()`, `get_payment()`, `get_payment_by_intent()`; added `provider_type` column to `bookings` via an idempotent migration in `init_db()`; added `timezone`/`start_time_utc` columns, `start_time_to_utc_iso()`, and `get_provider_busy_ranges()` for booking-calendar availability

---

## Export Files

`get_similar_cases()` and `get_approved_cases_with_lawyers()` on `LegalAgent` still export to the `exports/` directory when called directly (e.g. from a script or a future admin endpoint), but nothing in the CLI or web chat calls them automatically anymore — the same information now surfaces inline in the chat answer instead (see Feature 1 above).

### Similar Cases Export (`get_similar_cases`):
- **Filename:** `similar_cases_YYYYMMDD_HHMMSS.txt`
- **Content:** original query, number of cases found, and per case: case number, lawyer name, classification, DataLex link, judicial prehistory (if available)

### Approved Cases Export (`get_approved_cases_with_lawyers`):
- **Filename:** `approved_cases_YYYYMMDD_HHMMSS.txt`
- **Content:** total approved cases count, number of unique lawyers, and per case: case number, lawyer name, department, classification, verdict type, description

---

## Usage Examples

### Example 1: Typed Question — Divorce, with Inline Similar Cases and Approved Marker
```
Press: t
Input: "ինչպես կարող եմ ամուսնալուծվել"
Output:
  🎯 [CLASSIFIER MATCH FOUND]
  🔹 Դասակարգում: 4.1 Ամուսնալուծության վերաբերյալ
  🔹 Նմանատիպ գործ: ԱՐԴ/0161/02/24-1
  🔹 Հղում: http://www.datalex.am/?app=AppCaseSearch&case_id=ԱՐԴ/0161/02/24-1
  🔹 Առաջարկվող փաստաբան: Հրանտ Կարապետյան Հրաչյայի

  🏆 Ամենահաջողակ փաստաբանը նմանատիպ գործերում: Սևակ Մարաբյան Նորայրի
     Հաստատված գործեր: 1 (ընդհանուր 1 նմանատիպ գործից)

  📚 Նմանատիպ գործեր (օգտակար օրինակներ).
     1. ԵԴ2/8262/02/24-1 — 4.1 Ամուսնալուծության վերաբերյալ
        Փաստաբան: Աշխեն Դաշյան Յուրիկի, Արման Ներսիսյան Համբարձումի
     2. ԵԴ2/1882/02/24-1 — 4.1 Ամուսնալուծության վերաբերյալ
        Փաստաբան: Սևակ Մարաբյան Նորայրի ✅ Հաստատված
     3. ԱՐԴ/0731/02/24-1 — 4.1 Ամուսնալուծության վերաբերյալ
        Փաստաբան: Գայանե Հարությունյան Ռոբինզոնի

  📄 Գործի նախապատմություն / բովանդակության օրինակ:
  ...
```

### Example 2: Typed Question with Automatic Top Lawyer
```
Press: t
Input: "Ժառանգություն ընդունելու վերաբերյալ գործ"
Output:
  🎯 [CLASSIFIER MATCH FOUND]
  🔹 Դասակարգում: 8.7.5 Ժառանգությունն ընդունելու և ժառանգ...
  🔹 Առաջարկվող փաստաբան: Հայկ Վարդանյան

  🏆 Ամենահաջողակ փաստաբանը նմանատիպ գործերում: Հայկ Վարդանյան
     Հաստատված գործեր: 5 (ընդհանուր 5 նմանատիպ գործից)
  ...
```

---

## Keyboard Shortcuts Summary

| Key | Action |
|-----|--------|
| **m** | Speak via microphone |
| **t** | Type your legal question |
| **u** | Upload a legal document |
| **q** | Quit the application |

Typed (`t`) and spoken (`m`) questions that match a known case now automatically include the top lawyer for similar cases and a short list of other similar cases (with lawyer names and approval status) in the answer — no separate controls needed.

---

## Technical Details

### Similarity Scoring
- Uses TF-IDF vectorization from `sklearn`
- Cosine similarity threshold: > 0.1
- Returns cases sorted by similarity score (highest first)

### Approval Detection
- Scans case text for predefined keywords in Armenian and English
- Keywords include: approval terms, verdict types, case outcomes
- Can be easily extended by adding more keywords

### Export Formatting
- UTF-8 encoding with Armenian text support
- Organized sections with clear visual separators
- Includes metadata and timestamps
- Easy to read in any text editor

---

## Integration Notes

The new features are fully integrated with:
- ✅ Existing classifier system
- ✅ Vector database (ChromaDB)
- ✅ LLM (Ollama)
- ✅ Export directory management
- ✅ Keyboard event handling
- ✅ Unicode normalization for Armenian text

---

## Future Enhancements

Possible improvements:
1. Add filtering by case category/classification
2. Add filtering by verdict type (won/lost/pending)
3. Add statistical analysis of lawyer success rates
4. Export to PDF format
5. Add database search by date range
6. Add case similarity visualization
7. Email export functionality

---

## Troubleshooting

**Issue:** "No similar cases found"
- **Solution:** Ensure data files are loaded in the `data/` folder with proper HTML format

**Issue:** "Classifier not available"
- **Solution:** Check that `data/prehistory*.htm` files exist and are properly formatted

**Issue:** Export file not created
- **Solution:** Ensure `exports/` directory exists or can be created (permissions check)

---

## Support

For issues or feature requests, check:
1. Console output for error messages
2. Data folder structure and file formats
3. Ollama service status
4. ChromaDB connection status
