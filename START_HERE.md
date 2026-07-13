# Armenian Legal Portal Setup

## 1. Install Python dependencies

Run this from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then, as a **separate, required-for-crew-responses** command:

```bash
pip install --no-deps crewai==1.15.2
```

This has to be `--no-deps` and separate from `requirements.txt` — `crewai` hard-pins a `chromadb` version that cannot even open this project's existing `./chroma_legal_data` directory (see README.md "Vector search & multi-agent orchestration (CrewAI)" for the full explanation). Never run a plain `pip install crewai` in this venv. If you skip this step entirely, the app still runs — `/api/chat` and `/api/therapist-chat` just fall back to a simpler, non-crew response instead of erroring.

## 2. Run the web portal

Optional — set these before starting the server if you want payments to work (card/Apple Pay/Google Pay for lawyer and therapist consultations). Without them, everything else still runs; `/api/payments/create-intent` just returns a clear "not configured" error.

```bash
export STRIPE_SECRET_KEY="sk_test_..."        # from https://dashboard.stripe.com/apikeys
export STRIPE_PUBLISHABLE_KEY="pk_test_..."
export STRIPE_WEBHOOK_SECRET="whsec_..."      # only needed for /api/payments/webhook
```

Start the FastAPI app:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- http://localhost:8000
- http://localhost:8000/therapist — supportive Q&A chat demo (not a legal chat) + link to book/pay for a real session
- http://localhost:8000/mood-tracking — web fallback mood-tracking landing page
- http://localhost:8000/pay?consultation_type=lawyer (or `therapist`) — Stripe payment demo

### If port 8000 is already in use

`uvicorn` will fail with `[Errno 48] Address already in use` if something else (often a previous `uvicorn` run you forgot to stop) is already listening on that port. Two ways to fix it:

- **Run on a different port instead**, e.g.:
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8080
  ```
  then open http://localhost:8080.

- **Or find and stop whatever is using port 8000:**
  ```bash
  lsof -nP -iTCP:8000 -sTCP:LISTEN   # shows the PID and process name
  kill <PID>                          # stop it (add -9 if it won't stop)
  ```
  Only kill a PID you recognize from the command name `lsof` prints. Re-run the `lsof` command with no output to confirm the port is free.

## 3. What is included

The portal now provides:

- Individual and lawyer registration (passwords hashed, stored in SQLite — not plaintext, not in-memory)
- Sign in flow
- Dashboard view
- Calendar booking form
- WebRTC video call room with signaling
- Backend-ready sign-up / registration flow for future frontend integration
- Armenian legal AI chat (`/api/chat`) with multi-turn conversation memory, backed by the case-law classifier + RAG pipeline used by the CLI app

## 4. Frontend placeholder for sign-up / registration

Since the frontend is not ready yet, the sign-up and sign-in flow is currently exposed as a backend-ready placeholder that can be connected to a future frontend.

- Open the temporary demo page at http://localhost:8000/ to see the registration form.
- The registration endpoint is ready for integration: http://localhost:8000/api/auth/register
- The login endpoint is ready for integration: http://localhost:8000/api/auth/login

Example registration payload:

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "secret123",
  "role": "individual"
}
```

When the real frontend is built, you can connect the sign-up form to this endpoint and use the same payload structure.

## 5. Mobile app readiness for Android and iOS

The same backend authentication flow can be used by a future mobile app that will be distributed through Google Play and the Apple App Store.

Planned mobile app features:

- Sign up and sign in for individuals and lawyers
- Forgot password with OTP verification using email or phone number
- Lawyer registration with license number support
- Secure password reset flow
- App-based booking, dashboard, and video consultation features

Recommended app deployment path:

- Android: publish as a native Android app via Google Play
- iOS: publish as a native iPhone/iPad app via the Apple App Store
- Backend API: keep this FastAPI service as the shared backend for web, Android, and iOS

## 6. Mobile API contract

Use these endpoints from the mobile app as the backend contract for web, Android, and iOS.

### Authentication

- POST /api/auth/register
  - Body: name, email, phone_number, password, role, license_number
- POST /api/auth/login
  - Body: identifier, password, role
- POST /api/auth/forgot-password
  - Body: identifier, channel
- POST /api/auth/reset-password
  - Body: identifier, otp, new_password

### Booking and dashboard

- GET /api/dashboard
- GET /api/bookings
- POST /api/bookings
  - Body: title, client_name, lawyer_name (holds the provider's name — lawyer or therapist), start_time, role, `provider_type` (optional, `"lawyer"` or `"therapist"`, defaults to `"lawyer"`), `timezone` (optional IANA name e.g. `"Asia/Yerevan"`, defaults to `"UTC"` — used to interpret `start_time` if it has no UTC offset)
- GET /api/bookings/availability
  - Query params: `provider_name`, `date` (`YYYY-MM-DD`, interpreted in `timezone`), `timezone` (default `"UTC"`), `slot_minutes` (default 60), `start_hour`/`end_hour` (default 9/18 — fixed business hours, not yet per-provider)
  - Response: `success`, `slots: [{local_start, local_end, utc_start, utc_end, is_free}, ...]` — every slot is shown in both the requested timezone and UTC

### Legal AI chat

- POST /api/chat
  - Body: `message` (Armenian question text), `session_id` (optional — omit on the first message, then reuse the `session_id` returned in the response for every follow-up so the assistant keeps conversational context), `language` (optional short code e.g. `"hy"`/`"en"`, defaults to `"hy"` — see README.md "Language support")
  - Response: `success`, `session_id`, `response` (the assistant's answer, including the recommended lawyer and the lawyer with the strongest approved-case record for similar cases)
  - Before any legal-advice logic runs, every message is screened for self-harm/suicide risk language (keyword check + zero-shot classification — see README.md "Crisis/safety and mental-health risk screening"). If flagged, `response` is a fixed message with real emergency contact numbers instead of a legal answer.
- GET /api/chat/{session_id}
  - Returns the full message history for that session: `[{role: "user"|"bot", text, at}, ...]`
  - Chat history is in-memory per server process (resets on restart); user/booking data is persisted separately in SQLite

### Therapist chat (supportive Q&A, not legal advice, not a licensed therapist)

- POST /api/therapist-chat
  - Body: `message`, `session_id` (optional, same pattern as /api/chat), `language` (optional, defaults to `"en"`)
  - Response: `success`, `session_id`, `response` — a retrieved answer from `MentalHealthQAClassifier` (trained on `src/data/student_mh_counseling_100k_with_label_column.csv`), always screened for crisis signals first, same as /api/chat
- GET /api/therapist-chat/{session_id}
  - Returns the full message history for that session

### Payments (Stripe — card, Apple Pay, Google Pay)

- POST /api/payments/create-intent
  - Body: `consultation_type` (`"lawyer"` or `"therapist"`), `customer_name`, `amount_cents`, `currency` (default `"usd"`)
  - Response: `success`, `payment_id`, `client_secret`, `publishable_key` — feed `client_secret` + `publishable_key` into Stripe.js on the client to complete payment (see `/pay` for a reference implementation)
  - Returns `success: false` with a clear message if `STRIPE_SECRET_KEY` isn't set on the server
- GET /api/payments/{payment_id}
  - Returns the stored payment record (status, amount, consultation_type, etc.)
- POST /api/payments/webhook
  - Configure this URL in the Stripe Dashboard so payment status stays correct even if the client disconnects before confirming

### Real-time communication

- WS /ws/signaling/{room_id}

## 7. Main endpoints

- Home page: http://localhost:8000/
- Health check: http://localhost:8000/health
- Register: http://localhost:8000/api/auth/register
- Login: http://localhost:8000/api/auth/login
- Forgot password: http://localhost:8000/api/auth/forgot-password
- Reset password: http://localhost:8000/api/auth/reset-password
- Bookings: http://localhost:8000/api/bookings
- Booking availability: http://localhost:8000/api/bookings/availability
- Dashboard: http://localhost:8000/api/dashboard
- Legal AI chat: http://localhost:8000/api/chat
- Chat history: http://localhost:8000/api/chat/{session_id}
- Therapist chat: http://localhost:8000/api/therapist-chat
- Therapist chat history: http://localhost:8000/api/therapist-chat/{session_id}
- Payments: http://localhost:8000/api/payments/create-intent
- Mood tracking (web fallback): http://localhost:8000/mood-tracking
- Talk to a therapist (demo): http://localhost:8000/therapist
- Payment demo page: http://localhost:8000/pay
- WebSocket signaling: ws://localhost:8000/ws/signaling/room-name

## 8. Notes

- The current video call uses local browser media and a simple signaling server.
- Users, bookings, and payments are persisted in a local SQLite database (`portal.db`, gitignored) with salted/hashed passwords — this survives restarts, unlike the earlier in-memory version.
- Still missing for production: token/session-based auth (login currently just verifies the password per request, it doesn't issue a session token), a production-grade database if you outgrow SQLite, a TURN/STUN setup for reliable calls across networks, and real Stripe keys + Apple Pay domain verification (payments return a clear "not configured" error without them).
- Chat conversation history (both `/api/chat` and `/api/therapist-chat`) is still in-memory per server process and resets on restart; only user/booking/payment data is persisted so far.
- Therapist-matching to a specific human therapist (mirroring the lawyer top-lawyer ranking) is not built — `/api/therapist-chat` is supportive Q&A retrieval only.
- The `/api/chat` endpoint requires Ollama running locally with the `nomic-embed-text` and `armenia-lawyer-router` models pulled — see `OLLAMA setup.md`.
- The zero-shot mental-health risk model (`transformers` + `torch`, ~280MB) downloads on first use, the first time any message reaches the crisis-check step — expect a one-time delay on the very first chat message after starting the server.
- Both chat endpoints now draft answers via a two-agent CrewAI crew (Researcher, then Writer — see README.md), which means two sequential LLM calls instead of one, so responses are slower than before. If `crewai` isn't installed (step 1's second command) or a crew call fails, both endpoints fall back to their previous behavior instead of erroring.
- This project is proprietary (see `LICENSE`) — not open-source, no license is granted for reuse without permission.
