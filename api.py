# api.py (repo root) — the FastAPI WEB PORTAL: browser chat UI, REST API
# (auth, bookings, payments, /api/chat, /api/therapist-chat), WebRTC signaling.
# Run with: uvicorn api:app --reload
#
# This is a different entry point from src/main.py, which is the DESKTOP CLI
# app (webcam + microphone loop, keyboard-driven). They share the same
# underlying LegalAgent/classifier/vector-store code in src/, but are two
# separate ways to run this project — not two versions of the same file.
# Run the CLI with: python src/main.py
import audioop
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from collections import defaultdict
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Dict, Set
from zoneinfo import ZoneInfo

import stripe
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.db import portal_store
from src.services.crisis_detection import detect_crisis_signal, get_crisis_response
from src.services.voice import sanitize_transcript

app = FastAPI(title="Armenian Legal Portal", version="1.0.0")

# api.py lives at the project root, so this file's own directory *is* the
# project root -- anchoring the chroma path here (instead of the bare
# relative "./chroma_legal_data") means the vector store resolves to the
# same directory regardless of the process's cwd at launch (an IDE run
# config, a different launch script, etc. can all set cwd to something
# other than the project root). A cwd mismatch previously didn't error --
# chromadb's PersistentClient just creates a fresh empty directory wherever
# it's pointed -- so every similarity search silently returned nothing.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

portal_store.init_db()

# Payments (Apple Pay / Google Pay / card) via Stripe. Both keys must come from
# your own Stripe account (https://dashboard.stripe.com/apikeys) — never commit
# real keys to source control. Apple Pay/Google Pay show up automatically in the
# Stripe Payment Element for supported browsers/devices once you've verified your
# domain for Apple Pay in the Stripe Dashboard; that step happens outside this repo.
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
stripe.api_key = STRIPE_SECRET_KEY

CONSULTATION_TYPES = {"lawyer", "therapist"}

# Display names for _direct_therapist_answer's language instruction. Any code
# not listed here is passed through as-is — the local Ollama model's actual
# fluency beyond Armenian/English is unverified, so this is best-effort.
_THERAPIST_LANGUAGE_NAMES = {
    "hy": "Armenian",
    "en": "English",
}


def _direct_therapist_answer(message: str, qa_classifier, llm, language: str = "en") -> str:
    """Single direct LLM call for the therapist supportive-chat path: finds a
    relevant past Q&A example via the existing MentalHealthQAClassifier, then
    asks the LLM to draft a short supportive response inspired by it.

    Same model as the legal side (armenia-lawyer-router) — there's no
    separate specialized model for supportive conversation."""
    language_name = _THERAPIST_LANGUAGE_NAMES.get(language, language)
    match = qa_classifier.find_similar_answer(message)
    if match:
        found_example = (
            f"Similar past question (topic: {match.get('label', 'unknown')}): {match['question']}\n"
            f"Past supportive answer: {match['answer']}"
        )
    else:
        found_example = "No sufficiently similar past conversation was found."

    prompt = (
        f"You are a compassionate peer-support writer. Write a short, warm, supportive response "
        f"in {language_name} to what the person just said, using the retrieved past conversation "
        "below as inspiration (not a script to copy verbatim). Do not diagnose. Do not claim to be "
        "a licensed therapist. End by gently mentioning that /therapist can be used to book a real "
        "session for anything beyond this conversation.\n\n"
        f"The person just said: {message}\n\n"
        f"Retrieved past conversation:\n{found_example}"
    )
    return llm.invoke(prompt).strip()


room_clients: Dict[str, Set[WebSocket]] = defaultdict(set)
# Chat history (both /api/chat and /api/therapist-chat) is persisted in
# portal_store's chat_messages table, keyed by (session_id, session_type) —
# it survives restarts, unlike the earlier in-memory-dict version.

_legal_agent = None
_legal_agent_error: str | None = None
_mental_health_qa_classifier = None
_vision_service = None


def get_vision_service():
    """Lazily build the LegalVisionService (loads YOLO + MediaPipe on first
    use), reused across /api/upload video requests. Its own SystemState is
    private to this instance — see analyze_video_headless's docstring for
    why per-request state would be wrong here."""
    global _vision_service
    if _vision_service is None:
        from src.core.state import SystemState
        from src.services.vision import LegalVisionService
        _vision_service = LegalVisionService(SystemState())
    return _vision_service


def get_mental_health_qa_classifier():
    """Lazily build the MentalHealthQAClassifier (loads/indexes ~100k Q&A rows on
    first use), reused across requests. Independent of the legal agent/classifier."""
    global _mental_health_qa_classifier
    if _mental_health_qa_classifier is None:
        from src.services.classifier import MentalHealthQAClassifier
        _mental_health_qa_classifier = MentalHealthQAClassifier()
    return _mental_health_qa_classifier


def get_legal_agent():
    """Lazily initialize the LegalAgent (Chroma + classifier + Ollama LLM), reused across requests."""
    global _legal_agent, _legal_agent_error
    if _legal_agent is not None:
        return _legal_agent
    if _legal_agent_error is not None:
        raise RuntimeError(_legal_agent_error)

    try:
        from langchain_ollama import OllamaEmbeddings
        from src.core.state import SystemState
        from src.services.classifier import LegalCaseClassifier
        from src.agents.legal_agent import LegalAgent
        from src.db.repository import CompanyLegalRepo
        from src.db.vector_store import ChromaVectorStore, open_persistent_client

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        client = open_persistent_client(os.path.join(_PROJECT_ROOT, "chroma_legal_data"))
        vector_db = ChromaVectorStore(client=client, collection_name="company_legal_cases", embeddings=embeddings)
        classifier_service = LegalCaseClassifier(data_folder="src/data")
        _legal_agent = LegalAgent(CompanyLegalRepo(vector_db), SystemState(), classifier=classifier_service)
        return _legal_agent
    except Exception as exc:
        _legal_agent_error = str(exc)
        raise


@app.on_event("startup")
async def warm_up_legal_agent():
    from src.services.ollama_setup import ensure_ollama_models
    await run_in_threadpool(ensure_ollama_models)

    try:
        await run_in_threadpool(get_legal_agent)
        print("✅ Legal AI chat backend ready")
    except Exception as exc:
        print(f"⚠️ Legal AI chat backend failed to initialize: {exc}")
        print("   The portal will still run, but /api/chat will return an error until this is fixed.")

    # Vision (YOLO + MediaPipe) and Whisper both used to lazy-load on
    # whichever request first needed a video/audio analysis — fine for
    # Whisper's "tiny"/"base" sizes, but "medium" (the default, see
    # _WHISPER_MODEL_SIZE) re-verifies its full checkpoint's SHA256 on every
    # cold load, which alone can run past a minute even with the weights
    # already cached on disk. Stacked with YOLO/MediaPipe's own cold-load,
    # that easily pushed a first /api/upload or /api/upload-link request
    # past the browser's own patience for a hanging fetch (surfaced as
    # Safari's opaque "Load failed", with no indication anything was even
    # in progress). Warming both here instead moves that one-time cost to
    # server boot, where a slow startup is expected, instead of ambushing
    # the first real user request.
    try:
        # get_vision_service() alone only builds the LegalVisionService
        # shell -- the actual YOLO/MediaPipe load is behind its `.classifier`
        # property, touched lazily on first real frame. Access it here too,
        # or this warm-up is a no-op and the cold-load still lands on the
        # first real /api/upload(-link) request.
        vision_service = await run_in_threadpool(get_vision_service)
        await run_in_threadpool(lambda: vision_service.classifier)
        print("✅ Vision models (YOLO + MediaPipe) ready")
    except Exception as exc:
        print(f"⚠️ Vision models failed to initialize: {exc}")

    try:
        await run_in_threadpool(_get_whisper_model)
        print("✅ Whisper speech-to-text model ready")
    except Exception as exc:
        print(f"⚠️ Whisper model failed to initialize: {exc}")


class RegisterRequest(BaseModel):
    name: str
    email: str
    phone_number: str | None = None
    password: str
    role: str
    license_number: str | None = None


class LoginRequest(BaseModel):
    identifier: str
    password: str
    role: str


class ForgotPasswordRequest(BaseModel):
    identifier: str
    channel: str = "email"


class ResetPasswordRequest(BaseModel):
    identifier: str
    otp: str
    new_password: str


class BookingRequest(BaseModel):
    title: str
    client_name: str
    lawyer_name: str
    start_time: str  # ISO8601; if it has no offset/Z it's interpreted as local time in `timezone`
    role: str
    provider_type: str = "lawyer"  # "lawyer" or "therapist" — who client_name is booking with
    timezone: str = "UTC"  # IANA name (e.g. "Asia/Yerevan") the client picked start_time in


class ChatMessageRequest(BaseModel):
    message: str
    session_id: str | None = None
    language: str | None = None  # short code ("hy", "en", ...); each endpoint applies its own default when omitted


class PaymentIntentRequest(BaseModel):
    consultation_type: str  # "lawyer" or "therapist"
    customer_name: str
    amount_cents: int
    currency: str = "usd"
    # Optional: if provided, a successful payment (webhook payment_intent.succeeded)
    # auto-creates a calendar booking for this slot — see create_booking_from_payment_if_scheduled.
    provider_name: str | None = None
    start_time: str | None = None
    timezone: str = "UTC"


@app.get("/legacy-demo", response_class=HTMLResponse)
async def legacy_demo():
    """Original hand-rolled registration/dashboard/booking/video-call demo
    page. Moved off "/" to make room for the real frontend/ (Vite) chat
    console mounted at the end of this file — this stays reachable at its
    own path since it still demonstrates the raw registration/booking/
    payment/video-call endpoints directly, unlike the chat console."""
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Armenian Legal Portal</title>
      <style>
        :root { color-scheme: dark; }
        body { font-family: Arial, sans-serif; margin: 0; background: #08111d; color: #f5f7fa; }
        .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
        .hero { background: linear-gradient(135deg, #1d4ed8, #0f766e); padding: 26px; border-radius: 16px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin-top: 18px; }
        .card { background: #111c2c; border: 1px solid #23344f; border-radius: 14px; padding: 18px; }
        input, select, button, textarea { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #35506f; margin-top: 8px; font-size: 14px; }
        button { cursor: pointer; background: #2563eb; color: white; border: none; }
        button.secondary { background: #0f766e; }
        .stats { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
        .stat { background: #16253b; padding: 10px 14px; border-radius: 10px; }
        .video-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
        video { width: 100%; background: #000; border-radius: 10px; min-height: 220px; }
        .muted { opacity: 0.8; }
        .small { font-size: 12px; color: #9fb0cb; }
        .chat-card { margin-top: 18px; }
        .chat-messages { height: 380px; overflow-y: auto; background: #0c1726; border: 1px solid #23344f; border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
        .chat-bubble { max-width: 80%; padding: 10px 14px; border-radius: 14px; white-space: pre-wrap; line-height: 1.4; font-size: 14px; }
        .chat-bubble.user { align-self: flex-end; background: #2563eb; color: white; border-bottom-right-radius: 4px; }
        .chat-bubble.bot { align-self: flex-start; background: #16253b; color: #f5f7fa; border: 1px solid #23344f; border-bottom-left-radius: 4px; }
        .chat-bubble.typing { opacity: 0.6; font-style: italic; }
        .chat-input-row { display: flex; gap: 8px; margin-top: 10px; }
        .chat-input-row textarea { margin-top: 0; resize: none; height: 48px; }
        .chat-input-row button { width: auto; padding: 0 20px; }
        .chat-header-row { display: flex; justify-content: space-between; align-items: center; }
        .chat-header-row button { width: auto; padding: 6px 12px; font-size: 12px; }

        /* Auth card — deliberately its own light theme (cream/terracotta) distinct
           from the rest of the dark console, matching the reference patient-portal
           login design this was modeled on. */
        .auth-card { background: #f4ead9; color: #1a1410; border: 2px solid #1a1410; border-radius: 18px; padding: 20px; }
        .auth-card-header { text-align: center; margin-bottom: 14px; }
        .auth-card-header .auth-icon { font-size: 30px; }
        .auth-card-header h3 { margin: 6px 0 2px; font-size: 19px; color: #1a1410; }
        .auth-card-header p { margin: 0; font-size: 12px; color: #7a6a58; }
        .role-toggle { display: flex; background: #fff; border: 2px solid #1a1410; border-radius: 999px; padding: 3px; margin-bottom: 12px; }
        .role-toggle button { flex: 1; border: none; background: transparent; color: #1a1410; padding: 8px 4px; border-radius: 999px; font-size: 12px; font-weight: 600; cursor: pointer; margin: 0; }
        .role-toggle button.active { background: #c17d5f; color: #fff; }
        .auth-tabs { display: flex; margin-bottom: 12px; border-bottom: 1px solid #d8c8b4; }
        .auth-tabs button { flex: 1; border: none; background: transparent; color: #8a7a68; padding: 8px 2px; font-size: 11px; font-weight: 700; cursor: pointer; margin: 0; border-radius: 0; border-bottom: 2px solid transparent; }
        .auth-tabs button.active { color: #1a1410; border-bottom-color: #c17d5f; }
        .auth-panel-title { display: flex; align-items: center; gap: 6px; font-weight: 700; font-size: 15px; margin: 4px 0 12px; }
        .auth-card form input { background: #efe2ce; border: 1px solid #d8c8b4; color: #1a1410; border-radius: 10px; }
        .auth-card form input::placeholder { color: #8a7a68; }
        .auth-card button[type="submit"] { background: #c17d5f; border: 2px solid #1a1410; color: #fff; font-weight: 700; border-radius: 12px; padding: 12px; margin-top: 12px; }
        .auth-links { text-align: center; margin-top: 10px; }
        .auth-links a { color: #1a1410; font-size: 12px; text-decoration: underline; cursor: pointer; }
        .auth-forgot-panel { margin-top: 10px; padding-top: 10px; border-top: 1px solid #d8c8b4; }
        .auth-quick-panel, .auth-payments-panel { text-align: center; padding: 6px 0; }
        .auth-quick-panel p, .auth-payments-panel p { color: #7a6a58; }
        .auth-button-link { display: inline-block; margin-top: 8px; background: #c17d5f; color: #fff; border: 2px solid #1a1410; border-radius: 12px; padding: 10px 18px; text-decoration: none; font-weight: 700; font-size: 13px; }
        #authMessage { color: #1a1410; text-align: center; margin-top: 10px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="hero">
          <h1>Armenian Legal Portal</h1>
          <p>Register as an individual or lawyer, manage your dashboard, book consultations, and start online video calls.</p>
        </div>

        <div class="card chat-card">
          <div class="chat-header-row">
            <h3>⚖️ Legal AI Chat</h3>
            <button id="clearChatBtn" class="secondary">Clear chat</button>
          </div>
          <p class="small">Type your legal question in Armenian. The assistant answers using the case database and highlights the most successful lawyer for similar cases.</p>
          <div id="chatMessages" class="chat-messages"></div>
          <form id="chatForm" class="chat-input-row">
            <textarea id="chatInput" placeholder="Նկարագրեք ձեր իրավական հարցը..." required></textarea>
            <button type="submit">Send</button>
          </form>
        </div>

        <div class="grid">
          <div class="auth-card">
            <div class="auth-card-header">
              <div class="auth-icon">⚖️</div>
              <h3>Armenian Legal Portal</h3>
              <p>Legal &amp; Therapy Consultations</p>
            </div>

            <div class="role-toggle" id="roleToggle">
              <button type="button" class="active" data-role="individual">Individual</button>
              <button type="button" data-role="lawyer">Lawyer</button>
              <button type="button" data-role="therapist">Therapist</button>
            </div>

            <div class="auth-tabs" id="authTabs">
              <button type="button" data-tab="quick">Quick Chat</button>
              <button type="button" class="active" data-tab="signin">Sign In</button>
              <button type="button" data-tab="register">Register</button>
              <button type="button" data-tab="payments">Payments</button>
            </div>

            <div class="auth-panel" id="panelSignin">
              <div class="auth-panel-title">❤️ <span id="signinRoleLabel">Individual</span> Sign In</div>
              <form id="loginForm">
                <input id="loginEmail" type="email" placeholder="Email address" required />
                <input id="loginPassword" type="password" placeholder="Password" required />
                <button type="submit">Sign In</button>
              </form>
              <div class="auth-links"><a id="forgotPasswordLink">Forgot password?</a></div>
              <div class="auth-forgot-panel" id="forgotPanel" hidden>
                <input id="forgotIdentifier" placeholder="Email or phone" />
                <button type="button" id="sendOtpBtn" class="secondary">Send OTP</button>
                <input id="resetOtp" placeholder="OTP code" style="margin-top:8px;" />
                <input id="resetNewPassword" type="password" placeholder="New password" style="margin-top:8px;" />
                <button type="button" id="resetPasswordBtn" class="secondary" style="margin-top:8px;">Reset password</button>
                <div id="forgotMessage" class="small"></div>
              </div>
            </div>

            <div class="auth-panel" id="panelRegister" hidden>
              <div class="auth-panel-title">📝 <span id="registerRoleLabel">Individual</span> Register</div>
              <form id="registerForm">
                <input id="name" placeholder="Full name" required />
                <input id="email" type="email" placeholder="Email address" required />
                <input id="password" type="password" placeholder="Password" required />
                <button type="submit">Register</button>
              </form>
            </div>

            <div class="auth-panel auth-quick-panel" id="panelQuick" hidden>
              <p class="small">Skip sign-in and ask the Legal AI a quick question below.</p>
              <a href="#chatMessages" class="auth-button-link">Go to Quick Chat</a>
            </div>

            <div class="auth-panel auth-payments-panel" id="panelPayments" hidden>
              <p class="small">Book &amp; pay for a real consultation.</p>
              <a id="paymentsLink" href="/pay?consultation_type=lawyer" class="auth-button-link">Continue to Payments</a>
            </div>

            <div id="authMessage" class="small"></div>
          </div>

          <div class="card">
            <h3>Dashboard</h3>
            <div id="dashboardStats" class="stats"></div>
            <div id="recentBookings" class="small" style="margin-top: 10px;"></div>
          </div>

          <div class="card">
            <h3>Mobile app ready</h3>
            <p class="small">This backend is ready to power Android and iOS apps through Google Play and the Apple App Store with sign-up, sign-in, OTP recovery, and lawyer license support.</p>
          </div>
        </div>

        <div class="grid">
          <div class="card">
            <h3>Calendar booking</h3>
            <p class="small">Business hours default to 09:00–18:00 in the timezone below (no per-provider schedule yet). Check free times first, then click a free slot to fill the booking time — shown in both that local timezone and UTC.</p>
            <div class="chat-input-row" style="align-items:flex-end;">
              <div style="flex:1;">
                <label class="small">Provider name</label>
                <input id="availabilityProvider" placeholder="Lawyer or therapist name" />
              </div>
              <div style="flex:1;">
                <label class="small">Date</label>
                <input id="availabilityDate" type="date" />
              </div>
            </div>
            <label class="small">Timezone (IANA name)</label>
            <input id="bookingTimezone" placeholder="e.g. Asia/Yerevan" />
            <button id="checkAvailabilityBtn" type="button" class="secondary">Check free times</button>
            <div id="availabilitySlots" class="small" style="margin-top:8px; max-height:160px; overflow-y:auto;"></div>

            <form id="bookingForm" style="margin-top:14px;">
              <input id="bookingTitle" placeholder="Consultation topic" required />
              <input id="bookingClient" placeholder="Client name" required />
              <select id="bookingProviderType">
                <option value="lawyer">Booking with a lawyer</option>
                <option value="therapist">Booking with a therapist</option>
              </select>
              <input id="bookingLawyer" placeholder="Lawyer or therapist name" required />
              <input id="bookingTime" type="datetime-local" required />
              <select id="bookingRole">
                <option value="individual">Individual</option>
                <option value="lawyer">Lawyer</option>
                <option value="therapist">Therapist</option>
              </select>
              <button type="submit">Book appointment</button>
            </form>
            <div id="bookingMessage" class="small"></div>
          </div>

          <div class="card">
            <h3>Online video call</h3>
            <div>
              <input id="roomId" placeholder="Room name" value="legal-room-1" />
              <button id="createRoomBtn">Create room</button>
              <button id="joinRoomBtn" class="secondary">Join room</button>
              <button id="startCallBtn">Start call</button>
              <button id="hangUpBtn" class="secondary">End call</button>
            </div>
            <div>
              <button id="muteAudioBtn" class="secondary">🎤 Mute my audio</button>
              <button id="muteVideoBtn" class="secondary">📷 Mute my video</button>
            </div>
            <div class="video-grid">
              <video id="localVideo" autoplay muted playsinline></video>
              <video id="remoteVideo" autoplay playsinline></video>
            </div>
            <div id="callStatus" class="small">Ready to connect.</div>
            <div id="mutedNotice" class="small" style="display:none; margin-top:8px; padding:10px; background:#3a2410; border:1px solid #8a5a1f; border-radius:8px;">
              ⚠️ The other participant appears to be muted. Ask them to unmute, or continue in the chat above instead.
            </div>
          </div>
        </div>
      </div>

      <script>
        const authMessage = document.getElementById('authMessage');
        const bookingMessage = document.getElementById('bookingMessage');
        const bookingTimezoneInput = document.getElementById('bookingTimezone');
        const availabilitySlots = document.getElementById('availabilitySlots');
        bookingTimezoneInput.value = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
        document.getElementById('availabilityDate').value = new Date().toISOString().slice(0, 10);

        document.getElementById('checkAvailabilityBtn').addEventListener('click', async () => {
          const provider_name = document.getElementById('availabilityProvider').value.trim();
          const date = document.getElementById('availabilityDate').value;
          const tz = bookingTimezoneInput.value.trim() || 'UTC';
          if (!provider_name || !date) {
            availabilitySlots.textContent = 'Enter a provider name and date first.';
            return;
          }
          availabilitySlots.textContent = 'Loading...';
          const params = new URLSearchParams({ provider_name, date, timezone: tz });
          const res = await fetch(`/api/bookings/availability?${params}`);
          const data = await res.json();
          if (!data.success) {
            availabilitySlots.textContent = data.message;
            return;
          }
          availabilitySlots.innerHTML = '';
          data.slots.forEach(slot => {
            const row = document.createElement('div');
            const localTime = slot.local_start.slice(11, 16);
            const utcTime = slot.utc_start.slice(11, 16);
            row.textContent = `${slot.is_free ? '🟢' : '🔴'} ${localTime} ${tz} (${utcTime} UTC)`;
            if (slot.is_free) {
              row.style.cursor = 'pointer';
              row.style.textDecoration = 'underline';
              row.addEventListener('click', () => {
                document.getElementById('bookingLawyer').value = provider_name;
                document.getElementById('bookingTime').value = slot.local_start.slice(0, 16);
              });
            } else {
              row.style.opacity = '0.5';
            }
            availabilitySlots.appendChild(row);
          });
        });
        const dashboardStats = document.getElementById('dashboardStats');
        const recentBookings = document.getElementById('recentBookings');
        const roomIdInput = document.getElementById('roomId');
        const callStatus = document.getElementById('callStatus');
        const localVideo = document.getElementById('localVideo');
        const remoteVideo = document.getElementById('remoteVideo');

        let currentUser = null;
        let peerConnection = null;
        let localStream = null;
        let ws = null;
        let pendingOffer = false;

        const chatMessages = document.getElementById('chatMessages');
        const chatForm = document.getElementById('chatForm');
        const chatInput = document.getElementById('chatInput');
        const clearChatBtn = document.getElementById('clearChatBtn');
        let chatSessionId = localStorage.getItem('legalChatSessionId') || null;

        function appendChatBubble(role, text) {
          const bubble = document.createElement('div');
          bubble.className = `chat-bubble ${role}`;
          bubble.textContent = text;
          chatMessages.appendChild(bubble);
          chatMessages.scrollTop = chatMessages.scrollHeight;
          return bubble;
        }

        function showTyping() {
          const bubble = appendChatBubble('bot typing', '...');
          bubble.id = 'typingIndicator';
        }

        function hideTyping() {
          const el = document.getElementById('typingIndicator');
          if (el) el.remove();
        }

        async function loadChatHistory() {
          if (!chatSessionId) return;
          try {
            const res = await fetch(`/api/chat/${chatSessionId}`);
            const data = await res.json();
            (data.messages || []).forEach(m => appendChatBubble(m.role, m.text));
          } catch (err) {
            console.error('Failed to load chat history', err);
          }
        }

        chatForm.addEventListener('submit', async (event) => {
          event.preventDefault();
          const text = chatInput.value.trim();
          if (!text) return;
          appendChatBubble('user', text);
          chatInput.value = '';
          showTyping();

          try {
            const res = await fetch('/api/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: text, session_id: chatSessionId })
            });
            const data = await res.json();
            hideTyping();
            if (data.session_id) {
              chatSessionId = data.session_id;
              localStorage.setItem('legalChatSessionId', chatSessionId);
            }
            appendChatBubble('bot', data.success ? data.response : (data.message || 'Error contacting Legal AI.'));
          } catch (err) {
            hideTyping();
            appendChatBubble('bot', 'Network error: ' + err.message);
          }
        });

        clearChatBtn.addEventListener('click', () => {
          chatMessages.innerHTML = '';
          chatSessionId = null;
          localStorage.removeItem('legalChatSessionId');
        });

        loadChatHistory();

        async function refreshDashboard() {
          const res = await fetch('/api/dashboard');
          const data = await res.json();
          dashboardStats.innerHTML = `
            <div class="stat">Users: ${data.users}</div>
            <div class="stat">Bookings: ${data.bookings}</div>
            <div class="stat">Active roles: ${data.roles.join(', ')}</div>
          `;
          recentBookings.innerHTML = '<strong>Recent:</strong><br/>' + data.bookingsList.map(b => `${b.title} • ${b.start_time}`).join('<br/>');
        }

        // Role toggle (Individual / Lawyer / Therapist) drives both the
        // register/login payload's `role` field and the Payments tab's link,
        // replacing what used to be separate <select> dropdowns per form.
        let selectedRole = 'individual';
        const ROLE_LABELS = { individual: 'Individual', lawyer: 'Lawyer', therapist: 'Therapist' };
        const roleToggleButtons = document.querySelectorAll('#roleToggle button');
        const signinRoleLabel = document.getElementById('signinRoleLabel');
        const registerRoleLabel = document.getElementById('registerRoleLabel');
        const paymentsLink = document.getElementById('paymentsLink');

        function setRole(role) {
          selectedRole = role;
          roleToggleButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.role === role));
          signinRoleLabel.textContent = ROLE_LABELS[role];
          registerRoleLabel.textContent = ROLE_LABELS[role];
          paymentsLink.href = `/pay?consultation_type=${role === 'therapist' ? 'therapist' : 'lawyer'}`;
        }
        roleToggleButtons.forEach(btn => btn.addEventListener('click', () => setRole(btn.dataset.role)));

        const authTabButtons = document.querySelectorAll('#authTabs button');
        const authPanels = {
          quick: document.getElementById('panelQuick'),
          signin: document.getElementById('panelSignin'),
          register: document.getElementById('panelRegister'),
          payments: document.getElementById('panelPayments'),
        };
        function setAuthTab(tab) {
          authTabButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tab));
          Object.entries(authPanels).forEach(([key, panel]) => { panel.hidden = key !== tab; });
        }
        authTabButtons.forEach(btn => btn.addEventListener('click', () => setAuthTab(btn.dataset.tab)));

        const forgotPanel = document.getElementById('forgotPanel');
        const forgotMessage = document.getElementById('forgotMessage');
        document.getElementById('forgotPasswordLink').addEventListener('click', () => {
          forgotPanel.hidden = !forgotPanel.hidden;
        });
        document.getElementById('sendOtpBtn').addEventListener('click', async () => {
          const identifier = document.getElementById('forgotIdentifier').value.trim();
          if (!identifier) { forgotMessage.textContent = 'Enter your email or phone first.'; return; }
          const res = await fetch('/api/auth/forgot-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ identifier, channel: 'email' })
          });
          const data = await res.json();
          forgotMessage.textContent = data.message;
        });
        document.getElementById('resetPasswordBtn').addEventListener('click', async () => {
          const identifier = document.getElementById('forgotIdentifier').value.trim();
          const otp = document.getElementById('resetOtp').value.trim();
          const new_password = document.getElementById('resetNewPassword').value;
          if (!identifier || !otp || !new_password) { forgotMessage.textContent = 'Fill in email/phone, OTP, and new password.'; return; }
          const res = await fetch('/api/auth/reset-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ identifier, otp, new_password })
          });
          const data = await res.json();
          forgotMessage.textContent = data.message;
        });

        document.getElementById('registerForm').addEventListener('submit', async (event) => {
          event.preventDefault();
          const payload = {
            name: document.getElementById('name').value,
            email: document.getElementById('email').value,
            phone_number: '',
            password: document.getElementById('password').value,
            role: selectedRole,
            license_number: selectedRole === 'lawyer' ? '' : null
          };
          const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          authMessage.textContent = data.message;
          currentUser = data.user;
          refreshDashboard();
        });

        document.getElementById('loginForm').addEventListener('submit', async (event) => {
          event.preventDefault();
          const payload = {
            identifier: document.getElementById('loginEmail').value,
            password: document.getElementById('loginPassword').value,
            role: selectedRole
          };
          const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          authMessage.textContent = data.message;
          currentUser = data.user;
          refreshDashboard();
        });

        document.getElementById('bookingForm').addEventListener('submit', async (event) => {
          event.preventDefault();
          const payload = {
            title: document.getElementById('bookingTitle').value,
            client_name: document.getElementById('bookingClient').value,
            lawyer_name: document.getElementById('bookingLawyer').value,
            start_time: document.getElementById('bookingTime').value,
            role: document.getElementById('bookingRole').value,
            provider_type: document.getElementById('bookingProviderType').value,
            timezone: bookingTimezoneInput.value.trim() || 'UTC'
          };
          const res = await fetch('/api/bookings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          bookingMessage.textContent = data.message;
          refreshDashboard();
        });

        document.getElementById('createRoomBtn').addEventListener('click', () => {
          const room = `room-${Math.random().toString(36).slice(2, 8)}`;
          roomIdInput.value = room;
          callStatus.textContent = `Room created: ${room}`;
        });

        function getSocketUrl(room) {
          const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          return `${protocol}//${window.location.host}/ws/signaling/${encodeURIComponent(room)}`;
        }

        async function ensureMedia() {
          if (localStream) return localStream;
          localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
          localVideo.srcObject = localStream;
          return localStream;
        }

        // Practical fallback for a muted participant: no attempt at guessing
        // speech from lip movement (unreliable, and there's no viable
        // Armenian lipreading model to draw from) — just surface a clear
        // prompt so the call can move to chat/unmuting instead.
        // Note: browser support for firing track.onmute when the *sender*
        // sets track.enabled=false varies — this is best-effort, not a
        // guaranteed signal in every browser.
        const mutedNotice = document.getElementById('mutedNotice');
        const remoteTrackMuted = { audio: false, video: false };
        function updateMutedNotice() {
          mutedNotice.style.display = (remoteTrackMuted.audio || remoteTrackMuted.video) ? 'block' : 'none';
        }

        function createPeer() {
          if (peerConnection) return peerConnection;
          peerConnection = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
          peerConnection.ontrack = (event) => {
            remoteVideo.srcObject = event.streams[0];
            const track = event.track;
            remoteTrackMuted[track.kind] = track.muted;
            track.onmute = () => { remoteTrackMuted[track.kind] = true; updateMutedNotice(); };
            track.onunmute = () => { remoteTrackMuted[track.kind] = false; updateMutedNotice(); };
            updateMutedNotice();
          };
          peerConnection.onicecandidate = (event) => {
            if (event.candidate && ws) {
              ws.send(JSON.stringify({ type: 'candidate', candidate: event.candidate }));
            }
          };
          return peerConnection;
        }

        async function startCall() {
          const room = roomIdInput.value;
          if (!room) return;
          await ensureMedia();
          const pc = createPeer();
          localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
          ws = new WebSocket(getSocketUrl(room));
          ws.onopen = () => callStatus.textContent = 'Connected to signaling server';
          ws.onmessage = async (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'offer') {
              await pc.setRemoteDescription(new RTCSessionDescription(message.offer));
              const answer = await pc.createAnswer();
              await pc.setLocalDescription(answer);
              ws.send(JSON.stringify({ type: 'answer', answer }));
              callStatus.textContent = 'Answer sent';
            } else if (message.type === 'answer') {
              await pc.setRemoteDescription(new RTCSessionDescription(message.answer));
              callStatus.textContent = 'Call connected';
            } else if (message.type === 'candidate') {
              if (pc.remoteDescription) {
                await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
              } else {
                pendingOffer = true;
              }
            }
          };
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          ws.send(JSON.stringify({ type: 'offer', offer }));
          callStatus.textContent = 'Offer sent';
        }

        document.getElementById('joinRoomBtn').addEventListener('click', async () => {
          const room = roomIdInput.value;
          if (!room) return;
          await ensureMedia();
          const pc = createPeer();
          localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
          ws = new WebSocket(getSocketUrl(room));
          ws.onopen = () => callStatus.textContent = 'Joined room';
          ws.onmessage = async (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'offer') {
              await pc.setRemoteDescription(new RTCSessionDescription(message.offer));
              const answer = await pc.createAnswer();
              await pc.setLocalDescription(answer);
              ws.send(JSON.stringify({ type: 'answer', answer }));
              callStatus.textContent = 'Answer sent';
            } else if (message.type === 'answer') {
              await pc.setRemoteDescription(new RTCSessionDescription(message.answer));
              callStatus.textContent = 'Call connected';
            } else if (message.type === 'candidate') {
              if (pc.remoteDescription) {
                await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
              }
            }
          };
        });

        document.getElementById('startCallBtn').addEventListener('click', startCall);
        document.getElementById('hangUpBtn').addEventListener('click', () => {
          if (ws) ws.close();
          if (peerConnection) peerConnection.close();
          peerConnection = null;
          callStatus.textContent = 'Call ended';
        });

        const muteAudioBtn = document.getElementById('muteAudioBtn');
        const muteVideoBtn = document.getElementById('muteVideoBtn');
        muteAudioBtn.addEventListener('click', () => {
          if (!localStream) return;
          localStream.getAudioTracks().forEach(track => { track.enabled = !track.enabled; });
          const enabled = localStream.getAudioTracks()[0]?.enabled ?? true;
          muteAudioBtn.textContent = enabled ? '🎤 Mute my audio' : '🔇 Unmute my audio';
        });
        muteVideoBtn.addEventListener('click', () => {
          if (!localStream) return;
          localStream.getVideoTracks().forEach(track => { track.enabled = !track.enabled; });
          const enabled = localStream.getVideoTracks()[0]?.enabled ?? true;
          muteVideoBtn.textContent = enabled ? '📷 Mute my video' : '🚫 Unmute my video';
        });

        refreshDashboard();
      </script>
    </body>
    </html>
    """


@app.get("/pay", response_class=HTMLResponse)
async def pay_page(consultation_type: str = "lawyer"):
    """Backend-ready payment demo (card + Apple Pay + Google Pay via Stripe's
    Payment Element). Like the auth/booking demo on the home page, this is a
    minimal reference page — a B2B partner's own frontend would call
    /api/payments/create-intent directly with its own UI."""
    consultation_type = consultation_type if consultation_type in CONSULTATION_TYPES else "lawyer"
    publishable_key_notice = (
        "" if STRIPE_PUBLISHABLE_KEY
        else "<p style='color:#f87171;'>STRIPE_PUBLISHABLE_KEY is not set on the server — payments will not load.</p>"
    )
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Pay for Consultation</title>
      <script src="https://js.stripe.com/v3/"></script>
      <style>
        :root {{ color-scheme: dark; }}
        body {{ font-family: Arial, sans-serif; margin: 0; background: #08111d; color: #f5f7fa; }}
        .container {{ max-width: 520px; margin: 40px auto; padding: 24px; }}
        .card {{ background: #111c2c; border: 1px solid #23344f; border-radius: 14px; padding: 22px; }}
        input, select, button {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #35506f; margin-top: 8px; font-size: 14px; background: #0c1726; color: #f5f7fa; }}
        button {{ cursor: pointer; background: #2563eb; color: white; border: none; margin-top: 16px; }}
        #payment-element {{ margin-top: 14px; }}
        .small {{ font-size: 12px; color: #9fb0cb; }}
        a {{ color: #60a5fa; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="card">
          <h2>Pay for {consultation_type} consultation</h2>
          <p class="small">Demo checkout — card, Apple Pay, and Google Pay all go through this one form.</p>
          {publishable_key_notice}
          <form id="setupForm">
            <input id="customerName" placeholder="Your name" required />
            <input id="amount" type="number" min="1" placeholder="Amount (in cents, e.g. 5000 = $50.00)" required />
            <select id="currency">
              <option value="usd">USD</option>
              <option value="eur">EUR</option>
            </select>
            <button type="submit">Continue to payment</button>
          </form>
          <form id="paymentForm" style="display:none;">
            <div id="payment-element"></div>
            <button id="submitPayment" type="submit">Pay now</button>
            <div id="paymentMessage" class="small"></div>
          </form>
        </div>
        <p class="small" style="margin-top:12px;"><a href="/">&larr; Back to portal</a></p>
      </div>
      <script>
        const consultationType = {json.dumps(consultation_type)};
        let stripe = null;
        let elements = null;

        document.getElementById('setupForm').addEventListener('submit', async (event) => {{
          event.preventDefault();
          const customer_name = document.getElementById('customerName').value;
          const amount_cents = parseInt(document.getElementById('amount').value, 10);
          const currency = document.getElementById('currency').value;

          const res = await fetch('/api/payments/create-intent', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ consultation_type: consultationType, customer_name, amount_cents, currency }})
          }});
          const data = await res.json();
          if (!data.success) {{
            document.getElementById('paymentMessage').textContent = data.message;
            return;
          }}

          stripe = Stripe(data.publishable_key);
          elements = stripe.elements({{ clientSecret: data.client_secret }});
          const paymentElement = elements.create('payment');
          paymentElement.mount('#payment-element');

          document.getElementById('setupForm').style.display = 'none';
          document.getElementById('paymentForm').style.display = 'block';
        }});

        document.getElementById('paymentForm').addEventListener('submit', async (event) => {{
          event.preventDefault();
          const {{ error }} = await stripe.confirmPayment({{
            elements,
            confirmParams: {{ return_url: window.location.href }}
          }});
          if (error) {{
            document.getElementById('paymentMessage').textContent = error.message;
          }}
        }});
      </script>
    </body>
    </html>
    """


@app.get("/mood-tracking", response_class=HTMLResponse)
async def mood_tracking_page():
    """Web fallback for mood tracking when the user isn't on the mobile app.
    Backend-ready placeholder — the mobile app owns the real mood-tracking UI;
    this page exists so a web link can point somewhere useful in the meantime."""
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Mood Tracking</title>
      <style>
        :root { color-scheme: dark; }
        body { font-family: Arial, sans-serif; margin: 0; background: #08111d; color: #f5f7fa; }
        .container { max-width: 560px; margin: 40px auto; padding: 24px; }
        .card { background: #111c2c; border: 1px solid #23344f; border-radius: 14px; padding: 22px; }
        .small { font-size: 13px; color: #9fb0cb; line-height: 1.5; }
        a.button { display: inline-block; margin-top: 16px; padding: 10px 18px; border-radius: 8px; background: #2563eb; color: white; text-decoration: none; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="card">
          <h2>💙 Mood Tracking (web)</h2>
          <p class="small">
            This is a web fallback for mood tracking, for when you're not on the mobile app.
            The mobile app owns the full mood-tracking experience (daily check-ins, history,
            trends); this page is a placeholder for that flow on the web.
          </p>
          <p class="small">
            If the assistant's webcam feature has noticed a sustained low mood during a session,
            or you'd simply like to talk to someone, you can start a conversation with a therapist.
          </p>
          <a class="button" href="/therapist">Talk to a therapist</a>
        </div>
        <p class="small" style="margin-top:12px;"><a href="/">&larr; Back to portal</a></p>
      </div>
    </body>
    </html>
    """


@app.get("/therapist", response_class=HTMLResponse)
async def therapist_page():
    """Backend-ready placeholder for therapist consultations. Not a specific-
    therapist matching feature (that mirrors the lawyer-matching feature and is
    on hold pending a therapist dataset) — for now this only gets a user to a
    booking/payment flow, same shape as the existing lawyer booking form."""
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Talk to a Therapist</title>
      <style>
        :root { color-scheme: dark; }
        body { font-family: Arial, sans-serif; margin: 0; background: #08111d; color: #f5f7fa; }
        .container { max-width: 560px; margin: 40px auto; padding: 24px; }
        .card { background: #111c2c; border: 1px solid #23344f; border-radius: 14px; padding: 22px; margin-bottom: 16px; }
        .small { font-size: 13px; color: #9fb0cb; line-height: 1.5; }
        a.button { display: inline-block; margin-top: 16px; padding: 10px 18px; border-radius: 8px; background: #0f766e; color: white; text-decoration: none; }
        .chat-messages { height: 320px; overflow-y: auto; background: #0c1726; border: 1px solid #23344f; border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 10px; margin-top: 12px; }
        .chat-bubble { max-width: 85%; padding: 10px 14px; border-radius: 14px; white-space: pre-wrap; line-height: 1.4; font-size: 14px; }
        .chat-bubble.user { align-self: flex-end; background: #0f766e; color: white; border-bottom-right-radius: 4px; }
        .chat-bubble.bot { align-self: flex-start; background: #16253b; color: #f5f7fa; border: 1px solid #23344f; border-bottom-left-radius: 4px; }
        .chat-input-row { display: flex; gap: 8px; margin-top: 10px; }
        textarea, button { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #35506f; font-size: 14px; background: #0c1726; color: #f5f7fa; }
        .chat-input-row textarea { resize: none; height: 48px; }
        .chat-input-row button { width: auto; padding: 0 20px; cursor: pointer; background: #0f766e; color: white; border: none; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="card">
          <h2>🧑‍⚕️ Talk to a Therapist</h2>
          <p class="small">
            This is a supportive-conversation demo backed by a Q&amp;A dataset of past
            counseling conversations — it is <strong>not</strong> a licensed therapist and
            not a diagnosis. Matching to a specific human therapist (like the lawyer-matching
            feature) is a separate, not-yet-built feature.
          </p>
          <p class="small">
            If you are in immediate danger or thinking about suicide, please contact
            emergency services directly rather than using this chat.
          </p>
          <div id="chatMessages" class="chat-messages"></div>
          <form id="chatForm" class="chat-input-row">
            <textarea id="chatInput" placeholder="What's on your mind?" required></textarea>
            <button type="submit">Send</button>
          </form>
        </div>
        <div class="card">
          <a class="button" href="/pay?consultation_type=therapist">Book &amp; pay for a real session</a>
        </div>
        <p class="small"><a href="/">&larr; Back to portal</a></p>
      </div>
      <script>
        const chatMessages = document.getElementById('chatMessages');
        const chatForm = document.getElementById('chatForm');
        const chatInput = document.getElementById('chatInput');
        let sessionId = localStorage.getItem('therapistChatSessionId') || null;

        function appendBubble(role, text) {
          const bubble = document.createElement('div');
          bubble.className = `chat-bubble ${role}`;
          bubble.textContent = text;
          chatMessages.appendChild(bubble);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        chatForm.addEventListener('submit', async (event) => {
          event.preventDefault();
          const text = chatInput.value.trim();
          if (!text) return;
          appendBubble('user', text);
          chatInput.value = '';

          const res = await fetch('/api/therapist-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
          });
          const data = await res.json();
          if (data.session_id) {
            sessionId = data.session_id;
            localStorage.setItem('therapistChatSessionId', sessionId);
          }
          appendBubble('bot', data.success ? data.response : (data.message || 'Error contacting chat.'));
        });
      </script>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok"}


def _account_session_id(user_id: int) -> str:
    """The chat history session_id a logged-in user's messages are stored
    under, instead of the random per-guest uuid /api/chat otherwise
    generates -- deterministic on user_id so history is found again (the
    "cache") across reloads/devices on login, and so it can be located and
    wiped by delete_user_account on account deletion."""
    return f"user_{user_id}"


# 8-12 chars, at least one lowercase, one uppercase, one digit, one symbol --
# mirrors the frontend's STRONG_PASSWORD_RE (frontend/src/main.js) so
# register/reset-password reject a weak password server-side even if a
# request bypasses the browser form entirely, not just when the UI enforces it.
_STRONG_PASSWORD_RE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,12}$")
_PASSWORD_HINT = "Password must be 8-12 characters with upper & lower case, a number, and a symbol."


@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    if not _STRONG_PASSWORD_RE.match(request.password):
        return {"success": False, "message": _PASSWORD_HINT}

    existing = portal_store.find_user(email=request.email, phone_number=request.phone_number or None)
    if existing:
        return {"success": False, "message": "User already exists"}

    user = portal_store.create_user(
        name=request.name,
        email=request.email,
        phone_number=request.phone_number or "",
        password=request.password,
        role=request.role,
        license_number=request.license_number or "",
    )
    token = portal_store.create_session(user["id"], user["role"])
    return {
        "success": True, "message": "Registered successfully", "user": user, "token": token,
        "chat_session_id": _account_session_id(user["id"]),
    }


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    user = portal_store.authenticate_user(request.identifier, request.password, request.role)
    if not user:
        return {"success": False, "message": "Account not found or password is incorrect"}
    token = portal_store.create_session(user["id"], user["role"])
    return {
        "success": True, "message": "Signed in successfully", "user": user, "token": token,
        "chat_session_id": _account_session_id(user["id"]),
    }


class LogoutRequest(BaseModel):
    token: str


@app.post("/api/auth/logout")
async def logout(request: LogoutRequest):
    portal_store.delete_session(request.token)
    return {"success": True, "message": "Signed out"}


@app.get("/api/auth/me")
async def get_current_session(request: Request):
    """Validate a session token passed as `Authorization: Bearer <token>` and
    return the associated user. This is additive, opt-in session support —
    existing endpoints are unchanged and still don't require a token, so this
    doesn't break any current caller; a future auth-gated endpoint can depend
    on this same lookup."""
    auth_header = request.headers.get("authorization", "")
    token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
    session = portal_store.get_session(token)
    if not session:
        return {"success": False, "message": "Invalid or expired session"}
    return {
        "success": True, "user_id": session["user_id"], "role": session["role"],
        "chat_session_id": _account_session_id(session["user_id"]),
    }


@app.delete("/api/auth/account")
async def delete_account(request: Request):
    """Permanently deletes the signed-in user's account: every session token
    (all devices), and their chat history (see _account_session_id/
    delete_user_account) along with the user row itself. Irreversible --
    there's no confirmation step here since the frontend already gates this
    behind its own confirm dialog."""
    auth_header = request.headers.get("authorization", "")
    token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
    session = portal_store.get_session(token)
    if not session:
        return {"success": False, "message": "Invalid or expired session"}
    await run_in_threadpool(
        portal_store.delete_user_account, session["user_id"], _account_session_id(session["user_id"])
    )
    return {"success": True, "message": "Account deleted"}


@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    row = portal_store.find_user(email=request.identifier, phone_number=request.identifier)
    if not row:
        return {"success": False, "message": "Account not found"}

    otp = f"{len(row['email'] or '') % 10}{len(row['phone_number'] or '') % 10}{abs(hash(row['name'] or '')) % 10000:04d}"
    portal_store.set_password_reset_otp(request.identifier, otp, row["email"] or "")
    return {"success": True, "message": f"OTP sent via {request.channel}", "otp": otp}


@app.post("/api/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    if not _STRONG_PASSWORD_RE.match(request.new_password):
        return {"success": False, "message": _PASSWORD_HINT}

    reset_row = portal_store.get_password_reset(request.identifier)
    if not reset_row or reset_row["otp"] != request.otp:
        return {"success": False, "message": "Invalid OTP"}

    updated = portal_store.update_password(request.identifier, request.new_password)
    if not updated:
        return {"success": False, "message": "Account not found"}

    portal_store.clear_password_reset(request.identifier)
    return {"success": True, "message": "Password updated successfully"}


@app.get("/api/dashboard")
async def dashboard():
    roles = portal_store.distinct_roles()
    return {
        "users": portal_store.count_users(),
        "bookings": portal_store.count_bookings(),
        "roles": roles or ["individual", "lawyer", "therapist"],
        "bookingsList": portal_store.recent_bookings(5),
    }


@app.get("/api/bookings")
async def get_bookings():
    return {"bookings": portal_store.list_bookings()}


@app.post("/api/bookings")
async def create_booking(request: BookingRequest):
    booking = portal_store.create_booking(
        title=request.title,
        client_name=request.client_name,
        lawyer_name=request.lawyer_name,
        start_time=request.start_time,
        role=request.role,
        provider_type=request.provider_type,
        timezone=request.timezone,
    )
    return {"success": True, "message": "Appointment booked successfully", "booking": booking}


@app.get("/api/bookings/availability")
async def get_availability(
    provider_name: str,
    date: str,
    timezone: str = "UTC",
    slot_minutes: int = 60,
    start_hour: int | None = None,
    end_hour: int | None = None,
):
    """Free/busy slots for a provider on one local calendar date, in both that
    timezone and UTC. If the provider has a configured weekly schedule (see
    POST /api/providers/schedule) for that weekday, its hours are used;
    otherwise falls back to a fixed default window (09:00-18:00). Explicit
    start_hour/end_hour query params always win over both."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return {"success": False, "message": f"Unknown timezone: {timezone}"}
    try:
        day = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return {"success": False, "message": "date must be in YYYY-MM-DD format"}

    if start_hour is None or end_hour is None:
        schedule = await run_in_threadpool(portal_store.get_provider_schedule, provider_name, day.weekday())
        start_hour = start_hour if start_hour is not None else (schedule["start_hour"] if schedule else 9)
        end_hour = end_hour if end_hour is not None else (schedule["end_hour"] if schedule else 18)

    if not (0 <= start_hour < end_hour <= 24):
        return {"success": False, "message": "start_hour must be less than end_hour, both within 0-24"}
    if slot_minutes <= 0:
        return {"success": False, "message": "slot_minutes must be positive"}

    day_start_local = datetime(day.year, day.month, day.day, start_hour, 0, tzinfo=tz)
    day_end_local = datetime(day.year, day.month, day.day, end_hour, 0, tzinfo=tz)

    # Widen the DB query window so bookings whose *local* date differs from `date`
    # (because timezone offsets can shift a UTC instant across a calendar day)
    # are still caught if their instant actually falls inside the local window.
    query_start_utc = (day_start_local - timedelta(hours=14)).astimezone(dt_timezone.utc).isoformat()
    query_end_utc = (day_end_local + timedelta(hours=14)).astimezone(dt_timezone.utc).isoformat()
    busy_starts_utc = await run_in_threadpool(
        portal_store.get_provider_busy_ranges, provider_name, query_start_utc, query_end_utc
    )
    busy_instants = []
    for iso in busy_starts_utc:
        try:
            busy_instants.append(datetime.fromisoformat(iso))
        except ValueError:
            continue

    slots = []
    cursor_local = day_start_local
    slot_delta = timedelta(minutes=slot_minutes)
    while cursor_local + slot_delta <= day_end_local:
        slot_end_local = cursor_local + slot_delta
        slot_start_utc = cursor_local.astimezone(dt_timezone.utc)
        slot_end_utc = slot_end_local.astimezone(dt_timezone.utc)

        is_free = not any(slot_start_utc <= busy < slot_end_utc for busy in busy_instants)

        slots.append({
            "local_start": cursor_local.isoformat(),
            "local_end": slot_end_local.isoformat(),
            "utc_start": slot_start_utc.isoformat(),
            "utc_end": slot_end_utc.isoformat(),
            "is_free": is_free,
        })
        cursor_local = slot_end_local

    return {
        "success": True,
        "provider_name": provider_name,
        "date": date,
        "timezone": timezone,
        "slot_minutes": slot_minutes,
        "slots": slots,
    }


class ProviderScheduleRequest(BaseModel):
    provider_name: str
    weekday: int  # 0=Monday .. 6=Sunday
    start_hour: int
    end_hour: int
    timezone: str = "UTC"


@app.post("/api/providers/schedule")
async def set_provider_schedule(request: ProviderScheduleRequest):
    """Configure a provider's working hours for one weekday (upsert — call
    again with the same provider_name/weekday to change it). Used by
    GET /api/bookings/availability instead of the fixed 09:00-18:00 default
    once set. Days without a configured schedule still use the default."""
    if not (0 <= request.weekday <= 6):
        return {"success": False, "message": "weekday must be 0 (Monday) through 6 (Sunday)"}
    if not (0 <= request.start_hour < request.end_hour <= 24):
        return {"success": False, "message": "start_hour must be less than end_hour, both within 0-24"}
    schedule = await run_in_threadpool(
        portal_store.set_provider_schedule,
        request.provider_name, request.weekday, request.start_hour, request.end_hour, request.timezone,
    )
    return {"success": True, "schedule": schedule}


@app.get("/api/providers/schedule")
async def get_provider_schedule(provider_name: str):
    schedule = await run_in_threadpool(portal_store.list_provider_schedule, provider_name)
    return {"success": True, "provider_name": provider_name, "schedule": schedule}


@app.post("/api/chat")
async def chat(request: ChatMessageRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_message = request.message.strip()
    if not user_message:
        return {"success": False, "message": "Please type a message.", "session_id": session_id}

    history = await run_in_threadpool(portal_store.get_chat_messages, session_id, "legal")
    await run_in_threadpool(portal_store.append_chat_message, session_id, "legal", "user", user_message)

    try:
        agent = get_legal_agent()
        response_text = await run_in_threadpool(
            agent.get_advice, user_message, history, language=request.language or "hy"
        )
    except Exception as exc:
        response_text = (
            "⚠️ Legal AI backend is unavailable right now "
            f"(is Ollama running? error: {exc})"
        )

    await run_in_threadpool(portal_store.append_chat_message, session_id, "legal", "bot", response_text)
    return {"success": True, "session_id": session_id, "response": response_text}


@app.get("/api/chat/{session_id}")
async def get_chat_history(session_id: str):
    messages = await run_in_threadpool(portal_store.get_chat_messages, session_id, "legal")
    return {"session_id": session_id, "messages": messages}


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Display names for surfacing which language a transcript was actually
# detected in (see _whisper_transcribe) when it differs from the request's
# own `language`.
STT_LANGUAGE_DISPLAY_NAMES = {
    "hy": "Armenian",
    "en": "English",
    "ru": "Russian",
}

# A video's full audio track producing only a couple of characters/one short
# word (e.g. "I am") is far more likely a spurious misrecognition of
# music/singing/background noise than genuine dialogue. Below this length, a
# transcript is treated as not good enough to keep (see _whisper_transcribe's
# min_chars); if nothing clears the bar, the video is reported as having
# audio but no reliable speech (see _transcribe_video_audio) rather than
# confidently showing a couple of probably-wrong words. Not applied to the
# live mic endpoint (/api/speech-to-text), where a short "Yes"/"Stop" is a
# normal, expected utterance rather than noise.
MIN_VIDEO_TRANSCRIPT_CHARS = 8


def _extract_wav(src_path: str, timeout: int = 60) -> str:
    """ffmpeg: convert src_path into a 16kHz mono WAV alongside it. Works
    both for raw mic recordings (webm/ogg) and as an audio-track extractor
    for video uploads — ffmpeg pulls whichever audio stream is present and
    `-vn` just tells it to drop any video stream from the output."""
    wav_path = src_path + ".wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-vn", "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")
    return wav_path


# Whisper (openai-whisper) replaces recognize_google for both the mic
# endpoint and video transcription — real-world testing showed Google's free
# Web Speech API is noticeably less accurate for Armenian/Russian than
# English, especially on real microphone audio (accent, background noise),
# while Whisper is natively multilingual and was trained on Armenian data
# directly (see stt_service/app.py, the standalone Whisper service this
# reuses the model-loading/tuning from). "medium" trades a slower first
# transcribe for noticeably better accuracy on lower-resource languages than
# "small"/"base" — override with WHISPER_MODEL if that tradeoff needs
# revisiting.
_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
_whisper_model = None


def _get_whisper_model():
    """Lazy-loaded so the server starts instantly and never pays this cost
    if no request ever needs speech-to-text — same pattern as
    get_vision_service()/_get_audio_event_classifier()."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"🗣️ Loading Whisper speech-to-text model ({_WHISPER_MODEL_SIZE}) on first use...")
        _whisper_model = whisper.load_model(_WHISPER_MODEL_SIZE)
    return _whisper_model


def _whisper_join_transcript(result: dict) -> str:
    """result["text"] is Whisper's own concatenation of its segments, which
    doesn't reliably insert a space at every segment boundary — on longer
    clips (more than one segment) that shows up as two words from adjacent
    segments getting glued together. Rebuilding the text by explicitly
    joining each segment's (stripped) text with a single space guarantees a
    word boundary at every seam regardless of what Whisper itself included.
    Same fix as stt_service/app.py's _join_transcript."""
    segments = result.get("segments") or []
    if segments:
        text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
    else:
        text = (result.get("text") or "").strip()
    return re.sub(r"\s+", " ", text).strip()


def _whisper_transcribe(
    audio_path: str, language: str = None, min_chars: int = 0, filter_repetition: bool = True
) -> tuple[str, str]:
    """Transcribes an audio/video file with Whisper. `language` is an
    optional hint (Whisper's own ISO codes match this app's "hy"/"en"/"ru"
    exactly, no locale-string mapping needed like recognize_google's
    STT_LANGUAGE_MAP) — Whisper auto-detects the spoken language on its own
    when this is omitted, which is what video/link transcription wants
    (genuinely unknown spoken language) instead of manually looping over
    locales the way _recognize_wav's try_other_locales did for Google.

    no_speech_threshold=0.3 and condition_on_previous_text=False (not
    Whisper's tuned-for-English defaults of 0.6/True) avoid it misjudging a
    quieter or lower-resource-language segment as silence and skipping it,
    or drifting into a repetition loop that eats the rest of the transcript
    — both show up as "only got part of it" even though the whole thing was
    said clearly. Same tuning as stt_service/app.py.

    Returns (text, detected_language); text is "" if nothing usable was
    recognized (post sanitize_transcript + min_chars, matching
    _recognize_wav's contract)."""
    model = _get_whisper_model()
    result = model.transcribe(
        audio_path,
        language=language or None,
        no_speech_threshold=0.3,
        condition_on_previous_text=False,
        fp16=False,
    )
    detected_language = result.get("language") or language or ""
    text = sanitize_transcript(
        _whisper_join_transcript(result), language=detected_language, filter_repetition=filter_repetition
    )
    if text and len(text) >= min_chars:
        return text, detected_language
    return "", ""


def _wav_has_audio_content(wav_path: str, rms_threshold: int = 200) -> bool:
    """Cheap RMS-energy check on the extracted 16kHz mono WAV — distinguishes
    a track that's actually silent from one that has real sound (music,
    background noise, unclear speech, etc.) recognize_google just couldn't
    turn into a transcript, so the caller can say "no speech, but sound was
    heard" instead of a flat, misleading "nothing detected"."""
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_width = wf.getsampwidth()
        return bool(frames) and audioop.rms(frames, sample_width) >= rms_threshold
    except Exception:
        return False


def _transcribe_video_audio(video_path: str, language: str) -> tuple[str, bool, str]:
    """Best-effort: extract + transcribe whatever words are in a video's
    audio track for the upload analysis, using Whisper (see
    _whisper_transcribe) instead of recognize_google — same accuracy
    motivation as /api/speech-to-text. Transcribes the whole track,
    spoken or sung — Whisper's own lyrics generally come through fine, so
    there's no separate speech-vs-music gate here. Unlike
    /api/speech-to-text, a failure here (no audio track, unrecognized
    speech) is not fatal to the upload — it just means no transcript, so
    this always returns a (text, has_nonspeech_audio, detected_language)
    triple ("", False, "" on failure) instead of raising.
    has_nonspeech_audio is only meaningful when text is empty — it tells
    the caller whether the track had audible content (music, unclear
    speech, background noise) that just didn't end up as transcribed
    speech. detected_language is which language Whisper actually detected
    the speech to be in — since `language` isn't passed as a hard
    constraint here (the spoken language in an arbitrary video/link is
    genuinely unknown), this may differ from the `language` the request was
    made with."""
    wav_path = None
    try:
        wav_path = _extract_wav(video_path)
        # filter_repetition=False: a sung track's repeated hook/chorus words
        # ("la la la", "yes yes yes") are genuine lyrics here, not a mic's
        # misheard-silence artifact — see sanitize_transcript's docstring.
        text, detected_language = _whisper_transcribe(
            wav_path, min_chars=MIN_VIDEO_TRANSCRIPT_CHARS, filter_repetition=False
        )
        has_nonspeech_audio = (not text) and _wav_has_audio_content(wav_path)
        return text, has_nonspeech_audio, detected_language
    except Exception:
        has_nonspeech_audio = bool(wav_path and os.path.exists(wav_path) and _wav_has_audio_content(wav_path))
        return "", has_nonspeech_audio, ""
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


async def _analyze_video_upload(tmp_path: str, filename: str, language: str, session_id: str) -> dict:
    """Runs the vision + speech-to-text analysis pipeline on a local video
    file and, when session_id is given, appends a summary to that chat
    session. Shared by /api/upload's video branch and /api/upload-link
    (video URLs downloaded via yt-dlp), so both entry points produce
    identical results once a local file exists on disk."""
    vision_service = get_vision_service()
    result = await run_in_threadpool(vision_service.analyze_video_headless, tmp_path, 12, language)
    transcript, has_nonspeech_audio, detected_language = await run_in_threadpool(
        _transcribe_video_audio, tmp_path, language
    )
    result["transcript"] = transcript
    result["has_nonspeech_audio"] = has_nonspeech_audio
    result["transcript_language"] = detected_language

    actions = ", ".join(result.get("actions") or []) or "none detected"
    emotion_changes = result.get("emotion_changes") or []
    if transcript:
        # detected_language can differ from the request's own `language`
        # (e.g. a Russian-language video while the portal is set to
        # Armenian) — _recognize_wav already fell back across locales to
        # find it, so say which one explicitly instead of silently
        # presenting it as if it matched the UI language.
        if detected_language and detected_language != language:
            lang_name = STT_LANGUAGE_DISPLAY_NAMES.get(detected_language, detected_language)
            speech_note = f"; spoken (in {lang_name}): \"{transcript}\""
        else:
            speech_note = f"; spoken: \"{transcript}\""
    elif has_nonspeech_audio:
        # The track has audible sound (music, singing, background noise,
        # etc.) but nothing recognizable as speech in any supported
        # language — worth saying explicitly instead of implying the video
        # is silent.
        speech_note = "; no speech detected — sound was heard (possibly music/a song or background audio)"
    else:
        # No speech to go on — fall back to describing what the action
        # detection actually saw happening in the video instead.
        speech_note = f"; no speech detected (based on action detection: {actions})"
    emotion_note = (
        f"; emotion changed during video: {' → '.join(emotion_changes)}"
        if len(emotion_changes) > 1 else ""
    )
    objects_seen_list = result.get("objects") or []
    # Non-person objects (a vehicle, a phone, etc.) seen alongside whatever
    # actions/emotion were detected — e.g. a person next to a car — instead
    # of leaving vehicles/objects out of the summary entirely just because a
    # person was also in frame.
    objects_note = f"; objects seen: {', '.join(objects_seen_list)}" if objects_seen_list else ""
    video_summary = (
        f"[Video '{filename}' analyzed — "
        f"actions: {actions}; emotion: {result.get('emotion') or 'n/a'}"
        + emotion_note + objects_note + speech_note
        + "]"
    )

    # NOTE: this used to also ask the legal agent for an actual
    # interpretation/analysis of video_summary. Disabled — see the matching
    # comment in upload_file() above: the armenia-lawyer-router model is
    # unreliable for free-text generation (v1 pure gibberish, v2 collapses
    # into mixed-script noise or leaks its own system prompt within a
    # sentence or two, verified directly against Ollama's REST API). Re-enable
    # agent.get_advice(video_summary, None, language) only once the model
    # itself has actually been fixed and re-verified.

    if session_id:
        # Each video starts a new case — clear this session's prior history
        # first, so a follow-up chat about this video isn't fed a previous,
        # unrelated video's analysis as context.
        await run_in_threadpool(portal_store.clear_chat_messages, session_id, "legal")
        await run_in_threadpool(
            portal_store.append_chat_message, session_id, "legal", "bot", video_summary
        )

    return {"success": True, "kind": "video", **result}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    language: str = Form("hy"),
    session_id: str = Form(None),
):
    """Browser equivalent of the desktop CLI's [u]pload (src/main.py
    handle_upload): documents get embedded into the case vector store,
    videos get run through the same YOLO+MediaPipe action/emotion pipeline
    (headless — see LegalVisionService.analyze_video_headless) plus a
    speech-to-text pass over the video's audio track, so "what was said"
    is surfaced alongside the visual analysis. When session_id is given,
    the video analysis (actions/emotion/transcript) is also appended to
    that chat session as a bot message, so a follow-up question in the
    same session has it as conversational context. Any file that isn't a
    recognized video extension is treated as a document: IngestionService
    has a dedicated parser for the most common formats (txt/xlsx/csv/json/
    docx/pdf) and falls back to generic best-effort text extraction for
    anything else, so uploads are no longer rejected outright based on
    extension."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    is_video = suffix in VIDEO_EXTENSIONS
    if not file.filename:
        return {"success": False, "message": "No file was uploaded."}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if not is_video:
            from src.services.ingestion import IngestionService

            agent = await run_in_threadpool(get_legal_agent)
            ingestor = IngestionService(agent.repo.db)
            status = await run_in_threadpool(ingestor.process_file, tmp_path)

            # NOTE: this used to also run the extracted text through
            # agent.get_advice() for an actual analysis (matching the CLI's
            # handle_upload). Disabled — the armenia-lawyer-router Ollama
            # model produces unreliable output for free-text generation: v1
            # was pure gibberish from the first token, and v2 (a retrained
            # replacement, now the live weights behind this model name) was
            # tried too — verified directly via clean (non-terminal-escaped)
            # queries against Ollama's REST API, it opens with one coherent
            # Armenian clause but then reliably collapses into mixed-script
            # noise or leaks its own system-prompt text within a sentence or
            # two, regardless of prompt or repeat_penalty tuning. Surfacing
            # that to users as "legal analysis" is worse than not analyzing
            # at all. Re-enable get_advice(doc_text, None, language) only
            # after the model itself (fine-tune/GGUF conversion) is actually
            # fixed and re-verified — not just by swapping the Ollama tag.

            return {"success": True, "kind": "document", "message": status}

        return await _analyze_video_upload(tmp_path, file.filename, language, session_id)
    except Exception as exc:
        return {"success": False, "message": f"Could not process '{file.filename}': {exc}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# Cap on a linked video's length before it's even downloaded — this pipeline
# only samples up to 12 frames regardless of video length, so a multi-hour
# link buys no analysis benefit, only a slow/expensive download.
MAX_VIDEO_LINK_DURATION_SECONDS = 20 * 60


def _video_file_has_audio_stream(video_path: str, timeout: int = 30) -> bool:
    """ffprobe check: does this downloaded file actually contain an audio
    stream? Used by _download_video_from_url — see its docstring for why
    this matters (some yt-dlp format listings claim an audio codec that the
    actual downloaded stream doesn't contain)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def _download_video_from_url(url: str) -> str:
    """Downloads a video from a YouTube/TikTok/Instagram/etc. link (any site
    yt-dlp supports) into a fresh temp directory, so it can be run through
    the exact same analysis pipeline as an uploaded file
    (_analyze_video_upload). Checks the video's duration via yt-dlp's
    metadata-only extraction *before* downloading anything, and rejects
    anything over MAX_VIDEO_LINK_DURATION_SECONDS. The download itself is
    capped to a modest resolution (this pipeline downscales to 960px width
    internally anyway, and speech-to-text only needs the audio track) to
    keep it fast and small.

    Tries a short list of format selectors in order, verifying with ffprobe
    that the downloaded file actually has an audio stream before accepting
    it — checked against a real TikTok video, yt-dlp's format listing
    claimed every available format included AAC audio, but the actual
    highest-resolution CDN variant "best" resolved to was silently
    video-only, silently producing a video the speech pipeline could never
    find any words in no matter how good the classifier logic downstream
    was. `download` (TikTok's own default, watermarked link) is tried last
    specifically because it reliably included real audio in that test, as a
    fallback for whatever site-specific quirk causes the first format to
    come up silent. If every attempt downloads but none has audio (e.g. the
    video is genuinely silent), the last successful download is still
    returned — the vision analysis half of the pipeline doesn't need audio,
    so a video-only file is a worse-than-ideal result, not a failure.

    Returns the local path to the downloaded video file."""
    import yt_dlp

    # socket_timeout/retries bound how long a stalled or rate-limited host
    # (Instagram in particular routinely stalls or resets connections for
    # unauthenticated requests) can hang this call — without them a bad
    # network condition here blocks the request indefinitely, which is what
    # eventually surfaces to the browser as an opaque "Load failed" instead
    # of the clear JSON error message the except block below would return.
    NETWORK_OPTS = {"socket_timeout": 20, "retries": 3, "fragment_retries": 3}

    probe_opts = {"quiet": True, "no_warnings": True, "skip_download": True, **NETWORK_OPTS}
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration") or 0
    if duration and duration > MAX_VIDEO_LINK_DURATION_SECONDS:
        raise ValueError(
            f"Video is {duration // 60} min long — linked videos are limited to "
            f"{MAX_VIDEO_LINK_DURATION_SECONDS // 60} min."
        )

    tmp_dir = tempfile.mkdtemp(prefix="video_link_")
    format_attempts = ["best[height<=480]/best", "download"]

    downloaded_path = None
    for i, fmt in enumerate(format_attempts):
        # Each attempt gets its own output filename (attempt0_/attempt1_...)
        # — reusing the same path across attempts made yt-dlp silently skip
        # re-downloading when a file already existed there from the
        # previous (audio-less) attempt, defeating the whole fallback.
        download_opts = {
            "quiet": True,
            "no_warnings": True,
            "outtmpl": os.path.join(tmp_dir, f"attempt{i}_%(id)s.%(ext)s"),
            "format": fmt,
            "merge_output_format": "mp4",
            **NETWORK_OPTS,
        }
        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                candidate = ydl.prepare_filename(info)
        except Exception:
            continue  # this format isn't available/failed -- try the next one

        if not os.path.exists(candidate):
            # merge_output_format can change the actual extension (e.g. a
            # separately-downloaded webm video+audio pair gets merged into .mp4).
            candidate = os.path.splitext(candidate)[0] + ".mp4"
        if not os.path.exists(candidate):
            continue

        downloaded_path = candidate  # keep as best-so-far even if audio-less
        if _video_file_has_audio_stream(candidate):
            return downloaded_path

    if downloaded_path is None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Download completed but the output file wasn't found.")
    return downloaded_path


@app.post("/api/upload-link")
async def upload_video_link(
    url: str = Form(...),
    language: str = Form("hy"),
    session_id: str = Form(None),
):
    """Same video analysis pipeline as /api/upload's video branch
    (_analyze_video_upload), but for a YouTube/TikTok/Instagram/etc. link
    instead of a local file: downloads the video via yt-dlp first, then
    reuses the exact same analysis + summary + session-append logic so both
    entry points behave identically."""
    if not url or not url.strip():
        return {"success": False, "message": "No video URL was given."}

    tmp_path = None
    try:
        tmp_path = await run_in_threadpool(_download_video_from_url, url.strip())
        return await _analyze_video_upload(tmp_path, os.path.basename(tmp_path), language, session_id)
    except Exception as exc:
        return {"success": False, "message": f"Could not process video link: {exc}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            shutil.rmtree(os.path.dirname(tmp_path), ignore_errors=True)


@app.post("/api/speech-to-text")
async def speech_to_text(file: UploadFile = File(...), language: str = Form("hy")):
    """Browser equivalent of the desktop CLI's mic input (src/services/voice.py
    listen_once): the browser records with MediaRecorder (webm/opus, ogg, or
    similar — never raw WAV), so this converts to a 16kHz mono WAV via ffmpeg
    before transcribing with Whisper (see _whisper_transcribe) — replaces an
    earlier recognize_google-based version, which real-world testing showed
    was noticeably less accurate for Armenian/Russian than English on real
    microphone audio."""
    src_path = None
    wav_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src_tmp:
            src_tmp.write(await file.read())
            src_path = src_tmp.name

        def convert_and_transcribe():
            nonlocal wav_path
            wav_path = _extract_wav(src_path, timeout=30)
            return _whisper_transcribe(wav_path, language=language)

        text, detected_language = await run_in_threadpool(convert_and_transcribe)
        if not text:
            return {"success": False, "message": "Didn't catch valid speech — please try again."}
        return {"success": True, "text": text, "language": detected_language}
    except Exception as exc:
        return {"success": False, "message": f"Speech-to-text failed: {exc}"}
    finally:
        for p in (src_path, wav_path):
            if p and os.path.exists(p):
                os.remove(p)


@app.post("/api/therapist-chat")
async def therapist_chat(request: ChatMessageRequest):
    """Supportive-conversation chat backed by MentalHealthQAClassifier, separate
    from the legal /api/chat. Always screens for crisis signals first (keyword,
    then the Random Forest risk classifier if the legal agent happens to be
    available) and returns the crisis response before ever touching the Q&A
    retrieval — the dataset itself contains "suicidal thoughts"-labeled rows,
    so a retrieved answer must never be allowed to substitute for it."""
    session_id = request.session_id or str(uuid.uuid4())
    user_message = request.message.strip()
    if not user_message:
        return {"success": False, "message": "Please type a message.", "session_id": session_id}
    language = request.language or "en"

    await run_in_threadpool(portal_store.append_chat_message, session_id, "therapist", "user", user_message)

    response_text = None
    if detect_crisis_signal(user_message):
        response_text = get_crisis_response(language)

    agent = None
    if response_text is None:
        try:
            agent = get_legal_agent()
            if agent.risk_classifier:
                risk = await run_in_threadpool(agent.risk_classifier.classify_mental_health_risk, user_message)
                if risk and risk["is_risk"]:
                    response_text = get_crisis_response(language)
        except Exception:
            pass  # Legal agent/Ollama being unavailable shouldn't block therapist chat.

    if response_text is None:
        qa_classifier = get_mental_health_qa_classifier()
        try:
            if agent is None or not agent.llm:
                raise RuntimeError("LLM unavailable")
            llm_response = await run_in_threadpool(
                _direct_therapist_answer, user_message, qa_classifier, agent.llm, language
            )
            response_text = (
                f"{llm_response}\n\n"
                f"—\nThis is a supportive-conversation demo, not advice from a licensed "
                f"therapist. For an actual consultation, visit /therapist."
            )
        except Exception as exc:
            print(f"⚠️ Direct LLM answer unavailable, falling back to retrieval: {exc}")
            match = await run_in_threadpool(qa_classifier.find_similar_answer, user_message)
            if match:
                label_note = f" (topic: {match['label']})" if match.get("label") else ""
                response_text = (
                    f"{match['answer']}\n\n"
                    f"—\nThis is a supportive response from similar past conversations{label_note}, "
                    f"not advice from a licensed therapist. For an actual consultation, visit /therapist."
                )
            else:
                response_text = (
                    "I don't have a good match for that in similar past conversations. "
                    "For a real conversation with a licensed therapist, visit /therapist."
                )

    await run_in_threadpool(portal_store.append_chat_message, session_id, "therapist", "bot", response_text)
    return {"success": True, "session_id": session_id, "response": response_text}


@app.get("/api/therapist-chat/{session_id}")
async def get_therapist_chat_history(session_id: str):
    messages = await run_in_threadpool(portal_store.get_chat_messages, session_id, "therapist")
    return {"session_id": session_id, "messages": messages}


@app.post("/api/payments/create-intent")
async def create_payment_intent(request: PaymentIntentRequest):
    """Create a Stripe PaymentIntent for a lawyer or therapist consultation.

    With automatic_payment_methods enabled, Stripe's client-side Payment Element
    shows card, Apple Pay, and Google Pay automatically for eligible browsers/
    devices — no separate integration code needed for each payment method.
    """
    if not STRIPE_SECRET_KEY:
        return {"success": False, "message": "Payments are not configured on this server (STRIPE_SECRET_KEY missing)."}
    if request.consultation_type not in CONSULTATION_TYPES:
        return {"success": False, "message": f"consultation_type must be one of {sorted(CONSULTATION_TYPES)}"}
    if request.amount_cents <= 0:
        return {"success": False, "message": "amount_cents must be positive."}

    try:
        intent = await run_in_threadpool(
            stripe.PaymentIntent.create,
            amount=request.amount_cents,
            currency=request.currency,
            automatic_payment_methods={"enabled": True},
            metadata={"consultation_type": request.consultation_type, "customer_name": request.customer_name},
        )
    except stripe.error.StripeError as exc:
        return {"success": False, "message": f"Stripe error: {exc.user_message or str(exc)}"}

    payment = portal_store.create_payment(
        consultation_type=request.consultation_type,
        customer_name=request.customer_name,
        amount_cents=request.amount_cents,
        currency=request.currency,
        stripe_payment_intent_id=intent["id"],
        status=intent["status"],
        created_at=datetime.utcnow().isoformat(),
        provider_name=request.provider_name,
        start_time=request.start_time,
        timezone=request.timezone,
    )
    return {
        "success": True,
        "payment_id": payment["id"],
        "client_secret": intent["client_secret"],
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
    }


@app.get("/api/payments/{payment_id}")
async def get_payment_status(payment_id: int):
    payment = portal_store.get_payment(payment_id)
    if not payment:
        return {"success": False, "message": "Payment not found"}
    return {"success": True, "payment": payment}


@app.post("/api/payments/webhook")
async def stripe_webhook(request: Request):
    """Stripe calls this after a payment succeeds/fails so payment status stays
    correct even if the customer closes the tab before the client-side confirm
    call returns. Configure this URL in the Stripe Dashboard webhook settings."""
    if not STRIPE_WEBHOOK_SECRET:
        return {"success": False, "message": "Webhook is not configured on this server (STRIPE_WEBHOOK_SECRET missing)."}

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        return {"success": False, "message": "Invalid webhook signature or payload"}

    if event["type"] in ("payment_intent.succeeded", "payment_intent.payment_failed"):
        intent = event["data"]["object"]
        status = "succeeded" if event["type"] == "payment_intent.succeeded" else "failed"
        portal_store.update_payment_status(intent["id"], status)
        if status == "succeeded":
            payment = portal_store.get_payment_by_intent(intent["id"])
            if payment:
                portal_store.create_booking_from_payment_if_scheduled(payment)

    return {"received": True}


@app.websocket("/ws/signaling/{room_id}")
async def signaling_socket(websocket: WebSocket, room_id: str):
    await websocket.accept()
    room_clients[room_id].add(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)
            for client in list(room_clients[room_id]):
                if client is not websocket:
                    await client.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        room_clients[room_id].discard(websocket)
        if not room_clients[room_id]:
            room_clients.pop(room_id, None)


# Serves the built frontend/ (Vite) chat console -- must be the LAST route
# registered in this file: Starlette matches routes in registration order,
# so every /api/*, /health, /legacy-demo, /pay, /therapist, /mood-tracking,
# /ws/* route above is matched first; this mount only catches whatever none
# of those already claimed (i.e. "/" and the built JS/CSS asset paths).
# html=True makes it serve frontend/dist/index.html for "/" (and for any
# other unmatched path, since this is a single-page app with no client-side
# routes of its own beyond "/").
#
# Requires `cd frontend && npm run build` to have been run at least once
# (writes frontend/dist/, gitignored like frontend/node_modules/) --
# missing entirely during local frontend development is fine, since `npm
# run dev`'s Vite dev server (port 5173) proxies /api and /health to this
# backend instead of needing this mount at all. This mount is what makes a
# single `uvicorn api:app` serve the whole app with no separate dev server.
_frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
