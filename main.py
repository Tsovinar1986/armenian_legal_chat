import json
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set

import stripe
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.db import portal_store
from src.services.crisis_detection import CRISIS_RESPONSE_HY, detect_crisis_signal

app = FastAPI(title="Armenian Legal Portal", version="1.0.0")

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

room_clients: Dict[str, Set[WebSocket]] = defaultdict(set)
# Chat history is per-session and in-memory for v1 (resets on restart); it is fed back
# into LegalAgent.get_advice() as conversational context, unlike users/bookings which
# are now persisted in portal_store (SQLite) with hashed passwords.
chat_sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)

_legal_agent = None
_legal_agent_error: str | None = None
_mental_health_qa_classifier = None

# Therapist chat is a separate conversation domain from the legal chat above —
# supportive Q&A retrieval over student_mh_counseling_100k_with_label_column.csv,
# not legal advice. Kept as its own session store rather than sharing chat_sessions.
therapist_chat_sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)


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
        from langchain_chroma import Chroma
        import chromadb
        from src.core.state import SystemState
        from src.services.classifier import LegalCaseClassifier
        from src.agents.legal_agent import LegalAgent
        from src.db.repository import CompanyLegalRepo

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        client = chromadb.PersistentClient(path="./chroma_legal_data")
        vector_db = Chroma(collection_name="company_legal_cases", embedding_function=embeddings, client=client)
        classifier_service = LegalCaseClassifier(data_folder="src/data")
        _legal_agent = LegalAgent(CompanyLegalRepo(vector_db), SystemState(), classifier=classifier_service)
        return _legal_agent
    except Exception as exc:
        _legal_agent_error = str(exc)
        raise


@app.on_event("startup")
async def warm_up_legal_agent():
    try:
        await run_in_threadpool(get_legal_agent)
        print("✅ Legal AI chat backend ready")
    except Exception as exc:
        print(f"⚠️ Legal AI chat backend failed to initialize: {exc}")
        print("   The portal will still run, but /api/chat will return an error until this is fixed.")


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
    start_time: str
    role: str
    provider_type: str = "lawyer"  # "lawyer" or "therapist" — who client_name is booking with


class ChatMessageRequest(BaseModel):
    message: str
    session_id: str | None = None


class PaymentIntentRequest(BaseModel):
    consultation_type: str  # "lawyer" or "therapist"
    customer_name: str
    amount_cents: int
    currency: str = "usd"


@app.get("/", response_class=HTMLResponse)
async def index():
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
          <div class="card">
            <h3>Sign up / Sign in</h3>
            <p class="small">This backend supports email or phone sign-in, OTP password reset, and lawyer license registration for future web and mobile apps.</p>
            <form id="registerForm">
              <input id="name" placeholder="Full name" required />
              <input id="email" type="email" placeholder="Email" required />
              <input id="password" type="password" placeholder="Password" required />
              <select id="role">
                <option value="individual">Individual</option>
                <option value="lawyer">Lawyer</option>
                <option value="therapist">Therapist</option>
              </select>
              <button type="submit">Register</button>
            </form>
            <form id="loginForm" style="margin-top:12px;">
              <input id="loginEmail" type="email" placeholder="Email" required />
              <input id="loginPassword" type="password" placeholder="Password" required />
              <select id="loginRole">
                <option value="individual">Individual</option>
                <option value="lawyer">Lawyer</option>
                <option value="therapist">Therapist</option>
              </select>
              <button class="secondary" type="submit">Sign in</button>
            </form>
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
            <form id="bookingForm">
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
            <div class="video-grid">
              <video id="localVideo" autoplay muted playsinline></video>
              <video id="remoteVideo" autoplay playsinline></video>
            </div>
            <div id="callStatus" class="small">Ready to connect.</div>
          </div>
        </div>
      </div>

      <script>
        const authMessage = document.getElementById('authMessage');
        const bookingMessage = document.getElementById('bookingMessage');
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

        document.getElementById('registerForm').addEventListener('submit', async (event) => {
          event.preventDefault();
          const role = document.getElementById('role').value;
          const payload = {
            name: document.getElementById('name').value,
            email: document.getElementById('email').value,
            phone_number: '',
            password: document.getElementById('password').value,
            role,
            license_number: role === 'lawyer' ? '' : null
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
            role: document.getElementById('loginRole').value
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
            provider_type: document.getElementById('bookingProviderType').value
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

        function createPeer() {
          if (peerConnection) return peerConnection;
          peerConnection = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
          peerConnection.ontrack = (event) => {
            remoteVideo.srcObject = event.streams[0];
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


@app.post("/api/auth/register")
async def register(request: RegisterRequest):
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
    return {"success": True, "message": "Registered successfully", "user": user}


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    user = portal_store.authenticate_user(request.identifier, request.password, request.role)
    if not user:
        return {"success": False, "message": "Account not found or password is incorrect"}
    return {"success": True, "message": "Signed in successfully", "user": user}


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
    )
    return {"success": True, "message": "Appointment booked successfully", "booking": booking}


@app.post("/api/chat")
async def chat(request: ChatMessageRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_message = request.message.strip()
    if not user_message:
        return {"success": False, "message": "Please type a message.", "session_id": session_id}

    now = datetime.utcnow().isoformat()
    history = list(chat_sessions[session_id])  # turns before this message, for conversational context
    chat_sessions[session_id].append({"role": "user", "text": user_message, "at": now})

    try:
        agent = get_legal_agent()
        response_text = await run_in_threadpool(agent.get_advice, user_message, history)
    except Exception as exc:
        response_text = (
            "⚠️ Legal AI backend is unavailable right now "
            f"(is Ollama running? error: {exc})"
        )

    chat_sessions[session_id].append({"role": "bot", "text": response_text, "at": datetime.utcnow().isoformat()})
    return {"success": True, "session_id": session_id, "response": response_text}


@app.get("/api/chat/{session_id}")
async def get_chat_history(session_id: str):
    return {"session_id": session_id, "messages": chat_sessions.get(session_id, [])}


@app.post("/api/therapist-chat")
async def therapist_chat(request: ChatMessageRequest):
    """Supportive-conversation chat backed by MentalHealthQAClassifier, separate
    from the legal /api/chat. Always screens for crisis signals first (keyword,
    then zero-shot if the legal agent's classifier happens to be available) and
    returns CRISIS_RESPONSE_HY before ever touching the Q&A retrieval — the
    dataset itself contains "suicidal thoughts"-labeled rows, so a retrieved
    answer must never be allowed to substitute for the real crisis response."""
    session_id = request.session_id or str(uuid.uuid4())
    user_message = request.message.strip()
    if not user_message:
        return {"success": False, "message": "Please type a message.", "session_id": session_id}

    now = datetime.utcnow().isoformat()
    therapist_chat_sessions[session_id].append({"role": "user", "text": user_message, "at": now})

    response_text = None
    if detect_crisis_signal(user_message):
        response_text = CRISIS_RESPONSE_HY
    else:
        try:
            agent = get_legal_agent()
            if agent.classifier:
                risk = await run_in_threadpool(agent.classifier.classify_mental_health_risk, user_message)
                if risk and risk["is_risk"]:
                    response_text = CRISIS_RESPONSE_HY
        except Exception:
            pass  # Legal agent/Ollama being unavailable shouldn't block therapist chat.

    if response_text is None:
        qa_classifier = get_mental_health_qa_classifier()
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

    therapist_chat_sessions[session_id].append({"role": "bot", "text": response_text, "at": datetime.utcnow().isoformat()})
    return {"success": True, "session_id": session_id, "response": response_text}


@app.get("/api/therapist-chat/{session_id}")
async def get_therapist_chat_history(session_id: str):
    return {"session_id": session_id, "messages": therapist_chat_sessions.get(session_id, [])}


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
