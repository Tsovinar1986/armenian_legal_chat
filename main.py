import json
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.db import portal_store

app = FastAPI(title="Armenian Legal Portal", version="1.0.0")

portal_store.init_db()

room_clients: Dict[str, Set[WebSocket]] = defaultdict(set)
# Chat history is per-session and in-memory for v1 (resets on restart); it is fed back
# into LegalAgent.get_advice() as conversational context, unlike users/bookings which
# are now persisted in portal_store (SQLite) with hashed passwords.
chat_sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)

_legal_agent = None
_legal_agent_error: str | None = None


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


class ChatMessageRequest(BaseModel):
    message: str
    session_id: str | None = None


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
              </select>
              <button type="submit">Register</button>
            </form>
            <form id="loginForm" style="margin-top:12px;">
              <input id="loginEmail" type="email" placeholder="Email" required />
              <input id="loginPassword" type="password" placeholder="Password" required />
              <select id="loginRole">
                <option value="individual">Individual</option>
                <option value="lawyer">Lawyer</option>
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
              <input id="bookingLawyer" placeholder="Lawyer name" required />
              <input id="bookingTime" type="datetime-local" required />
              <select id="bookingRole">
                <option value="individual">Individual</option>
                <option value="lawyer">Lawyer</option>
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
            role: document.getElementById('bookingRole').value
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
        "roles": roles or ["individual", "lawyer"],
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
