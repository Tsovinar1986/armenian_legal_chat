import json
from collections import defaultdict
from typing import Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Armenian Legal Portal", version="1.0.0")

users_db: List[Dict[str, str]] = []
bookings_db: List[Dict[str, str]] = []
room_clients: Dict[str, Set[WebSocket]] = defaultdict(set)
password_reset_otps: Dict[str, Dict[str, str]] = {}


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
      </style>
    </head>
    <body>
      <div class="container">
        <div class="hero">
          <h1>Armenian Legal Portal</h1>
          <p>Register as an individual or lawyer, manage your dashboard, book consultations, and start online video calls.</p>
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
    existing = next(
        (
            item
            for item in users_db
            if (item.get("email") == request.email) or (request.phone_number and item.get("phone_number") == request.phone_number)
        ),
        None,
    )
    if existing:
        return {"success": False, "message": "User already exists"}

    user = {
        "name": request.name,
        "email": request.email,
        "phone_number": request.phone_number or "",
        "password": request.password,
        "role": request.role,
        "license_number": request.license_number or "",
    }
    users_db.append(user)
    return {"success": True, "message": "Registered successfully", "user": user}


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    user = next(
        (
            item
            for item in users_db
            if item["role"] == request.role
            and (item.get("email") == request.identifier or item.get("phone_number") == request.identifier)
            and item.get("password") == request.password
        ),
        None,
    )
    if not user:
        return {"success": False, "message": "Account not found or password is incorrect"}
    return {"success": True, "message": "Signed in successfully", "user": user}


@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    user = next(
        (
            item
            for item in users_db
            if item.get("email") == request.identifier or item.get("phone_number") == request.identifier
        ),
        None,
    )
    if not user:
        return {"success": False, "message": "Account not found"}

    otp = f"{len(user.get('email', '')) % 10}{len(user.get('phone_number', '')) % 10}{abs(hash(user.get('name', ''))) % 10000:04d}"
    password_reset_otps[request.identifier] = {"otp": otp, "user_email": user.get("email", "")}
    return {"success": True, "message": f"OTP sent via {request.channel}", "otp": otp}


@app.post("/api/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    reset_data = password_reset_otps.get(request.identifier)
    if not reset_data or reset_data["otp"] != request.otp:
        return {"success": False, "message": "Invalid OTP"}

    user = next(
        (
            item
            for item in users_db
            if item.get("email") == request.identifier or item.get("phone_number") == request.identifier
        ),
        None,
    )
    if not user:
        return {"success": False, "message": "Account not found"}

    user["password"] = request.new_password
    password_reset_otps.pop(request.identifier, None)
    return {"success": True, "message": "Password updated successfully"}


@app.get("/api/dashboard")
async def dashboard():
    roles = sorted({user["role"] for user in users_db})
    return {
        "users": len(users_db),
        "bookings": len(bookings_db),
        "roles": roles or ["individual", "lawyer"],
        "bookingsList": bookings_db[-5:][::-1],
    }


@app.get("/api/bookings")
async def get_bookings():
    return {"bookings": bookings_db}


@app.post("/api/bookings")
async def create_booking(request: BookingRequest):
    booking = {
        "title": request.title,
        "client_name": request.client_name,
        "lawyer_name": request.lawyer_name,
        "start_time": request.start_time,
        "role": request.role,
    }
    bookings_db.append(booking)
    return {"success": True, "message": "Appointment booked successfully", "booking": booking}


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
