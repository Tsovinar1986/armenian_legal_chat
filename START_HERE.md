# Armenian Legal Portal Setup

## 1. Install Python dependencies

Run this from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the web portal

Start the FastAPI app:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- http://localhost:8000

## 3. What is included

The portal now provides:

- Individual and lawyer registration
- Sign in flow
- Dashboard view
- Calendar booking form
- WebRTC video call room with signaling
- Backend-ready sign-up / registration flow for future frontend integration

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
- Dashboard: http://localhost:8000/api/dashboard
- WebSocket signaling: ws://localhost:8000/ws/signaling/room-name

## 8. Notes

- The current video call uses local browser media and a simple signaling server.
- For production, you should add real authentication, persistent storage, and a TURN/STUN setup for reliable calls.
