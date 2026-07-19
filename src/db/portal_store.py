"""SQLite-backed persistence for the FastAPI portal (api.py): users, bookings,
and password-reset OTPs. Replaces the previous in-memory lists, which lost all
data on every restart and stored passwords in plaintext.
"""
import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
# Overridable via PORTAL_DB_PATH (e.g. to point at a mounted Docker volume
# directory rather than the project root) — defaults to the same path used
# throughout local development.
DB_PATH = os.environ.get("PORTAL_DB_PATH") or os.path.join(_PROJECT_ROOT, "portal.db")

_PBKDF2_ITERATIONS = 390_000


def _connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                phone_number TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                license_number TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                client_name TEXT NOT NULL,
                lawyer_name TEXT NOT NULL,
                start_time TEXT NOT NULL,
                role TEXT NOT NULL
            )
        """)
        # Migration: older DBs created before provider_type existed. lawyer_name is
        # kept as-is (documented in the mobile API contract in START_HERE.md) and
        # now holds either a lawyer's or a therapist's name; provider_type says which.
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(bookings)").fetchall()}
        if "provider_type" not in existing_columns:
            conn.execute("ALTER TABLE bookings ADD COLUMN provider_type TEXT NOT NULL DEFAULT 'lawyer'")
        # Migration: timezone the client booked in, plus start_time_utc — a
        # normalized UTC instant computed from start_time + timezone, used to
        # reliably compare bookings across timezones for availability lookups.
        # start_time itself is left as whatever the client sent (unchanged, for
        # API-contract backward compatibility); start_time_utc is best-effort and
        # may be NULL for old rows that predate this migration.
        if "timezone" not in existing_columns:
            conn.execute("ALTER TABLE bookings ADD COLUMN timezone TEXT NOT NULL DEFAULT 'UTC'")
        if "start_time_utc" not in existing_columns:
            conn.execute("ALTER TABLE bookings ADD COLUMN start_time_utc TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                identifier TEXT PRIMARY KEY,
                otp TEXT NOT NULL,
                user_email TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_type TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                currency TEXT NOT NULL,
                stripe_payment_intent_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # Migration: optional scheduling fields so a successful payment can
        # auto-create a booking (see api.py's webhook handler and
        # create_booking_from_payment_if_scheduled below).
        payment_columns = {row["name"] for row in conn.execute("PRAGMA table_info(payments)").fetchall()}
        if "provider_name" not in payment_columns:
            conn.execute("ALTER TABLE payments ADD COLUMN provider_name TEXT")
        if "start_time" not in payment_columns:
            conn.execute("ALTER TABLE payments ADD COLUMN start_time TEXT")
        if "timezone" not in payment_columns:
            conn.execute("ALTER TABLE payments ADD COLUMN timezone TEXT NOT NULL DEFAULT 'UTC'")
        if "booking_id" not in payment_columns:
            conn.execute("ALTER TABLE payments ADD COLUMN booking_id INTEGER")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                session_type TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages (session_type, session_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS provider_schedules (
                provider_name TEXT NOT NULL,
                weekday INTEGER NOT NULL,
                start_hour INTEGER NOT NULL,
                end_hour INTEGER NOT NULL,
                timezone TEXT NOT NULL DEFAULT 'UTC',
                PRIMARY KEY (provider_name, weekday)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS).hex()
    return secrets.compare_digest(check, digest)


def _user_row_to_dict(row: sqlite3.Row) -> Dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"] or "",
        "phone_number": row["phone_number"] or "",
        "role": row["role"],
        "license_number": row["license_number"] or "",
    }


def find_user(email: Optional[str] = None, phone_number: Optional[str] = None, role: Optional[str] = None) -> Optional[sqlite3.Row]:
    conn = _connect()
    try:
        conditions = []
        params: List = []
        if role:
            conditions.append("role = ?")
            params.append(role)

        or_parts = []
        if email:
            or_parts.append("email = ?")
            params.append(email)
        if phone_number:
            or_parts.append("phone_number = ?")
            params.append(phone_number)
        if or_parts:
            conditions.append("(" + " OR ".join(or_parts) + ")")

        query = "SELECT * FROM users"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        return conn.execute(query, params).fetchone()
    finally:
        conn.close()


def create_user(name: str, email: str, phone_number: str, password: str, role: str, license_number: str) -> Dict:
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO users (name, email, phone_number, password_hash, role, license_number) VALUES (?, ?, ?, ?, ?, ?)",
            (name, email, phone_number, hash_password(password), role, license_number),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _user_row_to_dict(row)
    finally:
        conn.close()


def authenticate_user(identifier: str, password: str, role: str) -> Optional[Dict]:
    row = find_user(email=identifier, phone_number=identifier, role=role)
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return _user_row_to_dict(row)


def set_password_reset_otp(identifier: str, otp: str, user_email: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO password_resets (identifier, otp, user_email) VALUES (?, ?, ?)",
            (identifier, otp, user_email),
        )
        conn.commit()
    finally:
        conn.close()


def get_password_reset(identifier: str) -> Optional[sqlite3.Row]:
    conn = _connect()
    try:
        return conn.execute("SELECT * FROM password_resets WHERE identifier = ?", (identifier,)).fetchone()
    finally:
        conn.close()


def clear_password_reset(identifier: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM password_resets WHERE identifier = ?", (identifier,))
        conn.commit()
    finally:
        conn.close()


def update_password(identifier: str, new_password: str) -> bool:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ? OR phone_number = ?", (identifier, identifier)
        ).fetchone()
        if not row:
            return False
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new_password), row["id"]))
        conn.commit()
        return True
    finally:
        conn.close()


_SESSION_TTL_HOURS = 24 * 30  # 30 days


def create_session(user_id: int, role: str, ttl_hours: int = _SESSION_TTL_HOURS) -> str:
    """Issue a new session token for a user and store only its hash (the raw
    token is returned once, here, and never persisted — same principle as a
    password, so a DB leak alone can't be used to impersonate a session)."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(dt_timezone.utc)
    expires_at = now.replace(microsecond=0) + timedelta(hours=ttl_hours)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, role, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (token_hash, user_id, role, now.isoformat(), expires_at.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def get_session(token: str) -> Optional[Dict]:
    """Return {"user_id", "role"} for a valid, unexpired token, or None."""
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)).fetchone()
        if not row:
            return None
        if row["expires_at"] < datetime.now(dt_timezone.utc).isoformat():
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()
            return None
        return {"user_id": row["user_id"], "role": row["role"]}
    finally:
        conn.close()


def delete_session(token: str) -> None:
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    conn = _connect()
    try:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        conn.commit()
    finally:
        conn.close()


def append_chat_message(session_id: str, session_type: str, role: str, text: str) -> None:
    """session_type is "legal" or "therapist" — the two chat domains keep
    separate histories even if a session_id were ever reused across them."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO chat_messages (session_id, session_type, role, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, session_type, role, text, datetime.now(dt_timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_chat_messages(session_id: str, session_type: str) -> List[Dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT role, text, created_at AS at FROM chat_messages "
            "WHERE session_id = ? AND session_type = ? ORDER BY id ASC",
            (session_id, session_type),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def set_provider_schedule(provider_name: str, weekday: int, start_hour: int, end_hour: int, timezone: str = "UTC") -> Dict:
    """weekday: 0=Monday .. 6=Sunday (Python date.weekday() convention)."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO provider_schedules (provider_name, weekday, start_hour, end_hour, timezone) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(provider_name, weekday) DO UPDATE SET start_hour=excluded.start_hour, "
            "end_hour=excluded.end_hour, timezone=excluded.timezone",
            (provider_name, weekday, start_hour, end_hour, timezone),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM provider_schedules WHERE provider_name = ? AND weekday = ?",
            (provider_name, weekday),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def get_provider_schedule(provider_name: str, weekday: int) -> Optional[Dict]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM provider_schedules WHERE provider_name = ? AND weekday = ?",
            (provider_name, weekday),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_provider_schedule(provider_name: str) -> List[Dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM provider_schedules WHERE provider_name = ? ORDER BY weekday ASC",
            (provider_name,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def start_time_to_utc_iso(start_time: str, tz_name: str) -> Optional[str]:
    """Convert a booking's start_time to a normalized UTC ISO8601 string.

    If start_time already carries an offset/Z, that offset is trusted as-is and
    just converted to UTC. If it's naive (e.g. from an HTML datetime-local
    input, which has no timezone), it's interpreted as local time in tz_name
    before converting. Returns None if start_time or tz_name can't be parsed —
    callers should treat that booking as unknown/unschedulable for availability
    purposes rather than failing the whole request.
    """
    try:
        dt = datetime.fromisoformat(start_time)
    except ValueError:
        return None
    if dt.tzinfo is None:
        try:
            dt = dt.replace(tzinfo=ZoneInfo(tz_name))
        except Exception:
            dt = dt.replace(tzinfo=dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc).isoformat()


def create_booking(
    title: str,
    client_name: str,
    lawyer_name: str,
    start_time: str,
    role: str,
    provider_type: str = "lawyer",
    timezone: str = "UTC",
) -> Dict:
    start_time_utc = start_time_to_utc_iso(start_time, timezone)
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO bookings (title, client_name, lawyer_name, start_time, role, provider_type, "
            "timezone, start_time_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, client_name, lawyer_name, start_time, role, provider_type, timezone, start_time_utc),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM bookings WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_bookings() -> List[Dict]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM bookings ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_provider_busy_ranges(provider_name: str, start_utc_iso: str, end_utc_iso: str) -> List[str]:
    """Return start_time_utc values for a provider's bookings whose UTC instant
    falls within [start_utc_iso, end_utc_iso). Used to compute free/busy slots
    for GET /api/bookings/availability. Bookings with no start_time_utc (couldn't
    be parsed, or predate the timezone migration) are excluded — best-effort."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT start_time_utc FROM bookings WHERE lawyer_name = ? AND start_time_utc IS NOT NULL "
            "AND start_time_utc >= ? AND start_time_utc < ?",
            (provider_name, start_utc_iso, end_utc_iso),
        ).fetchall()
        return [r["start_time_utc"] for r in rows]
    finally:
        conn.close()


def recent_bookings(limit: int = 5) -> List[Dict]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM bookings ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_users() -> int:
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    finally:
        conn.close()


def count_bookings() -> int:
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) AS c FROM bookings").fetchone()["c"]
    finally:
        conn.close()


def distinct_roles() -> List[str]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT DISTINCT role FROM users").fetchall()
        return sorted(r["role"] for r in rows)
    finally:
        conn.close()


def create_payment(
    consultation_type: str,
    customer_name: str,
    amount_cents: int,
    currency: str,
    stripe_payment_intent_id: str,
    status: str,
    created_at: str,
    provider_name: str = None,
    start_time: str = None,
    timezone: str = "UTC",
) -> Dict:
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO payments (consultation_type, customer_name, amount_cents, currency, "
            "stripe_payment_intent_id, status, created_at, provider_name, start_time, timezone) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (consultation_type, customer_name, amount_cents, currency, stripe_payment_intent_id,
             status, created_at, provider_name, start_time, timezone),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM payments WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def update_payment_status(stripe_payment_intent_id: str, status: str) -> bool:
    conn = _connect()
    try:
        cursor = conn.execute(
            "UPDATE payments SET status = ? WHERE stripe_payment_intent_id = ?",
            (status, stripe_payment_intent_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def set_payment_booking_id(stripe_payment_intent_id: str, booking_id: int) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE payments SET booking_id = ? WHERE stripe_payment_intent_id = ?",
            (booking_id, stripe_payment_intent_id),
        )
        conn.commit()
    finally:
        conn.close()


def create_booking_from_payment_if_scheduled(payment: Dict) -> Optional[Dict]:
    """If a paid-for consultation included a start_time, auto-create the
    calendar booking now that payment succeeded (called from the Stripe
    webhook on payment_intent.succeeded). Returns the created booking, or
    None if this payment had no start_time (nothing to schedule) or already
    has a linked booking_id (avoid double-booking on a duplicate webhook)."""
    if not payment.get("start_time") or payment.get("booking_id"):
        return None
    booking = create_booking(
        title=f"{payment['consultation_type'].capitalize()} consultation",
        client_name=payment["customer_name"],
        lawyer_name=payment.get("provider_name") or "Unassigned",
        start_time=payment["start_time"],
        role="individual",
        provider_type=payment["consultation_type"],
        timezone=payment.get("timezone") or "UTC",
    )
    set_payment_booking_id(payment["stripe_payment_intent_id"], booking["id"])
    return booking


def get_payment(payment_id: int) -> Optional[Dict]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM payments WHERE id = ?", (payment_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_payment_by_intent(stripe_payment_intent_id: str) -> Optional[Dict]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM payments WHERE stripe_payment_intent_id = ?", (stripe_payment_intent_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def clear_all() -> None:
    """Wipe all rows from every table. Test-only helper — not used by the app itself."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM bookings")
        conn.execute("DELETE FROM password_resets")
        conn.execute("DELETE FROM payments")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM chat_messages")
        conn.execute("DELETE FROM provider_schedules")
        conn.commit()
    finally:
        conn.close()
