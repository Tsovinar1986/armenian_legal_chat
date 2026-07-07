"""SQLite-backed persistence for the FastAPI portal (main.py): users, bookings,
and password-reset OTPs. Replaces the previous in-memory lists, which lost all
data on every restart and stored passwords in plaintext.
"""
import hashlib
import os
import secrets
import sqlite3
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DB_PATH = os.path.join(_PROJECT_ROOT, "portal.db")

_PBKDF2_ITERATIONS = 390_000


def _connect() -> sqlite3.Connection:
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                identifier TEXT PRIMARY KEY,
                otp TEXT NOT NULL,
                user_email TEXT
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


def create_booking(title: str, client_name: str, lawyer_name: str, start_time: str, role: str) -> Dict:
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO bookings (title, client_name, lawyer_name, start_time, role) VALUES (?, ?, ?, ?, ?)",
            (title, client_name, lawyer_name, start_time, role),
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


def clear_all() -> None:
    """Wipe all rows from every table. Test-only helper — not used by the app itself."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM bookings")
        conn.execute("DELETE FROM password_resets")
        conn.commit()
    finally:
        conn.close()
