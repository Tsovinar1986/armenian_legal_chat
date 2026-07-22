-- Auth/login schema for the Armenian Legal Portal (api.py).
-- SQLite DDL for the tables backing register / sign in / sign out / forgot
-- password / reset password (see /api/auth/* in api.py). This mirrors what
-- init_db() in src/db/portal_store.py creates at runtime -- that Python
-- function is still the source of truth (it also runs migrations for older
-- DBs), this file is a plain reference/handoff copy of the same DDL.

-- One row per registered account. A single email/phone can hold more than
-- one role (e.g. the same person as both "individual" and "lawyer"), since
-- role is not part of any uniqueness constraint here -- authenticate_user()
-- always looks up by (identifier, role) together.
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    email           TEXT,
    phone_number    TEXT,
    -- "<hex salt>$<hex pbkdf2-hmac-sha256 digest>", never plaintext -- see
    -- hash_password()/verify_password() in portal_store.py.
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,           -- "individual" | "lawyer" | "therapist"
    license_number  TEXT
);

-- One-time password reset codes, keyed by the identifier (email or phone)
-- the user entered on the "Forgot password?" form. INSERT OR REPLACE means
-- requesting a new OTP invalidates any earlier unused one for that identifier.
CREATE TABLE IF NOT EXISTS password_resets (
    identifier  TEXT PRIMARY KEY,
    otp         TEXT NOT NULL,
    user_email  TEXT
);

-- Login sessions issued on register/login. Only the SHA-256 hash of the
-- bearer token is stored (the raw token is returned to the client once and
-- never persisted), so a DB leak alone can't be replayed as a session --
-- same principle as password_hash above. Expired rows are lazily deleted by
-- get_session() the next time that token_hash is looked up.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users (id),
    role        TEXT NOT NULL,
    created_at  TEXT NOT NULL,               -- ISO8601 UTC
    expires_at  TEXT NOT NULL                -- ISO8601 UTC; 30 days from creation by default
);
