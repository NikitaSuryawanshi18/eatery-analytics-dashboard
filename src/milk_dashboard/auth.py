"""Authentication and Square connection persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Any


SESSION_TTL_DAYS = 30
STATE_TTL_MINUTES = 10
PBKDF2_ITERATIONS = 600_000


@dataclass
class UserRecord:
    id: int
    email: str
    first_name: str
    last_name: str
    created_at_utc: str
    last_login_at_utc: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> datetime:
    out = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if out.tzinfo is None:
        return out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _require_fernet():
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Missing dependency 'cryptography'. Install project dependencies before using auth features."
        ) from exc
    key = os.getenv("TOKEN_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("TOKEN_ENCRYPTION_KEY is required for auth/token storage.")
    try:
        return Fernet(key.encode("utf-8"))
    except Exception as exc:  # pragma: no cover - invalid env
        raise RuntimeError("TOKEN_ENCRYPTION_KEY must be a valid Fernet key.") from exc


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return "pbkdf2_sha256${iterations}${salt}${digest}".format(
        iterations=PBKDF2_ITERATIONS,
        salt=base64.b64encode(salt).decode("ascii"),
        digest=base64.b64encode(derived).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algo, iterations, salt_b64, digest_b64 = password_hash.split("$", 3)
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations),
        )
    except Exception:
        return False
    return hmac.compare_digest(expected, derived)


def _hash_session_token(token: str) -> str:
    secret = os.getenv("APP_AUTH_SECRET", "").strip()
    if not secret:
        raise RuntimeError("APP_AUTH_SECRET is required for session management.")
    digest = hmac.new(secret.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest


class AuthStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    first_name TEXT NOT NULL DEFAULT '',
                    last_name TEXT NOT NULL DEFAULT '',
                    password_hash TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    last_login_at_utc TEXT
                );

                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    revoked_at_utc TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS oauth_states (
                    state TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    used_at_utc TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS square_connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE,
                    merchant_id TEXT NOT NULL UNIQUE,
                    access_token_encrypted TEXT NOT NULL,
                    refresh_token_encrypted TEXT NOT NULL,
                    token_type TEXT,
                    scope TEXT,
                    expires_at_utc TEXT,
                    square_env TEXT NOT NULL,
                    connected_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    revoked_at_utc TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "first_name" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN first_name TEXT NOT NULL DEFAULT ''")
            if "last_name" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN last_name TEXT NOT NULL DEFAULT ''")
            conn.commit()

    def create_user(self, *, email: str, password: str, first_name: str = "", last_name: str = "") -> UserRecord:
        normalized = email.strip().lower()
        clean_first_name = first_name.strip()
        clean_last_name = last_name.strip()
        if not normalized or "@" not in normalized:
            raise ValueError("A valid email is required.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        created = utc_now_iso()
        password_hash = _hash_password(password)
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO users (email, first_name, last_name, password_hash, created_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (normalized, clean_first_name, clean_last_name, password_hash, created),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("An account with this email already exists.") from exc
            conn.commit()
            return UserRecord(
                id=int(cur.lastrowid),
                email=normalized,
                first_name=clean_first_name,
                last_name=clean_last_name,
                created_at_utc=created,
                last_login_at_utc=None,
            )

    def authenticate_user(self, *, email: str, password: str) -> UserRecord | None:
        normalized = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, email, first_name, last_name, password_hash, created_at_utc, last_login_at_utc
                FROM users
                WHERE email = ?
                """,
                (normalized,),
            ).fetchone()
            if row is None or not verify_password(password, str(row["password_hash"])):
                return None
            now = utc_now_iso()
            conn.execute("UPDATE users SET last_login_at_utc = ? WHERE id = ?", (now, int(row["id"])))
            conn.commit()
            return UserRecord(
                id=int(row["id"]),
                email=str(row["email"]),
                first_name=str(row["first_name"] or ""),
                last_name=str(row["last_name"] or ""),
                created_at_utc=str(row["created_at_utc"]),
                last_login_at_utc=now,
            )

    def create_session(self, *, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = _hash_session_token(token)
        created = datetime.now(timezone.utc)
        expires = created + timedelta(days=SESSION_TTL_DAYS)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_sessions (user_id, token_hash, created_at_utc, expires_at_utc)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, token_hash, created.isoformat(), expires.isoformat()),
            )
            conn.commit()
        return token

    def revoke_session(self, token: str) -> None:
        token_hash = _hash_session_token(token)
        with self._connect() as conn:
            conn.execute(
                "UPDATE user_sessions SET revoked_at_utc = ? WHERE token_hash = ? AND revoked_at_utc IS NULL",
                (utc_now_iso(), token_hash),
            )
            conn.commit()

    def get_user_by_session(self, token: str | None) -> UserRecord | None:
        if not token:
            return None
        token_hash = _hash_session_token(token)
        now = utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.email, u.first_name, u.last_name, u.created_at_utc, u.last_login_at_utc
                FROM user_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                  AND s.revoked_at_utc IS NULL
                  AND s.expires_at_utc > ?
                """,
                (token_hash, now),
            ).fetchone()
            if row is None:
                return None
            return UserRecord(
                id=int(row["id"]),
                email=str(row["email"]),
                first_name=str(row["first_name"] or ""),
                last_name=str(row["last_name"] or ""),
                created_at_utc=str(row["created_at_utc"]),
                last_login_at_utc=str(row["last_login_at_utc"]) if row["last_login_at_utc"] else None,
            )

    def create_oauth_state(self, *, user_id: int) -> str:
        state = secrets.token_urlsafe(24)
        created = datetime.now(timezone.utc)
        expires = created + timedelta(minutes=STATE_TTL_MINUTES)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO oauth_states (state, user_id, created_at_utc, expires_at_utc)
                VALUES (?, ?, ?, ?)
                """,
                (state, user_id, created.isoformat(), expires.isoformat()),
            )
            conn.commit()
        return state

    def consume_oauth_state(self, state: str) -> int:
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, expires_at_utc, used_at_utc FROM oauth_states WHERE state = ?",
                (state,),
            ).fetchone()
            if row is None:
                raise ValueError("Invalid OAuth state.")
            if row["used_at_utc"]:
                raise ValueError("OAuth state already used.")
            if _parse_iso(str(row["expires_at_utc"])) <= now:
                raise ValueError("OAuth state expired.")
            conn.execute(
                "UPDATE oauth_states SET used_at_utc = ? WHERE state = ?",
                (now.isoformat(), state),
            )
            conn.commit()
            return int(row["user_id"])

    def upsert_square_connection(self, *, user_id: int, token_payload: dict[str, Any], square_env: str) -> dict[str, Any]:
        merchant_id = str(token_payload.get("merchant_id") or "").strip()
        access_token = str(token_payload.get("access_token") or "").strip()
        refresh_token = str(token_payload.get("refresh_token") or "").strip()
        if not merchant_id or not access_token or not refresh_token:
            raise ValueError("Square token payload is missing merchant_id/access_token/refresh_token.")
        fernet = _require_fernet()
        now = utc_now_iso()
        encrypted_access = fernet.encrypt(access_token.encode("utf-8")).decode("utf-8")
        encrypted_refresh = fernet.encrypt(refresh_token.encode("utf-8")).decode("utf-8")
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM square_connections WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO square_connections (
                        user_id, merchant_id, access_token_encrypted, refresh_token_encrypted,
                        token_type, scope, expires_at_utc, square_env, connected_at_utc, updated_at_utc, revoked_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        user_id,
                        merchant_id,
                        encrypted_access,
                        encrypted_refresh,
                        str(token_payload.get("token_type") or ""),
                        str(token_payload.get("scope") or ""),
                        str(token_payload.get("expires_at") or ""),
                        square_env,
                        now,
                        now,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE square_connections
                    SET merchant_id = ?, access_token_encrypted = ?, refresh_token_encrypted = ?,
                        token_type = ?, scope = ?, expires_at_utc = ?, square_env = ?,
                        updated_at_utc = ?, revoked_at_utc = NULL
                    WHERE user_id = ?
                    """,
                    (
                        merchant_id,
                        encrypted_access,
                        encrypted_refresh,
                        str(token_payload.get("token_type") or ""),
                        str(token_payload.get("scope") or ""),
                        str(token_payload.get("expires_at") or ""),
                        square_env,
                        now,
                        user_id,
                    ),
                )
            conn.commit()
        return self.get_square_connection_by_user(user_id, include_secrets=False) or {}

    def get_square_connection_by_user(self, user_id: int, *, include_secrets: bool = True) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM square_connections WHERE user_id = ? AND revoked_at_utc IS NULL",
                (user_id,),
            ).fetchone()
        return self._connection_row_to_dict(row, include_secrets=include_secrets)

    def get_square_connection_by_merchant(self, merchant_id: str, *, include_secrets: bool = True) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM square_connections WHERE merchant_id = ? AND revoked_at_utc IS NULL",
                (merchant_id.strip(),),
            ).fetchone()
        return self._connection_row_to_dict(row, include_secrets=include_secrets)

    def revoke_square_connection(self, *, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE square_connections SET revoked_at_utc = ?, updated_at_utc = ? WHERE user_id = ?",
                (utc_now_iso(), utc_now_iso(), user_id),
            )
            conn.commit()

    def _connection_row_to_dict(self, row: sqlite3.Row | None, *, include_secrets: bool) -> dict[str, Any] | None:
        if row is None:
            return None
        out = {
            "id": int(row["id"]),
            "user_id": int(row["user_id"]),
            "merchant_id": str(row["merchant_id"]),
            "token_type": str(row["token_type"] or ""),
            "scope": str(row["scope"] or ""),
            "expires_at": str(row["expires_at_utc"] or ""),
            "square_env": str(row["square_env"] or ""),
            "connected_at_utc": str(row["connected_at_utc"] or ""),
            "updated_at_utc": str(row["updated_at_utc"] or ""),
            "revoked_at_utc": str(row["revoked_at_utc"] or "") if row["revoked_at_utc"] else None,
        }
        if include_secrets:
            fernet = _require_fernet()
            out["access_token"] = fernet.decrypt(str(row["access_token_encrypted"]).encode("utf-8")).decode("utf-8")
            out["refresh_token"] = fernet.decrypt(str(row["refresh_token_encrypted"]).encode("utf-8")).decode("utf-8")
        return out
