"""Local user accounts with salted PBKDF2 password verification."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.ledger import LEDGER_PATH, _connection, initialize_ledger

_HASH_NAME = "sha256"
_HASH_ITERATIONS = 600_000
_SALT_BYTES = 16
_KEY_BYTES = 32
_USERNAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,31}\Z")
_MAX_PASSWORD_LENGTH = 1024
_LOGIN_WINDOW_SECONDS = 300
_LOGIN_MAX_FAILURES = 5
_LOGIN_LOCK_SECONDS = 300


from src.oidc import EXTERNAL_PASSWORD_MARKER, USER_ID_PREFIX  # noqa: E402


class AccountError(ValueError):
    """A user-correctable account creation or sign-in error."""


@dataclass(frozen=True)
class Account:
    user_id: str
    username: str


def initialize_accounts(db_path: Path = LEDGER_PATH) -> None:
    """Create the account table and migrate the ledger without assigning old rows."""
    with _connection(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                username TEXT PRIMARY KEY,
                failure_count INTEGER NOT NULL,
                window_started_at REAL NOT NULL,
                blocked_until REAL NOT NULL DEFAULT 0
            )
            """)
    initialize_ledger(db_path)


def _password_hash(password: str, salt: bytes, iterations: int = _HASH_ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac(
        _HASH_NAME, password.encode("utf-8"), salt, iterations, dklen=_KEY_BYTES
    )


def _encode_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _password_hash(password, salt)
    return f"pbkdf2_{_HASH_NAME}${_HASH_ITERATIONS}${salt.hex()}${digest.hex()}"


def _valid_password(password: str) -> bool:
    return 12 <= len(password) <= _MAX_PASSWORD_LENGTH


def create_account(username: str, password: str, db_path: Path = LEDGER_PATH) -> Account:
    """Create an account with a normalized username and a salted password hash."""
    initialize_accounts(db_path)
    normalized = username.strip().casefold()
    if not _USERNAME_PATTERN.fullmatch(normalized):
        raise AccountError("Username must be 3–32 characters: letters, numbers, _ or -.")
    if normalized.startswith(USER_ID_PREFIX):
        raise AccountError("That username is reserved. Choose another.")
    if not _valid_password(password):
        raise AccountError("Password must be at least 12 and no more than 1024 characters.")

    account = Account(user_id=str(uuid.uuid4()), username=normalized)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _connection(db_path) as connection:
            connection.execute(
                "INSERT INTO accounts (user_id, username, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (account.user_id, account.username, _encode_password(password), created_at),
            )
    except sqlite3.IntegrityError as exc:
        if "accounts.username" in str(exc):
            raise AccountError(
                "That username is already registered. Sign in or choose another."
            ) from exc
        raise
    return account


def ensure_external_account(user_id: str, username: str, db_path: Path = LEDGER_PATH) -> Account:
    """Create (once) the local row for a single-sign-on user; it can never be used for a password login."""
    initialize_accounts(db_path)
    with _connection(db_path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO accounts (user_id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (
                user_id,
                username,
                EXTERNAL_PASSWORD_MARKER,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return Account(user_id=user_id, username=username)


def authenticate(username: str, password: str, db_path: Path = LEDGER_PATH) -> Optional[Account]:
    """Authenticate credentials without distinguishing unknown usernames from bad passwords."""
    initialize_accounts(db_path)
    if len(password) > _MAX_PASSWORD_LENGTH:
        return None
    normalized = username.strip().casefold()
    now = datetime.now(timezone.utc).timestamp()
    if not _USERNAME_PATTERN.fullmatch(normalized):
        candidate = _password_hash(password, bytes(_SALT_BYTES))
        hmac.compare_digest(candidate, bytes(_KEY_BYTES))
        return None
    with _connection(db_path) as connection:
        connection.execute(
            "DELETE FROM login_attempts WHERE blocked_until < ? AND window_started_at < ?",
            (now, now - 2 * _LOGIN_WINDOW_SECONDS),
        )
        attempts = connection.execute(
            "SELECT failure_count, window_started_at, blocked_until "
            "FROM login_attempts WHERE username = ?",
            (normalized,),
        ).fetchone()
        if attempts is not None and attempts["blocked_until"] > now:
            return None
        row = connection.execute(
            "SELECT user_id, username, password_hash FROM accounts WHERE username = ?",
            (normalized,),
        ).fetchone()

    if row is None:
        candidate = _password_hash(password, bytes(_SALT_BYTES))
        hmac.compare_digest(candidate, bytes(_KEY_BYTES))
        _record_failed_login(normalized, now, db_path)
        return None

    if row["password_hash"] == EXTERNAL_PASSWORD_MARKER:
        # Single-sign-on account: it has no password, so a password login must never succeed.
        candidate = _password_hash(password, bytes(_SALT_BYTES))
        hmac.compare_digest(candidate, bytes(_KEY_BYTES))
        _record_failed_login(normalized, now, db_path)
        return None

    scheme, iterations_text, salt_hex, expected_hex = row["password_hash"].split("$")
    iterations = int(iterations_text)
    if scheme != f"pbkdf2_{_HASH_NAME}" or not _HASH_ITERATIONS <= iterations <= 10_000_000:
        raise RuntimeError("Stored password hash has an unsupported format or work factor.")
    candidate = _password_hash(password, bytes.fromhex(salt_hex), iterations)
    if not hmac.compare_digest(candidate.hex(), expected_hex):
        _record_failed_login(normalized, now, db_path)
        return None
    with _connection(db_path) as connection:
        connection.execute("DELETE FROM login_attempts WHERE username = ?", (normalized,))
    return Account(user_id=row["user_id"], username=row["username"])


def _record_failed_login(
    username: str,
    now: float,
    db_path: Path,
) -> None:
    with _connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous_attempts = connection.execute(
            "SELECT failure_count, window_started_at, blocked_until "
            "FROM login_attempts WHERE username = ?",
            (username,),
        ).fetchone()
        if previous_attempts is not None and previous_attempts["blocked_until"] > now:
            return
        if (
            previous_attempts is None
            or now - previous_attempts["window_started_at"] > _LOGIN_WINDOW_SECONDS
        ):
            count, window_start = 1, now
        else:
            count = previous_attempts["failure_count"] + 1
            window_start = previous_attempts["window_started_at"]
        blocked_until = now + _LOGIN_LOCK_SECONDS if count >= _LOGIN_MAX_FAILURES else 0.0
        connection.execute(
            """
            INSERT INTO login_attempts (username, failure_count, window_started_at, blocked_until)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                failure_count = excluded.failure_count,
                window_started_at = excluded.window_started_at,
                blocked_until = excluded.blocked_until
            """,
            (username, count, window_start, blocked_until),
        )


def reset_password(username: str, password: str, db_path: Path = LEDGER_PATH) -> bool:
    """Reset a local account password from a trusted administrator's offline session."""
    if not _valid_password(password):
        raise AccountError("Password must be at least 12 and no more than 1024 characters.")
    normalized = username.strip().casefold()
    initialize_accounts(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE accounts SET password_hash = ? WHERE username = ?",
            (_encode_password(password), normalized),
        )
        if cursor.rowcount:
            connection.execute("DELETE FROM login_attempts WHERE username = ?", (normalized,))
        return cursor.rowcount == 1
