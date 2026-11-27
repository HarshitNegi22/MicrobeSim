"""Append-only, tamper-evident audit trail stored in the ledger database.

Every security-relevant action (sign-in outcomes, account changes, run/project/profile changes,
uploads, model approvals) is written as one row. Rows are chained: each stores the SHA-256 of its own
content plus the previous row's hash, so editing or deleting any past row breaks ``verify_chain``.
SQLite triggers additionally refuse UPDATE and DELETE on the table. A database owner can still rebuild
the whole chain, so for high-assurance use ship the log (``log_event`` lines) to external storage.

Writing never raises into the application: a failure is logged and counted
(``microbesim_audit_write_failures_total``) so an audit outage cannot take the app down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ledger import LEDGER_PATH, _connection
from src.observability import METRICS, log_event

_GENESIS = "0" * 64
_initialised: set = set()
_lock = threading.Lock()


def initialize_audit(db_path: Path = LEDGER_PATH) -> None:
    with _connection(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                outcome TEXT NOT NULL,
                detail TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            )
            """)
        for operation in ("UPDATE", "DELETE"):
            connection.execute(f"""
                CREATE TRIGGER IF NOT EXISTS audit_log_no_{operation.lower()}
                BEFORE {operation} ON audit_log
                BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END
                """)


def _digest(
    prev_hash: str, ts: str, actor: str, action: str, target: str, outcome: str, detail: str
) -> str:
    material = json.dumps(
        [prev_hash, ts, actor, action, target, outcome, detail], ensure_ascii=False
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def record(
    actor: Optional[str],
    action: str,
    target: str = "",
    outcome: str = "ok",
    detail: Optional[Dict[str, Any]] = None,
    db_path: Path = LEDGER_PATH,
) -> bool:
    """Append one event. Never put passwords, tokens or file contents in ``detail``."""
    try:
        key = str(db_path)
        with _lock:
            if key not in _initialised:
                initialize_audit(db_path)
                _initialised.add(key)
        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        body = json.dumps(detail or {}, sort_keys=True, default=str, ensure_ascii=False)
        with _connection(db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            last = connection.execute(
                "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev = last["hash"] if last else _GENESIS
            digest = _digest(prev, ts, actor or "-", action, str(target), outcome, body)
            connection.execute(
                "INSERT INTO audit_log (ts, actor, action, target, outcome, detail, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, actor or "-", action, str(target), outcome, body, prev, digest),
            )
        log_event("audit", actor=actor or "-", action=action, target=str(target), outcome=outcome)
        return True
    except Exception as exc:  # audit must never break the application
        METRICS.inc("microbesim_audit_write_failures_total")
        log_event(
            "audit.write_failed", level=logging.ERROR, error=type(exc).__name__, action=action
        )
        return False


def verify_chain(db_path: Path = LEDGER_PATH) -> Tuple[bool, Optional[int]]:
    """Return ``(ok, first_bad_id)``; recomputes every hash and link."""
    initialize_audit(db_path)
    prev = _GENESIS
    with _connection(db_path) as connection:
        for row in connection.execute("SELECT * FROM audit_log ORDER BY id"):
            expected = _digest(
                prev,
                row["ts"],
                row["actor"],
                row["action"],
                row["target"],
                row["outcome"],
                row["detail"],
            )
            if row["prev_hash"] != prev or row["hash"] != expected:
                return False, int(row["id"])
            prev = row["hash"]
    return True, None


def list_events(
    actor: Optional[str] = None,
    action_prefix: Optional[str] = None,
    limit: int = 200,
    db_path: Path = LEDGER_PATH,
) -> List[Dict[str, Any]]:
    initialize_audit(db_path)
    clauses, params = [], []
    if actor:
        clauses.append("actor = ?")
        params.append(actor)
    if action_prefix:
        clauses.append("action LIKE ?")
        params.append(action_prefix.replace("%", "") + "%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT id, ts, actor, action, target, outcome, detail FROM audit_log {where} ORDER BY id DESC LIMIT ?",
            (*params, max(1, min(int(limit), 5000))),
        ).fetchall()
    return [dict(row) for row in rows]
