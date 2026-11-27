"""Persistent local ledger for completed MicrobeSim analyses."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("MICROBESIM_STATE_DIR", ROOT / "data"))
LEDGER_PATH = Path(os.environ.get("MICROBESIM_LEDGER_PATH", STATE_DIR / "analysis_ledger.sqlite3"))
_LEDGER_SCHEMA_VERSION = 1


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        _prepare_sqlite_file(db_path)
    connection = sqlite3.connect(db_path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    if os.name != "nt":
        for suffix in ("-wal", "-shm"):
            _restrict_sqlite_file(Path(f"{db_path}{suffix}"))
    return connection


def _prepare_sqlite_file(path: Path) -> None:
    """Create a private database file and reject unsafe POSIX parent directories."""
    owner = os.getuid()
    parent_stat = path.parent.stat()
    if parent_stat.st_uid != owner or parent_stat.st_mode & 0o022:
        raise RuntimeError(
            "SQLite database parent must be application-owned and not shared-writable."
        )
    if path.is_symlink():
        raise RuntimeError(f"SQLite database path '{path}' must not be a symbolic link.")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        file_stat = os.fstat(descriptor)
        if file_stat.st_uid != owner:
            raise RuntimeError("SQLite database must be owned by the application account.")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _restrict_sqlite_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return
    try:
        if os.fstat(descriptor).st_uid != os.getuid():
            raise RuntimeError(f"SQLite sidecar '{path}' must be owned by the application account.")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


@contextmanager
def _connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_ledger(db_path: Path = LEDGER_PATH) -> None:
    """Create or migrate the per-account analysis ledger."""
    with _connection(db_path) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _LEDGER_SCHEMA_VERSION:
            raise RuntimeError(
                f"Ledger schema version {version} is newer than supported "
                f"version {_LEDGER_SCHEMA_VERSION}."
            )
        if version == _LEDGER_SCHEMA_VERSION:
            return
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                run_type TEXT NOT NULL CHECK (run_type IN ('assay', 'prediction', 'simulation')),
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                metric_label TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                user_id TEXT REFERENCES accounts(user_id)
            )
            """)
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(analysis_runs)").fetchall()
        }
        if "user_id" not in columns:
            connection.execute("ALTER TABLE analysis_runs RENAME TO analysis_runs_legacy")
            connection.execute("""
                CREATE TABLE analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    run_type TEXT NOT NULL CHECK (run_type IN ('assay', 'prediction', 'simulation')),
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    metric_label TEXT NOT NULL,
                    metric_value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    user_id TEXT REFERENCES accounts(user_id)
                )
                """)
            connection.execute("""
                INSERT INTO analysis_runs
                    (run_id, fingerprint, run_type, title, summary, metric_label,
                     metric_value, created_at, payload_json, user_id)
                SELECT run_id, fingerprint, run_type, title, summary, metric_label,
                       metric_value, created_at, payload_json, NULL
                FROM analysis_runs_legacy
                """)
            connection.execute("DROP TABLE analysis_runs_legacy")
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(analysis_runs)").fetchall()
        }
        if "project_id" not in columns:
            connection.execute("ALTER TABLE analysis_runs ADD COLUMN project_id TEXT")
        if "notes" not in columns:
            connection.execute(
                "ALTER TABLE analysis_runs ADD COLUMN notes TEXT NOT NULL DEFAULT ''"
            )
        connection.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, name)
            )
            """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                profile_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,
                profile_type TEXT NOT NULL CHECK (profile_type IN ('product', 'scenario')),
                name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, profile_type, name)
            )
            """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at "
            "ON analysis_runs(user_id, created_at DESC)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_runs_user_fingerprint "
            "ON analysis_runs(user_id, fingerprint) WHERE user_id IS NOT NULL"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_runs_user_type_created "
            "ON analysis_runs(user_id, run_type, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_runs_user_project_created "
            "ON analysis_runs(user_id, project_id, created_at DESC)"
        )
        connection.execute(f"PRAGMA user_version = {_LEDGER_SCHEMA_VERSION}")


def save_run_record(
    user_id: str,
    run_type: str,
    title: str,
    summary: str,
    metric_label: str,
    metric_value: str,
    payload: Dict[str, Any],
    db_path: Path = LEDGER_PATH,
) -> tuple[bool, str]:
    """Persist a completed run once and return its ID (existing ID when deduplicated)."""
    if not user_id:
        raise ValueError("An authenticated account is required to save an analysis.")
    initialize_ledger(db_path)
    content = {
        "user_id": user_id,
        "run_type": run_type,
        "title": title,
        "summary": summary,
        "metric_label": metric_label,
        "metric_value": metric_value,
        "payload": payload,
    }
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO analysis_runs
                (run_id, fingerprint, run_type, title, summary, metric_label,
                 metric_value, created_at, payload_json, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                fingerprint,
                run_type,
                title,
                summary,
                metric_label,
                metric_value,
                created_at,
                json.dumps(payload, sort_keys=True, default=str),
                user_id,
            ),
        )
        if cursor.rowcount == 1:
            return True, run_id
        row = connection.execute(
            "SELECT run_id FROM analysis_runs WHERE user_id = ? AND fingerprint = ?",
            (user_id, fingerprint),
        ).fetchone()
        if row is None:
            raise RuntimeError("Saved analysis could not be retrieved after deduplication.")
        return False, str(row["run_id"])


def save_run(
    user_id: str,
    run_type: str,
    title: str,
    summary: str,
    metric_label: str,
    metric_value: str,
    payload: Dict[str, Any],
    db_path: Path = LEDGER_PATH,
) -> bool:
    """Persist a completed run once; return whether a new record was inserted."""
    return save_run_record(
        user_id,
        run_type,
        title,
        summary,
        metric_label,
        metric_value,
        payload,
        db_path,
    )[0]


def list_runs(
    user_id: str,
    run_type: Optional[str] = None,
    limit: int = 500,
    db_path: Path = LEDGER_PATH,
) -> List[Dict[str, Any]]:
    """Load recent saved runs, with decoded input and result payloads."""
    if not user_id:
        raise ValueError("An authenticated account is required to view saved analyses.")
    initialize_ledger(db_path)
    query = "SELECT * FROM analysis_runs WHERE user_id = ?"
    parameters: tuple[Any, ...] = (user_id,)
    if run_type:
        query += " AND run_type = ?"
        parameters += (run_type,)
    query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    parameters += (limit,)
    with _connection(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [
        {
            **dict(row),
            "payload": json.loads(row["payload_json"]),
        }
        for row in rows
    ]


def count_runs_by_type(user_id: str, db_path: Path = LEDGER_PATH) -> Dict[str, int]:
    """Return counts for the current account without loading saved payloads."""
    if not user_id:
        raise ValueError("An authenticated account is required to view saved analyses.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute(
            "SELECT run_type, COUNT(*) AS total FROM analysis_runs "
            "WHERE user_id = ? GROUP BY run_type",
            (user_id,),
        ).fetchall()
    counts = {run_type: 0 for run_type in ("assay", "prediction", "simulation")}
    counts.update({row["run_type"]: int(row["total"]) for row in rows})
    return counts


def list_run_summaries(
    user_id: str,
    run_type: Optional[str] = None,
    search: str = "",
    limit: int = 50,
    offset: int = 0,
    project_id: Optional[str] = None,
    db_path: Path = LEDGER_PATH,
) -> List[Dict[str, Any]]:
    """Load only searchable run metadata for one page of the current account's history."""
    if not user_id:
        raise ValueError("An authenticated account is required to view saved analyses.")
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset cannot be negative.")
    initialize_ledger(db_path)
    clauses = ["user_id = ?"]
    parameters: List[Any] = [user_id]
    if run_type:
        clauses.append("run_type = ?")
        parameters.append(run_type)
    if project_id is not None:
        if project_id:
            clauses.append("project_id = ?")
            parameters.append(project_id)
        else:
            clauses.append("project_id IS NULL")
    if search.strip():
        clauses.append(
            "instr(lower(title || ' ' || summary || ' ' || metric_value || ' ' || notes), lower(?)) > 0"
        )
        parameters.append(search.strip())
    query = (
        "SELECT run_id, run_type, title, summary, metric_label, metric_value, created_at, "
        "project_id, notes "
        "FROM analysis_runs WHERE "
        + " AND ".join(clauses)
        + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
    )
    parameters.extend((limit, offset))
    with _connection(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def count_run_summaries(
    user_id: str,
    run_type: Optional[str] = None,
    search: str = "",
    project_id: Optional[str] = None,
    db_path: Path = LEDGER_PATH,
) -> int:
    """Count filtered history entries without loading their saved payloads."""
    if not user_id:
        raise ValueError("An authenticated account is required to view saved analyses.")
    initialize_ledger(db_path)
    clauses = ["user_id = ?"]
    parameters: List[Any] = [user_id]
    if run_type:
        clauses.append("run_type = ?")
        parameters.append(run_type)
    if project_id is not None:
        if project_id:
            clauses.append("project_id = ?")
            parameters.append(project_id)
        else:
            clauses.append("project_id IS NULL")
    if search.strip():
        clauses.append(
            "instr(lower(title || ' ' || summary || ' ' || metric_value || ' ' || notes), lower(?)) > 0"
        )
        parameters.append(search.strip())
    with _connection(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM analysis_runs WHERE " + " AND ".join(clauses),
            parameters,
        ).fetchone()
    return int(row[0])


def get_run(user_id: str, run_id: str, db_path: Path = LEDGER_PATH) -> Optional[Dict[str, Any]]:
    """Load one saved record, including its potentially large payload."""
    if not user_id:
        raise ValueError("An authenticated account is required to view saved analyses.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM analysis_runs WHERE user_id = ? AND run_id = ?",
            (user_id, run_id),
        ).fetchone()
    if row is None:
        return None
    return {**dict(row), "payload": json.loads(row["payload_json"])}


def create_project(user_id: str, name: str, db_path: Path = LEDGER_PATH) -> str:
    if not user_id:
        raise ValueError("An authenticated account is required to create a project.")
    normalized = " ".join(name.split())
    if not normalized or len(normalized) > 80:
        raise ValueError("Project names must contain between 1 and 80 characters.")
    initialize_ledger(db_path)
    project_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _connection(db_path) as connection:
            connection.execute(
                "INSERT INTO projects (project_id, user_id, name, created_at) VALUES (?, ?, ?, ?)",
                (project_id, user_id, normalized, created_at),
            )
    except sqlite3.IntegrityError as exc:
        if "projects.user_id, projects.name" in str(exc):
            raise ValueError("A project with that name already exists.") from exc
        raise
    return project_id


def list_projects(user_id: str, db_path: Path = LEDGER_PATH) -> List[Dict[str, str]]:
    if not user_id:
        raise ValueError("An authenticated account is required to view projects.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute(
            "SELECT project_id, name, created_at FROM projects WHERE user_id = ? ORDER BY name",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_run_metadata(
    user_id: str,
    run_id: str,
    project_id: Optional[str],
    notes: str,
    db_path: Path = LEDGER_PATH,
) -> bool:
    if not user_id:
        raise ValueError("An authenticated account is required to update a saved run.")
    if len(notes) > 2000:
        raise ValueError("Notes cannot exceed 2000 characters.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        if project_id is not None:
            owned = connection.execute(
                "SELECT 1 FROM projects WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
            if owned is None:
                raise ValueError("The selected project does not belong to this account.")
        cursor = connection.execute(
            "UPDATE analysis_runs SET project_id = ?, notes = ? "
            "WHERE user_id = ? AND run_id = ?",
            (project_id, notes.strip(), user_id, run_id),
        )
    return cursor.rowcount == 1


def save_user_profile(
    user_id: str,
    profile_type: str,
    name: str,
    payload: Dict[str, Any],
    db_path: Path = LEDGER_PATH,
) -> str:
    if not user_id:
        raise ValueError("An authenticated account is required to save a profile.")
    if profile_type not in ("product", "scenario"):
        raise ValueError("profile_type must be 'product' or 'scenario'.")
    normalized = " ".join(name.split())
    if not normalized or len(normalized) > 80:
        raise ValueError("Profile names must contain between 1 and 80 characters.")
    initialize_ledger(db_path)
    profile_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _connection(db_path) as connection:
            connection.execute(
                "INSERT INTO user_profiles "
                "(profile_id, user_id, profile_type, name, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    profile_id,
                    user_id,
                    profile_type,
                    normalized,
                    json.dumps(payload, sort_keys=True, default=str),
                    created_at,
                ),
            )
    except sqlite3.IntegrityError as exc:
        if "user_profiles.user_id, user_profiles.profile_type, user_profiles.name" in str(exc):
            raise ValueError("A profile with that name already exists.") from exc
        raise
    return profile_id


def list_user_profiles(
    user_id: str, profile_type: str, db_path: Path = LEDGER_PATH
) -> List[Dict[str, Any]]:
    if not user_id:
        raise ValueError("An authenticated account is required to view profiles.")
    if profile_type not in ("product", "scenario"):
        raise ValueError("profile_type must be 'product' or 'scenario'.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute(
            "SELECT profile_id, name, payload_json FROM user_profiles "
            "WHERE user_id = ? AND profile_type = ? ORDER BY name",
            (user_id, profile_type),
        ).fetchall()
    return [
        {
            "profile_id": row["profile_id"],
            "name": row["name"],
            "payload": json.loads(row["payload_json"]),
        }
        for row in rows
    ]


def delete_user_profile(user_id: str, profile_id: str, db_path: Path = LEDGER_PATH) -> bool:
    if not user_id:
        raise ValueError("An authenticated account is required to delete a profile.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM user_profiles WHERE user_id = ? AND profile_id = ?",
            (user_id, profile_id),
        )
    return cursor.rowcount == 1


def delete_run(user_id: str, run_id: str, db_path: Path = LEDGER_PATH) -> bool:
    """Delete one of the current account's saved records."""
    if not user_id:
        raise ValueError("An authenticated account is required to delete a saved analysis.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM analysis_runs WHERE user_id = ? AND run_id = ?", (user_id, run_id)
        )
        return cursor.rowcount == 1


def clear_runs(user_id: str, db_path: Path = LEDGER_PATH) -> int:
    """Delete only the current account's saved records."""
    if not user_id:
        raise ValueError("An authenticated account is required to clear saved analyses.")
    initialize_ledger(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM analysis_runs WHERE user_id = ?", (user_id,))
        return cursor.rowcount


# --------------------------------------------------------------------------- #
# Audit trail: every successful change to saved data is recorded (src/audit.py)
# --------------------------------------------------------------------------- #
def _with_audit(func, action, describe):
    """Wrap a mutating function so a successful change is written to the audit trail.

    ``describe(arguments, result)`` returns ``(target, detail)`` or ``None`` when nothing changed. Auditing
    problems never break the operation itself.
    """
    import functools
    import inspect

    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            outcome = describe(bound.arguments, result)
            if outcome is not None:
                from src import audit

                audit.record(
                    bound.arguments.get("user_id"),
                    action,
                    outcome[0],
                    detail=outcome[1],
                    db_path=bound.arguments["db_path"],
                )
        except Exception:  # noqa: BLE001 - audit must never break the operation
            pass
        return result

    return wrapper


save_run_record = _with_audit(
    save_run_record, "run.save", lambda a, r: (r[1], {"run_type": a["run_type"]}) if r[0] else None
)
create_project = _with_audit(
    create_project, "project.create", lambda a, r: (r, {"name": str(a["name"])[:80]})
)
update_run_metadata = _with_audit(
    update_run_metadata, "run.update", lambda a, r: (a["run_id"], {}) if r else None
)
save_user_profile = _with_audit(
    save_user_profile, "profile.save", lambda a, r: (r, {"profile_type": a["profile_type"]})
)
delete_user_profile = _with_audit(
    delete_user_profile, "profile.delete", lambda a, r: (a["profile_id"], {}) if r else None
)
delete_run = _with_audit(delete_run, "run.delete", lambda a, r: (a["run_id"], {}) if r else None)
clear_runs = _with_audit(
    clear_runs, "run.clear", lambda a, r: ("all", {"deleted": r}) if r else None
)
