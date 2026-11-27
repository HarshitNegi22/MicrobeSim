"""Model registry: lineage, validation metrics, review status and model cards.

Each trained artifact (identified by its cache signature, which already covers the dataset content, the
model code and the library versions) gets one registry row recording what it was trained on, how it was
validated and who approved it. Re-training with changed data or code produces a NEW signature and
therefore a new ``draft`` row; old rows are kept as history. Status changes go through
:func:`set_status` and are written to the audit trail. By default the reviewer must differ from the
author (four-eyes); set ``MICROBESIM_FOUR_EYES=0`` to disable for single-person installations.
"""

from __future__ import annotations

import getpass
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import audit
from src.ledger import LEDGER_PATH, _connection

STATUSES = ("draft", "approved", "rejected", "retired")
_initialised: set = set()
_lock = threading.Lock()


class RegistryError(ValueError):
    """A registry operation was refused (unknown model, bad status, reviewer rule)."""


def initialize_registry(db_path: Path = LEDGER_PATH) -> None:
    with _connection(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                signature TEXT PRIMARY KEY,
                created_utc TEXT NOT NULL,
                label TEXT NOT NULL,
                kind TEXT NOT NULL,
                organism TEXT NOT NULL,
                training_source TEXT NOT NULL,
                group_name TEXT NOT NULL,
                dataset_hash TEXT NOT NULL,
                n_rows INTEGER NOT NULL,
                model_keys TEXT NOT NULL,
                settings TEXT NOT NULL,
                metrics TEXT NOT NULL,
                code_digest TEXT NOT NULL,
                environment TEXT NOT NULL,
                author TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                reviewed_by TEXT,
                reviewed_utc TEXT,
                review_note TEXT
            )
            """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_registry_label ON model_registry(label, created_utc)"
        )


def _ready(db_path: Path) -> None:
    key = str(db_path)
    with _lock:
        if key not in _initialised:
            initialize_registry(db_path)
            _initialised.add(key)


def default_author() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown")


def register_model(
    signature: str,
    label: str,
    kind: str,
    organism: str,
    dataset_hash: str,
    n_rows: int,
    model_keys: List[str],
    settings: Dict[str, Any],
    metrics: List[Dict[str, Any]],
    code_digest: str,
    environment: Dict[str, str],
    training_source: str = "",
    group_name: str = "",
    author: Optional[str] = None,
    db_path: Path = LEDGER_PATH,
) -> bool:
    """Insert a draft entry; an existing signature is left untouched (approvals are never reset)."""
    _ready(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO model_registry (signature, created_utc, label, kind, organism, training_source,"
            " group_name, dataset_hash, n_rows, model_keys, settings, metrics, code_digest, environment, author)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                signature,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                label,
                kind,
                organism,
                training_source or "",
                group_name or "",
                dataset_hash,
                int(n_rows),
                json.dumps(model_keys),
                json.dumps(settings, sort_keys=True, default=str),
                json.dumps(metrics, default=str),
                code_digest,
                json.dumps(environment, sort_keys=True),
                author or default_author(),
            ),
        )
        created = cursor.rowcount == 1
    if created:
        audit.record(
            author or default_author(),
            "model.register",
            signature,
            detail={"label": label},
            db_path=db_path,
        )
    return created


def _row_to_dict(row) -> Dict[str, Any]:
    data = dict(row)
    for key in ("model_keys", "settings", "metrics", "environment"):
        data[key] = json.loads(data[key])
    return data


def get_model(signature: str, db_path: Path = LEDGER_PATH) -> Optional[Dict[str, Any]]:
    _ready(db_path)
    with _connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM model_registry WHERE signature = ?", (signature,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_models(
    status: Optional[str] = None, label: Optional[str] = None, db_path: Path = LEDGER_PATH
) -> List[Dict[str, Any]]:
    _ready(db_path)
    clauses, params = [], []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if label:
        clauses.append("label = ?")
        params.append(label)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT * FROM model_registry {where} ORDER BY created_utc DESC, signature", params
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def set_status(
    signature: str, status: str, reviewer: str, note: str = "", db_path: Path = LEDGER_PATH
) -> Dict[str, Any]:
    """Approve/reject/retire a model. Approval needs a note and (by default) a reviewer other than the author."""
    if status not in STATUSES or status == "draft":
        raise RegistryError(
            f"status must be one of {', '.join(s for s in STATUSES if s != 'draft')}"
        )
    entry = get_model(signature, db_path)
    if entry is None:
        raise RegistryError("unknown model signature")
    if not reviewer.strip():
        raise RegistryError("a reviewer name is required")
    if status == "approved":
        if not note.strip():
            raise RegistryError("approving a model needs a review note (what was checked)")
        if (
            os.environ.get("MICROBESIM_FOUR_EYES", "1") != "0"
            and reviewer.strip() == entry["author"]
        ):
            raise RegistryError("the reviewer must be different from the author (four-eyes rule)")
    with _connection(db_path) as connection:
        connection.execute(
            "UPDATE model_registry SET status = ?, reviewed_by = ?, reviewed_utc = ?, review_note = ? WHERE signature = ?",  # noqa: E501
            (
                status,
                reviewer.strip(),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                note.strip(),
                signature,
            ),
        )
    audit.record(
        reviewer.strip(),
        f"model.{status}",
        signature,
        detail={"label": entry["label"], "note": note.strip()[:500]},
        db_path=db_path,
    )
    return get_model(signature, db_path)  # type: ignore[return-value]


def status_of(signature: str, db_path: Path = LEDGER_PATH) -> Optional[str]:
    entry = get_model(signature, db_path)
    return entry["status"] if entry else None


def model_card(signature: str, db_path: Path = LEDGER_PATH) -> str:
    """Markdown model card: intended use, data, validation, limits, review state and lineage."""
    entry = get_model(signature, db_path)
    if entry is None:
        raise RegistryError("unknown model signature")
    history = [
        m for m in list_models(label=entry["label"], db_path=db_path) if m["signature"] != signature
    ]
    lines = [
        f"# Model card: {entry['label']}",
        "",
        f"* **Status:** {entry['status']}"
        + (
            f" (reviewed by {entry['reviewed_by']} on {entry['reviewed_utc']})"
            if entry["reviewed_by"]
            else " (not reviewed)"
        ),
        f"* **Signature:** `{entry['signature']}`",
        f"* **Trained:** {entry['created_utc']} by {entry['author']}",
        "* **Intended use:** exploratory estimates of nominal microbial growth rate from "
        "temperature, pH and water activity. Not a calibrated safety decision; output needs "
        "human review.",
        "",
        "## Data",
        f"* Organism: {entry['organism']}; training source: {entry['training_source'] or 'default'}; "
        f"validation grouping: {entry['group_name'] or 'random split'}",
        f"* Rows: {entry['n_rows']}; dataset SHA-256 (first 16): `{entry['dataset_hash']}`",
        "",
        "## Validation (held-out data)",
        "",
    ]
    metrics = entry["metrics"]
    if metrics:
        columns = [c for c in ("Model", "R2", "RMSE", "Af", "Bf") if c in metrics[0]]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in metrics:
            lines.append(
                "| "
                + " | ".join(
                    f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c]) for c in columns
                )
                + " |"
            )
    else:
        lines.append("No validation metrics recorded.")
    lines += [
        "",
        "## Reproducibility",
        f"* Models: {', '.join(entry['model_keys'])}",
        f"* Training settings: `{json.dumps(entry['settings'], sort_keys=True)}`",
        f"* Code digest (first 16): `{entry['code_digest'][:16]}`",
        f"* Libraries: {', '.join(f'{k} {v}' for k, v in sorted(entry['environment'].items()))}",
        "",
        "## Review",
        f"{entry['review_note'] or 'No review note.'}",
        "",
        "## Lineage",
    ]
    lines += [f"* {m['created_utc']} `{m['signature'][:12]}` {m['status']}" for m in history] or [
        "* No earlier versions."
    ]
    return "\n".join(lines) + "\n"
