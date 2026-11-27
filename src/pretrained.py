"""Pretrained model store: train every dataset the app offers once, load instantly on every visit.

``build_jobs`` enumerates exactly what the UI can request (every real and synthetic organism x every
validation-grouping choice) plus the expanded-data training sources, using the same helpers the UI
uses, so the store cannot drift from what the app asks for. ``pretrain`` fills
``data/pretrained_models``; ``load_or_train`` (src/model_cache.py) reads from it before ever training.
Uploaded datasets are user data and cannot be pretrained; they are trained once and cached as before.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src import model_cache
from src.datasets import (
    expanded_data_available,
    load_expanded_growth_dataset,
    load_real_growth_dataset,
    load_synthetic_growth_dataset,
    published_plus_organisms,
    real_organisms,
    synthetic_organisms,
)
from src.models.benchmark import available_model_keys

PRETRAINED_DIR = model_cache._PRETRAINED_DIR
MANIFEST_NAME = "manifest.json"

FEATURE_COLUMNS = {"temperature_celsius", "ph", "water_activity", "mu_max_per_h", "lag_h"}
KNOWN_GROUP_COLUMNS = {
    "study_id",
    "study",
    "experiment_id",
    "batch_id",
    "record_id",
    "food_category",
    "series_id",
}
DEFAULT_GROUP_PREFERENCE = ("study_id", "study", "experiment_id", "batch_id", "food_category")
REAL_SUFFIX = " (installed real)"
SYNTHETIC_SUFFIX = " (synthetic)"
PUBLISHED_SUFFIX = " (local real + published)"


# --------------------------------------------------------------------------- #
# Grouping choices shared with the UI
# --------------------------------------------------------------------------- #
def grouping_options(df: pd.DataFrame) -> Tuple[List[Optional[str]], Optional[str]]:
    """``(options, default)`` for the 'Validation grouping' picker; ``None`` = random row split."""
    columns = [
        column
        for column in df.columns
        if column not in FEATURE_COLUMNS
        and (column.lower() in KNOWN_GROUP_COLUMNS or pd.api.types.is_object_dtype(df[column]))
        and df[column].nunique(dropna=False) > 1
    ]
    defaults = [column for column in DEFAULT_GROUP_PREFERENCE if column in columns]
    return [None, *columns], (defaults[0] if defaults else None)


def grouping_arguments(df: pd.DataFrame, group_column: Optional[str]) -> dict:
    """Keyword arguments for ``load_or_train`` for a chosen validation grouping."""
    if group_column is None:
        return {}
    if group_column not in df.columns:
        raise ValueError(f"Validation group column '{group_column}' is not in the dataset.")
    groups = df[group_column].fillna("Unknown").astype(str).to_numpy()
    if pd.Series(groups).nunique() < 2:
        raise ValueError(f"Validation group column '{group_column}' needs at least two groups.")
    return {"groups": groups, "group_name": group_column}


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def _load(job: Dict) -> pd.DataFrame:
    if job["kind"] == "real":
        return load_real_growth_dataset(job["organism"])
    if job["kind"] == "real_plus":
        return load_real_growth_dataset(job["organism"], include_external=True)
    if job["kind"] == "synthetic":
        return load_synthetic_growth_dataset(job["organism"])
    return load_expanded_growth_dataset(job["organism"])


def build_jobs(include_expanded: bool = True) -> List[Dict]:
    """Every (dataset, validation grouping, training source) the app or scripts can request."""
    jobs: List[Dict] = []
    for kind, organisms, suffix in (
        ("real", real_organisms(), REAL_SUFFIX),
        ("real_plus", published_plus_organisms(), PUBLISHED_SUFFIX),
        ("synthetic", synthetic_organisms(), SYNTHETIC_SUFFIX),
    ):
        for organism in organisms:
            frame = _load({"kind": kind, "organism": organism})
            options, _ = grouping_options(frame)
            for group_column in options:
                jobs.append(
                    {
                        "kind": kind,
                        "organism": organism,
                        "group_column": group_column,
                        "training_source": None,
                        "label": f"{organism}{suffix} | grouping: {group_column or 'random split'}",
                    }
                )
    if include_expanded and expanded_data_available():
        counts = load_expanded_growth_dataset(include_synthetic=False).groupby("organism").size()
        for organism in sorted(counts[counts >= 20].index):
            for training_source in ("real", "real+synthetic"):
                jobs.append(
                    {
                        "kind": "expanded",
                        "organism": organism,
                        "group_column": None,
                        "training_source": training_source,
                        "label": f"{organism} (expanded) | training: {training_source}",
                    }
                )
    return jobs


def _signature(job: Dict) -> str:
    frame = _load(job)
    return model_cache.signature_for(
        frame,
        available_model_keys(),
        training_source=job["training_source"],
        **grouping_arguments(frame, job["group_column"]),
    )


def _run_job(job: Dict, store_dir: str, force: bool) -> Dict:
    started = time.time()
    frame = _load(job)
    extra = grouping_arguments(frame, job["group_column"])
    signature, status, result = model_cache.train_and_store(
        frame,
        available_model_keys(),
        Path(store_dir),
        training_source=job["training_source"],
        force=force,
        **extra,
    )
    path = Path(store_dir) / f"{signature}.joblib"
    if (
        result is None
    ):  # already trained: read it back so the registry/manifest still get its metrics
        result = model_cache._read_artifact(path, signature)
    table = getattr(result, "table", None)
    columns = [
        c for c in ("key", "Model", "R2", "RMSE", "Af", "Bf") if table is not None and c in table
    ]
    metrics = (
        json.loads(table[columns].to_json(orient="records"))
        if table is not None and columns
        else []
    )
    return {
        **job,
        "metrics": metrics,
        "dataset_hash": hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()[:16],
        "signature": signature,
        "status": status,
        "rows": int(len(frame)),
        "megabytes": round(path.stat().st_size / 1e6, 2),
        "seconds": round(time.time() - started, 1),
    }


def pretrain(
    store_dir: Optional[Path] = None,
    include_expanded: bool = True,
    n_jobs: int = 1,
    force: bool = False,
    progress: Optional[Callable[[int, int, Dict], None]] = None,
    organisms: Optional[List[str]] = None,
) -> List[Dict]:
    """Train and store every job; resumable (existing valid artifacts are skipped)."""
    store = Path(store_dir) if store_dir else PRETRAINED_DIR
    jobs = build_jobs(include_expanded)
    if organisms:
        jobs = [job for job in jobs if job["organism"] in organisms]
    records: List[Dict] = []
    if n_jobs > 1:
        from joblib import Parallel, delayed

        for done, record in enumerate(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_run_job)(j, str(store), force) for j in jobs
            ),
            1,
        ):
            records.append(record)
            if progress:
                progress(done, len(jobs), record)
    else:
        for done, job in enumerate(jobs, 1):
            record = _run_job(job, str(store), force)
            records.append(record)
            if progress:
                progress(done, len(jobs), record)
    if not organisms:
        write_manifest(store, records)
    return records


# --------------------------------------------------------------------------- #
# Manifest and status
# --------------------------------------------------------------------------- #
def write_manifest(store: Path, records: List[Dict]) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": model_cache.environment_fingerprint(),
        "source_digest": model_cache.source_digest(),
        "model_keys": available_model_keys(),
        "settings": model_cache._BENCHMARK_OPTIONS,
        "entries": [
            {
                k: r.get(k)
                for k in (
                    "label",
                    "signature",
                    "rows",
                    "megabytes",
                    "seconds",
                    "kind",
                    "organism",
                    "training_source",
                    "group_column",
                    "dataset_hash",
                    "metrics",
                )
            }
            for r in records
        ],
    }
    (store / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def pretrained_status(store_dir: Optional[Path] = None, include_expanded: bool = True) -> Dict:
    """``state`` is ``ready``, ``partial``, ``stale`` (other library/code version) or ``missing``."""
    store = Path(store_dir) if store_dir else PRETRAINED_DIR
    manifest_path = store / MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "state": "missing",
            "ready": 0,
            "total": len(build_jobs(include_expanded)),
            "detail": "no pretrained models found",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = model_cache.environment_fingerprint()
    if manifest.get("environment") != current:
        diff = {
            k: (manifest.get("environment", {}).get(k), v)
            for k, v in current.items()
            if manifest.get("environment", {}).get(k) != v
        }
        return {
            "state": "stale",
            "ready": 0,
            "total": len(manifest.get("entries", [])),
            "detail": f"built with different library versions: {diff}",
        }
    if manifest.get("source_digest") != model_cache.source_digest():
        return {
            "state": "stale",
            "ready": 0,
            "total": len(manifest.get("entries", [])),
            "detail": "model code changed since the models were trained",
        }
    jobs = build_jobs(include_expanded)
    ready = sum(1 for job in jobs if (store / f"{_signature(job)}.joblib").is_file())
    state = "ready" if ready == len(jobs) else "partial"
    return {
        "state": state,
        "ready": ready,
        "total": len(jobs),
        "detail": f"{ready}/{len(jobs)} datasets pretrained",
    }


def sync_registry(store_dir: Optional[Path] = None, db_path: Optional[Path] = None) -> int:
    """Register the models listed in the manifest as ``draft`` entries (idempotent; approvals are kept).

    Reads only the manifest, so it is cheap enough to run at application start. A manifest built for other
    library versions or other model code is ignored because its signatures can no longer be loaded.
    """
    from src import registry
    from src.ledger import LEDGER_PATH

    store = Path(store_dir) if store_dir else PRETRAINED_DIR
    manifest_path = store / MANIFEST_NAME
    if not manifest_path.exists():
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("environment") != model_cache.environment_fingerprint()
        or manifest.get("source_digest") != model_cache.source_digest()
    ):
        return 0
    added = 0
    for entry in manifest.get("entries", []):
        if not entry.get("dataset_hash"):
            continue  # manifest from before registry support: run scripts/pretrain_models.py to refresh it
        created = registry.register_model(
            signature=entry["signature"],
            label=entry["label"],
            kind=entry.get("kind") or "",
            organism=entry.get("organism") or "",
            dataset_hash=entry["dataset_hash"],
            n_rows=entry["rows"],
            model_keys=manifest.get("model_keys", []),
            settings=manifest.get("settings", {}),
            metrics=entry.get("metrics", []),
            code_digest=manifest["source_digest"],
            environment=manifest["environment"],
            training_source=entry.get("training_source") or "",
            group_name=entry.get("group_column") or "",
            author="pretrain-script",
            db_path=db_path or LEDGER_PATH,
        )
        added += int(created)
    return added
