"""Persistent cache for fitted growth-rate models and their benchmark results."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from importlib.metadata import version
from pathlib import Path
from pickle import UnpicklingError
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from src.ledger import STATE_DIR
from src.models.benchmark import BenchmarkResult, benchmark

_CACHE_DIR = STATE_DIR / "trained_models"
# Read-only store of models trained ahead of time (scripts/pretrain_models.py). It lives inside the
# application tree (not the writable state directory) so it survives container volumes and upgrades.
_PRETRAINED_DIR = Path(__file__).resolve().parents[1] / "data" / "pretrained_models"
_CACHE_FORMAT = 2
_MAX_CACHE_ENTRIES = 300
_LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "joblib")
_BENCHMARK_OPTIONS = {
    "test_size": 0.25,
    "seed": 0,
    "min_obs_afbf": 0.05,
    "n_repeats": 2,
}
_MODEL_SOURCE_FILES = (
    "benchmark.py",
    "metrics.py",
    "ml_regressors.py",
    "secondary.py",
)
_MODELS_DIR = Path(__file__).resolve().parent / "models"


def _signature(
    df: pd.DataFrame,
    model_keys: List[str],
    groups: Optional[np.ndarray],
    group_name: str,
    training_source: Optional[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "format": _CACHE_FORMAT,
                "model_keys": model_keys,
                "group_name": group_name,
                "training_source": training_source,
                "benchmark_options": _BENCHMARK_OPTIONS,
                # Library versions decide whether a pickle can be read; the Python patch level does
                # not, so it is deliberately not part of the signature.
                "libraries": {name: version(name) for name in _LIBRARIES},
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update(
        json.dumps(
            [(str(column), str(dtype)) for column, dtype in zip(df.columns, df.dtypes)]
        ).encode("utf-8")
    )
    if groups is not None:
        digest.update(pd.util.hash_pandas_object(pd.Series(groups), index=True).values.tobytes())
    digest.update(source_digest().encode("ascii"))
    return digest.hexdigest()


def _normalised_bytes(path: Path) -> bytes:
    """File bytes with Windows line endings folded, so a checkout does not change the signature."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def source_digest() -> str:
    """Fingerprint of the model code: any change to it invalidates saved models."""
    digest = hashlib.sha256()
    for name in _MODEL_SOURCE_FILES:
        digest.update(_normalised_bytes(_MODELS_DIR / name))
    digest.update(_normalised_bytes(Path(__file__)))
    return digest.hexdigest()


def environment_fingerprint() -> dict:
    """Library versions a saved model depends on (recorded in the pretrained manifest)."""
    return {name: version(name) for name in _LIBRARIES}


def _save(path: Path, signature: str, result: BenchmarkResult) -> None:
    _prepare_cache_dir()
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(
            {"format": _CACHE_FORMAT, "signature": signature, "result": result},
            temporary_path,
            compress=3,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    _prune_cache(path)


def _prune_cache(current_path: Path) -> None:
    """Keep the app-managed model cache bounded while retaining the newest artifact."""
    entries = sorted(
        _CACHE_DIR.glob("*.joblib"),
        key=lambda item: (item == current_path, item.stat().st_mtime),
        reverse=True,
    )
    for stale_path in entries[_MAX_CACHE_ENTRIES:]:
        stale_path.unlink()


def _prepare_cache_dir() -> None:
    """Keep the pickle-backed cache private to its owning application account."""
    if _CACHE_DIR.is_symlink():
        raise RuntimeError(f"Model cache directory '{_CACHE_DIR}' must not be a symbolic link.")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        return
    owner = os.getuid()
    directory_stat = _CACHE_DIR.stat()
    parent_stat = _CACHE_DIR.parent.stat()
    if directory_stat.st_uid != owner or parent_stat.st_uid != owner:
        raise RuntimeError("Model cache and its parent must be owned by the application account.")
    if parent_stat.st_mode & 0o022:
        raise RuntimeError("Model cache parent must not be writable by group or other users.")
    os.chmod(_CACHE_DIR, 0o700)


def _read_artifact(path: Path, signature: str) -> Optional[BenchmarkResult]:
    """Return the cached result at ``path`` or None if it is not a valid artifact for ``signature``."""
    artifact = joblib.load(path)
    if (
        isinstance(artifact, dict)
        and artifact.get("format") == _CACHE_FORMAT
        and artifact.get("signature") == signature
        and isinstance(artifact.get("result"), BenchmarkResult)
    ):
        return artifact["result"]
    return None


def _load_pretrained(signature: str) -> Optional[BenchmarkResult]:
    """Models shipped with (or built for) this installation; an unusable file is ignored, not fatal."""
    path = _PRETRAINED_DIR / f"{signature}.joblib"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        return _read_artifact(path, signature)
    except (
        Exception
    ):  # corrupt/incompatible pickle: fall back to training rather than breaking the app
        return None


def load_or_train(
    df: pd.DataFrame,
    model_keys: List[str],
    groups: Optional[np.ndarray] = None,
    group_name: str = "supplied labels",
    force_retrain: bool = False,
    training_source: Optional[str] = None,
) -> BenchmarkResult:
    """Load saved models (pretrained store, then local cache) or train, benchmark and persist them.

    Order: local cache -> pretrained store -> train. ``result.cache_source`` is ``"cache"``,
    ``"pretrained"`` or ``"trained"``. Set ``force_retrain`` to replace saved results with fresh fits.
    """
    _prepare_cache_dir()
    signature = _signature(df, model_keys, groups, group_name, training_source)
    path = _CACHE_DIR / f"{signature}.joblib"
    if path.exists() and not force_retrain:
        try:
            result = _read_artifact(path, signature)
        except (EOFError, OSError, UnpicklingError, ValueError) as exc:
            raise RuntimeError(
                f"Could not load saved model cache '{path}'. Remove that cache file and retry."
            ) from exc
        if result is None:
            raise RuntimeError(
                f"Saved model cache '{path}' is invalid. Remove that cache file and retry."
            )
        result.loaded_from_disk = True
        result.cache_source = "cache"
        return result

    if not force_retrain:
        result = _load_pretrained(signature)
        if result is not None:
            result.loaded_from_disk = True
            result.cache_source = "pretrained"
            return result

    result = benchmark(
        df,
        model_keys,
        groups=groups,
        group_name=group_name,
        training_source=training_source,
        **_BENCHMARK_OPTIONS,
    )
    result.loaded_from_disk = False
    result.cache_source = "trained"
    _save(path, signature, result)
    return result


def train_and_store(
    df: pd.DataFrame,
    model_keys: List[str],
    store_dir: Path,
    groups: Optional[np.ndarray] = None,
    group_name: str = "supplied labels",
    training_source: Optional[str] = None,
    force: bool = False,
) -> tuple[str, str, Optional[BenchmarkResult]]:
    """Train (if needed) and write one artifact into ``store_dir``; used by scripts/pretrain_models.py.

    Returns ``(signature, status, result)`` with status ``"exists"`` or ``"trained"``.
    """
    signature = _signature(df, model_keys, groups, group_name, training_source)
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / f"{signature}.joblib"
    if path.exists() and not force:
        try:
            if _read_artifact(path, signature) is not None:
                return signature, "exists", None
        except Exception:
            pass  # unreadable: rebuild below
    result = benchmark(
        df,
        model_keys,
        groups=groups,
        group_name=group_name,
        training_source=training_source,
        **_BENCHMARK_OPTIONS,
    )
    result.loaded_from_disk = False
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(
            {"format": _CACHE_FORMAT, "signature": signature, "result": result},
            temporary_path,
            compress=3,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return signature, "trained", result


def signature_for(
    df: pd.DataFrame,
    model_keys: List[str],
    groups: Optional[np.ndarray] = None,
    group_name: str = "supplied labels",
    training_source: Optional[str] = None,
) -> str:
    """Public wrapper so tools can check which artifact a request would use."""
    return _signature(df, model_keys, groups, group_name, training_source)
