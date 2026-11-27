"""Structured logging, metrics, error tracking and health endpoints (stdlib only).

* ``configure_logging`` emits one JSON object per line (``MICROBESIM_LOG_LEVEL``, default INFO).
* ``METRICS`` is a small thread-safe registry rendered in Prometheus text format.
* ``timed`` records duration and failures of an operation as metrics plus a log line.
* ``start_ops_server`` serves ``/healthz`` (liveness), ``/readyz`` (state dir, ledger, pretrained
  models) and ``/metrics`` on a separate port. It is OFF unless ``MICROBESIM_OPS_PORT`` is set, binds to
  127.0.0.1 unless ``MICROBESIM_OPS_HOST`` says otherwise, and ``MICROBESIM_OPS_TOKEN`` (if set) is
  required as a bearer token for everything except ``/healthz``. Never publish it to the internet.
* ``init_error_tracking`` enables Sentry only if ``SENTRY_DSN`` is set and ``sentry_sdk`` is installed.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterator, Optional, Tuple

LOGGER_NAME = "microbesim"
_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 300)
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra ``logger.info(..., extra={...})`` fields are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is *now* (hosts and test runners swap it)."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # the base class assigns it; we always use the live stderr
        pass


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    """Install the JSON handler once (idempotent) and return the application logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_microbesim", False) for h in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(JsonFormatter())
        handler._microbesim = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel((level or os.environ.get("MICROBESIM_LOG_LEVEL", "INFO")).upper())
    return logger


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    """Log a named event with structured fields. Never put secrets or passwords in ``fields``."""
    configure_logging().log(level, event, extra={"event": event, **fields})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _label_key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _fmt_labels(key: Tuple[Tuple[str, str], ...], extra: str = "") -> str:
    parts = [f'{k}="{v}"' for k, v in key] + ([extra] if extra else [])
    return "{" + ",".join(parts) + "}" if parts else ""


class Metrics:
    """Counters, gauges and histograms with Prometheus text exposition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[tuple, float]] = {}
        self._gauges: Dict[str, Dict[tuple, float]] = {}
        self._hist: Dict[str, Dict[tuple, list]] = {}
        self._help: Dict[str, str] = {}

    def describe(self, name: str, text: str) -> None:
        self._help[name] = text

    def inc(self, name: str, amount: float = 1.0, **labels) -> None:
        with self._lock:
            series = self._counters.setdefault(name, {})
            series[_label_key(labels)] = series.get(_label_key(labels), 0.0) + amount

    def set(self, name: str, value: float, **labels) -> None:
        with self._lock:
            self._gauges.setdefault(name, {})[_label_key(labels)] = float(value)

    def observe(self, name: str, value: float, **labels) -> None:
        with self._lock:
            row = self._hist.setdefault(name, {}).setdefault(
                _label_key(labels), [[0] * len(_BUCKETS), 0.0, 0]
            )
            for i, bound in enumerate(_BUCKETS):
                if value <= bound:
                    row[0][i] += 1
            row[1] += value
            row[2] += 1

    def value(self, name: str, **labels) -> float:
        with self._lock:
            return self._counters.get(name, {}).get(_label_key(labels), 0.0)

    def render(self) -> str:
        lines = []
        with self._lock:
            for kind, store in (("counter", self._counters), ("gauge", self._gauges)):
                for name in sorted(store):
                    lines.append(f"# HELP {name} {self._help.get(name, name)}")
                    lines.append(f"# TYPE {name} {kind}")
                    for key, val in sorted(store[name].items()):
                        lines.append(f"{name}{_fmt_labels(key)} {val:g}")
            for name in sorted(self._hist):
                lines.append(f"# HELP {name} {self._help.get(name, name)}")
                lines.append(f"# TYPE {name} histogram")
                for key, (counts, total, n) in sorted(self._hist[name].items()):
                    for bound, count in zip(_BUCKETS, counts):
                        le = 'le="%g"' % bound
                        lines.append(f"{name}_bucket{_fmt_labels(key, le)} {count}")
                    inf = 'le="+Inf"'
                    lines.append(f"{name}_bucket{_fmt_labels(key, inf)} {n}")
                    lines.append(f"{name}_sum{_fmt_labels(key)} {total:g}")
                    lines.append(f"{name}_count{_fmt_labels(key)} {n}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()
METRICS.describe("microbesim_operation_seconds", "Duration of application operations")
METRICS.describe("microbesim_operation_errors_total", "Failed application operations")
METRICS.describe(
    "microbesim_model_loads_total", "Model loads by source (pretrained, cache, trained)"
)
METRICS.describe("microbesim_audit_write_failures_total", "Audit events that could not be written")
METRICS.describe("microbesim_uploads_total", "Uploaded files by outcome")


@contextmanager
def timed(operation: str, **fields) -> Iterator[None]:
    """Record duration/failure of ``operation`` as metrics and a structured log line."""
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        METRICS.inc("microbesim_operation_errors_total", operation=operation)
        log_event(f"{operation}.failed", level=logging.ERROR, error=type(exc).__name__, **fields)
        raise
    finally:
        seconds = time.perf_counter() - started
        METRICS.observe("microbesim_operation_seconds", seconds, operation=operation)
    log_event(f"{operation}.done", seconds=round(seconds, 3), **fields)


def init_error_tracking() -> bool:
    """Enable Sentry when configured; returns whether tracking is active. No hard dependency."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk  # type: ignore
    except ImportError:
        log_event(
            "error_tracking.unavailable", level=logging.WARNING, reason="sentry_sdk not installed"
        )
        return False
    sentry_sdk.init(dsn=dsn, send_default_pii=False, traces_sample_rate=0.0)
    return True


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def readiness() -> Dict[str, object]:
    """Cheap readiness checks: writable state dir, ledger reachable, pretrained models match."""
    checks: Dict[str, object] = {}
    from src import ledger, model_cache, pretrained

    try:
        ledger.STATE_DIR.mkdir(parents=True, exist_ok=True)
        probe = ledger.STATE_DIR / f".ready-{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
        checks["state_dir"] = "ok"
    except OSError as exc:
        checks["state_dir"] = f"error: {exc.strerror or exc}"
    try:
        ledger.initialize_ledger()
        checks["ledger"] = "ok"
    except Exception as exc:  # any failure means not ready
        checks["ledger"] = f"error: {type(exc).__name__}"
    manifest = pretrained.PRETRAINED_DIR / pretrained.MANIFEST_NAME
    if not manifest.exists():
        checks["pretrained_models"] = "missing"
    else:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        same = (
            data.get("environment") == model_cache.environment_fingerprint()
            and data.get("source_digest") == model_cache.source_digest()
        )
        checks["pretrained_models"] = "ok" if same else "stale"
    # Models are optional for readiness (the app trains on demand); state dir and ledger are not.
    checks["ready"] = checks["state_dir"] == "ok" and checks["ledger"] == "ok"
    return checks


class _Handler(BaseHTTPRequestHandler):
    server_version = "microbesim-ops"

    def log_message(self, *args) -> None:  # silence default stderr access log
        return

    def _authorised(self) -> bool:
        token = os.environ.get("MICROBESIM_OPS_TOKEN")
        if not token:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {token}")

    def _send(self, status: int, body: str, content_type: str = "application/json") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/healthz":
            return self._send(200, json.dumps({"status": "alive"}))
        if not self._authorised():
            return self._send(401, json.dumps({"error": "unauthorised"}))
        if self.path == "/readyz":
            report = readiness()
            return self._send(200 if report["ready"] else 503, json.dumps(report))
        if self.path == "/metrics":
            return self._send(200, METRICS.render(), "text/plain; version=0.0.4")
        return self._send(404, json.dumps({"error": "not found"}))


_server_lock = threading.Lock()
_server: Optional[ThreadingHTTPServer] = None


def start_ops_server(host: Optional[str] = None, port: Optional[int] = None) -> Optional[int]:
    """Start the ops endpoints once per process; returns the bound port or None when disabled."""
    global _server
    port = port if port is not None else int(os.environ.get("MICROBESIM_OPS_PORT", "0") or 0)
    if not port and port != 0:
        return None
    if port == 0 and "MICROBESIM_OPS_PORT" not in os.environ and host is None:
        return None
    with _server_lock:
        if _server is not None:
            return _server.server_address[1]
        _server = ThreadingHTTPServer(
            (host or os.environ.get("MICROBESIM_OPS_HOST", "127.0.0.1"), port), _Handler
        )
        threading.Thread(target=_server.serve_forever, name="microbesim-ops", daemon=True).start()
        log_event("ops_server.started", port=_server.server_address[1])
        return _server.server_address[1]


def stop_ops_server() -> None:
    global _server
    with _server_lock:
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None
