"""Run long operations (training, pretraining) in a background worker with progress.

One worker thread by default so heavy fits cannot starve the UI or each other. Jobs with the same name are
de-duplicated while one is running. State lives in memory of this process: fine for a single Streamlit
container; running several replicas needs an external queue (e.g. RQ/Celery), which is not implemented.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.observability import METRICS, log_event

Progress = Callable[[int, int, str], None]


@dataclass
class Job:
    id: int
    name: str
    status: str = "queued"  # queued | running | done | failed
    done: int = 0
    total: int = 0
    message: str = ""
    error: Optional[str] = None
    started: Optional[float] = None
    finished: Optional[float] = None
    result: object = field(default=None, repr=False)

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total else (1.0 if self.status == "done" else 0.0)


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="microbesim-job"
        )
        self._jobs: Dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def submit(self, name: str, fn: Callable[[Progress], object]) -> Job:
        """Queue ``fn(progress)``; returns the already-active job of the same name if there is one."""
        with self._lock:
            for job in self._jobs.values():
                if job.name == name and job.status in ("queued", "running"):
                    return job
            job = Job(id=next(self._ids), name=name)
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Progress], object]) -> None:
        def progress(done: int, total: int, message: str = "") -> None:
            job.done, job.total, job.message = done, total, message

        job.status, job.started = "running", time.time()
        log_event("job.started", job=job.name, job_id=job.id)
        try:
            job.result = fn(progress)
            job.status = "done"
        except Exception as exc:  # report, never crash the worker
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            METRICS.inc("microbesim_operation_errors_total", operation=f"job.{job.name}")
            log_event(
                "job.failed", level=logging.ERROR, job=job.name, job_id=job.id, error=job.error
            )
        finally:
            job.finished = time.time()
            METRICS.observe(
                "microbesim_operation_seconds",
                job.finished - (job.started or job.finished),
                operation=f"job.{job.name}",
            )
            log_event("job.finished", job=job.name, job_id=job.id, status=job.status)

    def get(self, job_id: int) -> Optional[Job]:
        return self._jobs.get(job_id)

    def latest(self, name: str) -> Optional[Job]:
        matches: List[Job] = [j for j in self._jobs.values() if j.name == name]
        return max(matches, key=lambda j: j.id) if matches else None

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
