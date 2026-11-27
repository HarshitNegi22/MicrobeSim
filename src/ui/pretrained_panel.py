"""Sidebar status for pretrained models, with a one-click background job to prepare them."""

from __future__ import annotations

import streamlit as st

from src import registry
from src.jobs import get_manager
from src.pretrained import pretrain, pretrained_status

_JOB = "pretrain_models"


@st.cache_resource(ttl=300, show_spinner=False)
def _status() -> dict:
    return pretrained_status()


@st.fragment(run_every=2)
def _job_progress() -> None:
    job = get_manager().latest(_JOB)
    if job is None:
        return
    if job.status in ("queued", "running"):
        st.progress(
            job.fraction,
            text=f"{job.done}/{job.total}: {job.message}" if job.total else "Starting...",
        )
    elif job.status == "failed":
        st.error(f"Preparing models failed: {job.error}")
    else:
        _status.clear()
        st.success("Pretrained models are ready. Reload the page to use them.")


def _review_summary() -> None:
    try:
        counts: dict = {}
        for entry in registry.list_models():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    except Exception:  # the registry is informational here
        return
    if counts:
        st.caption("Model review: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))


def render() -> None:
    status = _status()
    with st.sidebar:
        st.subheader("Pretrained models")
        if status["state"] == "ready":
            st.success(f"Ready: {status['detail']}. Pages load instantly.")
            _review_summary()
            return
        reasons = {
            "missing": "No pretrained models are installed for this environment.",
            "stale": f"Saved models do not match this installation ({status['detail']}).",
            "partial": f"Only part of the models are pretrained ({status['detail']}).",
        }
        st.warning(
            reasons.get(status["state"], status["detail"])
            + " Datasets without a pretrained model are trained the first time you open them. "
            "To prepare everything once, use the button below or run `python scripts/pretrain_models.py`."
        )
        if st.button("Prepare pretrained models now", key="pretrain_now"):
            get_manager().submit(
                _JOB, lambda progress: pretrain(progress=lambda d, t, r: progress(d, t, r["label"]))
            )
        _job_progress()
