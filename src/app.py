"""MicrobeSim - Streamlit entry point.  Run with:  streamlit run src/app.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="MicrobeSim", page_icon="🧫", layout="wide", initial_sidebar_state="collapsed"
)

from src.ui import (
    page_assays,
    page_hurdles,
    page_overview,
    page_runs,
    page_simulator,
    pretrained_panel,
)  # noqa: E402
from src.ui.auth import logout, render_login  # noqa: E402
from src.observability import (
    configure_logging,
    init_error_tracking,
    log_event,
    start_ops_server,
)  # noqa: E402
from src.ui.theme import inject_css, render_topbar  # noqa: E402


@st.cache_resource(show_spinner=False)
def _process_init() -> bool:
    """Once per server process: JSON logging, optional error tracking/ops endpoints, model registry sync."""
    configure_logging()
    init_error_tracking()
    start_ops_server()
    try:
        from src.pretrained import sync_registry

        log_event("registry.sync", added=sync_registry())
    except Exception as exc:  # the registry is informational; never block start-up
        log_event("registry.sync_failed", level=40, error=type(exc).__name__)
    return True


_process_init()

inject_css()
if not render_login():
    st.stop()
pretrained_panel.render()


def overview() -> None:
    page_overview.render(PAGES)


PAGES = {
    "overview": st.Page(overview, title="Overview", default=True),
    "assays": st.Page(page_assays.render, title="Fit growth curves", url_path="assays"),
    "hurdles": st.Page(page_hurdles.render, title="Predict growth rate", url_path="hurdles"),
    "simulator": st.Page(
        page_simulator.render, title="Simulate a cold-chain failure", url_path="simulator"
    ),
    "runs": st.Page(page_runs.render, title="Saved runs", url_path="runs"),
}

current_page = st.navigation(list(PAGES.values()), position="hidden")
active_key = next((key for key, page in PAGES.items() if page == current_page), "overview")
render_topbar(PAGES, active_key, st.session_state["username"], logout)
current_page.run()
