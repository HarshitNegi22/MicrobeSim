"""Runs the real Streamlit app headlessly (skipped if streamlit is not installed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


def test_default_page_renders_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception


def test_assay_page_defaults_to_sample_and_controls_affect_processing(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.ui import page_assays

    saved_payloads = []

    def capture_run(*args):
        saved_payloads.append(args[6])
        return True, f"ui-test-{len(saved_payloads)}"

    monkeypatch.setattr(page_assays, "save_run_record", capture_run)
    at = AppTest.from_string("from src.ui.page_assays import render\nrender()", default_timeout=120)
    at.session_state["user_id"] = "ui_assay_regression"
    at.run()

    assert not at.exception
    assert at.selectbox[0].value == page_assays.SRC_SAMPLE
    assert len(at.get("plotly_chart")) >= 1
    assert any("Cleaning applied:" in caption.value for caption in at.caption)
    assert not saved_payloads
    next(button for button in at.button if button.label.startswith("Save ")).click().run()
    assert not at.exception
    assert saved_payloads
    assert all(not payload["cleaning"]["override_metadata"] for payload in saved_payloads)

    override = next(
        checkbox
        for checkbox in at.checkbox
        if checkbox.label == "Use these defaults even when the file provides metadata"
    )
    saved_before_override = len(saved_payloads)
    override.set_value(True).run()
    assert not at.exception
    assert len(saved_payloads) == saved_before_override
    next(button for button in at.button if button.label.startswith("Save ")).click().run()
    assert not at.exception
    overridden_payloads = saved_payloads[saved_before_override:]
    assert overridden_payloads
    assert all(payload["cleaning"]["override_metadata"] for payload in overridden_payloads)
    assert all(payload["organism"] == "Unspecified organism" for payload in overridden_payloads)
    assert all(payload["temperature_celsius"] == 37.0 for payload in overridden_payloads)

    smoothing = next(
        checkbox for checkbox in at.checkbox if checkbox.label == "Smooth noise (Savitzky-Golay)"
    )
    smoothing.set_value(False).run()
    assert not at.exception
    assert any("Smoothing off" in caption.value for caption in at.caption)


def test_hurdle_page_renders_predictions_and_charts(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.ui import page_hurdles

    saved_records = []

    def capture_run(*args):
        saved_records.append(args)
        return True, str(args[2])

    monkeypatch.setattr(page_hurdles, "save_run_record", capture_run)
    at = AppTest.from_string(
        "from src.ui.page_hurdles import render\nrender()",
        default_timeout=180,
    )
    at.session_state["user_id"] = "ui_hurdle_regression"
    at.run()

    assert not at.exception
    assert not at.error
    assert len(at.get("plotly_chart")) >= 3
    assert any("How growth rate changes" in item.value for item in at.markdown)
    assert not saved_records
    save_button = next(
        button for button in at.button if button.label == "Save this prediction to history"
    )
    save_button.click().run()
    assert not at.exception
    assert saved_records


def test_simulator_page_saves_only_after_explicit_action(monkeypatch):
    import numpy as np
    import pandas as pd
    from streamlit.testing.v1 import AppTest

    from src.ui import page_simulator

    class FixedGrowthModel:
        def predict(self, X):
            return np.full(len(X), 0.2)

        def predict_interval(self, X):
            mean = self.predict(X)
            return mean, mean * 0.8, mean * 1.2

    class FixedBenchmark:
        models = {"rf": FixedGrowthModel()}
        validation_description = "3 grouped holdouts by study_id"
        loaded_from_disk = False
        errors = {}

    frame = pd.DataFrame(
        {
            "temperature_celsius": [4.0, 25.0],
            "ph": [6.5, 6.5],
            "water_activity": [0.98, 0.98],
        }
    )
    saved_records = []
    monkeypatch.setattr(page_simulator, "list_user_profiles", lambda *args: [])
    monkeypatch.setattr(
        page_simulator,
        "dataset_selector",
        lambda prefix: ("Test dataset", frame, FixedBenchmark()),
    )

    def capture_run(*args):
        saved_records.append(args)
        return True, "sim-ui-test"

    monkeypatch.setattr(page_simulator, "save_run_record", capture_run)
    at = AppTest.from_string(
        "from src.ui.page_simulator import render\nrender()",
        default_timeout=120,
    )
    at.session_state["user_id"] = "ui_simulator_regression"
    at.run()

    assert not at.exception
    assert not saved_records
    save_button = next(
        button for button in at.button if button.label == "Save this simulation to history"
    )
    save_button.click().run()
    assert not at.exception
    assert len(saved_records) == 1


def test_account_creation_and_logout_returns_to_login(tmp_path, monkeypatch):
    from src.accounts import (
        authenticate,
        create_account,
        initialize_accounts,
    )
    from src.ui import auth

    db_path = tmp_path / "ui-accounts.sqlite3"
    monkeypatch.setattr(auth, "initialize_accounts", lambda: initialize_accounts(db_path))
    monkeypatch.setattr(
        auth, "authenticate", lambda username, password: authenticate(username, password, db_path)
    )
    monkeypatch.setattr(
        auth,
        "create_account",
        lambda username, password: create_account(username, password, db_path),
    )
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.tabs[1].run()
    inputs = {element.label: element for element in at.text_input}
    inputs["Choose a username"].set_value("ui_smoke_user")
    inputs["Choose a password"].set_value("ui-smoke-password-2026")
    inputs["Confirm password"].set_value("ui-smoke-password-2026")
    next(button for button in at.button if button.label == "Create account").click().run()
    assert not at.exception
    assert at.session_state["username"] == "ui_smoke_user"

    next(button for button in at.button if button.label == "Sign out").click().run()
    assert not at.exception
    assert "user_id" not in at.session_state
    fields_after_logout = {element.label: element for element in at.text_input}
    fields_after_logout["Username"].set_value("ui_smoke_user")
    fields_after_logout["Password"].set_value("ui-smoke-password-2026")
    next(button for button in at.button if button.label == "Sign in").click().run()
    assert not at.exception
    assert at.session_state["username"] == "ui_smoke_user"
    assert any(button.label == "Sign out" for button in at.button)


def test_authenticated_session_idle_and_absolute_expiry():
    from src.ui.auth import _session_is_valid

    assert _session_is_valid(100.0, 200.0, 300.0)
    assert not _session_is_valid(100.0, 100.0, 100.0 + 30 * 60)
    assert not _session_is_valid(100.0, 1000.0, 100.0 + 12 * 60 * 60)
    assert not _session_is_valid(None, 100.0, 200.0)
