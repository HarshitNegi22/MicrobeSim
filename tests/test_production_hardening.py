import io
import json
import sqlite3
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

import pandas as pd
import pytest

from src import audit, guardrails, jobs, observability, oidc, registry, upload_security
from src.accounts import AccountError, authenticate, create_account, ensure_external_account
from src.ingestion import IngestionError, read_table


@pytest.fixture()
def db(tmp_path):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    return state / "ledger.sqlite3"


# ----------------------------------------------------------------- observability
def test_json_logging_includes_structured_fields(capsys):
    observability.configure_logging("INFO")
    observability.log_event("unit.test", rows=3)
    line = capsys.readouterr().err.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["event"] == "unit.test" and data["rows"] == 3 and data["level"] == "INFO"


def test_metrics_render_prometheus_text():
    m = observability.Metrics()
    m.inc("things_total", 2, kind="a")
    m.set("level", 7)
    m.observe("op_seconds", 0.2, op="x")
    text = m.render()
    assert 'things_total{kind="a"} 2' in text and "level 7" in text
    assert 'op_seconds_bucket{op="x",le="0.5"} 1' in text and 'op_seconds_count{op="x"} 1' in text


def test_timed_records_success_and_failure():
    before = observability.METRICS.value("microbesim_operation_errors_total", operation="unit_fail")
    with observability.timed("unit_ok"):
        pass
    with pytest.raises(RuntimeError):
        with observability.timed("unit_fail"):
            raise RuntimeError("boom")
    assert (
        observability.METRICS.value("microbesim_operation_errors_total", operation="unit_fail")
        == before + 1
    )
    assert 'operation="unit_ok"' in observability.METRICS.render()


def _get(port, path, token=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_ops_server_health_ready_metrics_and_token(monkeypatch):
    monkeypatch.delenv("MICROBESIM_OPS_TOKEN", raising=False)
    port = observability.start_ops_server("127.0.0.1", 0)
    try:
        assert _get(port, "/healthz")[0] == 200
        status, body = _get(port, "/readyz")
        assert status in (200, 503) and "state_dir" in json.loads(body)
        assert _get(port, "/metrics")[0] == 200
        assert _get(port, "/nope")[0] == 404
        monkeypatch.setenv("MICROBESIM_OPS_TOKEN", "s3cret")
        assert _get(port, "/metrics")[0] == 401
        assert _get(port, "/metrics", "wrong")[0] == 401
        assert _get(port, "/metrics", "s3cret")[0] == 200
        assert _get(port, "/healthz")[0] == 200  # liveness needs no secret
    finally:
        observability.stop_ops_server()


# ----------------------------------------------------------------- audit
def test_audit_chain_verifies_and_blocks_edits(db):
    for i in range(3):
        assert audit.record("u1", "run.save", f"r{i}", detail={"i": i}, db_path=db)
    assert audit.verify_chain(db) == (True, None)
    assert [e["target"] for e in audit.list_events(db_path=db)] == ["r2", "r1", "r0"]
    raw = sqlite3.connect(db)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute("UPDATE audit_log SET actor = 'mallory' WHERE id = 1")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute("DELETE FROM audit_log WHERE id = 2")
    raw.close()


def test_audit_detects_tampering_even_if_triggers_are_removed(db):
    for i in range(3):
        audit.record("u1", "run.save", f"r{i}", db_path=db)
    raw = sqlite3.connect(db)
    raw.execute("DROP TRIGGER audit_log_no_update")
    raw.execute("UPDATE audit_log SET target = 'forged' WHERE id = 2")
    raw.commit()
    raw.close()
    assert audit.verify_chain(db) == (False, 2)


def test_audit_failure_never_raises(tmp_path):
    before = observability.METRICS.value("microbesim_audit_write_failures_total")
    assert audit.record("u", "x", db_path=tmp_path) is False  # a directory cannot be a database
    assert observability.METRICS.value("microbesim_audit_write_failures_total") == before + 1


def test_ledger_changes_are_audited(db):
    from src import ledger

    ledger.initialize_ledger(db)
    account = ensure_external_account("oidc-" + "a" * 32, "oidc-aaaaaaaaaaaa", db)
    inserted, run_id = ledger.save_run_record(
        account.user_id, "prediction", "t", "s", "m", "1", {"a": 1}, db
    )
    assert inserted
    assert ledger.delete_run(account.user_id, run_id, db) is True
    actions = [e["action"] for e in audit.list_events(db_path=db)]
    assert "run.save" in actions and "run.delete" in actions
    assert audit.verify_chain(db)[0]


# ----------------------------------------------------------------- registry
def _register(db, signature="sig1", author="builder"):
    return registry.register_model(
        signature,
        "Test organism (real)",
        "real",
        "Test organism",
        "abc123",
        120,
        ["rf", "ensemble"],
        {"n_repeats": 5},
        [{"Model": "Random Forest", "R2": 0.71, "RMSE": 0.2, "Af": 1.4, "Bf": 1.0}],
        "digest" * 4,
        {"scikit-learn": "1.9.1"},
        author=author,
        db_path=db,
    )


def test_registry_register_is_idempotent_and_keeps_approval(db):
    assert _register(db) is True and _register(db) is False
    registry.set_status("sig1", "approved", "alice", "checked parity plots", db)
    assert _register(db) is False
    assert registry.status_of("sig1", db) == "approved"


def test_registry_approval_rules(db, monkeypatch):
    _register(db)
    with pytest.raises(registry.RegistryError, match="note"):
        registry.set_status("sig1", "approved", "alice", "", db)
    with pytest.raises(registry.RegistryError, match="four-eyes"):
        registry.set_status("sig1", "approved", "builder", "ok", db)
    with pytest.raises(registry.RegistryError, match="unknown"):
        registry.set_status("nope", "approved", "alice", "ok", db)
    with pytest.raises(registry.RegistryError):
        registry.set_status("sig1", "draft", "alice", "ok", db)
    monkeypatch.setenv("MICROBESIM_FOUR_EYES", "0")
    assert (
        registry.set_status("sig1", "approved", "builder", "solo install", db)["status"]
        == "approved"
    )
    assert "model.approved" in [e["action"] for e in audit.list_events(db_path=db)]


def test_model_card_has_data_validation_and_lineage(db):
    _register(db, "sig1")
    _register(db, "sig2")
    card = registry.model_card("sig2", db)
    for text in (
        "Model card: Test organism (real)",
        "abc123",
        "Random Forest",
        "0.71",
        "scikit-learn 1.9.1",
        "`sig1`",
    ):
        assert text in card
    assert "not reviewed" in card


# ----------------------------------------------------------------- upload security
def _zip(names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"x")
    return buffer.getvalue()


def test_upload_accepts_plain_csv_and_workbook():
    ok = upload_security.inspect_upload("data.csv", b"a,b\n1,2\n")
    assert len(ok["sha256"]) == 64
    frame = pd.DataFrame({"a": [1]})
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    assert upload_security.inspect_upload("book.xlsx", buffer.getvalue())["extension"] == ".xlsx"


@pytest.mark.parametrize(
    "name,data",
    [
        ("evil.exe", b"MZ"),
        ("noext", b"a,b"),
        ("empty.csv", b""),
        ("binary.csv", b"a,b\x00\x01\n"),
        ("fake.xlsx", b"not a zip at all"),
        ("macro.xlsm", _zip(["[Content_Types].xml", "xl/vbaProject.bin"])),
        ("embed.xlsx", _zip(["[Content_Types].xml", "xl/embeddings/oleObject1.bin"])),
        ("links.xlsx", _zip(["[Content_Types].xml", "xl/externalLinks/externalLink1.xml"])),
        ("traversal.xlsx", _zip(["[Content_Types].xml", "../../etc/passwd"])),
        ("fake.xls", b"hello"),
        ("macro.xls", bytes.fromhex("D0CF11E0A1B11AE1") + b"..._VBA_PROJECT..."),
    ],
)
def test_upload_rejects_hostile_or_mismatched_files(name, data):
    with pytest.raises(upload_security.UploadRejected):
        upload_security.inspect_upload(name, data)


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(upload_security, "MAX_BYTES", 10)
    with pytest.raises(upload_security.UploadRejected, match="limit"):
        upload_security.inspect_upload("big.csv", b"a,b\n" * 10)


def test_scanner_hook_fails_closed(monkeypatch):
    ok = f"{sys.executable} -c 'import sys; sys.exit(0)' {{path}}"
    monkeypatch.setenv("MICROBESIM_UPLOAD_SCAN_CMD", ok)
    upload_security.inspect_upload("data.csv", b"a,b\n1,2\n")
    monkeypatch.setenv(
        "MICROBESIM_UPLOAD_SCAN_CMD", f"{sys.executable} -c 'import sys; sys.exit(1)' {{path}}"
    )
    with pytest.raises(upload_security.UploadRejected, match="virus scanner"):
        upload_security.inspect_upload("data.csv", b"a,b\n1,2\n")
    monkeypatch.setenv("MICROBESIM_UPLOAD_SCAN_CMD", "/no/such/scanner {path}")
    with pytest.raises(upload_security.UploadRejected, match="could not be scanned"):
        upload_security.inspect_upload("data.csv", b"a,b\n1,2\n")


def test_read_table_applies_upload_checks():
    macro = io.BytesIO(_zip(["[Content_Types].xml", "xl/vbaProject.bin"]))
    macro.name = "x.xlsm"
    with pytest.raises(IngestionError, match="macros"):
        read_table(macro)
    csv = io.BytesIO(b"temperature_celsius,ph\n20,6.5\n")
    csv.name = "ok.csv"
    assert len(read_table(csv)) == 1


# ----------------------------------------------------------------- guardrails
ENVELOPE = {"temperature_celsius": (5.0, 40.0), "ph": (4.5, 8.0), "water_activity": (0.9, 1.0)}


def _assess(T=25.0, ph=6.5, aw=0.98, **kw):
    base = dict(
        mean=1.0, lower=0.7, upper=1.3, model_status="approved", validation_r2=0.8, n_train=100
    )
    base.update(kw)
    return guardrails.assess(ENVELOPE, T, ph, aw, **base)


def test_guardrail_tiers():
    assert _assess().tier == "validated"
    assert _assess(T=55).tier == "out_of_scope" and "temperature" in _assess(T=55).reasons[0]
    assert _assess(model_status="draft").tier == "exploratory"
    assert _assess(model_status=None).tier == "exploratory"
    assert _assess(validation_r2=0.2).tier == "exploratory"
    assert _assess(n_train=10).tier == "exploratory"
    assert _assess(lower=0.0, upper=3.0).tier == "exploratory"
    near_edge = _assess(T=6.0)
    assert near_edge.domain == "edge" and near_edge.tier == "exploratory"
    assert all(a.human_review_required for a in (_assess(), _assess(T=55)))


# ----------------------------------------------------------------- jobs
def test_job_runs_with_progress_and_dedupes():
    manager = jobs.JobManager()
    release = []

    def work(progress):
        for i in range(3):
            progress(i + 1, 3, f"step {i + 1}")
        while not release:
            time.sleep(0.01)
        return "done"

    job = manager.submit("t", work)
    assert manager.submit("t", work) is job  # same name while active -> same job
    deadline = time.time() + 5
    while job.done < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert job.done == 3 and job.status == "running" and job.fraction == 1.0
    release.append(1)
    deadline = time.time() + 5
    while job.status != "done" and time.time() < deadline:
        time.sleep(0.01)
    assert job.status == "done" and job.result == "done"
    assert manager.latest("t") is job
    manager.shutdown()


def test_job_failure_is_captured():
    manager = jobs.JobManager()

    def broken(progress):
        raise ValueError("bad input")

    job = manager.submit("bad", broken)
    deadline = time.time() + 5
    while job.status not in ("failed", "done") and time.time() < deadline:
        time.sleep(0.01)
    assert job.status == "failed" and "bad input" in job.error
    manager.shutdown()


# ----------------------------------------------------------------- OIDC
CLAIMS = {
    "sub": "123",
    "iss": "https://idp.example",
    "email": "Ann@Corp.example",
    "email_verified": True,
    "name": "Ann",
}


def test_oidc_identity_rules(monkeypatch):
    for name in (
        "MICROBESIM_OIDC_ALLOWED_DOMAINS",
        "MICROBESIM_OIDC_ALLOWED_EMAILS",
        "MICROBESIM_OIDC_ALLOW_ANY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(oidc.OidcRejected, match="allowed"):  # no allow-list configured -> deny
        oidc.identity_from_claims(CLAIMS)
    monkeypatch.setenv("MICROBESIM_OIDC_ALLOWED_DOMAINS", "corp.example, other.example")
    identity = oidc.identity_from_claims(CLAIMS)
    assert identity.email == "ann@corp.example" and identity.user_id.startswith("oidc-")
    assert identity.user_id == oidc.identity_from_claims(CLAIMS).user_id  # stable
    other = oidc.identity_from_claims({**CLAIMS, "sub": "999"})
    assert other.user_id != identity.user_id
    with pytest.raises(oidc.OidcRejected, match="verified"):
        oidc.identity_from_claims({**CLAIMS, "email_verified": False})
    with pytest.raises(oidc.OidcRejected, match="verified"):
        oidc.identity_from_claims({k: v for k, v in CLAIMS.items() if k != "email_verified"})
    with pytest.raises(oidc.OidcRejected, match="allowed"):
        oidc.identity_from_claims({**CLAIMS, "email": "eve@evil.example"})
    with pytest.raises(oidc.OidcRejected, match="usable"):
        oidc.identity_from_claims({**CLAIMS, "sub": ""})
    monkeypatch.delenv("MICROBESIM_OIDC_ALLOWED_DOMAINS")
    monkeypatch.setenv("MICROBESIM_OIDC_ALLOWED_EMAILS", "ann@corp.example")
    assert oidc.identity_from_claims(CLAIMS).email == "ann@corp.example"


def test_sso_accounts_can_never_log_in_with_a_password(db):
    account = ensure_external_account("oidc-" + "b" * 32, "oidc-bbbbbbbbbbbb", db)
    assert ensure_external_account(account.user_id, account.username, db) == account  # idempotent
    for password in ("", "!external", "correct horse battery staple"):
        assert authenticate(account.username, password, db) is None
    with pytest.raises(AccountError, match="reserved"):
        create_account("oidc-attacker", "a-long-enough-password", db)


def test_auth_mode_defaults_to_local(monkeypatch):
    monkeypatch.delenv("MICROBESIM_AUTH_MODE", raising=False)
    assert oidc.auth_mode() == "local"
    monkeypatch.setenv("MICROBESIM_AUTH_MODE", "OIDC")
    assert oidc.auth_mode() == "oidc"


def test_app_session_starts_ops_endpoints_when_configured(monkeypatch):
    """The ops server starts with the first Streamlit session (the script is not run before that)."""
    import socket

    from streamlit.testing.v1 import AppTest

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("MICROBESIM_OPS_PORT", str(port))
    monkeypatch.delenv("MICROBESIM_OPS_TOKEN", raising=False)
    try:
        app = Path(__file__).resolve().parents[1] / "src" / "app.py"
        at = AppTest.from_file(str(app), default_timeout=60).run()
        assert not at.exception
        assert _get(port, "/healthz")[0] == 200
        assert "microbesim" in _get(port, "/metrics")[1] or _get(port, "/metrics")[0] == 200
    finally:
        observability.stop_ops_server()
