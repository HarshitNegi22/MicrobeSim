import os
import stat

import pytest

from src.accounts import create_account
from src.ledger import (
    clear_runs,
    count_run_summaries,
    count_runs_by_type,
    create_project,
    delete_run,
    get_run,
    list_projects,
    list_run_summaries,
    list_user_profiles,
    list_runs,
    save_run,
    save_run_record,
    save_user_profile,
    update_run_metadata,
)


def test_runs_are_private_and_deduplicated_per_account(tmp_path):
    db_path = tmp_path / "analysis.sqlite3"
    payload = {"organism": "Listeria monocytogenes", "mu_max_per_h": 0.16}
    account_one = create_account("analyst_one", "very-secure-password-1", db_path)
    account_two = create_account("analyst_two", "very-secure-password-2", db_path)

    assert save_run(
        account_one.user_id,
        "assay",
        "Listeria fit",
        "Gompertz fit",
        "μmax",
        "0.160 / h",
        payload,
        db_path,
    )
    assert not save_run(
        account_one.user_id,
        "assay",
        "Listeria fit",
        "Gompertz fit",
        "μmax",
        "0.160 / h",
        payload,
        db_path,
    )
    assert save_run(
        account_two.user_id,
        "assay",
        "Listeria fit",
        "Gompertz fit",
        "μmax",
        "0.160 / h",
        payload,
        db_path,
    )

    runs = list_runs(account_one.user_id, db_path=db_path)
    assert len(runs) == 1
    assert runs[0]["payload"] == payload
    assert len(list_runs(account_two.user_id, db_path=db_path)) == 1
    assert not delete_run(account_two.user_id, runs[0]["run_id"], db_path)
    assert delete_run(account_one.user_id, runs[0]["run_id"], db_path)
    assert list_runs(account_one.user_id, db_path=db_path) == []
    assert len(list_runs(account_two.user_id, db_path=db_path)) == 1


def test_clear_runs_deletes_only_the_current_accounts_records(tmp_path):
    db_path = tmp_path / "analysis.sqlite3"
    account_one = create_account("analyst_one", "very-secure-password-1", db_path)
    account_two = create_account("analyst_two", "very-secure-password-2", db_path)
    for title in ("fit one", "fit two"):
        save_run(
            account_one.user_id,
            "assay",
            title,
            "completed",
            "μmax",
            "0.1 / h",
            {"title": title},
            db_path,
        )
    save_run(
        account_two.user_id,
        "assay",
        "private fit",
        "completed",
        "μmax",
        "0.1 / h",
        {"title": "private fit"},
        db_path,
    )

    assert clear_runs(account_one.user_id, db_path) == 2
    assert list_runs(account_one.user_id, db_path=db_path) == []
    assert len(list_runs(account_two.user_id, db_path=db_path)) == 1


def test_pre_account_history_is_preserved_but_not_assigned_to_a_user(tmp_path):
    import sqlite3

    from src.accounts import initialize_accounts

    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE analysis_runs (
                run_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                run_type TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                metric_label TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """)
        connection.execute(
            "INSERT INTO analysis_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-id",
                "legacy-hash",
                "assay",
                "Old run",
                "Old result",
                "μmax",
                "0.1 / h",
                "2025-01-01T00:00:00+00:00",
                '{"old": true}',
            ),
        )

    initialize_accounts(db_path)
    account = create_account("new_analyst", "very-secure-password-3", db_path)
    save_run(account.user_id, "assay", "New run", "Done", "μmax", "0.2 / h", {}, db_path)

    assert [run["title"] for run in list_runs(account.user_id, db_path=db_path)] == ["New run"]
    with sqlite3.connect(db_path) as connection:
        old_owner = connection.execute(
            "SELECT user_id FROM analysis_runs WHERE run_id = 'legacy-id'"
        ).fetchone()
    assert old_owner == (None,)


def test_history_summaries_paginate_and_load_payload_on_demand(tmp_path):
    db_path = tmp_path / "analysis.sqlite3"
    account = create_account("history_owner", "very-secure-password-4", db_path)
    for index in range(3):
        save_run(
            account.user_id,
            "simulation",
            f"Excursion {index}",
            f"Peak temperature {index + 4}",
            "Threshold",
            "No breach",
            {"trajectory": [{"time": step} for step in range(100)]},
            db_path,
        )

    page = list_run_summaries(account.user_id, limit=2, offset=1, db_path=db_path)
    assert len(page) == 2
    assert all("payload" not in run and "payload_json" not in run for run in page)
    assert count_run_summaries(account.user_id, "simulation", db_path=db_path) == 3
    assert count_run_summaries(account.user_id, search="peak temperature 5", db_path=db_path) == 1
    assert count_runs_by_type(account.user_id, db_path)["simulation"] == 3
    loaded = get_run(account.user_id, page[0]["run_id"], db_path)
    assert loaded is not None and len(loaded["payload"]["trajectory"]) == 100
    assert get_run("another-user", page[0]["run_id"], db_path) is None


def test_projects_notes_and_profiles_are_account_scoped(tmp_path):
    db_path = tmp_path / "workspace.sqlite3"
    owner = create_account("workspace_owner", "very-secure-password-5", db_path)
    other = create_account("workspace_other", "very-secure-password-6", db_path)
    project_id = create_project(owner.user_id, "Cold-chain study", db_path)
    run_id = save_run_record(
        owner.user_id,
        "simulation",
        "Baseline",
        "No breach",
        "Threshold",
        "No breach",
        {"trajectory": [1, 2]},
        db_path,
    )[1]

    assert update_run_metadata(owner.user_id, run_id, project_id, "Pilot shipment", db_path)
    assert count_run_summaries(owner.user_id, project_id=project_id, db_path=db_path) == 1
    assert count_run_summaries(owner.user_id, project_id="", db_path=db_path) == 0
    assert (
        list_run_summaries(owner.user_id, project_id=project_id, db_path=db_path)[0]["notes"]
        == "Pilot shipment"
    )
    with pytest.raises(ValueError, match="does not belong"):
        update_run_metadata(other.user_id, run_id, project_id, "Cross-user", db_path)

    profile_id = save_user_profile(
        owner.user_id, "product", "Dairy test", {"ph": 6.4, "water_activity": 0.97}, db_path
    )
    assert list_user_profiles(owner.user_id, "product", db_path)[0]["payload"]["ph"] == 6.4
    assert list_user_profiles(other.user_id, "product", db_path) == []
    assert profile_id
    assert [item["name"] for item in list_projects(owner.user_id, db_path)] == ["Cold-chain study"]


def test_ledger_schema_initializes_once_with_query_indexes(tmp_path):
    import sqlite3

    from src.ledger import initialize_ledger

    db_path = tmp_path / "versioned.sqlite3"
    initialize_ledger(db_path)
    with sqlite3.connect(db_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    assert version == 1
    assert "idx_analysis_runs_user_type_created" in indexes
    assert "idx_analysis_runs_user_project_created" in indexes

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA user_version = 999")
    with pytest.raises(RuntimeError, match="newer than supported"):
        initialize_ledger(db_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not meaningful on Windows")
def test_ledger_database_file_is_private_to_the_application_account(tmp_path):
    from src.ledger import initialize_ledger

    db_path = tmp_path / "private.sqlite3"
    initialize_ledger(db_path)

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
