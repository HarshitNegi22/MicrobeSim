import sqlite3

import pytest

from src.accounts import AccountError, authenticate, create_account, reset_password


def test_account_passwords_are_hashed_and_authentication_is_case_insensitive(tmp_path):
    db_path = tmp_path / "accounts.sqlite3"
    account = create_account("Food_Safety", "a-long-unique-password", db_path)

    assert authenticate("food_safety", "a-long-unique-password", db_path) == account
    assert authenticate("FOOD_SAFETY", "incorrect-password", db_path) is None
    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            "SELECT password_hash FROM accounts WHERE user_id = ?", (account.user_id,)
        ).fetchone()[0]
    assert "a-long-unique-password" not in stored
    assert stored.startswith("pbkdf2_sha256$")


def test_account_creation_rejects_weak_password_and_duplicate_username(tmp_path):
    db_path = tmp_path / "accounts.sqlite3"
    create_account("food_safety", "a-long-unique-password", db_path)

    with pytest.raises(AccountError, match="at least"):
        create_account("second_user", "too-short", db_path)
    with pytest.raises(AccountError, match="already registered"):
        create_account("FOOD_SAFETY", "another-long-password", db_path)


def test_account_creation_validates_username(tmp_path):
    with pytest.raises(AccountError, match="Username"):
        create_account("bad name", "a-long-unique-password", tmp_path / "accounts.sqlite3")


def test_failed_sign_ins_are_temporarily_throttled_and_password_can_be_reset(tmp_path):
    db_path = tmp_path / "accounts.sqlite3"
    create_account("limited_user", "original-long-password", db_path)

    for _ in range(5):
        assert authenticate("limited_user", "wrong-password", db_path) is None
    assert authenticate("limited_user", "original-long-password", db_path) is None

    assert reset_password("limited_user", "replacement-long-password", db_path)
    assert authenticate("limited_user", "replacement-long-password", db_path) is not None
    assert not reset_password("missing_user", "replacement-long-password", db_path)
