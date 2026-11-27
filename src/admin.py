"""Offline administration commands for a trusted MicrobeSim host operator."""

from __future__ import annotations

import argparse
from getpass import getpass
from pathlib import Path

from src import audit
from src.accounts import AccountError, reset_password
from src.ledger import LEDGER_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage local MicrobeSim accounts.")
    parser.add_argument("username", help="Existing local account username")
    parser.add_argument(
        "--database",
        type=Path,
        default=LEDGER_PATH,
        help="SQLite database path (defaults to the configured MicrobeSim ledger)",
    )
    args = parser.parse_args()
    password = getpass("New password (minimum 12 characters): ")
    confirmation = getpass("Confirm new password: ")
    if password != confirmation:
        parser.error("The passwords do not match.")
    try:
        updated = reset_password(args.username, password, args.database)
    except AccountError as exc:
        parser.error(str(exc))
    if not updated:
        parser.error(f"No account named {args.username!r} exists in {args.database}.")
    audit.record(
        "admin-cli",
        "account.password_reset",
        args.username.strip().casefold()[:32],
        db_path=args.database,
    )
    print(f"Password reset for {args.username.strip().casefold()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
