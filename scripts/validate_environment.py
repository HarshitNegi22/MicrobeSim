"""Check that this installation matches the tested environment before you start or deploy.

    python scripts/validate_environment.py          # exit code 1 on any failure
    python scripts/validate_environment.py --skip-pretrained   # e.g. CI, where models are not built

Checks: Python >= 3.11; every pin in requirements-tested.txt installed at exactly that version; the state
directory is writable and not group/world-writable; pretrained models match this environment.
"""

import os
import re
import stat
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MICROBESIM_LOG_LEVEL", "WARNING")


def main() -> int:
    problems, notes = [], []
    if sys.version_info < (3, 11):
        problems.append(f"Python {sys.version.split()[0]} is too old; 3.11 or newer is required")
    pins = {}
    for line in (ROOT / "requirements-tested.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
        if match:
            pins[match.group(1)] = match.group(2)
    for name, wanted in sorted(pins.items()):
        try:
            found = version(name)
        except PackageNotFoundError:
            problems.append(f"{name} is not installed (tested: {wanted})")
            continue
        if found != wanted:
            problems.append(f"{name} {found} differs from the tested {wanted}")
    from src.ledger import STATE_DIR

    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        mode = STATE_DIR.stat().st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            problems.append(
                f"{STATE_DIR} is group/world-writable; the app refuses to use it (chmod go-w)"
            )
        probe = STATE_DIR / ".validate-probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        problems.append(f"state directory {STATE_DIR} is not writable: {exc}")
    if "--skip-pretrained" in sys.argv[1:]:
        notes.append("pretrained models: check skipped (--skip-pretrained)")
    else:
        from src.pretrained import pretrained_status

        status = pretrained_status()
        (notes if status["state"] == "ready" else problems).append(
            f"pretrained models: {status['state']} ({status['detail']})"
        )
    for item in notes:
        print("OK     ", item)
    for item in problems:
        print("PROBLEM", item)
    print(
        f"\n{len(problems)} problem(s) found."
        if problems
        else "\nEnvironment matches the tested configuration."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
