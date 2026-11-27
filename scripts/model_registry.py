"""Model registry CLI: list models, print model cards, approve/reject/retire, verify the audit trail.

    python scripts/model_registry.py sync                      # register the pretrained models as drafts
    python scripts/model_registry.py list [--status draft]
    python scripts/model_registry.py card SIGNATURE            # model card (markdown)
    python scripts/model_registry.py approve SIGNATURE --reviewer alice --note "checked parity plots, R2 0.81"
    python scripts/model_registry.py reject  SIGNATURE --reviewer alice --note "extrapolates badly"
    python scripts/model_registry.py audit [--verify] [--limit 50]

Approval needs a note and a reviewer different from the author (set MICROBESIM_FOUR_EYES=0 to relax this on
a single-person installation). Every status change is written to the audit trail.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MICROBESIM_LOG_LEVEL", "WARNING")  # keep CLI output readable

from src import audit, registry  # noqa: E402
from src.pretrained import sync_registry  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("sync")
    ls = sub.add_parser("list")
    ls.add_argument("--status", choices=registry.STATUSES)
    card = sub.add_parser("card")
    card.add_argument("signature")
    for name in ("approve", "reject", "retire"):
        s = sub.add_parser(name)
        s.add_argument("signature")
        s.add_argument("--reviewer", required=True)
        s.add_argument("--note", default="")
    au = sub.add_parser("audit")
    au.add_argument("--verify", action="store_true")
    au.add_argument("--limit", type=int, default=50)
    args = p.parse_args(argv)

    try:
        if args.command == "sync":
            print(f"Registered {sync_registry()} new model(s).")
        elif args.command == "list":
            for m in registry.list_models(status=args.status):
                print(f"{m['signature'][:12]}  {m['status']:<8} {m['created_utc']}  {m['label']}")
        elif args.command == "card":
            print(registry.model_card(args.signature))
        elif args.command in ("approve", "reject", "retire"):
            status = {"approve": "approved", "reject": "rejected", "retire": "retired"}[
                args.command
            ]
            entry = registry.set_status(args.signature, status, args.reviewer, args.note)
            print(f"{entry['label']}: {entry['status']} by {entry['reviewed_by']}")
        elif args.command == "audit":
            if args.verify:
                ok, bad = audit.verify_chain()
                print("Audit chain OK." if ok else f"AUDIT CHAIN BROKEN at event id {bad}.")
                return 0 if ok else 2
            for e in audit.list_events(limit=args.limit):
                print(
                    f"{e['id']:>6} {e['ts']} {e['actor'][:12]:<12} {e['action']:<22} {e['outcome']:<7} {e['target'][:40]}"  # noqa: E501
                )
    except registry.RegistryError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
