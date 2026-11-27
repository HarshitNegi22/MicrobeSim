"""Train every model the app can show, once, so the app never trains while you wait.

    python scripts/pretrain_models.py              # build/refresh data/pretrained_models (resumable)
    python scripts/pretrain_models.py --jobs 4     # train datasets in parallel
    python scripts/pretrain_models.py --check      # exit 1 unless every dataset is pretrained for THIS environment
    python scripts/pretrain_models.py --force      # retrain everything

The store is tied to the exact library versions (requirements-tested.txt) and model code in use. After
upgrading libraries or editing src/models/*.py, run this again (the Dockerfile does it at build time).
Uploaded datasets cannot be pretrained; they are trained once on first use and then cached.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MICROBESIM_LOG_LEVEL", "WARNING")  # keep CLI output readable

from src.pretrained import PRETRAINED_DIR, pretrain, pretrained_status  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--jobs", type=int, default=1, help="Datasets to train in parallel (default 1).")
    p.add_argument(
        "--force", action="store_true", help="Retrain even if a valid saved model exists."
    )
    p.add_argument("--check", action="store_true", help="Only verify the store; train nothing.")
    p.add_argument(
        "--no-expanded", action="store_true", help="Skip the expanded-dataset training sources."
    )
    p.add_argument(
        "--organism",
        action="append",
        help="Only this organism (repeatable); no manifest is written.",
    )
    p.add_argument("--store", type=Path, default=PRETRAINED_DIR, help="Output directory.")
    args = p.parse_args(argv)

    if args.check:
        status = pretrained_status(args.store, include_expanded=not args.no_expanded)
        print(f"{status['state'].upper()}: {status['detail']}")
        return 0 if status["state"] == "ready" else 1

    started = time.time()

    def show(done, total, record):
        print(
            f"[{done:>3}/{total}] {record['status']:<7} {record['seconds']:>6.1f}s {record['megabytes']:>6.2f} MB  {record['label']}",  # noqa: E501
            flush=True,
        )

    records = pretrain(
        args.store,
        include_expanded=not args.no_expanded,
        n_jobs=args.jobs,
        force=args.force,
        progress=show,
        organisms=args.organism,
    )
    size = sum(r["megabytes"] for r in records)
    print(
        f"\nDone: {len(records)} datasets, {size:.1f} MB, {time.time() - started:.0f}s -> {args.store}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
