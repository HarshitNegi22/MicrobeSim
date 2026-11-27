"""Add externally sourced growth rates (published studies, author data, your lab) with provenance checks.

    # 1. dry run: validates and reports, changes nothing
    python scripts/import_external_data.py my_rows.csv --source-id Smith2025_Foods

    # 2. first time for a source: register it (licence and DOI are checked) and apply
    python scripts/import_external_data.py my_rows.csv --source-id Smith2025_Foods --apply \
        --register-source --citation "Smith J et al. ..." --doi 10.xxxx/yyyy --url https://... \
        --license cc-by-4.0 --obtained-how "downloaded Table S2 from the journal site" --notes "..." \
        --attribution "Data from Smith et al. (2025), CC BY 4.0"

Input columns: see data/external/templates/external_growth_rates_template.csv. ``mu_max_per_h`` must be the
NATURAL-LOG maximum specific growth rate per hour (multiply a published log10 rate by 2.302585); ``lag_h`` in hours.
Only licences in src/external_data.py BUNDLE_LICENSES are accepted (no NC/ND). Nothing is imported if any check
fails. After a successful import, run scripts/pretrain_models.py so the new rows reach the models.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.external_data import (  # noqa: E402
    EXTERNAL_RATES_CSV,
    EXTERNAL_SOURCES_CSV,
    load_sources,
    validate_external_frame,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv")
    p.add_argument("--source-id", required=True)
    p.add_argument("--apply", action="store_true", help="write the rows (default is a dry run)")
    p.add_argument("--register-source", action="store_true")
    for name in (
        "citation",
        "doi",
        "url",
        "license",
        "obtained-how",
        "notes",
        "attribution",
        "checks",
    ):
        p.add_argument(f"--{name}", default="")
    args = p.parse_args(argv)

    frame = pd.read_csv(args.csv)
    if "source_id" not in frame.columns:
        frame["source_id"] = args.source_id
    frame["source_id"] = frame["source_id"].fillna(args.source_id)
    wrong_source = frame["source_id"] != args.source_id
    if "source" not in frame.columns:
        frame["source"] = f"external ({args.source_id})"
    if "assumed" not in frame.columns:
        frame["assumed"] = ""

    sources = load_sources()
    if args.register_source:
        if args.source_id in set(sources["source_id"]):
            print(f"Source {args.source_id} is already registered.")
        else:
            new = {
                "source_id": args.source_id,
                "status": "imported",
                "citation": args.citation,
                "doi": args.doi,
                "url": args.url,
                "license": args.license.lower(),
                "license_url": "",
                "retrieved_on": str(pd.Timestamp.today().date()),
                "obtained_how": args.obtained_how,
                "checks": args.checks,
                "notes": args.notes,
                "attribution": args.attribution,
            }
            if not args.citation or not args.obtained_how:
                print("--citation and --obtained-how are required to register a source.")
                return 1
            sources = pd.concat([sources, pd.DataFrame([new])], ignore_index=True)

    problems = validate_external_frame(frame, sources)
    if wrong_source.any():
        problems.append(
            f"{int(wrong_source.sum())} row(s) carry a source_id different from --source-id {args.source_id}"
        )
    existing = pd.read_csv(EXTERNAL_RATES_CSV)
    clash = frame["record_id"].isin(existing["record_id"])
    if clash.any():
        problems.append(f"record_id already present: {sorted(frame.loc[clash, 'record_id'])[:5]}")
    key = ["organism", "temperature_celsius", "ph", "water_activity"]
    merged = frame.merge(existing[key + ["mu_max_per_h"]], on=key, suffixes=("", "_old"))
    duplicate = merged[(merged["mu_max_per_h"] - merged["mu_max_per_h_old"]).abs() < 1e-4]
    if len(duplicate):
        problems.append(
            f"{len(duplicate)} row(s) duplicate existing rows (same conditions and rate)"
        )

    print(f"{len(frame)} row(s); organisms: {sorted(frame['organism'].unique())}")
    if problems:
        print("NOT IMPORTED. Problems:\n  - " + "\n  - ".join(problems))
        return 1
    print("All checks passed." + ("" if args.apply else " Dry run: nothing written (use --apply)."))
    if args.apply:
        pd.concat([existing, frame], ignore_index=True).to_csv(EXTERNAL_RATES_CSV, index=False)
        sources.to_csv(EXTERNAL_SOURCES_CSV, index=False)
        print(f"Imported {len(frame)} row(s). Next: python scripts/pretrain_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
