"""Build the local real-data files in ./data from raw ComBase Browser exports (not distributed).

Usage:
    python scripts/build_real_datasets.py /path/to/export_folder [more_folders ...]

Each argument is a folder containing ComBaseExport*.xlsx files (organised in any
number of subfolders - the loader searches recursively). Re-run this whenever you
add another export (e.g. a growth-filtered re-export for an organism that came up
short) - it re-reads everything from scratch and overwrites the CSVs below.

Writes:
    data/combase_growth_dataset.csv    - real (organism, T, pH, aw, mu_max) rows,
                                          used to train the secondary/ML models
    data/combase_kinetic_curves.csv    - real raw growth curves, used for Page 1
    data/combase_dynamic_profiles.csv  - real time-varying T/pH/aw records, set
                                          aside (not currently used by the app -
                                          see docs/REAL_DATA.md)
    data/combase_load_report.csv       - the classification/exclusion counts, for
                                          the record
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.combase import load_combase_folder  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def main(folders):
    if not folders:
        print(__doc__)
        raise SystemExit(1)
    kin, growth, dyn, report = load_combase_folder(folders)
    DATA.mkdir(exist_ok=True)
    growth.to_csv(DATA / "combase_growth_dataset.csv", index=False)
    kin.to_csv(DATA / "combase_kinetic_curves.csv", index=False)
    dyn.to_csv(DATA / "combase_dynamic_profiles.csv", index=False)
    report.summary_table().to_csv(DATA / "combase_load_report.csv", index=False)

    print(report.summary_table().to_string(index=False))
    print()
    print("Usable growth rows per organism:")
    print(growth.groupby("organism").size().sort_values(ascending=False).to_string())
    print()
    print("Wrote:", DATA / "combase_growth_dataset.csv", f"({len(growth)} rows)")
    print(
        "Wrote:",
        DATA / "combase_kinetic_curves.csv",
        f"({kin['series_id'].nunique() if len(kin) else 0} curves)",
    )
    print(
        "Wrote:",
        DATA / "combase_dynamic_profiles.csv",
        f"({dyn['record_id'].nunique() if len(dyn) else 0} records)",
    )
    for w in report.warnings[:20]:
        print("warning:", w)


if __name__ == "__main__":
    main(sys.argv[1:])
