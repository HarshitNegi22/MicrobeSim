"""Regenerate data/suspect_records.csv: records that contradict the rest of their organism's data.

    python scripts/flag_suspect_records.py          # rewrite the list
    python scripts/flag_suspect_records.py --show   # print only, change nothing

The list is the single source of truth the loaders use to set rows aside (never delete them). Review it
and edit by hand if you know better: removing a row from the CSV brings that record back into training.
Method and thresholds: src/models/data_quality.py. Run scripts/pretrain_models.py afterwards.
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.datasets import EXPANDED_DATA_DIR, REAL_GROWTH_CSV, SUSPECT_RECORDS_CSV  # noqa: E402
from src.models.data_quality import suspect_records_table  # noqa: E402

warnings.filterwarnings("ignore")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)
    tables = []
    sources = {"combase_growth_dataset.csv": REAL_GROWTH_CSV}
    expanded = EXPANDED_DATA_DIR / "microbesim_growth_rates_model_ready.csv"
    if expanded.exists():
        sources["expanded growth rates (real)"] = expanded
    for name, path in sources.items():
        frame = pd.read_csv(path)
        if "is_synthetic" in frame:
            frame = frame[~frame["is_synthetic"].astype(bool)]
        table = suspect_records_table(frame)
        table["dataset"] = name
        tables.append(table)
        print(f"{name}: {len(table)} suspect rows")
    result = pd.concat(tables, ignore_index=True)
    if not result.empty:
        print(result.groupby(["organism", "dataset"]).size().to_string())
    if not args.show:
        result.to_csv(SUSPECT_RECORDS_CSV, index=False)
        print(f"Wrote {SUSPECT_RECORDS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
