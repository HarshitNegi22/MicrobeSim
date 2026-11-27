"""Score the models, trained ONLY on the ComBase data, on independent external data.

    python scripts/validate_on_external.py [--out docs/model_results/external_validation.csv]

For each organism with external rows: train every model on that organism's ComBase rows (the app's dataset,
suspect records set aside), predict the external conditions, and compare with the published rates. Rows
outside the training range are reported separately because extrapolation is not what the models are for.
Passing here would be evidence of transfer to a new lab/product; it is NOT a safety validation.
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.datasets import load_real_growth_dataset  # noqa: E402
from src.external_data import external_organisms, load_external_growth_dataset  # noqa: E402
from src.models.benchmark import MODEL_FACTORIES, MODEL_LABELS  # noqa: E402
from src.models.metrics import all_metrics  # noqa: E402
from src.models.secondary import frame_to_X  # noqa: E402

warnings.filterwarnings("ignore")


def _inside(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(test), bool)
    for column in ("temperature_celsius", "ph", "water_activity"):
        ok &= test[column].between(train[column].min(), train[column].max()).to_numpy()
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    rows = []
    for organism in external_organisms():
        train = load_real_growth_dataset(organism)
        test = load_external_growth_dataset(organism)
        Xtr, ytr, Xte, yte = (
            frame_to_X(train),
            train.mu_max_per_h.to_numpy(),
            frame_to_X(test),
            test.mu_max_per_h.to_numpy(),
        )
        inside = _inside(train, test)
        for subset, mask in (
            ("all external rows", np.ones(len(test), bool)),
            ("inside ComBase range", inside),
        ):
            if mask.sum() < 4:
                continue
            baseline = np.full(mask.sum(), np.median(ytr))
            base = all_metrics(yte[mask], baseline, min_obs=0.02)
            rows.append(
                {
                    "organism": organism,
                    "subset": subset,
                    "model": "baseline: training median",
                    "n": int(mask.sum()),
                    "R2": base["R2"],
                    "RMSE": base["RMSE"],
                    "Af": base["Af"],
                    "Bf": base["Bf"],
                    "n_missed": 0,
                }
            )
            for key, factory in MODEL_FACTORIES.items():
                try:
                    pred = factory().fit(Xtr, ytr).predict(Xte[mask])
                except (ValueError, RuntimeError) as exc:
                    rows.append(
                        {
                            "organism": organism,
                            "subset": subset,
                            "model": MODEL_LABELS[key],
                            "n": int(mask.sum()),
                            "error": str(exc)[:60],
                        }
                    )
                    continue
                m = all_metrics(yte[mask], pred, min_obs=0.02)
                rows.append(
                    {
                        "organism": organism,
                        "subset": subset,
                        "model": MODEL_LABELS[key],
                        "n": int(mask.sum()),
                        "R2": m["R2"],
                        "RMSE": m["RMSE"],
                        "Af": m["Af"],
                        "Bf": m["Bf"],
                        "n_missed": m.get("n_missed", np.nan),
                    }
                )
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    if args.out:
        table.round(4).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
