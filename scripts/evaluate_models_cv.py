"""Repeated, record-grouped cross-validation of every growth-rate model on the REAL expanded data.

Why this exists: the preassigned train/validation/test split leaves only 2-33 real test rows per
organism, so one split gives very noisy R2. Here every real row is scored out-of-fold, whole
ComBase records are kept together (no leakage between replicate conditions of one record), and
the whole procedure is repeated with different groupings.

    python scripts/evaluate_models_cv.py                       # all organisms, all models
    python scripts/evaluate_models_cv.py --organism "Escherichia coli" --repeats 1
    python scripts/evaluate_models_cv.py --with-synthetic      # add donor-safe synthetic rows to TRAIN folds

Metrics are computed on the pooled out-of-fold predictions of each repeat, then averaged.
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.datasets import load_expanded_growth_dataset
from src.models.benchmark import MODEL_FACTORIES, MODEL_LABELS
from src.models.metrics import all_metrics
from src.models.secondary import frame_to_X

warnings.filterwarnings("ignore")


def _record(frame: pd.DataFrame) -> pd.Series:
    """Group key: the record a row came from (synthetic rows use their donor's record)."""
    donor = frame["synthetic_donor_curve_id"].astype("string").str.split("__").str[-1]
    return frame["record_id"].astype("string").where(~frame["is_synthetic"], donor).astype(str)


def evaluate(organism, models, repeats, folds, with_synthetic, seed=0):
    frame = load_expanded_growth_dataset(organism)
    real = frame[~frame["is_synthetic"]].reset_index(drop=True)
    synth = frame[frame["is_synthetic"]].reset_index(drop=True)
    Xr, yr = frame_to_X(real), real["mu_max_per_h"].to_numpy()
    Xs, ys = frame_to_X(synth), synth["mu_max_per_h"].to_numpy()
    grp_real, grp_syn = _record(real), _record(synth)
    uniq = grp_real.unique()
    rows = []
    for key in models:
        per_repeat = []
        for rep in range(repeats):
            rng = np.random.default_rng(seed + rep)
            order = dict(zip(uniq, rng.permutation(len(uniq))))
            gid = grp_real.map(order).to_numpy()
            oof = np.full(len(real), np.nan)
            try:
                for tr, te in GroupKFold(folds).split(Xr, groups=gid):
                    Xt, yt = Xr[tr], yr[tr]
                    if with_synthetic and len(synth):
                        keep = ~grp_syn.isin(set(grp_real.iloc[te])).to_numpy()
                        Xt, yt = np.vstack([Xt, Xs[keep]]), np.concatenate([yt, ys[keep]])
                    oof[te] = MODEL_FACTORIES[key]().fit(Xt, yt).predict(Xr[te])
            except (ValueError, RuntimeError) as exc:
                rows.append(
                    {"organism": organism, "model": MODEL_LABELS[key], "error": str(exc)[:80]}
                )
                per_repeat = None
                break
            per_repeat.append(all_metrics(yr, oof, min_obs=0.05))
        if per_repeat:
            m = pd.DataFrame(per_repeat).mean()
            rows.append(
                {
                    "organism": organism,
                    "model": MODEL_LABELS[key],
                    "n_real_rows": len(real),
                    "R2": m["R2"],
                    "RMSE": m["RMSE"],
                    "Af": m["Af"],
                    "Bf": m["Bf"],
                }
            )
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--organism", action="append")
    p.add_argument(
        "--models", nargs="+", choices=list(MODEL_FACTORIES), default=list(MODEL_FACTORIES)
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--with-synthetic", action="store_true")
    p.add_argument("--out", default=None, help="Optional CSV path for the results table.")
    args = p.parse_args(argv)
    real = load_expanded_growth_dataset(include_synthetic=False)
    counts = real.groupby("organism").size()
    organisms = args.organism or sorted(counts[counts >= 20].index)
    rows = []
    for org in organisms:
        print(f"... {org}", flush=True)
        rows += evaluate(org, args.models, args.repeats, args.folds, args.with_synthetic)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    if "R2" in table:
        print(table.pivot(index="organism", columns="model", values="R2").round(3).to_string())
        print("\nMean over organisms:")
        print(table.groupby("model")[["R2", "RMSE", "Af", "Bf"]].mean().round(3).to_string())
    if args.out:
        table.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
