"""Detect batches of growth-rate records that contradict the rest of an organism's data.

Why: ComBase exports contain study batches whose reported rate is wildly inconsistent with every
other record at neighbouring conditions (e.g. 60 E. coli records at 39-40 C reporting ~0.03/h while
the same pH/aw gives ~1.7/h at 37 C and ~1.4/h at 45 C). A smooth secondary model cannot honour
both, and a model that learns the bad batch will tell users "no growth" at conditions where the
organism grows - the unsafe direction for a food-safety tool.

The check is deliberately conservative and explainable. A batch is flagged only if ALL hold:
  * it has >= ``min_rows`` rows and is < ``max_share`` of the organism's rows;
  * a mechanistic model fitted on the *other* rows is good (R2 >= ``min_rest_r2``);
  * the batch's conditions lie inside the range spanned by those other rows (no extrapolation);
  * the batch is systematically off: >= ``min_off_share`` of its rows differ from the
    prediction by more than ``factor``-fold.
Only the individual rows that are off by that factor are flagged, so consistent rows of a bad batch
survive. Flagged rows are NOT deleted: they are listed in ``data/suspect_records.csv`` and excluded
from training and scoring by default; pass ``exclude_suspect=False`` to the loaders to include them.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.secondary import MechanisticModel

_PREFIX = re.compile(r"^([A-Za-z_]+?)_*(?=\d)")


def batch_key(record_id) -> str:
    """Study-batch label of a ComBase record id: its leading letters (``CA_Ec886`` -> ``CA_Ec``)."""
    match = _PREFIX.match(str(record_id))
    return match.group(1) if match else str(record_id)


def _inside(rest: pd.DataFrame, part: pd.DataFrame) -> bool:
    for column in ("temperature_celsius", "ph", "water_activity"):
        if rest[column].notna().any() and part[column].notna().any():
            if (
                part[column].min() < rest[column].min() - 1e-9
                or part[column].max() > rest[column].max() + 1e-9
            ):
                return False
    return True


def find_suspect_batches(
    frame: pd.DataFrame,
    min_rows: int = 10,
    max_share: float = 0.6,
    min_rest_r2: float = 0.6,
    factor: float = 3.0,
    min_off_share: float = 0.8,
) -> List[Dict]:
    """Return one report dict per flagged (organism, batch). ``frame`` needs organism, record_id, T, pH, aw, mu."""
    report: List[Dict] = []
    cols = ["temperature_celsius", "ph", "water_activity"]
    for organism, group in frame.groupby("organism"):
        group = group.dropna(subset=cols + ["mu_max_per_h"])
        group = group[group["mu_max_per_h"] > 0]
        keys = group["record_id"].map(batch_key)
        for key, part in group.groupby(keys):
            rest = group[keys != key]
            if len(part) < min_rows or len(part) > max_share * len(group) or len(rest) < 20:
                continue
            if not _inside(rest, part):
                continue
            try:
                model = MechanisticModel("ratkowsky").fit(
                    rest[cols].to_numpy(float), rest["mu_max_per_h"].to_numpy(float)
                )
            except (ValueError, RuntimeError):
                continue
            pred_rest = model.predict(rest[cols].to_numpy(float))
            obs = rest["mu_max_per_h"].to_numpy(float)
            r2 = 1.0 - float(np.sum((obs - pred_rest) ** 2) / np.sum((obs - obs.mean()) ** 2))
            if r2 < min_rest_r2:
                continue
            pred = model.predict(part[cols].to_numpy(float))
            ratio = np.maximum(pred, 1e-6) / np.maximum(part["mu_max_per_h"].to_numpy(float), 1e-6)
            off = (ratio > factor) | (ratio < 1.0 / factor)
            if off.mean() >= min_off_share:
                report.append(
                    {
                        "organism": organism,
                        "batch": key,
                        "n_rows": int(off.sum()),
                        "n_batch_rows": int(len(part)),
                        "row_index": list(part.index[off]),
                        "rest_r2": round(r2, 3),
                        "median_observed": float(np.median(part["mu_max_per_h"])),
                        "median_expected": float(np.median(pred)),
                        "share_off_by_factor": round(float(off.mean()), 2),
                        "record_ids": sorted(part.loc[part.index[off], "record_id"].astype(str)),
                    }
                )
    return report


def flag_rate_quality(frame: pd.DataFrame, **kwargs) -> pd.Series:
    """'ok' or 'suspect_batch' for every row of ``frame`` (index preserved)."""
    flags = pd.Series("ok", index=frame.index, dtype=object)
    for item in find_suspect_batches(frame, **kwargs):
        flags.loc[item["row_index"]] = "suspect_batch"
    return flags


def suspect_records_table(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Reviewable list (organism, record_id, observed, expected, reason) of flagged rows."""
    rows = []
    for item in find_suspect_batches(frame, **kwargs):
        sub = frame.loc[item["row_index"]]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "organism": item["organism"],
                    "record_id": str(r["record_id"]),
                    "temperature_celsius": r["temperature_celsius"],
                    "ph": r["ph"],
                    "water_activity": r["water_activity"],
                    "observed_mu_max_per_h": r["mu_max_per_h"],
                    "expected_mu_max_per_h": item["median_expected"],
                    "reason": (
                        f"batch '{item['batch']}': {item['share_off_by_factor']:.0%} of its rows differ >3-fold from a "
                        f"model that fits the organism's other rows (R2={item['rest_r2']})"
                    ),
                }
            )
    return pd.DataFrame(rows)
