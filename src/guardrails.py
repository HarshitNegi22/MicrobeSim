"""Separate exploratory estimates from validated predictions, and flag out-of-scope conditions.

Every prediction gets an :class:`Assessment` with one of three tiers:

``out_of_scope``  a condition lies outside the range of the training data (extrapolation);
``exploratory``   inside the data but not backed by an approved model, or the evidence is weak
                  (few rows, low validation R2, wide interval, near the edge of the data);
``validated``     inside the data, approved model, adequate validation and a usable interval.

Even ``validated`` output requires human review before it is used for a food-safety decision; the tiers
only say how much the number can be trusted as an estimate. Thresholds are explicit parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

VARIABLES = (
    ("temperature_celsius", "temperature", "C"),
    ("ph", "pH", ""),
    ("water_activity", "water activity", ""),
)
REVIEW_NOTICE = "Model output must be reviewed by a qualified person before it is used for any food-safety decision."


@dataclass(frozen=True)
class Assessment:
    tier: str
    reasons: Tuple[str, ...]
    domain: str  # inside | edge | outside
    human_review_required: bool = True

    @property
    def label(self) -> str:
        return {
            "validated": "Validated estimate",
            "exploratory": "Exploratory estimate",
            "out_of_scope": "Out of scope (extrapolation)",
        }[self.tier]


def training_envelope(frame: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Min/max of each model input over the training rows."""
    envelope = {}
    for column, _, _ in VARIABLES:
        values = frame[column].dropna()
        if len(values):
            envelope[column] = (float(values.min()), float(values.max()))
    return envelope


def domain_position(
    envelope: Dict[str, Tuple[float, float]],
    temperature: float,
    ph: float,
    water_activity: float,
    edge_fraction: float = 0.1,
) -> Tuple[str, Tuple[str, ...]]:
    """``inside`` / ``edge`` / ``outside`` plus a human-readable reason per offending variable."""
    query = {"temperature_celsius": temperature, "ph": ph, "water_activity": water_activity}
    reasons, outside, edge = [], False, False
    for column, label, unit in VARIABLES:
        if column not in envelope:
            continue
        low, high = envelope[column]
        value, suffix = query[column], f" {unit}" if unit else ""
        if value < low or value > high:
            outside = True
            reasons.append(
                f"{label} {value:g}{suffix} is outside the training range {low:g}-{high:g}{suffix}"
            )
        else:
            margin = edge_fraction * (high - low)
            if margin > 0 and (value - low < margin or high - value < margin):
                edge = True
                reasons.append(
                    f"{label} {value:g}{suffix} is near the edge of the training range {low:g}-{high:g}{suffix}"
                )
    return ("outside" if outside else "edge" if edge else "inside"), tuple(reasons)


def assess(
    envelope: Dict[str, Tuple[float, float]],
    temperature: float,
    ph: float,
    water_activity: float,
    mean: Optional[float] = None,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    model_status: Optional[str] = None,
    validation_r2: Optional[float] = None,
    n_train: Optional[int] = None,
    max_relative_width: float = 1.5,
    min_validation_r2: float = 0.5,
    min_training_rows: int = 30,
) -> Assessment:
    domain, reasons = domain_position(envelope, temperature, ph, water_activity)
    if domain == "outside":
        return Assessment("out_of_scope", reasons + ("do not rely on this number",), domain)
    notes = list(reasons)
    if model_status != "approved":
        notes.append(
            "the model has not been approved by a reviewer"
            if model_status in (None, "draft")
            else f"model status is '{model_status}'"
        )
    if validation_r2 is None or validation_r2 < min_validation_r2:
        notes.append(
            "validation R2 is "
            + (
                "unknown"
                if validation_r2 is None
                else f"{validation_r2:.2f}, below {min_validation_r2:g}"
            )
        )
    if n_train is None or n_train < min_training_rows:
        notes.append(
            f"only {n_train if n_train is not None else 'an unknown number of'} training rows (need {min_training_rows})"  # noqa: E501
        )
    if (
        None not in (mean, lower, upper)
        and mean
        and mean > 0
        and (upper - lower) / mean > max_relative_width
    ):
        notes.append("the prediction interval is wider than 150% of the estimate")
    return Assessment("exploratory" if notes else "validated", tuple(notes), domain)
