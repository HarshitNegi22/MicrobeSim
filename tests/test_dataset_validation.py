import numpy as np
import pandas as pd
import pytest

from src.datasets import MAX_DATASET_ROWS, load_growth_dataset


def _dataset():
    return pd.DataFrame(
        {
            "temperature_celsius": [20.0, 25.0],
            "ph": [6.5, 7.0],
            "water_activity": [0.98, 0.95],
            "mu_max_per_h": [0.2, 0.3],
        }
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("temperature_celsius", "warm", "non-numeric"),
        ("temperature_celsius", np.inf, "finite"),
        ("ph", 14.1, "between 0 and 14"),
        ("water_activity", 0.0, "greater than 0"),
        ("mu_max_per_h", -0.1, ">= 0"),
    ],
)
def test_growth_dataset_rejects_invalid_model_inputs(column, value, message):
    frame = _dataset()
    if isinstance(value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        load_growth_dataset(frame)


def test_growth_dataset_rejects_missing_required_inputs_and_duplicate_columns():
    missing = _dataset()
    missing.loc[0, "ph"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        load_growth_dataset(missing)

    duplicate = _dataset()
    duplicate.columns = [
        "temperature_celsius",
        "ph",
        "ph",
        "mu_max_per_h",
    ]
    with pytest.raises(ValueError, match="unique"):
        load_growth_dataset(duplicate)


def test_growth_dataset_rejects_row_limit_overflow():
    frame = pd.concat([_dataset()] * (MAX_DATASET_ROWS // 2 + 1), ignore_index=True)
    with pytest.raises(ValueError, match="row limit"):
        load_growth_dataset(frame)
