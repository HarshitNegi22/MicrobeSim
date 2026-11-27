import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.datasets import (
    EXPANDED_DATA_DIR,
    load_expanded_growth_dataset,
    load_expanded_kinetic_curves,
    load_expanded_observations,
)
from src.models.benchmark import benchmark

# These assert facts about the locally built, ComBase-derived data/expanded bundle (not shipped).
pytestmark = pytest.mark.needs_expanded


def test_expanded_observations_have_4000_rows_per_organism():
    frame = load_expanded_observations(only_modeling_rows=False)

    assert frame.groupby("organism").size().eq(4000).all()
    assert frame["organism"].nunique() == 10


def test_combined_real_observations_preserve_the_clean_real_source():
    real = load_expanded_observations(
        include_synthetic=False,
        only_modeling_rows=False,
    )
    source = pd.read_csv(
        EXPANDED_DATA_DIR / "microbesim_clean_real_observations.csv",
        low_memory=False,
    )

    assert_frame_equal(
        real.astype(object).where(real.notna(), None),
        source.astype(object).where(source.notna(), None),
        check_dtype=False,
    )


def test_synthetic_observations_are_flagged_and_linked_to_real_curves():
    observations = load_expanded_observations(only_modeling_rows=False)
    summary = pd.read_csv(
        EXPANDED_DATA_DIR / "microbesim_curve_summary.csv",
        low_memory=False,
        dtype={"curve_id": "string", "synthetic_donor_curve_id": "string"},
    )
    synthetic = observations.loc[observations["is_synthetic"]]
    real_curve_ids = set(summary.loc[~summary["is_synthetic"], "curve_id"].dropna())

    assert not synthetic.empty
    assert synthetic["synthetic_donor_curve_id"].notna().all()
    assert set(synthetic["synthetic_donor_curve_id"].dropna()) <= real_curve_ids


def test_curve_splits_are_disjoint_and_synthetic_inherits_donor_split():
    summary = pd.read_csv(
        EXPANDED_DATA_DIR / "microbesim_curve_summary.csv",
        low_memory=False,
        dtype={"curve_id": "string", "synthetic_donor_curve_id": "string"},
    )
    assert summary.groupby("curve_id")["split"].nunique(dropna=False).le(1).all()

    real_splits = (
        summary.loc[~summary["is_synthetic"]].set_index("curve_id")["split"].astype("string")
    )
    synthetic = summary.loc[summary["is_synthetic"]]
    donor_splits = synthetic["synthetic_donor_curve_id"].map(real_splits).astype("string")
    assert donor_splits.notna().all()
    assert (
        synthetic["split"]
        .astype("string")
        .reset_index(drop=True)
        .equals(donor_splits.reset_index(drop=True))
    )


def test_expanded_loaders_return_documented_columns_and_cached_copies():
    growth = load_expanded_growth_dataset()
    observations = load_expanded_observations()
    curves = load_expanded_kinetic_curves(
        organism="Bacillus cereus",
        include_synthetic=False,
    )
    repeated = load_expanded_kinetic_curves(
        organism="Bacillus cereus",
        include_synthetic=False,
    )

    assert growth.columns.tolist() == [
        "organism",
        "temperature_celsius",
        "ph",
        "water_activity",
        "mu_max_per_h",
        "lag_h",
        "record_id",
        "food_category",
        "assumed",
        "source",
        "is_synthetic",
        "synthetic_donor_curve_id",
        "split",
    ]
    assert {
        "time_hours",
        "log10_count",
        "curve_id",
        "is_synthetic",
        "split",
    } <= set(observations.columns)
    assert curves.columns.tolist() == [
        "series_id",
        "organism",
        "temperature_celsius",
        "ph",
        "water_activity",
        "time_hours",
        "measurement",
        "food_category",
    ]
    assert curves["series_id"].is_unique is False
    assert curves is not repeated
    assert_frame_equal(curves, repeated)


def test_growth_loader_filters_organism_source_and_split():
    frame = load_expanded_growth_dataset(
        organism="Bacillus cereus",
        include_synthetic=False,
        split="train",
    )

    assert frame["organism"].eq("Bacillus cereus").all()
    assert not frame["is_synthetic"].any()
    assert frame["split"].eq("train").all()


def _benchmark_frame():
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for row_index in range(8):
            rows.append(
                {
                    "organism": "Test organism",
                    "temperature_celsius": 10.0 + row_index * 3,
                    "ph": 6.5,
                    "water_activity": 0.98,
                    "mu_max_per_h": 0.1 + split_index * 0.02 + row_index * 0.001,
                    "is_synthetic": False,
                    "split": split,
                }
            )
        for row_index in range(2):
            rows.append(
                {
                    "organism": "Test organism",
                    "temperature_celsius": 15.0 + row_index,
                    "ph": 6.5,
                    "water_activity": 0.98,
                    "mu_max_per_h": (9.0, 80.0, 99.0)[split_index],
                    "is_synthetic": True,
                    "split": split,
                }
            )
    rows.append(
        {
            "organism": "Test organism",
            "temperature_celsius": 20.0,
            "ph": 6.5,
            "water_activity": 0.98,
            "mu_max_per_h": 123.0,
            "is_synthetic": True,
            "split": None,
        }
    )
    return pd.DataFrame(rows)


def test_benchmark_scores_real_splits_and_compares_training_sources(monkeypatch):
    class TrackingModel:
        name = "Tracking model"

        def fit(self, X, mu, limits=None):
            self.training_mu = np.array(mu)
            self.limits = None
            return self

        def predict(self, X):
            return np.full(len(X), 0.25)

    import src.models.benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "build_model", lambda _key: TrackingModel())
    frame = _benchmark_frame()

    result = benchmark(frame, ["tracking"], training_source="real+synthetic")

    assert result.real_test_count == 8
    assert result.table["n_test"].eq(8).all()
    assert len(result.predictions) == 8
    assert not frame.iloc[result.test_index]["is_synthetic"].any()
    assert set(result.comparison_table["training_source"]) == {
        "Real only",
        "Real + synthetic",
    }
    assert result.comparison_table["n_test"].eq(8).all()
    assert result.validation_description.find("real rows only") >= 0
    assert result.models["tracking"].training_mu.max() < 80.0
    assert np.isclose(result.models["tracking"].training_mu.max(), 9.0)


def test_synthetic_only_benchmark_trains_only_on_synthetic_training_rows(monkeypatch):
    class TrackingModel:
        name = "Tracking model"

        def fit(self, X, mu, limits=None):
            self.training_mu = np.array(mu)
            self.limits = None
            return self

        def predict(self, X):
            return np.full(len(X), 0.25)

    import src.models.benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "build_model", lambda _key: TrackingModel())
    frame = _benchmark_frame()
    result = benchmark(frame, ["tracking"], training_source="synthetic")

    assert result.models["tracking"].training_mu.min() == 9.0
    assert result.models["tracking"].training_mu.max() == 9.0
    assert result.real_test_count == 8


def test_every_real_growth_row_has_a_split_and_organisms_have_test_rows():
    """Regression: rows from ComBase summary rates once had a blank split and were silently unused."""
    from src.datasets import load_expanded_growth_dataset

    real = load_expanded_growth_dataset(include_synthetic=False)
    assert real["split"].notna().all()
    assert set(real["split"]) <= {"train", "validation", "test"}
    for organism, group in real.groupby("organism"):
        if group["record_id"].nunique() >= 10:
            assert (group["split"] == "test").any(), organism
        # a record must never straddle splits
        assert group.groupby("record_id")["split"].nunique().le(1).all(), organism
