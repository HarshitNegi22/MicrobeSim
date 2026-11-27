import numpy as np
import pandas as pd
import pytest

from src.datasets import (
    SUSPECT_RECORDS_CSV,
    load_expanded_growth_dataset,
    load_real_growth_dataset,
    suspect_record_count,
)
from src.models.data_quality import batch_key, find_suspect_batches, flag_rate_quality
from src.models.secondary import ratkowsky_mu


def _organism_frame(seed=0, bad_batch=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(80):
        T = rng.uniform(8, 44)
        ph = rng.choice([5.5, 6.5, 7.0])
        aw = rng.choice([0.97, 0.99, 0.997])
        mu = (
            float(ratkowsky_mu(np.array([T]), 0.04, 3.0, 0.2, 48.0)[0]) * (1 - (7.0 - ph) * 0.1) * 6
        )
        rows.append(
            ("Test organism", f"A{i:03d}_Tc", T, ph, aw, max(mu * rng.lognormal(0, 0.05), 0.01))
        )
    if bad_batch:
        for i in range(30):  # replicate bundle at 36-37 C reporting ~100x too low a rate
            rows.append(("Test organism", f"ZZ_Tc{i:03d}", 36.0 + (i % 2), 7.0, 0.997, 0.03))
    return pd.DataFrame(
        rows,
        columns=[
            "organism",
            "record_id",
            "temperature_celsius",
            "ph",
            "water_activity",
            "mu_max_per_h",
        ],
    )


def test_batch_key_uses_leading_letters():
    assert batch_key("CA_Ec886") == "CA_Ec"
    assert batch_key("B58_Ec") == "B"
    assert batch_key("GoEca42") == "GoEca"
    assert batch_key("12345") == "12345"


def test_detector_flags_contradicting_batch_only():
    frame = _organism_frame()
    flags = flag_rate_quality(frame)
    assert (flags[frame.record_id.str.startswith("ZZ_")] == "suspect_batch").all()
    assert (flags[~frame.record_id.str.startswith("ZZ_")] == "ok").all()


def test_detector_is_silent_on_consistent_data():
    assert find_suspect_batches(_organism_frame(bad_batch=False)) == []


def test_detector_ignores_batches_outside_the_supported_range():
    frame = _organism_frame()
    frame.loc[frame.record_id.str.startswith("ZZ_"), "temperature_celsius"] = (
        60.0  # would be extrapolation
    )
    assert find_suspect_batches(frame) == []


@pytest.mark.needs_combase
def test_shipped_suspect_list_is_the_ecoli_ca_ec_batch():
    table = pd.read_csv(SUSPECT_RECORDS_CSV)
    assert set(table.organism) == {"Escherichia coli"}
    assert table.record_id.str.startswith("CA_Ec").all()
    assert table.record_id.nunique() == 60
    assert (table.expected_mu_max_per_h > 10 * table.observed_mu_max_per_h).all()


@pytest.mark.needs_combase
def test_loaders_set_suspect_rows_aside_unless_asked():
    clean = load_real_growth_dataset("Escherichia coli")
    full = load_real_growth_dataset("Escherichia coli", exclude_suspect=False)
    assert len(full) - len(clean) == suspect_record_count("Escherichia coli") == 60
    assert not clean.record_id.isin(pd.read_csv(SUSPECT_RECORDS_CSV).record_id).any()
    other = load_real_growth_dataset("Bacillus cereus")
    assert len(other) == len(load_real_growth_dataset("Bacillus cereus", exclude_suspect=False))


@pytest.mark.needs_expanded
def test_expanded_loader_excludes_only_real_suspect_rows():
    clean = load_expanded_growth_dataset("Escherichia coli", include_synthetic=False)
    full = load_expanded_growth_dataset(
        "Escherichia coli", include_synthetic=False, exclude_suspect=False
    )
    assert len(full) - len(clean) == 60
    with_syn = load_expanded_growth_dataset("Escherichia coli")
    assert with_syn.is_synthetic.any()


@pytest.mark.needs_combase
def test_mechanistic_models_recover_once_the_bundle_is_set_aside():
    from src.models.secondary import MechanisticModel, frame_to_X

    full = load_real_growth_dataset("Escherichia coli", exclude_suspect=False)
    clean = load_real_growth_dataset("Escherichia coli")
    scores = {}
    for name, frame in (("full", full), ("clean", clean)):
        X, y = frame_to_X(frame), frame.mu_max_per_h.to_numpy()
        pred = MechanisticModel("ratkowsky").fit(X, y).predict(X)
        scores[name] = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert scores["clean"] > 0.8 > 0.4 > scores["full"] - 0.2
