from src.datasets import (
    load_synthetic_growth_dataset,
    synthetic_organisms,
)


def test_archived_synthetic_datasets_are_available_and_valid():
    organisms = synthetic_organisms()

    assert len(organisms) == 11
    for organism in organisms:
        frame = load_synthetic_growth_dataset(organism)
        assert len(frame) == 700
        assert frame["organism"].eq(organism).all()
        assert {
            "temperature_celsius",
            "ph",
            "water_activity",
            "mu_max_per_h",
        } <= set(frame.columns)
        assert (frame["mu_max_per_h"] >= 0).all()
