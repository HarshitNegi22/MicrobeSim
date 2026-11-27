"""Shared pytest configuration.

MicrobeSim does not ship ComBase-derived data (its redistribution terms are unresolved; see
``data/README.md``). The suite therefore has to pass on a fresh clone:

* tests that exercise loader/pretraining logic use the ``real_bundle`` fixture, which provides a
  clearly labelled **synthetic stand-in** in the same layout as ``data/combase_growth_dataset.csv``
  (and uses the genuine local bundle instead when you have built one);
* tests that assert facts about the genuine ComBase-derived files carry ``needs_combase`` or
  ``needs_expanded`` and are skipped when those local files are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import datasets

# stand-in label -> (cardinal-parameter profile used to generate it, rows)
_STANDIN_ORGANISMS = {
    "Bacillus cereus": ("Staphylococcus aureus", 90),
    "Escherichia coli": ("Escherichia coli", 90),
    "Salmonella spp.": ("Salmonella spp.", 60),
    "Listeria monocytogenes": ("Listeria monocytogenes", 60),
}
_FOOD_CATEGORIES = ("Culture medium", "Meat", "Dairy", "Vegetable")


def standin_growth_frame() -> pd.DataFrame:
    """Synthetic rows in the ComBase growth-dataset layout. Never real measurements."""
    parts = []
    for organism, (profile, n) in _STANDIN_ORGANISMS.items():
        frame = datasets.generate_growth_dataset(profile, n=n * 2, seed=7, include_no_growth=False)
        frame = frame.head(n).reset_index(drop=True)
        frame["organism"] = organism
        slug = "".join(ch for ch in organism if ch.isalpha())[:3].upper()
        frame["record_id"] = [f"STANDIN_{slug}{i // 2:03d}" for i in range(len(frame))]
        frame["food_category"] = [
            _FOOD_CATEGORIES[i % len(_FOOD_CATEGORIES)] for i in range(len(frame))
        ]
        frame["assumed"] = np.nan
        frame["source"] = "synthetic_standin"
        parts.append(
            frame[
                datasets.DATASET_COLUMNS[:5] + ["record_id", "food_category", "assumed", "source"]
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    return out[
        [
            "record_id",
            "organism",
            "temperature_celsius",
            "ph",
            "water_activity",
            "mu_max_per_h",
            "food_category",
            "assumed",
            "source",
        ]
    ]


@pytest.fixture()
def real_bundle(tmp_path, monkeypatch):
    """The genuine local ComBase bundle if built, else a synthetic stand-in with the same layout."""
    if datasets.REAL_GROWTH_CSV.exists():
        return "genuine"
    path = tmp_path / "combase_growth_dataset.csv"
    standin_growth_frame().to_csv(path, index=False)
    monkeypatch.setattr(datasets, "REAL_GROWTH_CSV", path)
    return "standin"


def pytest_collection_modifyitems(config, items):
    gates = {
        "needs_combase": (
            datasets.real_data_available(),
            "needs the locally built ComBase bundle (data/README.md)",
        ),
        "needs_expanded": (
            datasets.expanded_data_available(),
            "needs the locally built data/expanded bundle (data/README.md)",
        ),
    }
    for item in items:
        for marker, (present, reason) in gates.items():
            if marker in item.keywords and not present:
                item.add_marker(pytest.mark.skip(reason=reason))
