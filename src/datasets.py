"""Synthetic growth-rate datasets.

These generators produce one row per growth experiment under a fixed T / pH / aw
combination from illustrative cardinal parameters, so the secondary-model and
benchmarking pipeline can be exercised offline.

The generating process is deliberately not identical to either mechanistic model
in ``secondary.py``: a mild pH x aw interaction and multiplicative log-normal
noise are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.secondary import cardinal_gamma, rosso_ctm, water_activity_gamma

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REAL_GROWTH_CSV = DATA_DIR / "combase_growth_dataset.csv"
REAL_KINETIC_CSV = DATA_DIR / "combase_kinetic_curves.csv"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
EXPANDED_DATA_DIR = DATA_DIR / "expanded"
SUSPECT_RECORDS_CSV = DATA_DIR / "suspect_records.csv"
REAL_MIN_ROWS = 20  # below this a per-organism secondary-model fit is not attempted
MAX_DATASET_ROWS = 100_000
MAX_DATASET_COLUMNS = 100
MAX_DATASET_BYTES = 25 * 1024 * 1024

DATASET_COLUMNS = [
    "organism",
    "temperature_celsius",
    "ph",
    "water_activity",
    "mu_max_per_h",
    "lag_h",
]


@dataclass(frozen=True)
class OrganismProfile:
    """Illustrative cardinal parameters (mu in 1/h, natural-log units)."""

    name: str
    mu_opt: float
    tmin: float
    topt: float
    tmax: float
    phmin: float
    phopt: float
    phmax: float
    awmin: float
    awopt: float = 0.997


ORGANISMS: Dict[str, OrganismProfile] = {
    p.name: p
    for p in [
        OrganismProfile("Escherichia coli", 2.0, 3.5, 39.5, 47.5, 4.2, 7.0, 9.5, 0.950),
        OrganismProfile("Salmonella spp.", 1.7, 5.2, 39.4, 47.0, 3.8, 7.0, 9.5, 0.940),
        OrganismProfile("Listeria monocytogenes", 1.1, -1.5, 37.0, 45.5, 4.4, 7.1, 9.4, 0.920),
        OrganismProfile("Staphylococcus aureus", 1.4, 7.0, 37.0, 48.0, 4.0, 7.0, 9.8, 0.860),
        OrganismProfile("Pseudomonas spp.", 1.3, -4.0, 30.0, 40.0, 4.5, 7.0, 9.0, 0.960),
        OrganismProfile("Bacillus cereus", 1.25, 8.0, 35.0, 50.0, 4.9, 7.0, 9.5, 0.930),
        OrganismProfile("Campylobacter jejuni", 0.75, 30.0, 42.0, 45.0, 5.5, 7.0, 8.5, 0.970),
        OrganismProfile("Clostridium perfringens", 2.3, 12.0, 43.0, 52.0, 5.0, 7.2, 9.0, 0.930),
        OrganismProfile(
            "Clostridium botulinum non-proteolytic", 0.55, 3.0, 25.0, 37.0, 5.0, 7.0, 9.0, 0.940
        ),
        OrganismProfile(
            "Clostridium botulinum proteolytic", 0.65, 10.0, 35.0, 48.0, 4.6, 7.0, 9.0, 0.930
        ),
        OrganismProfile("Yersinia enterocolitica", 0.8, -1.0, 29.0, 42.0, 4.2, 7.2, 9.6, 0.940),
    ]
}


def true_mu(profile: OrganismProfile, T, ph, aw) -> np.ndarray:
    """Noise-free generating function (CTM x cardinal pH x aw, with interaction)."""
    g_t = rosso_ctm(T, profile.mu_opt, profile.tmin, profile.topt, profile.tmax)
    g_ph = cardinal_gamma(ph, profile.phmin, profile.phopt, profile.phmax, 1.0)
    g_aw = water_activity_gamma(aw, profile.awmin, profile.awopt)
    interaction = 1.0 - 0.3 * (1.0 - g_ph) * (1.0 - g_aw)
    return g_t * g_ph * g_aw * interaction


def generate_growth_dataset(
    organism: str = "Salmonella spp.",
    n: int = 600,
    seed: int = 42,
    noise_sigma: float = 0.08,
    include_no_growth: bool = True,
) -> pd.DataFrame:
    """Draw ``n`` growth experiments over the organism's plausible (T, pH, aw) space."""
    if organism not in ORGANISMS:
        raise KeyError(f"Unknown organism '{organism}'. Options: {list(ORGANISMS)}")
    prof = ORGANISMS[organism]
    rng = np.random.default_rng(seed)

    T = rng.uniform(prof.tmin - 3.0, prof.tmax + 2.0, n)
    ph = rng.uniform(prof.phmin - 0.3, min(prof.phmax + 0.3, 10.5), n)
    aw = rng.uniform(prof.awmin - 0.02, 0.999, n)
    mu = true_mu(prof, T, ph, aw)
    growth = mu > 0
    mu = np.where(growth, mu * np.exp(rng.normal(0.0, noise_sigma, n)), 0.0)

    h0 = np.clip(rng.normal(2.0, 0.4, n), 0.8, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        lag = np.where(growth, h0 / np.where(growth, mu, 1.0), np.nan)

    df = pd.DataFrame(
        {
            "organism": organism,
            "temperature_celsius": np.round(T, 2),
            "ph": np.round(ph, 2),
            "water_activity": np.round(aw, 3),
            "mu_max_per_h": mu,
            "lag_h": lag,
        }
    )
    if not include_no_growth:
        df = df[growth].reset_index(drop=True)
    return df[DATASET_COLUMNS]


def load_growth_dataset(source) -> pd.DataFrame:
    """Read and validate a bounded ComBase-like CSV/DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        if isinstance(source, (str, Path)):
            if Path(source).stat().st_size > MAX_DATASET_BYTES:
                size_limit_mb = MAX_DATASET_BYTES // (1024 * 1024)
                raise ValueError(f"Dataset exceeds the {size_limit_mb} MB file-size limit.")
        elif hasattr(source, "getvalue") and len(source.getvalue()) > MAX_DATASET_BYTES:
            size_limit_mb = MAX_DATASET_BYTES // (1024 * 1024)
            raise ValueError(f"Dataset exceeds the {size_limit_mb} MB file-size limit.")
        df = pd.read_csv(source, nrows=MAX_DATASET_ROWS + 1)
    if len(df) > MAX_DATASET_ROWS:
        raise ValueError(f"Dataset exceeds the {MAX_DATASET_ROWS:,}-row limit.")
    if len(df.columns) > MAX_DATASET_COLUMNS:
        raise ValueError(f"Dataset exceeds the {MAX_DATASET_COLUMNS}-column limit.")
    if not df.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")
    rename = {}
    aliases = {
        "temperature_celsius": ["temperature", "temp", "temperature_c", "t"],
        "ph": ["ph"],
        "water_activity": ["aw", "wateractivity", "water_activity"],
        "mu_max_per_h": ["mu_max", "mumax", "growth_rate", "mu", "mu_max_1_h"],
    }
    lowered = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    for target, names in aliases.items():
        for col, low in lowered.items():
            if low == target or low in names:
                rename[col] = target
                break
    df = df.rename(columns=rename)
    if not df.columns.is_unique:
        raise ValueError("Dataset columns or aliases resolve to duplicate names.")
    missing = [
        c for c in ("temperature_celsius", "ph", "water_activity", "mu_max_per_h") if c not in df
    ]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")
    for column in ("temperature_celsius", "ph", "water_activity", "mu_max_per_h"):
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"{column} contains {int(invalid.sum())} non-numeric value(s).")
        df[column] = converted
    if "organism" not in df:
        df["organism"] = "Uploaded dataset"
    required = ("temperature_celsius", "ph", "water_activity", "mu_max_per_h")
    if df[list(required)].isna().any().any():
        missing_rows = int(df[list(required)].isna().any(axis=1).sum())
        raise ValueError(f"Required model inputs are missing in {missing_rows} row(s).")
    values = df[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Required model inputs must contain only finite numbers.")
    if not df["temperature_celsius"].between(-100.0, 100.0).all():
        raise ValueError("temperature_celsius must be between -100 and 100 °C.")
    if not df["ph"].between(0.0, 14.0).all():
        raise ValueError("ph must be between 0 and 14.")
    if not df["water_activity"].between(0.0, 1.0, inclusive="right").all():
        raise ValueError("water_activity must be greater than 0 and at most 1.")
    if (df["mu_max_per_h"] < 0).any():
        raise ValueError("mu_max_per_h must be finite and >= 0.")
    return df.reset_index(drop=True)


def synthetic_organisms() -> List[str]:
    """List bundled synthetic datasets and built-in generators, without duplicates."""
    names = set(ORGANISMS)
    if SYNTHETIC_DATA_DIR.exists():
        for path in sorted(SYNTHETIC_DATA_DIR.glob("*.csv")):
            frame = pd.read_csv(path, usecols=["organism"], nrows=1)
            if not frame.empty and pd.notna(frame["organism"].iloc[0]):
                names.add(str(frame["organism"].iloc[0]))
    return sorted(names)


def load_synthetic_growth_dataset(organism: str) -> pd.DataFrame:
    """Load a bundled per-organism synthetic dataset or generate a demonstration set."""
    if SYNTHETIC_DATA_DIR.exists():
        for path in sorted(SYNTHETIC_DATA_DIR.glob("*.csv")):
            frame = pd.read_csv(path, usecols=["organism"], nrows=1)
            if not frame.empty and str(frame["organism"].iloc[0]) == organism:
                return load_growth_dataset(path)
    if organism in ORGANISMS:
        return generate_growth_dataset(organism, n=700, seed=42)
    raise KeyError(f"No synthetic dataset available for '{organism}'.")


# --------------------------------------------------------------------------- #
# Real data, built from ComBase Browser exports (see scripts/build_real_datasets.py
# and docs/REAL_DATA.md). These CSVs are built locally and are NOT distributed with the
# repository (see data/README.md); the app falls back to synthetic data when they are absent.
# --------------------------------------------------------------------------- #
def real_data_available() -> bool:
    return REAL_GROWTH_CSV.exists()


def expanded_data_available() -> bool:
    """True when the optional locally built ``data/expanded`` bundle (derived from ComBase) exists."""
    return _EXPANDED_GROWTH_CSV.is_file() and _EXPANDED_OBSERVATIONS_CSV.is_file()


def published_plus_organisms() -> List[str]:
    """Organisms offered as "ComBase + published studies": needs the local ComBase bundle as its base."""
    if not real_data_available():
        return []
    from src.external_data import external_organisms

    return external_organisms()


def real_organisms(min_rows: int = REAL_MIN_ROWS) -> List[str]:
    """Organisms in the locally built real dataset with enough rows to fit a model."""
    if not real_data_available():
        return []
    counts = pd.read_csv(REAL_GROWTH_CSV, usecols=["organism"])["organism"].value_counts()
    return sorted(counts[counts >= min_rows].index.tolist())


@lru_cache(maxsize=1)
def _suspect_keys() -> frozenset:
    """(organism, record_id) pairs listed in ``data/suspect_records.csv`` (see src/models/data_quality.py)."""
    if not SUSPECT_RECORDS_CSV.exists():
        return frozenset()
    table = pd.read_csv(SUSPECT_RECORDS_CSV, usecols=["organism", "record_id"])
    return frozenset(zip(table["organism"], table["record_id"].astype(str)))


def _suspect_mask(frame: pd.DataFrame) -> pd.Series:
    keys = _suspect_keys()
    if not keys or frame.empty:
        return pd.Series(False, index=frame.index)
    return pd.Series(
        [(o, str(r)) in keys for o, r in zip(frame["organism"], frame["record_id"])],
        index=frame.index,
    )


def suspect_record_count(organism: Optional[str] = None) -> int:
    """Number of real records set aside as inconsistent with the rest of their organism's data."""
    keys = _suspect_keys()
    return sum(1 for o, _ in keys if organism is None or o == organism)


def load_real_growth_dataset(
    organism: Optional[str] = None, exclude_suspect: bool = True, include_external: bool = False
) -> pd.DataFrame:
    """Real (organism, T, pH, aw, mu_max) rows extracted from ComBase exports.

    mu_max is in natural-log units (1/h); ComBase's own log10-based rate has already
    been converted, and every row already passed the growth/decline/confidence
    filtering in :mod:`src.combase`. Raises ``FileNotFoundError`` if the bundle
    hasn't been built (see ``scripts/build_real_datasets.py``).
    """
    if not real_data_available():
        raise FileNotFoundError(
            f"{REAL_GROWTH_CSV} not found - run scripts/build_real_datasets.py on a ComBase export first"
        )
    df = pd.read_csv(REAL_GROWTH_CSV)
    if organism is not None:
        df = df[df["organism"] == organism].reset_index(drop=True)
    if exclude_suspect and "record_id" in df.columns:
        # Records that contradict the rest of the organism's data (data/suspect_records.csv) are
        # set aside, not deleted; pass exclude_suspect=False to get them back.
        df = df.loc[~_suspect_mask(df)].reset_index(drop=True)
    if include_external:
        # Published studies with a bundle-compatible licence (data/external, src/external_data.py).
        from src.external_data import load_external_growth_dataset

        extra = load_external_growth_dataset(organism)
        df = pd.concat([df, extra.reindex(columns=df.columns)], ignore_index=True)
    return df


def real_kinetic_organisms() -> List[str]:
    """Organisms with at least one real curve in the locally built kinetic-curve file."""
    if not REAL_KINETIC_CSV.exists():
        return []
    df = pd.read_csv(REAL_KINETIC_CSV, usecols=["organism", "series_id"])
    return sorted(df.groupby("organism")["series_id"].nunique().index.tolist())


def load_real_kinetic_curves(organism: Optional[str] = None) -> pd.DataFrame:
    """Real raw growth curves (long format), for :func:`src.ingestion.load_kinetic_series`."""
    if not REAL_KINETIC_CSV.exists():
        raise FileNotFoundError(
            f"{REAL_KINETIC_CSV} not found - run scripts/build_real_datasets.py first"
        )
    df = pd.read_csv(REAL_KINETIC_CSV)
    if organism is not None:
        df = df[df["organism"] == organism].reset_index(drop=True)
    return df


_EXPANDED_GROWTH_CSV = EXPANDED_DATA_DIR / "microbesim_growth_rates_model_ready.csv"
_EXPANDED_OBSERVATIONS_CSV = EXPANDED_DATA_DIR / "microbesim_combined_observations_40k.csv"
_EXPANDED_GROWTH_COLUMNS = (
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
)
_EXPANDED_OBSERVATION_COLUMNS = (
    "curve_id",
    "organism",
    "temperature_celsius",
    "ph",
    "water_activity",
    "time_hours",
    "log10_count",
    "is_synthetic",
    "synthetic_donor_curve_id",
    "split",
    "include_for_modeling",
    "curve_behavior",
    "usable_for_primary_fit",
    "food_category",
)
_EXPANDED_KINETIC_COLUMNS = (
    "series_id",
    "organism",
    "temperature_celsius",
    "ph",
    "water_activity",
    "time_hours",
    "measurement",
    "food_category",
)
_EXPANDED_SPLITS = {"train", "validation", "test"}


def _read_expanded_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Expanded dataset file '{path}' is missing. Restore the expanded data bundle "
            "or regenerate it with scripts/build_clean_and_synthetic.py."
        )
    frame = pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "curve_id": "string",
            "observation_id": "string",
            "record_id": "string",
            "synthetic_donor_curve_id": "string",
        },
    )
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Expanded dataset '{path}' is missing required columns: {missing}")
    return frame


@lru_cache(maxsize=1)
def _expanded_growth_data() -> pd.DataFrame:
    return _read_expanded_csv(_EXPANDED_GROWTH_CSV, _EXPANDED_GROWTH_COLUMNS)


@lru_cache(maxsize=1)
def _expanded_observation_data() -> pd.DataFrame:
    return _read_expanded_csv(_EXPANDED_OBSERVATIONS_CSV, _EXPANDED_OBSERVATION_COLUMNS)


@lru_cache(maxsize=32)
def _expanded_kinetic_data(organism: Optional[str], include_synthetic: bool) -> pd.DataFrame:
    frame = _expanded_observation_data()
    usable = frame["curve_behavior"].eq("growth") & frame["usable_for_primary_fit"].fillna(False)
    if organism is not None:
        usable &= frame["organism"].eq(organism)
    if not include_synthetic:
        usable &= ~frame["is_synthetic"].fillna(False)
    frame = frame.loc[
        usable,
        [
            "organism",
            "temperature_celsius",
            "ph",
            "water_activity",
            "time_hours",
            "food_category",
            "curve_id",
            "log10_count",
        ],
    ]
    return frame.rename(columns={"curve_id": "series_id", "log10_count": "measurement"})[
        list(_EXPANDED_KINETIC_COLUMNS)
    ].reset_index(drop=True)


def load_expanded_growth_dataset(
    organism: Optional[str] = None,
    include_synthetic: bool = True,
    split: Optional[str] = None,
    exclude_suspect: bool = True,
) -> pd.DataFrame:
    """Load model-ready growth rates from the expanded real/synthetic dataset.

    Real rows listed in ``data/suspect_records.csv`` are excluded unless ``exclude_suspect=False``.
    """
    if split is not None and split not in _EXPANDED_SPLITS:
        raise ValueError(f"split must be one of {sorted(_EXPANDED_SPLITS)} or None.")
    frame = _expanded_growth_data()
    selected = pd.Series(True, index=frame.index)
    if organism is not None:
        selected &= frame["organism"].eq(organism)
    if not include_synthetic:
        selected &= ~frame["is_synthetic"].fillna(False)
    if split is not None:
        selected &= frame["split"].eq(split)
    if exclude_suspect:
        selected &= ~(_suspect_mask(frame) & ~frame["is_synthetic"].fillna(False))
    return frame.loc[selected].reset_index(drop=True).copy()


def load_expanded_observations(
    organism: Optional[str] = None,
    include_synthetic: bool = True,
    only_modeling_rows: bool = True,
) -> pd.DataFrame:
    """Load expanded observation-level data, optionally restricting to modeling rows."""
    frame = _expanded_observation_data()
    selected = pd.Series(True, index=frame.index)
    if organism is not None:
        selected &= frame["organism"].eq(organism)
    if not include_synthetic:
        selected &= ~frame["is_synthetic"].fillna(False)
    if only_modeling_rows:
        selected &= frame["include_for_modeling"].fillna(False)
    return frame.loc[selected].reset_index(drop=True).copy()


def load_expanded_kinetic_curves(
    organism: Optional[str] = None,
    include_synthetic: bool = True,
) -> pd.DataFrame:
    """Load usable expanded growth curves in the ComBase kinetic-curve column format."""
    return _expanded_kinetic_data(organism, include_synthetic).copy()
