"""Externally sourced growth-rate data (published studies, lab measurements) with provenance.

Every row must trace to a source in ``data/external/external_sources.csv`` that records the DOI, the licence,
how the numbers were obtained and what was checked. Only data whose licence permits redistribution with
attribution may be bundled (see ``BUNDLE_LICENSES``); anything else must stay out of the repository.

Rows use the same units as the rest of MicrobeSim: ``mu_max_per_h`` is the natural-log maximum specific growth
rate per hour (published log10 rates are multiplied by ln 10), ``lag_h`` is in hours.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

LN10 = float(np.log(10.0))
EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
EXTERNAL_RATES_CSV = EXTERNAL_DIR / "external_growth_rates.csv"
EXTERNAL_SOURCES_CSV = EXTERNAL_DIR / "external_sources.csv"
EXCLUDED_ROWS_CSV = EXTERNAL_DIR / "external_excluded_rows.csv"

# Licences that allow redistribution of the data with attribution. NC/ND licences are deliberately absent:
# a non-commercial or no-derivatives term is not compatible with bundling data into a product.
BUNDLE_LICENSES = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-2.0",
    "public-domain",
    "author-permission-written",  # written permission from the data owner to redistribute (keep the letter)
    "owner-contributed",  # your own measurements, e.g. from your accredited laboratory
}
NO_DOI_NEEDED = {"author-permission-written", "owner-contributed"}
STANDARD_COLUMNS = [
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
REQUIRED_COLUMNS = [
    "record_id",
    "organism",
    "temperature_celsius",
    "ph",
    "water_activity",
    "mu_max_per_h",
    "lag_h",
    "food_category",
    "food_name",
    "source_id",
    "table_ref",
    "n_replicates",
]
ORGANISMS = {
    "Bacillus cereus",
    "Campylobacter jejuni",
    "Clostridium botulinum (non-proteolytic)",
    "Clostridium botulinum (proteolytic)",
    "Clostridium perfringens",
    "Escherichia coli",
    "Listeria monocytogenes",
    "Salmonella spp.",
    "Staphylococcus aureus",
    "Yersinia enterocolitica",
}
FOOD_CATEGORIES = {
    "Culture medium",
    "Pork",
    "Beef",
    "Poultry",
    "Milk",
    "Cheese",
    "Sausage",
    "Seafood/fish",
    "Dessert food",
    "Vegetable or fruit and their products",
    "Other or unknown type of meat",
    "Other or unknown type of dairy",
    "Other, mixed, uncategorised or unknown type of food",
}


class ExternalDataError(ValueError):
    """External data failed validation; the message lists every problem."""


def load_sources() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_SOURCES_CSV)


def validate_external_frame(
    frame: pd.DataFrame, sources: Optional[pd.DataFrame] = None
) -> List[str]:
    """Return a list of problems (empty = valid). Used by the importer, the build script and the tests."""
    problems: List[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        return [f"missing columns: {', '.join(missing)}"]
    sources = load_sources() if sources is None else sources
    by_id = sources.set_index("source_id")
    for column in (
        "temperature_celsius",
        "ph",
        "water_activity",
        "mu_max_per_h",
        "lag_h",
        "n_replicates",
    ):
        frame = frame.assign(**{column: pd.to_numeric(frame[column], errors="coerce")})
    if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        problems.append("record_id must be present and unique")
    bad = ~frame["organism"].isin(ORGANISMS)
    if bad.any():
        problems.append(f"unknown organism: {sorted(frame.loc[bad, 'organism'].unique())}")
    bad = ~frame["food_category"].isin(FOOD_CATEGORIES)
    if bad.any():
        problems.append(
            f"food_category not in the project vocabulary: {sorted(frame.loc[bad, 'food_category'].unique())}"
        )
    for column, low, high in (
        ("temperature_celsius", -5, 70),
        ("ph", 2, 11),
        ("water_activity", 0.5, 1.0),
        ("mu_max_per_h", 1e-4, 8.0),
    ):
        outside = frame[column].isna() | (frame[column] < low) | (frame[column] > high)
        if outside.any():
            problems.append(
                f"{column} missing or outside [{low}, {high}] in rows {list(frame.index[outside][:5])}"
            )
    if (frame["lag_h"].dropna() < 0).any():
        problems.append("lag_h must not be negative")
    unknown = ~frame["source_id"].isin(by_id.index)
    if unknown.any():
        problems.append(
            f"source_id not in external_sources.csv: {sorted(frame.loc[unknown, 'source_id'].unique())}"
        )
    for source_id in frame.loc[~unknown, "source_id"].unique():
        row = by_id.loc[source_id]
        if str(row["license"]).strip().lower() not in BUNDLE_LICENSES:
            problems.append(
                f"source {source_id}: licence '{row['license']}' does not permit bundling"
            )
        if not str(row.get("doi", "")).strip() or str(row.get("doi", "")).strip().lower() == "nan":
            if str(row["license"]).strip().lower() not in NO_DOI_NEEDED:
                problems.append(f"source {source_id}: a DOI is required")
        if str(row.get("status", "")).strip() != "imported":
            problems.append(f"source {source_id}: status is '{row.get('status')}', not 'imported'")
    return problems


def load_external_growth_dataset(
    organism: Optional[str] = None, standard_columns: bool = True
) -> pd.DataFrame:
    """Validated external rows; ``standard_columns`` returns only the columns of the ComBase growth dataset."""
    frame = pd.read_csv(EXTERNAL_RATES_CSV)
    problems = validate_external_frame(frame)
    if problems:
        raise ExternalDataError("; ".join(problems))
    if organism is not None:
        frame = frame[frame["organism"] == organism].reset_index(drop=True)
    return frame[STANDARD_COLUMNS].copy() if standard_columns else frame


def external_organisms(min_rows: int = 5) -> List[str]:
    if not EXTERNAL_RATES_CSV.exists():
        return []
    counts = pd.read_csv(EXTERNAL_RATES_CSV).groupby("organism").size()
    return sorted(counts[counts >= min_rows].index)
