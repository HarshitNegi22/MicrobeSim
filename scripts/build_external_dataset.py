"""Build data/external/ from published tables, with provenance and transcription checks.

The numbers below were transcribed once from the CC BY 4.0 article cited in SOURCE and are the audit trail.
Run:  python scripts/build_external_dataset.py   (deterministic; fails loudly if any check fails)

Checks (a failed check stops the build):
  1. Every transcribed rate at T >= 25 C must agree with the paper's own fitted secondary model
     (its Table 4 square-root equation) within 25% (36 C within 6%): catches digit slips.
  2. Within each series the rate must not decrease and the lag must not increase with temperature.
  3. Rows whose block duplicates another block's rates and densities are EXCLUDED, not imported.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.external_data import (  # noqa: E402
    EXCLUDED_ROWS_CSV,
    EXTERNAL_RATES_CSV,
    EXTERNAL_SOURCES_CSV,
    LN10,
    validate_external_frame,
)

SOURCE_ID = "Son2024_Foods_vegetables"
ACCESSED = "2026-11-04"

# matrix -> (pH, aw) from the paper's Table 1 (means)
MATRIX = {
    "Mung bean sprouts": (6.30, 0.951),
    "Onion": (6.30, 0.955),
    "Cabbage": (6.71, 0.948),
    "RTE shredded cabbage salad": (6.80, 0.962),
    "Cabbage juice": (5.20, 0.964),
}
# (T C, lag h, lag SD, SGR log10 CFU/h, SGR SD, max density log10, MPD SD)   -- paper Table 2 (Salmonella)
SALMONELLA = {
    "Mung bean sprouts": [
        (8, 170.1, 3.07, 0.012, 0.000, 8.47, 0.01),
        (10, 16.47, 1.93, 0.035, 0.001, 8.31, 0.04),
        (17, 7.11, 0.12, 0.209, 0.001, 8.82, 0.00),
        (25, 3.36, 0.28, 0.467, 0.025, 9.28, 0.13),
        (36, 1.85, 0.07, 1.138, 0.004, 9.30, 0.04),
    ],
    "Onion": [
        (9, 169.50, 4.80, 0.012, 0.000, 3.43, 0.05),
        (15, 24.33, 0.26, 0.114, 0.004, 4.70, 0.07),
        (25, 6.18, 0.14, 0.332, 0.004, 5.96, 0.02),
        (36, 4.79, 0.00, 1.308, 0.040, 7.54, 0.08),
    ],
    "Cabbage": [
        (9, 292.35, 59.25, 0.013, 0.005, 3.89, 0.03),
        (10, 68.48, 0.80, 0.015, 0.000, 4.42, 0.02),
        (17, 14.95, 0.18, 0.120, 0.000, 6.04, 0.06),
        (25, 5.34, 0.11, 0.312, 0.007, 6.69, 0.03),
        (36, 4.10, 0.00, 0.716, 0.000, 6.73, 0.00),
    ],
    "RTE shredded cabbage salad": [
        (10, 121.10, 1.80, 0.024, 0.000, 4.42, 0.02),
        (17, 17.11, 0.69, 0.058, 0.000, 4.76, 0.03),
        (25, 4.46, 0.37, 0.210, 0.005, 5.59, 0.01),
        (36, 1.02, 0.16, 0.432, 0.000, 7.49, 0.01),
    ],
    "Cabbage juice": [
        (17, 42.37, 1.11, 0.096, 0.005, 3.37, 0.01),
        (25, 9.65, 0.02, 0.240, 0.006, 4.93, 0.00),
        (36, 6.36, 0.03, 0.709, 0.008, 7.08, 0.01),
    ],
}
LISTERIA = {  # paper Table 3
    "Mung bean sprouts": [
        (4, 83.86, 1.81, 0.018, 0.000, 8.45, 0.01),
        (10, 13.08, 0.92, 0.048, 0.001, 6.55, 0.09),
        (17, 4.38, 0.05, 0.090, 0.001, 6.17, 0.01),
        (25, 3.13, 0.19, 0.187, 0.017, 6.38, 0.06),
        (36, 1.05, 0.06, 0.306, 0.001, 6.51, 0.00),
    ],
    "Onion": [
        (4, 363.00, 0.50, 0.011, 0.000, 4.67, 0.06),
        (10, 25.37, 1.37, 0.030, 0.001, 4.81, 0.07),
        (17, 3.89, 0.14, 0.086, 0.001, 4.85, 0.00),
        (25, 2.73, 0.02, 0.260, 0.001, 5.27, 0.00),
        (36, 1.15, 0.12, 0.435, 0.007, 5.45, 0.07),
    ],
    "Cabbage": [
        (4, 371.57, 6.31, 0.010, 0.001, 4.67, 0.06),
        (10, 15.37, 6.00, 0.030, 0.001, 4.81, 0.07),
        (17, 7.04, 0.38, 0.086, 0.001, 4.85, 0.00),
        (25, 4.50, 0.51, 0.260, 0.001, 5.27, 0.00),
        (36, 1.19, 0.00, 0.435, 0.007, 5.45, 0.07),
    ],
    "RTE shredded cabbage salad": [
        (4, 306.77, 6.32, 0.012, 0.001, 4.68, 0.00),
        (10, 16.90, 1.87, 0.025, 0.001, 4.95, 0.01),
        (17, 7.75, 0.13, 0.075, 0.001, 4.94, 0.00),
        (25, 1.52, 0.38, 0.200, 0.004, 5.21, 0.03),
        (36, 1.04, 0.38, 0.364, 0.003, 6.01, 0.07),
    ],
    "Cabbage juice": [
        (17, 29.14, 2.46, 0.026, 0.000, 2.87, 0.04),
        (25, 7.54, 0.92, 0.087, 0.003, 5.00, 0.18),
        (36, 1.06, 0.03, 0.322, 0.001, 6.34, 0.03),
    ],
}
# Paper's own secondary model for the rate: SGR = (b * (T - Tmin))^2 (Table 4)
TABLE4 = {
    ("S", "Mung bean sprouts"): (0.03344, 4.169),
    ("S", "Onion"): (0.04756, 12.07),
    ("S", "Cabbage"): (0.02666, 4.202),
    ("S", "RTE shredded cabbage salad"): (0.02014, 3.198),
    ("S", "Cabbage juice"): (0.02986, 7.91),
    ("L", "Mung bean sprouts"): (0.01286, -7.497),
    ("L", "Onion"): (0.01789, -1.381),
    ("L", "Cabbage"): (0.01688, -3.122),
    ("L", "RTE shredded cabbage salad"): (0.01649, -0.8631),
    ("L", "Cabbage juice"): (0.02345, 11.87),
}
CODE = {
    "Mung bean sprouts": "MBS",
    "Onion": "ONI",
    "Cabbage": "CAB",
    "RTE shredded cabbage salad": "RTE",
    "Cabbage juice": "CJU",
}
EXCLUDE = {
    (
        "L",
        "Cabbage",
    ): "Four of five growth rates (10-36 C) and all five maximum densities are identical to the Onion block "
    "of the same table (lags differ), which indicates a copy-paste error in the published table"
}


def build():
    rows, excluded = [], []
    for tag, organism, table, tref, strains in (
        (
            "S",
            "Salmonella spp.",
            SALMONELLA,
            "Table 2",
            "Salmonella cocktail (S. Typhimurium, S. Enteritidis; S. Bareilly in validation)",
        ),
        (
            "L",
            "Listeria monocytogenes",
            LISTERIA,
            "Table 3",
            "L. monocytogenes cocktail (serotypes 1/2a and 1/2b)",
        ),
    ):
        for matrix, series in table.items():
            ph, aw = MATRIX[matrix]
            ts = [s[0] for s in series]
            sgr = [s[3] for s in series]
            lag = [s[1] for s in series]
            assert ts == sorted(ts), (organism, matrix)
            assert all(
                b >= a for a, b in zip(sgr, sgr[1:])
            ), f"rate decreases with T: {organism} {matrix}"
            assert all(
                b <= a for a, b in zip(lag, lag[1:])
            ), f"lag increases with T: {organism} {matrix}"
            b_, tmin = TABLE4[(tag, matrix)]
            for t, lt, lt_sd, rate, rate_sd, mpd, _ in series:
                pred = (b_ * (t - tmin)) ** 2
                if t >= 25:
                    tol = 0.06 if t == 36 else 0.25
                    assert (
                        abs(pred / rate - 1) <= tol
                    ), f"transcription check failed: {organism} {matrix} {t} C: {rate} vs model {pred:.3f}"
                rec = {
                    "record_id": f"EXT_Son2024_{tag}_{CODE[matrix]}_{t:g}C",
                    "organism": organism,
                    "temperature_celsius": float(t),
                    "ph": ph,
                    "water_activity": aw,
                    "mu_max_per_h": round(rate * LN10, 6),
                    "lag_h": lt,
                    "food_category": "Vegetable or fruit and their products",
                    "food_name": matrix.lower(),
                    "assumed": "",
                    "source": "published_table (Son 2024)",
                    "source_id": SOURCE_ID,
                    "table_ref": f"{tref}; pH/aw from Table 1",
                    "n_replicates": 4,
                    "rate_log10_per_h_published": rate,
                    "rate_sd_log10_per_h": rate_sd,
                    "lag_sd_h": lt_sd,
                    "max_density_log10": mpd,
                    "strain_note": strains,
                    "evaluation_role": "independent_validation",
                }
                if (tag, matrix) in EXCLUDE:
                    excluded.append({**rec, "exclusion_reason": EXCLUDE[(tag, matrix)]})
                else:
                    rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(excluded)


SOURCES = [
    {
        "source_id": SOURCE_ID,
        "status": "imported",
        "citation": "Son SB, Lee HK, Kim SJ, Yoon KS. Modeling Behavior of Salmonella spp. and Listeria monocytogenes in Raw and "
        "Processed Vegetables. Foods 2024;13(18):2972.",
        "doi": "10.3390/foods13182972",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11430865/",
        "license": "cc-by-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "retrieved_on": ACCESSED,
        "obtained_how": "Transcribed from the article's Tables 1-3 (rate converted from log10 to ln by x ln 10); cross-checked "
        "against the article's Table 4 equations",
        "checks": "rates at >=25 C agree with the paper's own model; monotone rate/lag in each series; duplicate block excluded",
        "notes": "Growth kinetics fitted with the modified Gompertz model (n=4 per mean). Surface growth on vegetables, strain "
        "cocktails. Rows with no growth (ND) are not represented. Listeria on cabbage excluded (see external_excluded_rows.csv).",
        "attribution": "Data from Son et al. (2024), Foods 13(18):2972, doi:10.3390/foods13182972, CC BY 4.0; converted to natural-log "
        "units and combined with other data.",
    },
    {
        "source_id": "ArtiSaneFood_Zenodo_8088477",
        "status": "candidate_not_imported",
        "citation": "Valero A, Gonzales-Barron U, Cadavez V, et al. Database containing raw data of growth and survival of pathogens "
        "and microbiota from fate studies in artisanal fermented foods (ArtiSaneFood D5.5).",
        "doi": "10.5281/zenodo.8088477",
        "url": "https://zenodo.org/records/8088477",
        "license": "cc-by-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "retrieved_on": ACCESSED,
        "obtained_how": "not obtained",
        "checks": "",
        "notes": "CC BY 4.0, raw challenge-test data; the single .xlsx could not be downloaded from the build environment (network "
        "allow-list; the fetch tool cannot read spreadsheets). A newer version exists. Download it yourself, convert to the "
        "template in data/external/templates/ and import with scripts/import_external_data.py.",
        "attribution": "",
    },
    {
        "source_id": "Wang2024_Toxins_Bcereus",
        "status": "candidate_not_imported",
        "citation": "Wang Y, Liu Y, Yang S, et al. Effect of temperature, pH, and aw on cereulide synthesis ... with respect to Bacillus "
        "cereus growth and cereulide production. Toxins 2024;16(1):32.",
        "doi": "10.3390/toxins16010032",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10818934/",
        "license": "cc-by-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "retrieved_on": ACCESSED,
        "obtained_how": "not obtained",
        "checks": "",
        "notes": "Measured B. cereus growth at 8 temperature/pH/aw combinations (15-45 C, pH 5-8, aw 0.945-0.996) but the article reports "
        "only maximum population and cereulide, not growth rates. Data available on request from the first author: ask for the "
        "fitted mu_max and lag per condition (and cite the correspondence).",
        "attribution": "",
    },
    {
        "source_id": "ComBase_browser",
        "status": "candidate_not_imported",
        "citation": "ComBase (USDA-ARS ERRC / University of Tasmania / IFR). https://www.combase.cc",
        "doi": "",
        "url": "https://combasebrowser.errc.ars.usda.gov/",
        "license": "combase-terms-nontransferable",
        "license_url": "https://combasebrowser.errc.ars.usda.gov/membership/disclaimer.aspx",
        "retrieved_on": ACCESSED,
        "obtained_how": "not obtained",
        "checks": "",
        "notes": "Terms grant a non-transferable, non-exclusive licence to use ComBase data; a data.gov listing says public domain. The two "
        "do not obviously agree, so confirm with the ComBase partners before redistributing exports. Needs a registered browser "
        "session; keep record identifiers and cite ComBase.",
        "attribution": "",
    },
]


def main() -> int:
    main_df, excluded_df = build()
    sources = pd.DataFrame(SOURCES)
    problems = validate_external_frame(main_df, sources)
    if problems:
        print("VALIDATION FAILED:\n  " + "\n  ".join(problems))
        return 1
    EXTERNAL_RATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(EXTERNAL_RATES_CSV, index=False)
    excluded_df.to_csv(EXCLUDED_ROWS_CSV, index=False)
    sources.to_csv(EXTERNAL_SOURCES_CSV, index=False)
    print(main_df.groupby(["organism", "food_name"]).size().to_string())
    print(
        f"\nWrote {len(main_df)} rows ({len(excluded_df)} excluded) and {len(sources)} source records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
