# External data: sources, provenance and independent validation

## What was added
`data/external/` holds growth rates from published studies, each row tied to a source record with DOI, licence,
how the numbers were obtained and what was checked.

| Source | Organisms | Rows | Licence | Status |
|---|---|---|---|---|
| Son et al. 2024, *Foods* 13(18):2972, doi:10.3390/foods13182972 (vegetable matrices, 4-36 C) | *Salmonella* spp. (21), *L. monocytogenes* (18) | 39 | CC BY 4.0 | imported |

Five *Listeria*-on-cabbage rows were **excluded**: in the published table four of the five rates (10-36 C) and all five
maximum densities repeat the onion block exactly (the lags differ), which points to a copy-paste error
(`external_excluded_rows.csv`). The authors could be asked to confirm.
Rows are natural-log rates (published log10 rate x 2.302585) with the published lag, SD and replicate count.
Rates were transcribed by hand from the article's tables; the build script (`scripts/build_external_dataset.py`)
cross-checks every rate at 25 C and above against the paper's own fitted equation and fails on a mismatch. Low-
temperature rows have no such cross-check beyond monotonicity.

The data is **opt-in**: the app offers *"... (local real + published)"* next to the plain *"(installed real)"* ComBase-based entry for
*Salmonella* and *Listeria*. Nothing else changed. Attribution text: `data/external/ATTRIBUTION.md`.

## Independent validation (the important result)
Models trained **only on ComBase rows** were scored on these external rows (`scripts/validate_on_external.py`,
`docs/model_results/external_validation.csv`). Af and Bf are accuracy and bias factors (1.0 is perfect; Bf < 1 means the
model **under-predicts** growth, the unsafe direction).

*Salmonella* spp. (21 external rows, only 7 inside the ComBase range):

| Model | R² | Af | Bf |
|---|---|---|---|
| baseline: training median | -0.05 | 5.06 | 1.79 |
| Ratkowsky (mechanistic) | -0.44 | 5.24 | 0.23 |
| Rosso CTM (mechanistic) | -0.44 | 5.14 | 0.24 |
| Gaussian Process (ML) | -0.19 | 3.21 | 0.91 |
| Random Forest (ML) | -0.20 | 2.77 | 0.55 |
| Hybrid: Ratkowsky + Extra Trees | -0.48 | 7.01 | 0.17 |
| Ensemble (mechanistic + ML) | -0.33 | 3.35 | 0.43 |

*Listeria monocytogenes* (18 external rows, 14 inside the ComBase range):

| Model | R² | Af | Bf |
|---|---|---|---|
| baseline: training median | -0.84 | 6.06 | 0.26 |
| Ratkowsky (mechanistic) | -0.05 | 3.44 | 0.32 |
| Rosso CTM (mechanistic) | -0.85 | 3.18 | 0.34 |
| Gaussian Process (ML) | 0.63 | 4.26 | 0.40 |
| Random Forest (ML) | 0.57 | 2.55 | 0.56 |
| Hybrid: Ratkowsky + Extra Trees | -0.06 | 4.00 | 0.30 |
| Ensemble (mechanistic + ML) | 0.30 | 2.90 | 0.41 |

Reading this honestly:
* For *Salmonella* every model has **negative R²** (worse than predicting the training median) and under-predicts by
  roughly 2-6x. ComBase *Salmonella* data (35 rows) did not prepare the models for fast growth on vegetables at 36 C.
* For *Listeria* the Gaussian Process and Random Forest transfer moderately (R² 0.57-0.63) but still under-predict by
  about 2x; the mechanistic models and the Ensemble are weaker.
* Caveats: 18-21 rows, one study, strain cocktails, surface growth fitted with the Gompertz model, part of the gap is
  extrapolation beyond the ComBase range. None of this makes the numbers wrong, but it does mean **ComBase-trained
  MicrobeSim models must not be assumed to hold for a specific product without a product-specific check.**

## Sources considered and why they are or are not in the repository
| Source | Finding | Action |
|---|---|---|
| **ComBase** | Terms grant a *non-transferable, non-exclusive* licence to use data; a data.gov entry says public domain. These do not obviously agree. Access needs a registered browser session that this environment cannot operate. | Not added. Ask the ComBase partners whether exports may be redistributed in your product. Keep record identifiers and cite ComBase. Prefer raw time courses plus metadata. |
| **ArtiSaneFood challenge-test database** (Zenodo 10.5281/zenodo.8088477, CC BY 4.0) | Raw pathogen growth/survival data in fermented foods. A single `.xlsx`; not downloadable here (network allow-list, and the fetch tool cannot read spreadsheets). A newer version exists. | **You can add it**: download, convert to the template, import (below). |
| **Wang et al. 2024, *Toxins* 16(1):32** (*B. cereus*, CC BY) | Measured growth at 8 temperature/pH/aw combinations but reports only maximum population and cereulide, not rates. | Nothing importable. Email the authors for fitted mu_max and lag per condition (data are stated to be available on request); cite the correspondence. |
| **USDA-ARS PMP** | Model outputs for specific conditions, not bulk raw data; its tutorial stresses validating on the food of interest. | Use as an independent *comparison*, never as evidence MicrobeSim is validated. |
| **A qualified food-microbiology lab** | Product-specific data are worth more than many unrelated broth rows. | Run a small designed study in your product with an accredited lab and import it as `owner-contributed`. |

## How to add data you obtain
1. Get the data legally (check the licence; NC/ND licences cannot be bundled; ComBase needs its owners' answer).
2. Put it in the template `data/external/templates/external_growth_rates_template.csv` (natural-log rate per hour, lag in
   hours, project food categories, the DOI/table reference).
3. Dry run, then apply:

       python scripts/import_external_data.py rows.csv --source-id Author2025_Journal
       python scripts/import_external_data.py rows.csv --source-id Author2025_Journal --apply --register-source \
           --citation "..." --doi 10.xxxx/yyyy --url https://... --license cc-by-4.0 --obtained-how "Table S2" \
           --attribution "Data from ..., CC BY 4.0"

4. `python scripts/validate_on_external.py` to see how the existing models fare on it **before** training on it, then
   `python scripts/pretrain_models.py` to include it in the saved models.
Keep external rows for a genuinely independent test set whenever you can: data you train on cannot validate.
