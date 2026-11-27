# Real ComBase data

> **Note on data.** This document describes the optional real-data bundle. The ComBase-derived files it refers to (`data/combase_*.csv`, `data/expanded/`,
> `data/suspect_records.csv`) are **not distributed with this repository**; see [`data/README.md`](../data/README.md)
> for how to build them locally. Numbers below were produced with such a local bundle.

MicrobeSim's secondary models can now train on **real growth data** extracted from
ComBase Browser exports, alongside the original synthetic (illustrative-parameter)
datasets. This document says exactly what's in it, how it was built, and what its
limits are.

## What was provided

Two ComBase Browser export batches (`.xlsx`, 20 records/file, the site's per-search
cap), covering 10 organism labels: *Escherichia coli, Salmonella* spp., *Listeria
monocytogenes, Staphylococcus aureus, Campylobacter jejuni, Bacillus cereus,
Clostridium perfringens, Clostridium botulinum* (proteolytic and non-proteolytic),
*Yersinia enterocolitica*.

## How it was cleaned (`src/combase.py`)

ComBase exports mix several kinds of record that this project is not scoped to model,
so every record is classified and every exclusion is counted (nothing is silently
dropped) - run `scripts/build_real_datasets.py` on any export to see the exact
counts. A record is:

1. **Kept** if it's an isothermal (constant T/pH/Aw) record with a positive growth
   rate - either ComBase's own fitted rate, or one fitted here from the raw curve.
2. **Excluded as decline/inactivation** if the rate is negative or the curve trends
   downward (freezing, thermal death, curing studies - most of what's in a typical
   ComBase export for some organisms; see Campylobacter below).
3. **Excluded as "No growth"** when ComBase records that explicitly.
4. **Excluded as no significant change** ("flat" - rate is exactly zero, or the curve
   moves less than 0.2 log10 units either way).
5. **Set aside as a dynamic profile** if T, pH or Aw *varied during the experiment*
   (ComBase stores these as a semicolon-packed time-series string instead of a single
   number, with companion `Temperatures`/`pHs`/`Aws` sheets). These aren't used for
   secondary-model training - they need a different model (see "Future work" below).
6. **Excluded on data-quality grounds** - and this is the part worth knowing about,
   because two of these were discovered by testing on the real files, not anticipated
   in advance:
   - *Curve too short* (fewer than 6 points) to fit a primary growth model.
   - *Degenerate curve*: every point stamped at the same time value. This turned out
     to be how ComBase encodes replicate plate counts at a single instant, not a time
     series - naively curve-fitting it produces a meaningless result.
   - *Low-confidence fit*: the fit is rejected if it's pinned at the optimizer's floor,
     has R² < 0.8, **or has a relative standard error on mu_max above 50%**. That last
     check matters most: a curve where the entire log-count rise happens inside a
     single gap between samples can still show a deceptively good R² (the fit
     "connects the dots" fine) while the growth-rate estimate itself is almost
     completely unconstrained. One real example (Bacillus cereus, 28°C) fit to
     mu_max = 28.6/h - a ~1.4-minute doubling time, faster than any known bacterium -
     with a parameter standard error 100,000x the estimate itself. Requiring the
     *relative* standard error to be reasonable catches this while still keeping
     genuinely fast organisms (Clostridium perfringens is a real ~7-8 min doubler at
     its optimum, and those points pass the check with 20-40% relative error).

Rates are converted from ComBase's log10-per-hour units to the natural-log units
(1/h) the rest of MicrobeSim uses (`mu_max_ln = mu_max_log10 * ln(10)`).

## Per-organism coverage (after cleaning)

The score column below is a **legacy single random-holdout snapshot**. The live
benchmark now uses repeated holdouts grouped by food category when that field is
available, and shows split-to-split standard deviation. `record_id` is not a verified
study identifier, so food-category grouping tests matrix transfer but cannot guarantee
that all related records from one study stay together.

| Organism | Usable growth rows | Distinct temperatures | Legacy random-holdout snapshot |
|---|---|---|---|
| Escherichia coli | 190 | 46 | Rosso CTM (R²=0.88, Af=1.28) |
| Bacillus cereus | 173 | 19 | Ratkowsky (R²=0.20, Af=1.72) |
| Staphylococcus aureus | 162 | 39 | Random Forest (R²=0.98, Af=1.48) |
| Yersinia enterocolitica | 162 | 17 | Random Forest (R²=0.79, Af=1.41) |
| Clostridium perfringens | 107 | 16 | Random Forest (R²=0.37, Af=1.61) |
| Clostridium botulinum (proteolytic) | 93 | 14 | Rosso CTM (R²=0.82, Af=1.51) |
| Listeria monocytogenes | 84 | 25 | Random Forest (R²=0.99, Af=1.49) |
| Clostridium botulinum (non-proteolytic) | 71 | 20 | Rosso CTM (Af=1.79, **R²=-0.25**) |
| Salmonella spp. | 35 | 11 | Random Forest (R²=0.37, Af=1.46) |
| Campylobacter jejuni | 6 | 1 | **not trained** |

Af (accuracy factor) close to 1 is a good sign regardless of R²; it's the standard
predictive-microbiology accuracy metric and is less sensitive to a handful of
outliers than R². `scripts/build_real_datasets.py`'s printed report and
`data/combase_load_report.csv` have the full record-level accounting.

## Known weak spots - read before trusting a number

- **Campylobacter jejuni has no usable growth data.** All 100 records pulled by that
  search were survival/decline studies (frozen storage, thermal death) - which is
  plausible biology (Campylobacter is notoriously hard to grow in food matrices and
  is mostly studied for die-off), but it means this organism still falls back to the
  synthetic dataset in the app. **Fix**: re-run the ComBase search filtered
  specifically to growth-type experiments for this organism.
- **Salmonella (35 rows, 11 temperatures) is thin.** Its models work but are the
  least-supported of the well-populated organisms; treat its predictions as more
  approximate, especially near the edges of the temperature range.
- **Clostridium botulinum (non-proteolytic)'s Rosso CTM fit has negative R²** on
  held-out data - worse than predicting the mean. The Ratkowsky fit for the same
  organism is fine (R²=0.53); this looks like the CTM's cardinal-parameter
  optimization landing in a bad local optimum for this particular data spread, not a
  fundamental data problem. **If you use this organism, prefer Ratkowsky or an ML
  model over CTM**, or investigate refitting CTM with different seeds.
- **Staphylococcus aureus's mechanistic models (Ratkowsky/CTM) show Af ≈ 17** despite
  a good R² (0.88) - a few held-out points near the growth boundary, where the
  mechanistic model predicts a very small but nonzero rate, are blowing up the
  ratio-based accuracy metric. The ML models don't show this (Af ≈ 1.5), so **prefer
  Random Forest/GPR over the mechanistic models for this organism** if accuracy near
  the boundary matters to you.
- **All data comes from mixed food matrices** (broth, poultry, dairy, dessert, etc.),
  not a single controlled substrate. That's realistic, but matrix-specific inhibitors
  (nitrite, smoke compounds, competing flora) aren't modeled by the three-variable
  (T, pH, Aw) surface - they show up as unexplained scatter, which is part of why R²
  is modest for some organisms (e.g. Bacillus cereus, R²≈0.2).
- **A handful of `Assumed` values.** Some records have Aw or pH marked `Assumed` by
  ComBase (estimated rather than measured) rather than `NaN` - these are kept (the
  `assumed` column in `data/combase_growth_dataset.csv` flags which), since ComBase's
  assumptions are usually reasonable defaults, but it's a soft spot worth knowing
  about if you're auditing a specific prediction.

## Regenerating / extending this data

```bash
python scripts/build_real_datasets.py /path/to/export_folder_1 /path/to/export_folder_2 ...
```

Each argument is a folder of `ComBaseExport*.xlsx` files (searched recursively, any
subfolder structure). Re-run any time you add another export - it rebuilds
`data/combase_growth_dataset.csv`, `data/combase_kinetic_curves.csv`,
`data/combase_dynamic_profiles.csv` and `data/combase_load_report.csv` from scratch.
Organism names are normalised (see `ORGANISM_ALIASES` in `src/combase.py`) so
differently-worded exports for the same organism merge automatically; add an alias
there if a new export uses yet another spelling.

**To fix Campylobacter or thicken Salmonella**: re-export from ComBase Browser with
the search filtered to growth experiments (not survival/inactivation) for that
organism, and feed the new folder in alongside the existing ones.

## Future work: the dynamic-profile data

146 records (129 unique) had a genuinely time-varying temperature, pH or water
activity - mostly frozen-storage studies. These are set aside in
`data/combase_dynamic_profiles.csv` (`record_id, organism, time_hours,
temperature_celsius, ph, water_activity, log10_count`) rather than discarded, because
they're exactly the right shape to become **real validation data for the cold-chain
simulator** (Page 3): each one is a real measured environmental history *and* a real
measured outcome, which can be used to check the ODE simulator against observations.
The simulator accepts uploaded time-varying pH and water-activity columns in addition
to temperature, but these ComBase profiles are not automatically loaded as scenarios
or used for validation. A separate evaluation workflow is still needed to match the
model's initial population and matrix assumptions to each record before comparing
predicted and observed `log10_count`.
