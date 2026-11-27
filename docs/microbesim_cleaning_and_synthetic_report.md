# MicrobeSim: data cleaning and synthetic expansion report

> **Note on data.** This report describes how the optional `data/expanded` bundle was built. The ComBase-derived files it refers to (`data/combase_*.csv`, `data/expanded/`,
> `data/suspect_records.csv`) are **not distributed with this repository**; see [`data/README.md`](../data/README.md)
> for how to build them locally. Numbers below were produced with such a local bundle.

Built 2026-10-31 by `build_clean_and_synthetic.py` (deterministic, seed 20261031).

## 1. What was wrong with the data, and what was done

| Issue found | Rows | Action |
|---|---|---|
| `-0.01` used as a "not detected" sentinel inside the count column | 1,480 | Count set to NaN, `below_detection_limit=True`, original kept in `log10_count_source_value`. Rows kept so sampling design is preserved. |
| Impossible water activity (aw = 0.08, biscuit records O091_1/O091_2) | 20 | aw set to NaN, flagged `aw_implausible_removed` |
| Organism labels differ from the project (`Salmonella spp`, `Campylobacteraceae (...)`, `... (prot.)`) | 11,722 | Mapped to the names in `src/datasets.py` so real and synthetic files interoperate; original in `organism_source_label` |
| `food_name` case/whitespace variants (148 -> 145 names) | 3 | Normalised to lower case |
| Constant/useless columns (`source_max_rate_original`, `provenance_note`) and a 10 MB raw string column (`source_logcs_original`) | - | Dropped from the working file (still in your original CSV) |
| Exact duplicates | 0 | None left (already removed upstream); verified |
| Missing T / pH / aw | 1,097 / 3,565 / 5,573 | **Not imputed.** Flagged (`qc_flags`, `metadata_complete`). Inventing measurements would contaminate the real data. |

Checked and left alone: temperatures from -76 to 100 C (real freezing and thermal-death studies), counts up to 11.1 log (broth cultures; flagged `count_gt_10_log`), times up to 8,926 h.

## 2. Columns added (see `microbesim_data_dictionary.csv`)
`curve_id`, `is_synthetic`/`data_origin`, `synthetic_donor_curve_id`, `temperature_band`, `ph_class` (4.6 cut-off), `aw_class` (0.85 / 0.92), `log10_change_from_initial`, `ln_change_from_initial`, `nacl_pct`, `assumed_*` flags, `metadata_complete`, `curve_behavior` (growth / decline / flat / single_time), per-curve Gompertz fit (`curve_mu_max_per_h`, `curve_lag_h`, `curve_fit_r2`, `curve_growth_fit_ok`), `curve_slope_log10_per_h`, `usable_for_primary_fit`, `usable_for_secondary_model`, `include_for_modeling`, **`split`** (train/validation/test by curve) and `qc_flags`.

The curve fits reuse your project's `src.models.primary.fit_primary_curve` and the acceptance rules in `src/combase.py` (mu > 1.5e-3, R2 >= 0.8, relative SE <= 50%), so results are consistent with the app.

Result: 2,202 real curves; 979 growth, 1096 decline, 70 flat, 34 single-time. 574 growth curves fit acceptably; 387 also have complete T/pH/aw.

## 3. Synthetic data
Goal: take every organism from its real count to exactly 4,000 observations (the target in your coverage report): **18,045 synthetic rows**, total **40,000**.

How each synthetic curve is made (all from the real data, no outside numbers except the project's cardinal-parameter profiles as a prior):
1. Pick a real curve of that organism at random (the "donor"). This reproduces your real mix of foods, temperatures, pH/aw grids, missing-value patterns, number of points and time design. About 35% of curves get a small off-grid perturbation of T / pH / aw.
2. Growth donors: Baranyi curve using the donor's fitted rate, lag and rise (scaled for any condition perturbation with the project's `true_mu` surface and an organism calibration factor); donors without a usable fit use the mechanistic prior or the empirical rise.
3. Decline donors: Weibull-type die-off with the donor's total log reduction; flat donors: flat with tiny drift.
4. Noise: replicate scatter measured from your real data (median per organism, floor 0.05, default 0.15 log10), rounded to 2 decimals, values below 0 turned into the same `-0.01` sentinel convention.
5. The last curve per organism is truncated so the total is exactly 4,000.

Every synthetic curve inherits its donor's `split`, so a synthetic copy of a training curve can never land in test.

### Coverage
| organism                                |   real_observations |   synthetic_observations |   total_observations |   synthetic_share_pct |
|:----------------------------------------|--------------------:|-------------------------:|---------------------:|----------------------:|
| Bacillus cereus                         |                1247 |                     2753 |                 4000 |                  68.8 |
| Campylobacter jejuni                    |                3191 |                      809 |                 4000 |                  20.2 |
| Clostridium botulinum (non-proteolytic) |                1438 |                     2562 |                 4000 |                  64   |
| Clostridium botulinum (proteolytic)     |                2936 |                     1064 |                 4000 |                  26.6 |
| Clostridium perfringens                 |                1969 |                     2031 |                 4000 |                  50.8 |
| Escherichia coli                        |                1891 |                     2109 |                 4000 |                  52.7 |
| Listeria monocytogenes                  |                2724 |                     1276 |                 4000 |                  31.9 |
| Salmonella spp.                         |                1433 |                     2567 |                 4000 |                  64.2 |
| Staphylococcus aureus                   |                2744 |                     1256 |                 4000 |                  31.4 |
| Yersinia enterocolitica                 |                2382 |                     1618 |                 4000 |                  40.4 |

### How closely does it match the real data?
Kolmogorov-Smirnov distance real vs synthetic (0 = identical; under ~0.15 is a close match). `syn_mu_recovery` = fitted / true growth rate on synthetic curves (1.0 = perfect; ~1.1 is the known Gompertz-vs-Baranyi bias).

| organism                                |   KS_T |   KS_pH |   KS_aw |   KS_time |   KS_count |   syn_mu_recovery_median_ratio |
|:----------------------------------------|-------:|--------:|--------:|----------:|-----------:|-------------------------------:|
| Bacillus cereus                         |   0.06 |    0.05 |    0.05 |      0.04 |       0.07 |                           1.17 |
| Campylobacter jejuni                    |   0.08 |    0.1  |    0.15 |      0.05 |       0.11 |                         nan    |
| Clostridium botulinum (non-proteolytic) |   0.07 |    0.09 |    0.09 |      0.07 |       0.15 |                           1.08 |
| Clostridium botulinum (proteolytic)     |   0.18 |    0.09 |    0.07 |      0.05 |       0.09 |                           1.11 |
| Clostridium perfringens                 |   0.03 |    0.14 |    0.13 |      0.03 |       0.06 |                           1.13 |
| Escherichia coli                        |   0.07 |    0.05 |    0.08 |      0.04 |       0.04 |                           1.18 |
| Listeria monocytogenes                  |   0.07 |    0.11 |    0.05 |      0.07 |       0.08 |                           1.1  |
| Salmonella spp.                         |   0.09 |    0.07 |    0.07 |      0.06 |       0.05 |                           1.1  |
| Staphylococcus aureus                   |   0.1  |    0.07 |    0.11 |      0.03 |       0.12 |                           1.13 |
| Yersinia enterocolitica                 |   0.12 |    0.08 |    0.06 |      0.08 |       0.06 |                           1.13 |

Behaviour mix (growth/decline/flat) per organism matches the real curves within a few percentage points (`microbesim_synthetic_validation.csv`).

### Does it help a model? (honest check)
Random Forest on ln(mu_max) from (organism, T, pH, aw), trained on real-only vs real+synthetic training curves, scored on **real held-out test curves** only (5 seeds).

| scope                                   |   n_train_real |   n_train_syn |   n_test_real |   R2_real_only |   R2_real_plus_syn |   RMSE_ln_real_only |   RMSE_ln_real_plus_syn |
|:----------------------------------------|---------------:|--------------:|--------------:|---------------:|-------------------:|--------------------:|------------------------:|
| ALL organisms pooled                    |            321 |           348 |            62 |          0.783 |              0.786 |               0.634 |                   0.63  |
| Clostridium botulinum (non-proteolytic) |             41 |            63 |             9 |         -1.227 |             -2.415 |               0.933 |                   1.155 |
| Clostridium botulinum (proteolytic)     |             55 |            20 |            12 |          0.42  |              0.424 |               0.852 |                   0.848 |
| Staphylococcus aureus                   |             47 |            20 |            11 |          0.796 |              0.711 |               0.379 |                   0.451 |
| Yersinia enterocolitica                 |             79 |            59 |            15 |          0.703 |              0.809 |               0.528 |                   0.424 |

Reading: pooled accuracy is essentially unchanged (R2 0.78 either way); per-organism results move up and down and the test sets are tiny (9-15 curves), so no claim can be made that synthetic data improves accuracy. It is safe for volume, pipeline testing and demos; it is not a substitute for real measurements.

## 4. Limits you should know
- Synthetic rows only recombine structure already in the real data; they cannot add new biology. Conditions outside what was observed are not explored.
- **Campylobacter**: all 261 real curves are decline curves, so the synthetic ones are too. There is still no growth data for it.
- **Salmonella** has only 4 real curves with an accepted growth fit, so its growth calibration falls back to the mechanistic prior. Synthetic Salmonella growth is the least trustworthy.
- Synthetic data has fewer below-detection rows than real (1.8% vs 6.7%).
- Synthetic share is high for Bacillus cereus (69%), Salmonella (64%), non-proteolytic C. botulinum (64%). Always report metrics on real-only test data and keep `is_synthetic` in every analysis.
- Pseudomonas (in your project's synthetic set) has no real data here and was not generated.
- Split by `curve_id`/`split`, never by random row.

## 5. Files
| File | Contents |
|---|---|
| `microbesim_clean_real_observations.csv` | 21,955 cleaned real rows with all new columns |
| `microbesim_synthetic_observations.csv` | 18,045 synthetic rows, same columns |
| `microbesim_combined_observations_40k.csv` | both, 4,000 per organism |
| `microbesim_curve_summary.csv` | one row per curve (real + synthetic) with fits, behaviour, split |
| `microbesim_growth_rates_model_ready.csv` | 1,439 rows in the project's secondary-model format (`organism, temperature_celsius, ph, water_activity, mu_max_per_h, lag_h` + `is_synthetic`, `source`, `split`) |
| `microbesim_dataset_cleaned_and_expanded.xlsx` | Everything above in one workbook with README, dictionary, log |
| `microbesim_organism_coverage_updated.csv`, `..._cleaning_log.csv`, `..._synthetic_validation.csv`, `..._synthetic_utility_check.csv`, `..._synthetic_calibration.csv`, `..._data_dictionary.csv` | Reports |
| `build_clean_and_synthetic.py` | Reproduces all of it. Put in `microbesim/scripts/` and run with `--raw`, `--rates-xlsx`, `--out` |
