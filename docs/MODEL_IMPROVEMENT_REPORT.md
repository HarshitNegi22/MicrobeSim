# Model improvement report (corrected)

> **Note on data.** This report describes results on the optional real-data bundle. The ComBase-derived files it refers to (`data/combase_*.csv`, `data/expanded/`,
> `data/suspect_records.csv`) are **not distributed with this repository**; see [`data/README.md`](../data/README.md)
> for how to build them locally. Numbers below were produced with such a local bundle.

**Correction.** An earlier version of this report (and the summary Doc derived from it) claimed a mean R2 of
0.73. That number came from a lenient validation: whole ComBase *records* were kept together, but many records
share identical conditions, so replicate conditions leaked between folds and tree models scored well by memorising
them. Every number below uses a strict protocol and **supersedes** the earlier ones.

## Evaluation protocol
Repeated (3x) 5-fold cross-validation on real rows in which **every identical (T, pH, aw) condition is held out
together**, i.e. the model is always scored at conditions it has never seen. Scores are computed on the same
clean rows for before and after. Reproduce the machinery with `python scripts/evaluate_models_cv.py`
(note: that script groups by record; the strict-condition runs behind the tables below are in
`docs/model_results/`). Campylobacter (6 growth rows) cannot be modelled and is excluded.

## 1. Headline (mean over 9 organisms)

| Model | R2 before | R2 after | Af before | Af after |
|---|---|---|---|---|
| Rosso CTM | 0.074 | 0.361 | 2.12 | 1.89 |
| Ratkowsky | 0.439 | 0.531 | 2.03 | 1.81 |
| Gaussian Process | 0.154 | 0.458 | 2.16 | 1.74 |
| Random Forest | 0.429 | 0.486 | 1.74 | 1.69 |
| Hybrid: Ratkowsky + Extra Trees (new) | n/a | 0.530 | n/a | 1.73 |
| **Ensemble: mechanistic + ML (new)** | n/a | **0.555** | n/a | **1.60** |

Every original model improved. The best new model (Ensemble) is +0.12 R2 over the best original model
(Ratkowsky 0.44), with Af 1.60 vs 1.74. Worst organism: 0.29 (Ensemble) vs 0.03 (original Random Forest).
The honest expectation for a *new, unseen* condition is therefore R2 around 0.5, not 0.7.

## 2. What was wrong, and what changed

1. **A contradicting block of E. coli records (data).** Records `CA_Ec886`-`CA_Ec945` (60 rows, 39-40 C, pH 7.3,
   aw 0.997) report about 0.03/h, while the same pH/aw gives about 1.7/h at 37 C and 1.4/h at 45 C. No smooth
   temperature response can do that. Fitted on the other 104 E. coli rows, the mechanistic model reaches R2 0.92 and
   predicts about 2.1/h for the block's conditions. A robust loss and a multi-start fit did not help, so the
   equations were never the problem. The same 60 records are in the app's own `combase_growth_dataset.csv`.
   A conservative detector (`src/models/data_quality.py`: flags a batch only if the organism's other rows fit well,
   the batch lies inside their range, and >=80% of the batch is more than 3-fold off) found **only this block**
   in either dataset. The rows are listed with reasons in `data/suspect_records.csv`, set aside by default,
   never deleted. `load_real_growth_dataset(..., exclude_suspect=False)` restores them. **This is a judgement
   call: please check those records against your ComBase source.**
   Effect: E. coli Ratkowsky 0.19 -> 0.72, CTM 0.04 -> **0.83**, Gaussian Process 0.13 -> 0.80.
2. **Maximum-growth-temperature bug (`secondary.py`).** Tmax was bounded using only rows near the pH/aw optimum, so
   it could land below temperatures where growth was observed (B. cereus: 37.8 C despite growth at 45 C), which zeroed
   those predictions in every model. Fixed, with a regression test. B. cereus Ratkowsky 0.13 -> 0.37; C. perfringens CTM
   -0.90 -> 0.48. **Cost:** S. aureus Ratkowsky fell 0.45 -> 0.32 under the strict test.
3. **Blank train/test splits (data).** 639 real rows had no `split` and were silently unused by the split-aware
   benchmark. Now assigned by record; every organism with >= 10 records has test rows.
4. **Better models (`ml_regressors.py`).** Tuned Random Forest and Matern-kernel Gaussian Process; new Hybrid
   (Ratkowsky backbone + Extra Trees on its residuals); new Ensemble = equal-weight average of Ratkowsky, Gaussian
   Process, Random Forest and Hybrid. Weights are deliberately not tuned: weights tuned on 8 organisms scored
   R2 0.49 on the held-out ninth, below the 0.555 of equal weights.

## 3. Per-organism R2 (strict, scored on clean rows)

| Organism | Rat. before | CTM before | GPR before | RF before | Rat. after | CTM after | GPR after | RF after | Hybrid | Ensemble |
|---|---|---|---|---|---|---|---|---|---|---|
| Bacillus cereus | 0.13 | -0.25 | -0.14 | 0.04 | 0.37 | 0.03 | 0.04 | 0.13 | 0.31 | **0.29** |
| Clostridium botulinum (non-proteolytic) | 0.78 | 0.25 | 0.36 | 0.40 | 0.78 | 0.25 | 0.67 | 0.58 | 0.79 | **0.74** |
| Clostridium botulinum (proteolytic) | 0.45 | 0.45 | -0.05 | 0.32 | 0.45 | 0.45 | 0.41 | 0.29 | 0.46 | **0.44** |
| Clostridium perfringens | 0.51 | -0.90 | -0.17 | 0.54 | 0.51 | 0.48 | 0.54 | 0.50 | 0.36 | **0.54** |
| Escherichia coli | 0.19 | 0.04 | 0.13 | 0.77 | 0.72 | 0.83 | 0.80 | 0.74 | 0.72 | **0.78** |
| Listeria monocytogenes | 0.98 | 0.86 | 0.97 | 0.96 | 0.98 | 0.84 | 0.97 | 0.96 | 0.98 | **0.98** |
| Salmonella spp. | 0.38 | 0.07 | 0.11 | 0.03 | 0.38 | 0.07 | 0.37 | 0.02 | 0.21 | **0.38** |
| Staphylococcus aureus | 0.45 | 0.08 | 0.09 | 0.51 | 0.32 | 0.04 | 0.18 | 0.62 | 0.72 | **0.49** |
| Yersinia enterocolitica | 0.07 | 0.06 | 0.08 | 0.30 | 0.26 | 0.26 | 0.15 | 0.53 | 0.22 | **0.36** |

## 4. What did not improve, and why
- **No single model wins everywhere.** CTM is best for E. coli, the Hybrid for S. aureus (0.72), Random Forest for
  Yersinia (0.53), Ratkowsky for Salmonella. The Ensemble is the safe default, not the winner everywhere.
- **The tree residual learner hurts where the mechanistic backbone is already good** (C. perfringens, Salmonella,
  Yersinia): it memorises replicate conditions. That is why the Hybrid alone is not better than Ratkowsky.
- **B. cereus (0.29-0.37), Yersinia (0.36), Salmonella (0.38; 24 rows), C. botulinum proteolytic (0.44)** stay weak.
  Temperature, pH and water activity do not explain these data; adding food category did not help.
- **Campylobacter**: 6 growth rows, all other curves are decline curves. Not modellable yet.
- **Synthetic rows** gave no consistent gain (mixed by organism), so real-only remains the default.

## 5. Pretrained models
`data/pretrained_models/` holds every model the app can show (55 datasets: 9 real organisms x validation groupings,
11 synthetic, 18 expanded training sources), trained with the exact library versions in `requirements-tested.txt`.
`load_or_train` looks in the local cache, then the pretrained store, and only then trains. The sidebar shows
whether pretrained models are ready for the running environment. Saved models are tied to library versions and to
`src/models/*.py` / `src/model_cache.py`: after upgrading libraries or editing those files, run
`python scripts/pretrain_models.py` (the Dockerfile does it at build time). Uploaded datasets cannot be pretrained;
they train once on first use and are then cached.

## 6. Cautions
- The model and blend choices were made on this same cross-validation, so "after" is somewhat optimistic for brand-new
  data (choices were generic, not per-organism).
- I could not verify that the `CA_Ec` records are erroneous, only that they contradict everything around them.
- The Streamlit app was exercised through automated UI smoke tests, not by hand in a browser.

## 7. Files
`docs/model_results/strict_cv_summary.csv`, `strict_cv_per_organism.csv`, `blend_search_strict_cv.csv`;
`data/suspect_records.csv`; `scripts/pretrain_models.py`, `scripts/flag_suspect_records.py`,
`scripts/evaluate_models_cv.py`.
