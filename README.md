# MicrobeSim

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-software predictive-microbiology workbench. It fits primary growth curves, predicts the maximum
growth rate across temperature / pH / water activity with classical and machine-learning models, and
simulates cold-chain temperature deviations to find when a population breaches a spoilage or legal
threshold. No lab hardware is required.

> **Status: research / screening prototype, not a food-safety tool.** Read [Limitations](#limitations-read-this-first)
> before trusting any number it produces.

## Features

* **Primary fitting.** Modified Gompertz and Baranyi fits (Levenberg-Marquardt with derivative seeding), with
  gap filling, Hampel outlier repair and optional Savitzky-Golay smoothing for CSV/Excel curves.
* **Secondary models.** Ratkowsky and Rosso cardinal-temperature models with pH / water-activity factors,
  Gaussian Process, Random Forest, a Ratkowsky + Extra-Trees hybrid and a mechanistic + ML ensemble. Every model
  returns exactly zero outside its estimated biological limits.
* **Dynamic simulation.** A Baranyi-type ODE system solved with `solve_ivp` (RK45), integrated in ln N, split at
  telemetry time stamps so temperature steps never fall inside a solver step, with the time to breach (`t_crit`)
  found as an exact solver event. Monotone `pchip` interpolation avoids the overshoot of plain cubic splines.
* **Honest validation.** Condition-grouped cross-validation, Af / Bf / RMSE / R² scorecards, flags for
  predictions outside the training range, and an independent external-data validation.
* **Streamlit app.** Fit, predict and simulate workspaces, a saved-runs ledger with account-private history,
  CSV / Excel / PDF export with provenance, plus an audit log, model registry and optional OIDC sign-in.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements-dev.txt
export PYTHONPATH=.                                    # Windows: set PYTHONPATH=.
streamlit run src/app.py
```

Create a local account on first use (passwords need at least 12 characters). The first time you open a dataset
the models are trained and cached, which can take a minute; run `python scripts/pretrain_models.py --jobs 4`
once to prepare every bundled dataset up front. Or use Docker (the image pretrains during the build):

```bash
docker build -t microbesim . && docker run --rm -p 8501:8501 microbesim   # http://localhost:8501
```

### Python API

```python
from src.datasets import generate_growth_dataset
from src.models.secondary import MechanisticModel, frame_to_X
from src.schemas import SimulationScenario
from src.simulation.dynamic import build_mu_lookup, simulate
from src.simulation.scenarios import PRESETS

df = generate_growth_dataset("Listeria monocytogenes")
model = MechanisticModel("ratkowsky").fit(frame_to_X(df), df["mu_max_per_h"].to_numpy())
scenario = SimulationScenario(organism="Listeria", environmental_log=PRESETS["cold_chain_break"].build())
trace = simulate(scenario, build_mu_lookup(model, ph=6.5, aw=0.98), h0=1.0)
print(trace.spoilage_breached, trace.time_to_breach_hours)
```

## Data: what is and is not in this repository

| Data | Included? | Notes |
|---|---|---|
| Synthetic growth datasets (`data/synthetic/`) | Yes | Generated from **illustrative** cardinal parameters. For demonstration only. |
| Sample curves and cold-chain telemetry | Yes | Synthetic examples used by the app. |
| Published growth rates (`data/external/`) | Yes | CC BY 4.0 table with full provenance; see `data/external/ATTRIBUTION.md`. |
| ComBase-derived datasets | **No** | ComBase's terms do not clearly allow redistribution, so they are excluded. Results in `docs/` were produced with a local bundle built from the author's own ComBase exports. |
| Trained models, accounts, saved runs | **No** | Built or created locally; git-ignored. |

Out of the box the app therefore runs on **synthetic data only**. If you have ComBase access you can rebuild the
real-data bundle locally from your own exports and the app picks it up automatically; see
[`data/README.md`](data/README.md). The accuracy figures below come from such a local bundle, not from the
synthetic data shipped here.

## Limitations (read this first)

* **Modest accuracy on unseen conditions.** With a strict protocol (every identical T / pH / aw condition held out
  together), the best model, the ensemble, reaches a mean R² of about **0.55** across nine organisms, and
  individual organisms are as low as about 0.3. An earlier claim of R² 0.73 was retracted after replicate leakage was
  found; see [`docs/MODEL_IMPROVEMENT_REPORT.md`](docs/MODEL_IMPROVEMENT_REPORT.md).
* **Poor transfer to independent data.** On an independent published *Salmonella* / *Listeria* vegetable dataset,
  ComBase-trained models mostly score negative R² and under-predict growth rates (Bf typically 0.2-0.6). Details in
  [`docs/EXTERNAL_DATA.md`](docs/EXTERNAL_DATA.md) and `docs/model_results/external_validation.csv`.
* **Thin data.** About 1,000 real rows across nine organisms in the author's local bundle; *Campylobacter* has too
  little growth data to model at all, and 60 contradicting *E. coli* records are set aside (never deleted).
* **Synthetic results mean nothing about real accuracy.** Model rankings on synthetic data only test the code.
* **Uncertainty bands are nominal**, not calibrated probability intervals.
* **Not validated for regulatory, HACCP or release decisions.** Confirm any result against product-specific
  evidence. The app is a screening and teaching aid.

## Repository layout

| Path | Purpose |
|---|---|
| `src/models/` | Primary (Gompertz, Baranyi), secondary (Ratkowsky, CTM) and ML models; metrics and benchmarking |
| `src/simulation/` | Baranyi-type ODE solver, `t_crit` events, preset failure scenarios |
| `src/ingestion.py`, `src/combase.py` | CSV/Excel parsing and cleaning; ComBase Browser export loader |
| `src/datasets.py`, `src/external_data.py` | Dataset loaders, synthetic generator, licence-checked external data |
| `src/ui/`, `src/app.py` | Streamlit multi-page app |
| `src/accounts.py`, `src/ledger.py`, `src/audit.py`, `src/registry.py`, `src/oidc.py` | Local accounts, saved runs, audit trail, model registry, optional SSO |
| `scripts/` | Data building, training / pretraining, cross-validation and external validation |
| `data/` | Synthetic and published data (see above) |
| `docs/` | PRD, technical spec, data schemas, results and operations notes |
| `tests/` | Pytest suite (about 160 tests) |

Documentation index: [`docs/README.md`](docs/README.md). Accounts, backups, Docker and security notes:
[`docs/ACCOUNTS_AND_DEPLOYMENT.md`](docs/ACCOUNTS_AND_DEPLOYMENT.md).

## Development

```bash
pytest                                    # runs on a fresh clone, no local data needed
black src scripts tests && flake8 src scripts tests
python scripts/generate_sample_data.py   # regenerate the synthetic data in ./data
```

Tests that assert facts about the ComBase-derived files are skipped automatically when those local files are absent.
`requirements-tested.txt` pins the versions the project was verified against; `python scripts/validate_environment.py`
checks your environment against it. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Modelling notes

* **Units.** `mu_max` is in natural-log units (1/h), matching `y = ln(N/N0)` and the ODE; log10 input is converted.
* **Lag and N_max.** `simulate()` takes `h0` (alpha0 = exp(-h0)) and `n_max_cfu_per_g` as arguments because the scenario
  schema has no such fields.
* **Af / Bf** are computed on held-out points with observed `mu_max` > 0.05 1/h; ratios explode near the growth boundary.
* **Uncertainty.** GPR uses its Bayesian predictive interval; RF and mechanistic models use a constant-width residual
  band in sqrt(mu) space.
* **Pretrained models** are pickled (joblib) and tied to exact library versions. Only load model files you built
  yourself; they are never shipped in this repository.

## Acknowledgements

* **ComBase** (combase.cc), the predictive-microbiology database whose exports the author used for the real-data
  results in `docs/`. If you build the local bundle, follow ComBase's terms of use and cite it as they request.
* Son SB, Lee HK, Kim SJ, Yoon KS. *Modeling Behavior of Salmonella spp. and Listeria monocytogenes in Raw and Processed
  Vegetables.* Foods 2024;13(18):2972 (CC BY 4.0), used for the independent external validation. See
  `data/external/ATTRIBUTION.md`.

## License

Source code and synthetic data: [MIT](LICENSE). Third-party data keeps its own licence; see `data/README.md`.
