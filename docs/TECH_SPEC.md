# Technical Specification: MicrobeSim

## 1. Tech Stack
* **Runtime:** Python 3.11+
* **Numerical & ODE Solvers:** `numpy`, `scipy` (`scipy.optimize.curve_fit`, `scipy.integrate.solve_ivp`, `scipy.interpolate.interp1d`)
* **Machine Learning:** `scikit-learn` (GaussianProcessRegressor, RandomForestRegressor)
* **Data Ingestion & Validation:** `pandas`, `pydantic`
* **Presentation Tier:** `streamlit`, `plotly`
* **Testing & Code Quality:** `pytest`, `black`, `flake8`

## 2. Mathematical Formulations

### Primary Growth Model: Modified Gompertz
$$y(t) = A \cdot \exp\left\{ -\exp\left[ \frac{\mu_{\max} \cdot e}{A} (\lambda - t) + 1 \right] \right\}$$
* $y(t) = \ln(N(t) / N_0)$
* $A$: Asymptotic carrying capacity ($y_{\max}$)
* $\mu_{\max}$: Maximum specific growth rate ($1/\text{h}$)
* $\lambda$: Duration of the lag phase ($\text{h}$)

### Secondary Mechanistic Model: Ratkowsky Extended Square-Root
$$\sqrt{\mu_{\max}} = b \cdot (T - T_{\min}) \cdot \left(1 - \exp\left(c \cdot (T - T_{\max})\right)\right)$$
* $T_{\min}, T_{\max}$: Theoretical cardinal temperature boundaries
* $b, c$: Fitted regression constants

### Dynamic Growth Solver (ODE System)
$$\frac{dN}{dt} = \alpha(t) \cdot \mu_{\max}\big(T(t), \text{pH}, a_w\big) \cdot \left(1 - \frac{N(t)}{N_{\max}}\right) \cdot N(t)$$
$$\frac{d\alpha}{dt} = \mu_{\max} \cdot \alpha(t) \cdot (1 - \alpha(t))$$
* Solved via `scipy.integrate.solve_ivp` (Runge-Kutta 4th/5th order method `RK45`).

## 3. Machine Learning Architectures
* **Gaussian Process Regressor (GPR):** Uses a composite kernel `ConstantKernel * RBF(length_scale) + WhiteKernel(noise_level)` to capture non-linear thermal denaturation and return nominal predictive ranges. These ranges are not calibrated for independent studies or simulation outcomes.
* **Tree-Based Regressor (Random Forest):** Trained on multi-hurdle inputs ($T$, pH, $a_w$) to model non-linear boundaries and parameter interactions.

## 4. Validation, Input Limits, and Interpretation

* Uploaded tables are limited to 25 MB, 100,000 rows, and 100 columns. Compressed Excel workbook
  contents are additionally limited to 100 MB expanded size. Growth-rate model inputs must be
  finite and numeric; pH must be 0–14 and water activity must be greater than 0 and at most 1.
  Kinetic files are capped at 500 curves and 200,000 cleaned points per curve, including
  interpolated gap-fill points.
* Validation can group records by a supplied study, experiment, batch, or other identifier so
  that a group does not appear in both train and test partitions. Food-category grouping is not
  a substitute for verified study-level separation.
* Reported model ranges are nominal intervals. Their empirical coverage has not been calibrated
  for independent studies, full simulated trajectories, or threshold-crossing times.
* Random Forest training is capped at four worker threads to reduce contention on shared hosts.

The simulator still requires validation against measured dynamic histories before its screening
outputs should be treated as operational evidence. No calibration data or verified study IDs are
created by this implementation.