"""Primary growth models and automated non-linear least-squares fitting (FR-1).

Both models describe ``y(t) = ln(N(t) / N0)`` and share the parameter triple
``(A, mu_max, lam)``:

* ``A``       asymptotic increase in ln-count (carrying capacity, ``y_max``)
* ``mu_max``  maximum specific growth rate (1/h, natural-log units)
* ``lam``     lag-phase duration (h)

Fitting uses Levenberg-Marquardt (``scipy.optimize.curve_fit``) seeded from
numerical derivatives of the data, falling back to a bounded trust-region fit
when LM diverges or returns a biologically invalid solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit

E = np.e
_EXP_CLIP = 500.0


class FitError(RuntimeError):
    """Raised when a growth curve cannot be fitted."""


# --------------------------------------------------------------------------- #
# Model equations
# --------------------------------------------------------------------------- #
def gompertz(t, A, mu_max, lam):
    """Modified Gompertz (Zwietering et al., 1990).

    y(t) = A * exp{ -exp[ mu_max * e / A * (lam - t) + 1 ] }
    """
    t = np.asarray(t, dtype=float)
    inner = np.clip(mu_max * E / A * (lam - t) + 1.0, -_EXP_CLIP, _EXP_CLIP)
    return A * np.exp(-np.exp(inner))


def baranyi(t, A, mu_max, lam):
    """Baranyi-Roberts model with ``nu = mu_max`` and ``y(0) = 0``.

    y(t) = mu*A(t) - ln(1 + (exp(mu*A(t)) - 1) / exp(A)),
    A(t) = t + (1/mu) * ln(exp(-mu t) + exp(-h0) - exp(-mu t - h0)),  h0 = mu * lam
    """
    t = np.asarray(t, dtype=float)
    h0 = mu_max * lam
    e_mt = np.exp(-mu_max * t)
    inner = e_mt + np.exp(-h0) * (1.0 - e_mt)
    x = mu_max * t + np.log(np.maximum(inner, 1e-300))  # x = mu * A(t)
    log_one_minus = np.log1p(-np.exp(-A)) if A > 0 else -np.inf
    return x - np.logaddexp(log_one_minus, x - A)


PRIMARY_MODELS: Dict[str, Callable] = {"gompertz": gompertz, "baranyi": baranyi}


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PrimaryFit:
    model: str
    A: float
    mu_max: float
    lam: float
    r_squared: float
    rmse: float
    n_points: int
    method: str
    param_std: Tuple[float, float, float] = (np.nan, np.nan, np.nan)
    extras: Dict[str, float] = field(default_factory=dict)

    @property
    def params(self) -> Tuple[float, float, float]:
        return (self.A, self.mu_max, self.lam)

    @property
    def doubling_time_h(self) -> float:
        return float(np.log(2.0) / self.mu_max)

    def predict(self, t) -> np.ndarray:
        return PRIMARY_MODELS[self.model](t, *self.params)


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def _local_slopes(t: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Least-squares slope in a sliding window (robust to point-wise noise)."""
    n = len(t)
    half = window // 2
    slopes = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        tt, yy = t[lo:hi], y[lo:hi]
        tc = tt - tt.mean()
        denom = float(np.dot(tc, tc))
        slopes[i] = float(np.dot(tc, yy - yy.mean()) / denom) if denom > 0 else 0.0
    return slopes


def initial_guess(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Derivative-based priors for (A, mu_max, lam)."""
    window = int(min(7, max(3, (len(t) // 8) | 1)))
    slopes = _local_slopes(t, y, window)
    k = int(np.argmax(slopes))
    mu0 = max(float(slopes[k]), 1e-3)
    y_base = float(np.min(y[: max(3, len(y) // 10)]))
    lam0 = float(t[k] - (y[k] - y_base) / mu0)
    lam0 = float(np.clip(lam0, 0.0, max(float(t[-1]) * 0.9, 1e-3)))
    a0 = max(float(np.max(y) - y_base) * 1.05, 0.2)
    return a0, mu0, lam0


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _valid(params: np.ndarray, t_max: float) -> bool:
    A, mu, lam = params
    return bool(
        np.all(np.isfinite(params)) and A > 0 and mu > 0 and 0.0 <= lam <= t_max * 1.5 + 1.0
    )


def _goodness(y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    resid = y - y_hat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return r2, float(np.sqrt(sse / len(y)))


def fit_primary_curve(t, y, model: str = "gompertz", min_points: int = 6) -> PrimaryFit:
    """Fit a primary growth model to ``y = ln(N/N0)`` data.

    Raises
    ------
    FitError
        If the input is unusable or no valid solution can be found.
    """
    if model not in PRIMARY_MODELS:
        raise ValueError(f"Unknown primary model '{model}'. Choose from {list(PRIMARY_MODELS)}")
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise FitError("time and response arrays must be one-dimensional and equal length")
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < min_points:
        raise FitError(f"need at least {min_points} valid points, got {len(t)}")
    if float(np.ptp(t)) <= 0:
        raise FitError(
            "all time values are identical (not a time series - likely replicate counts)"
        )
    order = np.argsort(t)
    t, y = t[order], y[order]
    if float(np.ptp(y)) < 0.1:
        raise FitError("no measurable growth (range of ln-response < 0.1)")

    func = PRIMARY_MODELS[model]
    p0 = np.array(initial_guess(t, y))
    t_max = float(t[-1])

    popt = pcov = None
    method = "lm"
    try:
        popt, pcov = curve_fit(func, t, y, p0=p0, method="lm", maxfev=4000)
        if not _valid(popt, t_max):
            popt = None
    except (RuntimeError, ValueError, FloatingPointError):
        popt = None

    if popt is None:  # bounded trust-region fallback
        method = "trf"
        lower = [1e-3, 1e-4, 0.0]
        upper = [max(float(np.max(y)) * 4.0, 2.0), max(50.0 * p0[1], 5.0), max(t_max, 1.0)]
        p0_b = np.clip(p0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
        try:
            popt, pcov = curve_fit(func, t, y, p0=p0_b, bounds=(lower, upper), maxfev=4000)
        except (RuntimeError, ValueError) as exc:
            raise FitError(f"{model} fit did not converge: {exc}") from exc
        if not _valid(popt, t_max):
            raise FitError(f"{model} fit produced non-physical parameters: {popt}")

    with np.errstate(invalid="ignore"):
        std = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(3, np.nan)
    r2, rmse = _goodness(y, func(t, *popt))
    return PrimaryFit(
        model=model,
        A=float(popt[0]),
        mu_max=float(popt[1]),
        lam=float(popt[2]),
        r_squared=r2,
        rmse=rmse,
        n_points=len(t),
        method=method,
        param_std=tuple(float(s) for s in std),
    )
