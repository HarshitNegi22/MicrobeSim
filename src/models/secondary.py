"""Secondary models: growth rate as a function of T, pH and water activity (FR-2).

Contains the classical mechanistic models

* extended square-root (Ratkowsky) model for temperature,
* Rosso cardinal temperature model (CTM),
* cardinal (gamma) factors for pH and water activity,

wrapped in :class:`MechanisticModel`, plus the :class:`GrowthRateModel`
interface shared with the machine-learning regressors in ``ml_regressors``.

All models enforce biological boundaries: no growth at or beyond the
cardinal limits (e.g. ``T >= Tmax  =>  mu_max = 0``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

FEATURES = ("temperature_celsius", "ph", "water_activity")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_X(T, ph, aw) -> np.ndarray:
    """Stack broadcastable (T, pH, aw) inputs into an ``(n, 3)`` design matrix."""
    T, ph, aw = np.broadcast_arrays(
        np.atleast_1d(np.asarray(T, dtype=float)),
        np.atleast_1d(np.asarray(ph, dtype=float)),
        np.atleast_1d(np.asarray(aw, dtype=float)),
    )
    return np.column_stack([T.ravel(), ph.ravel(), aw.ravel()])


def frame_to_X(df) -> np.ndarray:
    """Extract the feature matrix from a DataFrame with the canonical column names."""
    return df[list(FEATURES)].to_numpy(dtype=float)


@dataclass(frozen=True)
class GrowthLimits:
    """Cardinal growth boundaries. Outside them ``mu_max`` is exactly zero."""

    tmin: float = -np.inf
    tmax: float = np.inf
    phmin: float = -np.inf
    phmax: float = np.inf
    awmin: float = -np.inf

    def inside(self, X: np.ndarray) -> np.ndarray:
        T, ph, aw = X[:, 0], X[:, 1], X[:, 2]
        return (
            (T > self.tmin)
            & (T < self.tmax)
            & (ph > self.phmin)
            & (ph < self.phmax)
            & (aw > self.awmin)
        )

    @classmethod
    def from_data(cls, X: np.ndarray, mu: np.ndarray) -> "GrowthLimits":
        """Fallback: the envelope of conditions where growth was observed."""
        g = np.asarray(mu) > 0
        if not g.any():
            return cls()
        Xg = X[g]
        eps = 1e-9
        return cls(
            tmin=float(Xg[:, 0].min() - eps),
            tmax=float(Xg[:, 0].max() + 1e-6),
            phmin=float(Xg[:, 1].min() - eps),
            phmax=float(Xg[:, 1].max() + 1e-6),
            awmin=float(Xg[:, 2].min() - eps),
        )


# --------------------------------------------------------------------------- #
# Model equations
# --------------------------------------------------------------------------- #
def ratkowsky_sqrt_mu(T, b, tmin, c, tmax):
    """sqrt(mu_max) = b (T - Tmin) (1 - exp(c (T - Tmax)))  [raw, unclipped]."""
    T = np.asarray(T, dtype=float)
    return b * (T - tmin) * (1.0 - np.exp(np.clip(c * (T - tmax), -500, 500)))


def ratkowsky_mu(T, b, tmin, c, tmax):
    """mu_max from the extended square-root model, zero outside (Tmin, Tmax)."""
    T = np.asarray(T, dtype=float)
    root = ratkowsky_sqrt_mu(T, b, tmin, c, tmax)
    return np.where((T > tmin) & (T < tmax), np.maximum(root, 0.0) ** 2, 0.0)


def cardinal_gamma(x, xmin, xopt, xmax, n: float = 1.0):
    """Rosso cardinal gamma factor in [0, 1] (1 at ``xopt``, 0 outside (xmin, xmax))."""
    x = np.asarray(x, dtype=float)
    num = (x - xmax) * (x - xmin) ** n
    den = (xopt - xmin) ** (n - 1.0) * (
        (xopt - xmin) * (x - xopt) - (xopt - xmax) * ((n - 1.0) * xopt + xmin - n * x)
    )
    inside = (x > xmin) & (x < xmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(inside, num / den, 0.0)
    return np.clip(np.nan_to_num(g), 0.0, 1.0)


def rosso_ctm(T, mu_opt, tmin, topt, tmax):
    """Rosso cardinal temperature model: mu_max = mu_opt * gamma_T(T), n = 2."""
    return mu_opt * cardinal_gamma(T, tmin, topt, tmax, n=2.0)


def water_activity_gamma(aw, awmin, awopt):
    """Square-root-type aw factor: ((aw - awmin) / (awopt - awmin))**2, capped at 1."""
    aw = np.asarray(aw, dtype=float)
    g = ((aw - awmin) / max(awopt - awmin, 1e-9)) ** 2
    return np.where(aw > awmin, np.clip(g, 0.0, 1.0), 0.0)


# --------------------------------------------------------------------------- #
# Model interface
# --------------------------------------------------------------------------- #
class GrowthRateModel(ABC):
    """Common interface: ``fit(X, mu)``, ``predict(X)``, ``predict_interval(X)``.

    ``X`` is an ``(n, 3)`` array ordered ``[temperature_celsius, ph, water_activity]``.
    Subclasses model ``sqrt(mu_max)`` and implement :meth:`_sqrt_mean_std`; the base
    class squares, gates by :class:`GrowthLimits` and builds intervals.
    """

    name: str = "model"
    limits: Optional[GrowthLimits] = None
    fitted: bool = False

    @abstractmethod
    def fit(
        self, X: np.ndarray, mu: np.ndarray, limits: Optional[GrowthLimits] = None
    ) -> "GrowthRateModel": ...

    @abstractmethod
    def _sqrt_mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean and std of sqrt(mu_max) *before* boundary gating."""

    def _check(self, X) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"{self.name} has not been fitted")
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 3)

    def _gate(self, X: np.ndarray, values: np.ndarray) -> np.ndarray:
        if self.limits is None:
            return values
        return np.where(self.limits.inside(X), values, 0.0)

    def predict(self, X) -> np.ndarray:
        X = self._check(X)
        mean, _ = self._sqrt_mean_std(X)
        return self._gate(X, np.maximum(mean, 0.0) ** 2)

    def predict_interval(self, X, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mean, lower, upper)`` for mu_max (1/h) with a ``level`` band."""
        X = self._check(X)
        z = float(norm.ppf(0.5 + level / 2.0))
        mean, std = self._sqrt_mean_std(X)
        m = np.maximum(mean, 0.0)
        return (
            self._gate(X, m**2),
            self._gate(X, np.maximum(m - z * std, 0.0) ** 2),
            self._gate(X, (m + z * std) ** 2),
        )

    def predict_env(self, T, ph, aw) -> np.ndarray:
        return self.predict(make_X(T, ph, aw))


# --------------------------------------------------------------------------- #
# Mechanistic model (T x pH x aw)
# --------------------------------------------------------------------------- #
def _seed_temperature(kind: str, T: np.ndarray, mu: np.ndarray, t_range=None):
    """Starting values and bounds for the temperature part; growth rows only.

    ``T``/``mu`` are the reference rows used for seeding. ``t_range`` is the (min, max)
    temperature of *all* growth rows: Tmin/Tmax bounds must respect it, otherwise a fit seeded
    from a subset can place Tmax below temperatures where growth was actually observed.
    """
    Tu = np.unique(T)
    s = np.array([np.sqrt(np.median(mu[T == u])) for u in Tu])
    k = int(np.argmax(s))
    low = Tu <= Tu[k]
    if low.sum() < 2:
        low = np.arange(len(Tu)) < 2
    slope, intercept = np.polyfit(Tu[low], s[low], 1)
    slope = max(float(slope), 1e-3)
    tmin0 = min(-intercept / slope, float(Tu.min()) - 0.5)
    tmax0 = float(Tu.max()) + 2.0
    t_lo, t_hi = float(Tu.min()), float(Tu.max())
    g_lo, g_hi = (
        (t_lo, t_hi)
        if t_range is None
        else (min(t_lo, float(t_range[0])), max(t_hi, float(t_range[1])))
    )
    tmax0 = max(tmax0, g_hi + 1.0)
    tmin_b = (g_lo - 30.0, g_lo - 1e-3)
    tmax_b = (g_hi + 1e-3, g_hi + 20.0)
    if kind == "ratkowsky":
        names = ["b", "tmin", "c", "tmax"]
        x0 = [slope, tmin0, 0.2, tmax0]
        lb = [1e-4, tmin_b[0], 0.005, tmax_b[0]]
        ub = [5.0, tmin_b[1], 2.0, tmax_b[1]]
    else:
        names = ["mu_opt", "tmin", "topt", "tmax"]
        x0 = [float(s[k] ** 2), tmin0, float(Tu[k]), tmax0]
        lb = [1e-3, tmin_b[0], g_lo, tmax_b[0]]
        ub = [10.0 * float(mu.max()) + 1.0, tmin_b[1], g_hi, tmax_b[1]]
    return names, x0, lb, ub


class MechanisticModel(GrowthRateModel):
    """Classical secondary model: ``mu = f_T(T) * gamma_pH(pH) * gamma_aw(aw)``.

    ``kind='ratkowsky'`` uses the extended square-root model for ``f_T``;
    ``kind='ctm'`` uses the Rosso cardinal temperature model.
    pH uses the cardinal (n = 1) factor, water activity the square-root factor.
    A factor is disabled (gamma = 1) if the training data do not vary in it.
    """

    def __init__(self, kind: str = "ratkowsky"):
        if kind not in ("ratkowsky", "ctm"):
            raise ValueError("kind must be 'ratkowsky' or 'ctm'")
        self.kind = kind
        self.name = "Ratkowsky (mechanistic)" if kind == "ratkowsky" else "Rosso CTM (mechanistic)"
        self.params_: Dict[str, float] = {}
        self.use_ph = False
        self.use_aw = False
        self.resid_std_ = 0.0

    # -- equations ---------------------------------------------------------- #
    def _mu(self, X: np.ndarray, p: Dict[str, float]) -> np.ndarray:
        T, ph, aw = X[:, 0], X[:, 1], X[:, 2]
        if self.kind == "ratkowsky":
            mu = ratkowsky_mu(T, p["b"], p["tmin"], p["c"], p["tmax"])
        else:
            mu = rosso_ctm(T, p["mu_opt"], p["tmin"], p["topt"], p["tmax"])
        if self.use_ph:
            mu = mu * cardinal_gamma(ph, p["phmin"], p["phopt"], p["phmax"], 1.0)
        if self.use_aw:
            mu = mu * water_activity_gamma(aw, p["awmin"], p["awopt"])
        return mu

    # -- fitting ------------------------------------------------------------ #
    def fit(self, X, mu, limits: Optional[GrowthLimits] = None) -> "MechanisticModel":
        X = np.asarray(X, dtype=float).reshape(-1, 3)
        mu = np.asarray(mu, dtype=float)
        grow = mu > 0
        if grow.sum() < 8 or len(np.unique(X[grow, 0])) < 4:
            raise ValueError("need >= 8 growth observations at >= 4 distinct temperatures")
        Xg, mug = X[grow], mu[grow]

        # Temperature seeds use conditions closest to the pH/aw optimum.
        ref = np.ones(len(mug), dtype=bool)
        if len(np.unique(Xg[:, 2])) > 1:
            ref &= Xg[:, 2] >= np.quantile(Xg[:, 2], 0.75) - 1e-12
        if len(np.unique(Xg[:, 1])) > 1:
            phs = np.round(Xg[ref, 1], 1)
            best = phs[np.argmax(mug[ref])]
            ref &= np.abs(np.round(Xg[:, 1], 1) - best) < 0.55
        if len(np.unique(Xg[ref, 0])) < 4:
            ref = np.ones(len(mug), dtype=bool)
        names, x0, lb, ub = _seed_temperature(
            self.kind, Xg[ref, 0], mug[ref], t_range=(float(Xg[:, 0].min()), float(Xg[:, 0].max()))
        )

        ph_g, aw_g = Xg[:, 1], Xg[:, 2]
        self.use_ph = len(np.unique(np.round(ph_g, 6))) >= 2
        self.use_aw = len(np.unique(np.round(aw_g, 6))) >= 2
        fixed: Dict[str, float] = {}
        if self.use_ph:
            lo, hi = float(ph_g.min()), float(ph_g.max())
            best_ph = float(np.clip(Xg[np.argmax(mug), 1], lo, hi))
            names += ["phmin", "phopt", "phmax"]
            x0 += [max(lo - 0.4, 0.6), best_ph, min(hi + 0.6, 13.9)]
            lb += [0.5, lo, hi + 1e-3]
            ub += [lo - 1e-3, hi, 14.0]
        if self.use_aw:
            lo = float(aw_g.min())
            fixed["awopt"] = float(aw_g.max())
            names += ["awmin"]
            x0 += [max(lo - 0.02, 0.31)]
            lb += [0.3]
            ub += [lo - 1e-3]
        x0 = np.clip(np.array(x0, float), np.array(lb) + 1e-6, np.array(ub) - 1e-6)
        lb, ub = np.array(lb), np.array(ub)

        sqrt_obs = np.sqrt(np.maximum(mu, 0.0))

        def residuals(theta):
            p = dict(zip(names, theta))
            p.update(fixed)
            return np.sqrt(np.maximum(self._mu(X, p), 0.0)) - sqrt_obs

        sol = least_squares(residuals, x0, bounds=(lb, ub), x_scale="jac", max_nfev=400)
        p = dict(zip(names, sol.x))
        p.update(fixed)
        self.params_ = {k: float(v) for k, v in p.items()}
        res = residuals(sol.x)
        self.resid_std_ = float(np.sqrt(np.mean(res[grow] ** 2)))

        q = self.params_
        self.limits = limits or GrowthLimits(
            tmin=q["tmin"],
            tmax=q["tmax"],
            phmin=q["phmin"] if self.use_ph else -np.inf,
            phmax=q["phmax"] if self.use_ph else np.inf,
            awmin=q["awmin"] if self.use_aw else -np.inf,
        )
        self.fitted = True
        return self

    def _sqrt_mean_std(self, X):
        mu = self._mu(X, self.params_)
        return np.sqrt(np.maximum(mu, 0.0)), np.full(len(X), self.resid_std_)


def estimate_limits(X: np.ndarray, mu: np.ndarray) -> GrowthLimits:
    """Estimate cardinal boundaries from data via a mechanistic fit (data envelope fallback)."""
    try:
        return MechanisticModel("ratkowsky").fit(X, mu).limits
    except (ValueError, RuntimeError):
        return GrowthLimits.from_data(np.asarray(X, float).reshape(-1, 3), np.asarray(mu, float))
