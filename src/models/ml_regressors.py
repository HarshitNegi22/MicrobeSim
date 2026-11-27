"""Machine-learning secondary models (FR-2).

All regressors learn ``sqrt(mu_max)`` from growth observations (variance-stabilising,
and it keeps predictions non-negative once squared) and are wrapped by the
:class:`~src.models.secondary.GrowthRateModel` interface, which zeroes predictions
outside the biological growth limits regardless of what the regressor extrapolates.

GPR uncertainty uses its Bayesian predictive standard deviation, including the fitted
noise term. Random Forest uses a constant-width band from out-of-bag residuals.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.models.secondary import (
    GrowthLimits,
    GrowthRateModel,
    MechanisticModel,
    estimate_limits,
)

MIN_TRAIN_ROWS = 10


def _prepare(model: GrowthRateModel, X, mu, limits: Optional[GrowthLimits]):
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    mu = np.asarray(mu, dtype=float)
    grow = mu > 0
    if grow.sum() < MIN_TRAIN_ROWS:
        raise ValueError(f"{model.name}: need >= {MIN_TRAIN_ROWS} growth observations")
    model.limits = limits if limits is not None else estimate_limits(X, mu)
    return X[grow], np.sqrt(mu[grow])


# --------------------------------------------------------------------------- #
# Gaussian process
# --------------------------------------------------------------------------- #
class GPRModel(GrowthRateModel):
    """GPR with kernel ``ConstantKernel * Matern(nu=1.5, ARD length scales) + WhiteKernel``.

    The Matern 3/2 kernel is rougher than the RBF kernel, which suits the abrupt changes in growth
    rate near cardinal limits; wider length-scale bounds stop the optimiser stalling at an edge.
    """

    name = "Gaussian Process (ML)"

    def __init__(self, n_restarts: int = 1, max_train: int = 300, random_state: int = 0):
        self.n_restarts = n_restarts
        self.max_train = max_train
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.gpr_: Optional[GaussianProcessRegressor] = None

    def fit(self, X, mu, limits: Optional[GrowthLimits] = None) -> "GPRModel":
        Xg, yg = _prepare(self, X, mu, limits)
        if len(Xg) > self.max_train:  # O(n^3): thin out deterministically
            idx = np.random.default_rng(self.random_state).choice(
                len(Xg), self.max_train, replace=False
            )
            Xg, yg = Xg[idx], yg[idx]
        Z = self.scaler_.fit_transform(Xg)
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(3), length_scale_bounds=(5e-2, 1e3), nu=1.5
        ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
        self.gpr_ = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
        ).fit(Z, yg)
        self.fitted = True
        return self

    def _sqrt_mean_std(self, X):
        mean, std = self.gpr_.predict(self.scaler_.transform(X), return_std=True)
        return mean, np.nan_to_num(std)


# --------------------------------------------------------------------------- #
# Tree ensembles
# --------------------------------------------------------------------------- #
class _TreeModel(GrowthRateModel):
    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.est_ = None
        self.resid_std_ = 0.0

    def _make(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, mu, limits: Optional[GrowthLimits] = None):
        Xg, yg = _prepare(self, X, mu, limits)
        self.est_ = self._make().fit(Xg, yg)
        self.resid_std_ = self._residual_std(Xg, yg)
        self.fitted = True
        return self

    def _sqrt_mean_std(self, X):
        return self.est_.predict(X), np.full(len(X), self.resid_std_)


class RandomForestModel(_TreeModel):
    name = "Random Forest (ML)"  # leaf size 1 + 2/3 of features per split: tuned by grouped CV

    def __init__(self, n_estimators: int = 180, random_state: int = 0):
        super().__init__(random_state)
        self.n_estimators = n_estimators

    def _make(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=1,
            max_features=0.67,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=min(4, os.cpu_count() or 1),
        )

    def _residual_std(self, Xg, yg):
        oob = self.est_.oob_prediction_  # free out-of-sample residuals
        ok = np.isfinite(oob)
        return float(np.sqrt(np.mean((oob[ok] - yg[ok]) ** 2))) if ok.any() else 0.0


def _condition_groups(X: np.ndarray) -> np.ndarray:
    """Group id per distinct (T, pH, aw) so replicate conditions never straddle a CV split."""
    _, inverse = np.unique(np.round(X, 4), axis=0, return_inverse=True)
    return inverse.ravel()


def _grouped_cv_residual_std(make, X: np.ndarray, y: np.ndarray, folds: int = 3) -> float:
    """Out-of-sample RMSE with replicate conditions held together (honest interval width)."""
    groups = _condition_groups(X)
    if len(np.unique(groups)) < folds:
        return float(np.std(y)) * 0.5
    oof = np.zeros(len(y))
    for train, test in GroupKFold(folds).split(X, groups=groups):
        oof[test] = make().fit(X[train], y[train]).predict(X[test])
    return float(np.sqrt(np.mean((oof - y) ** 2)))


class ExtraTreesModel(_TreeModel):
    """Extremely randomised trees on sqrt(mu_max): smoother than a forest on small, replicated data."""

    name = "Extra Trees (ML)"

    def __init__(self, n_estimators: int = 220, random_state: int = 0):
        super().__init__(random_state)
        self.n_estimators = n_estimators

    def _make(self):
        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=min(4, os.cpu_count() or 1),
        )

    def _residual_std(self, Xg, yg):
        return _grouped_cv_residual_std(
            lambda: ExtraTreesRegressor(150, random_state=self.random_state), Xg, yg
        )


# --------------------------------------------------------------------------- #
# Hybrid and ensemble
# --------------------------------------------------------------------------- #
class HybridModel(GrowthRateModel):
    """Ratkowsky backbone plus a learned correction of its residuals (sqrt scale).

    The mechanistic part keeps sensible shape and boundaries; the residual learner captures what
    the equation cannot (food/matrix clusters, interactions). ``residual="et"`` uses Extra Trees,
    ``residual="gpr"`` a Matern Gaussian process. If the backbone cannot be fitted the model
    degrades to the learner on the raw target rather than failing.
    """

    _NAMES = {
        "et": "Hybrid: Ratkowsky + Extra Trees",
        "gpr": "Hybrid: Ratkowsky + Gaussian Process",
    }

    def __init__(self, residual: str = "et", random_state: int = 0):
        if residual not in self._NAMES:
            raise ValueError("residual must be 'et' or 'gpr'")
        self.residual = residual
        self.name = self._NAMES[residual]
        self.random_state = random_state
        self.mech_: Optional[MechanisticModel] = None
        self.resid_model_ = None
        self.scaler_ = StandardScaler()
        self.resid_std_ = 0.0

    def _base(self, X):
        if self.mech_ is None:
            return np.zeros(len(X))
        return np.sqrt(np.maximum(self.mech_._mu(X, self.mech_.params_), 0.0))

    def _make(self, n_estimators: int = 180):
        if self.residual == "et":
            return ExtraTreesRegressor(
                n_estimators,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=min(4, os.cpu_count() or 1),
            )
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(3), length_scale_bounds=(5e-2, 1e3), nu=1.5
        ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
        return GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=self.random_state
        )

    def _features(self, X, fit: bool = False):
        if self.residual == "et":
            return X
        return self.scaler_.fit_transform(X) if fit else self.scaler_.transform(X)

    def fit(self, X, mu, limits: Optional[GrowthLimits] = None):
        Xg, yg = _prepare(self, X, mu, limits)
        try:
            self.mech_ = MechanisticModel("ratkowsky").fit(X, mu)
        except ValueError:
            self.mech_ = None
        residual = yg - self._base(Xg)
        Z = self._features(Xg, fit=True)
        self.resid_model_ = self._make().fit(Z, residual)
        self.resid_std_ = _grouped_cv_residual_std(lambda: self._make(150), Z, residual)
        self.fitted = True
        return self

    def _sqrt_mean_std(self, X):
        mean = self._base(X) + self.resid_model_.predict(self._features(X))
        return np.maximum(mean, 0.0), np.full(len(X), self.resid_std_)


class EnsembleModel(GrowthRateModel):
    """Equal-weight average (sqrt scale) of Ratkowsky, Gaussian process, Random Forest and Hybrid.

    The members err differently: the mechanistic equation extrapolates sensibly but cannot see
    clusters, the learners fit clusters but memorise replicate conditions. Averaging them was the
    most accurate and most stable choice when scored at *unseen* conditions (grouped CV that keeps
    identical conditions together). Weights are deliberately NOT tuned: weights tuned on some
    organisms did not transfer to others. Members that cannot be fitted for an organism are skipped;
    the spread between members adds to the reported uncertainty.
    """

    name = "Ensemble (mechanistic + ML)"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.members_: list = []

    def _candidates(self):
        return [
            MechanisticModel("ratkowsky"),
            GPRModel(random_state=self.random_state),
            RandomForestModel(random_state=self.random_state),
            HybridModel("et", random_state=self.random_state),
        ]

    def fit(self, X, mu, limits: Optional[GrowthLimits] = None):
        X = np.asarray(X, dtype=float).reshape(-1, 3)
        mu = np.asarray(mu, dtype=float)
        self.limits = limits if limits is not None else estimate_limits(X, mu)
        self.members_ = []
        for member in self._candidates():
            try:
                self.members_.append(member.fit(X, mu, limits=self.limits))
            except (ValueError, RuntimeError):
                continue
        if not self.members_:
            raise ValueError(f"{self.name}: no member could be fitted")
        self.fitted = True
        return self

    def _sqrt_mean_std(self, X):
        parts = [m._sqrt_mean_std(X) for m in self.members_]
        means = np.array([p[0] for p in parts])
        stds = np.array([p[1] for p in parts])
        mean = means.mean(axis=0)
        var = (stds**2).mean(axis=0) + means.var(axis=0)
        return mean, np.sqrt(var)
