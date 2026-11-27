import time

import numpy as np
import pytest

from src.models.primary import FitError, baranyi, fit_primary_curve, gompertz


def _curve(model, A, mu, lam, noise, seed, n=49):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, n)
    f = gompertz if model == "gompertz" else baranyi
    return t, f(t, A, mu, lam) + rng.normal(0, noise, n)


def test_gompertz_shape_and_max_slope():
    t = np.linspace(0, 60, 6001)
    A, mu, lam = 6.0, 0.5, 4.0
    y = gompertz(t, A, mu, lam)
    assert abs(y[-1] - A) < 1e-2
    assert abs(np.max(np.gradient(y, t)) - mu) < 5e-3  # mu_max is the maximal slope


def test_baranyi_limits():
    t = np.linspace(0, 60, 601)
    y = baranyi(t, 5.0, 0.4, 3.0)
    assert abs(y[0]) < 1e-9 and abs(y[-1] - 5.0) < 1e-3
    assert np.all(np.diff(y) >= -1e-12)


def test_recovers_parameters_from_noisy_curves():
    for model in ("gompertz", "baranyi"):
        for seed in range(10):
            A, mu, lam = 6.0, 0.6, 3.0
            t, y = _curve(model, A, mu, lam, 0.05, seed)
            fit = fit_primary_curve(t, y, model)
            assert abs(fit.mu_max - mu) / mu < 0.12
            assert abs(fit.A - A) / A < 0.06
            assert abs(fit.lam - lam) < 0.8
            assert fit.r_squared > 0.99


def test_single_fit_is_fast():
    t, y = _curve("gompertz", 6, 0.6, 3, 0.05, 0)
    start = time.perf_counter()
    fit_primary_curve(t, y)
    assert time.perf_counter() - start < 0.15


def test_rejects_unusable_input():
    t = np.linspace(0, 10, 20)
    with pytest.raises(FitError):
        fit_primary_curve(t, np.zeros(20))  # no growth
    with pytest.raises(FitError):
        fit_primary_curve(t[:4], np.arange(4.0))  # too few points
    with pytest.raises(ValueError):
        fit_primary_curve(t, t, model="nope")
