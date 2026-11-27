import numpy as np

from src.datasets import ORGANISMS, generate_growth_dataset, true_mu
from src.models.metrics import accuracy_factor, all_metrics, bias_factor, r_squared, rmse
from src.models.secondary import (
    GrowthLimits,
    MechanisticModel,
    cardinal_gamma,
    frame_to_X,
    make_X,
    ratkowsky_mu,
    rosso_ctm,
)


def test_cardinal_factors_peak_and_bounds():
    assert abs(float(cardinal_gamma(7.0, 4.0, 7.0, 9.5, 1)) - 1.0) < 1e-12
    assert float(cardinal_gamma(4.0, 4.0, 7.0, 9.5, 1)) == 0.0
    assert float(cardinal_gamma(9.6, 4.0, 7.0, 9.5, 1)) == 0.0
    assert abs(float(rosso_ctm(39.0, 2.0, 5.0, 39.0, 47.0)) - 2.0) < 1e-12


def test_ratkowsky_zero_growth_boundaries():
    args = (0.03, 5.0, 0.3, 46.0)
    T = np.array([-10.0, 5.0, 46.0, 60.0])
    assert np.all(ratkowsky_mu(T, *args) == 0.0)
    assert float(ratkowsky_mu(30.0, *args)) > 0.0


def test_mechanistic_fit_recovers_boundaries():
    prof = ORGANISMS["Salmonella spp."]
    df = generate_growth_dataset("Salmonella spp.", n=600, seed=1)
    model = MechanisticModel("ctm").fit(frame_to_X(df), df["mu_max_per_h"].to_numpy())
    assert abs(model.limits.tmin - prof.tmin) < 2.0
    assert abs(model.limits.tmax - prof.tmax) < 2.0
    # zero growth outside the biological envelope
    assert model.predict_env(model.limits.tmax + 1, 7.0, 0.99)[0] == 0.0
    assert model.predict_env(model.limits.tmin - 1, 7.0, 0.99)[0] == 0.0
    assert model.predict_env(30.0, model.limits.phmin - 0.1, 0.99)[0] == 0.0
    assert model.predict_env(30.0, 7.0, model.limits.awmin - 0.01)[0] == 0.0


def test_mechanistic_accuracy_on_holdout():
    df = generate_growth_dataset("Listeria monocytogenes", n=700, seed=5)
    X, mu = frame_to_X(df), df["mu_max_per_h"].to_numpy()
    model = MechanisticModel("ratkowsky").fit(X[:500], mu[:500])
    assert r_squared(mu[500:], model.predict(X[500:])) > 0.9


def test_interval_brackets_mean():
    df = generate_growth_dataset("Salmonella spp.", n=400, seed=2)
    model = MechanisticModel().fit(frame_to_X(df), df["mu_max_per_h"].to_numpy())
    mean, lo, hi = model.predict_interval(make_X(np.linspace(10, 40, 7), 7.0, 0.99))
    assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)


def test_truth_function_is_zero_outside_limits():
    prof = ORGANISMS["Escherichia coli"]
    assert float(true_mu(prof, prof.tmax + 1, 7.0, 0.99)) == 0.0


def test_metrics_known_values():
    obs, pred = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 2.0])
    assert abs(bias_factor(obs, pred) - 1.0) < 1e-12
    assert abs(accuracy_factor(obs, pred) - 1.7611241081939726) < 1e-9
    assert abs(rmse(obs, pred) - np.sqrt(5 / 3)) < 1e-12
    assert accuracy_factor(obs, obs) == 1.0
    m = all_metrics(obs, np.array([0.0, 2.0, 4.0]))
    assert m["n_missed"] == 1


def test_limits_gate():
    lim = GrowthLimits(tmin=0, tmax=40)
    assert list(lim.inside(make_X([-1, 20, 40], 7, 0.99))) == [False, True, False]
