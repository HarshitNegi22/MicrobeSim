import numpy as np
import pytest

from src.datasets import generate_growth_dataset
from src.models.benchmark import available_model_keys, benchmark
from src.models.metrics import r_squared
from src.models.ml_regressors import (
    EnsembleModel,
    ExtraTreesModel,
    GPRModel,
    HybridModel,
    RandomForestModel,
)
from src.models.secondary import MechanisticModel, frame_to_X, make_X

DF = generate_growth_dataset("Salmonella spp.", n=500, seed=11)
X, MU = frame_to_X(DF), DF["mu_max_per_h"].to_numpy()


def _check_model(model):
    model.fit(X[:380], MU[:380])
    assert r_squared(MU[380:], model.predict(X[380:])) > 0.75
    # biological realism: zero growth at/above Tmax and below Tmin
    hot = make_X([model.limits.tmax, model.limits.tmax + 5], 7.0, 0.99)
    cold = make_X([model.limits.tmin, model.limits.tmin - 5], 7.0, 0.99)
    assert np.all(model.predict(hot) == 0.0) and np.all(model.predict(cold) == 0.0)
    mean, lo, hi = model.predict_interval(make_X([20, 30, 37], 7.0, 0.99))
    assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12) and np.all(hi > 0)


def test_gpr():
    _check_model(GPRModel(n_restarts=0))


def test_random_forest():
    _check_model(RandomForestModel(n_estimators=150))


def test_extra_trees():
    _check_model(ExtraTreesModel(n_estimators=150))


@pytest.mark.parametrize("residual", ["et", "gpr"])
def test_hybrid(residual):
    _check_model(HybridModel(residual))


def test_ensemble_averages_members_and_widens_uncertainty():
    model = EnsembleModel()
    _check_model(model)
    assert len(model.members_) == 4
    Z = make_X([20, 30, 37], 7.0, 0.99)
    member_means = np.array([m.predict(Z) for m in model.members_])
    assert member_means.min(axis=0).min() <= model.predict(Z).max()
    assert np.all(model.predict(Z) <= member_means.max(axis=0) + 1e-9)


def test_hybrid_rejects_unknown_residual_learner():
    with pytest.raises(ValueError):
        HybridModel("svm")


def test_mechanistic_tmax_never_below_observed_growth_temperature():
    """Regression: Tmax was bounded using only rows near the pH/aw optimum, so it could fall below
    temperatures where growth was observed (B. cereus: Tmax 37.8 C despite growth at 45 C)."""
    rng = np.random.default_rng(3)
    T = np.concatenate([np.linspace(8, 37, 40), np.full(14, 45.0)])
    ph = np.concatenate([np.full(40, 7.0), np.full(14, 6.0)])
    aw = np.concatenate([np.full(40, 0.997), np.full(14, 0.989)])
    mu = np.maximum(0.0, 0.04 * (T - 3) ** 2 / 100 * (1 - np.exp(0.2 * (T - 52)))) + rng.normal(
        0, 0.01, len(T)
    )
    mu = np.maximum(mu, 0.02)
    for kind in ("ratkowsky", "ctm"):
        model = MechanisticModel(kind).fit(np.column_stack([T, ph, aw]), mu)
        assert model.limits.tmax > 45.0
        assert model.limits.inside(np.array([[45.0, 6.0, 0.989]])).all()


def test_model_registry_uses_supported_models():
    assert set(available_model_keys()) == {"ratkowsky", "ctm", "gpr", "rf", "hybrid", "ensemble"}
    res = benchmark(DF, ["ratkowsky"])
    assert "ratkowsky" in res.models


def test_benchmark_scorecard():
    res = benchmark(DF, available_model_keys())
    assert {"R2", "R2_std", "RMSE", "Af", "Bf"} <= set(res.table.columns)
    assert len(res.table) >= 3 and not res.errors
    assert (res.table["Af"] >= 1.0).all()
    assert len(res.predictions) == len(res.test_index)
    assert len(res.validation_splits) == 5


def test_grouped_benchmark_keeps_groups_out_of_train_partitions():
    groups = np.arange(len(DF)) % 8
    res = benchmark(DF, ["ratkowsky", "rf"], groups=groups, n_repeats=3)

    assert "grouped by supplied labels" in res.validation_description
    for train_indices, test_indices in res.validation_splits:
        assert not set(groups[train_indices]) & set(groups[test_indices])


def test_too_little_data_is_rejected():
    with pytest.raises(ValueError):
        GPRModel().fit(X[:5], MU[:5])
