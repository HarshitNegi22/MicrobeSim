import numpy as np
import pytest

from src.datasets import generate_growth_dataset
from src.models.primary import baranyi
from src.models.secondary import MechanisticModel, frame_to_X
from src.schemas import DynamicTelemetryPoint, SimulationScenario
from src.simulation.dynamic import (
    SimulationError,
    TemperatureProfile,
    build_environment_mu_interval_lookup,
    build_environment_mu_lookup,
    build_mu_lookup,
    simulate,
)
from src.simulation.scenarios import PRESETS, sudden_freeze


def _scenario(temps_by_time, n0=100.0, ncrit=1e5):
    log = [
        DynamicTelemetryPoint(timestamp_hours=t, temperature_celsius=T) for t, T in temps_by_time
    ]
    return SimulationScenario(
        organism="x",
        initial_cfu_per_g=n0,
        critical_threshold_cfu_per_g=ncrit,
        environmental_log=log,
    )


def test_isothermal_matches_analytic_exponential_and_tcrit():
    mu = 0.5
    tr = simulate(_scenario([(0, 10), (48, 10)]), lambda T: mu, n_max_cfu_per_g=1e30, h0=0.0)
    assert abs(tr.time_to_breach_hours - np.log(1e3) / mu) < 1e-4
    assert abs(tr.cfu_per_g[-1] / (100 * np.exp(mu * 48)) - 1) < 1e-6
    assert tr.spoilage_breached


def test_matches_baranyi_closed_form_with_lag():
    mu, h0 = 0.5, 2.0
    tr = simulate(_scenario([(0, 10), (48, 10)]), lambda T: mu, n_max_cfu_per_g=1e30, h0=h0)
    expected = baranyi(tr.time_hours, 60.0, mu, h0 / mu)
    assert np.max(np.abs(np.log(tr.cfu_per_g / 100) - expected)) < 1e-5


def test_logistic_ceiling_respected():
    tr = simulate(
        _scenario([(0, 10), (200, 10)], ncrit=1e12), lambda T: 0.8, n_max_cfu_per_g=1e9, h0=0.0
    )
    assert tr.cfu_per_g.max() <= 1e9 * 1.0001 and not tr.spoilage_breached


def test_initial_load_above_threshold_breaches_immediately():
    tr = simulate(_scenario([(0, 5), (10, 5)], n0=1e6, ncrit=1e5), lambda T: 0.1)
    assert tr.spoilage_breached and tr.time_to_breach_hours == 0.0


def test_no_growth_when_mu_zero():
    tr = simulate(_scenario([(0, -18), (24, -18)]), lambda T: 0.0)
    assert np.allclose(tr.cfu_per_g, 100.0)


def test_time_varying_ph_and_water_activity_change_growth_rate():
    dynamic_scenario = SimulationScenario(
        organism="x",
        environmental_log=[
            DynamicTelemetryPoint(
                timestamp_hours=0, temperature_celsius=10, ph=6.0, water_activity=0.98
            ),
            DynamicTelemetryPoint(
                timestamp_hours=12, temperature_celsius=10, ph=8.0, water_activity=0.90
            ),
        ],
    )
    fixed_scenario = SimulationScenario(
        organism="x",
        environmental_log=[
            DynamicTelemetryPoint(timestamp_hours=0, temperature_celsius=10),
            DynamicTelemetryPoint(timestamp_hours=12, temperature_celsius=10),
        ],
    )

    def rate(_temperature, ph, aw):
        return 0.1 + max(ph - 6.0, 0.0) * 0.05 + (0.98 - aw)

    dynamic = simulate(
        dynamic_scenario,
        lambda _temperature: 0.1,
        environment_mu_fn=rate,
        h0=0.0,
    )
    fixed = simulate(fixed_scenario, lambda _temperature: 0.1, h0=0.0)

    assert dynamic.ph[0] == 6.0 and dynamic.ph[-1] == 8.0
    assert dynamic.water_activity[0] == 0.98 and dynamic.water_activity[-1] == 0.90
    assert dynamic.cfu_per_g[-1] > fixed.cfu_per_g[-1]
    assert {"ph", "water_activity"} <= set(dynamic.to_frame().columns)
    assert dynamic.to_output().ph == dynamic.ph.tolist()


def test_growth_lookup_propagates_pointwise_model_intervals():
    class Model:
        def predict(self, X):
            return 0.1 + 0.01 * (X[:, 1] - 6.5)

        def predict_interval(self, X):
            mean = self.predict(X)
            return mean, mean * 0.5, mean * 1.5

    model = Model()
    mean_lookup = build_environment_mu_lookup(model, 6.5, 0.98)
    interval_lookup = build_environment_mu_interval_lookup(model, 6.5, 0.98)

    assert mean_lookup(20, 7.5, 0.98) == pytest.approx(0.11)
    mean, lower, upper = interval_lookup(20, 6.5, 0.98)
    assert lower < mean < upper
    assert interval_lookup(20, 7.5, 0.98)[0] == pytest.approx(0.11)


def test_pchip_does_not_overshoot_step_data():
    times, temps = [0, 1, 2, 3, 4], [4, 4, 25, 25, 25]
    prof = TemperatureProfile(times, temps, "pchip")
    vals = prof.evaluate(np.linspace(0, 4, 200))
    assert vals.min() >= 4 - 1e-9 and vals.max() <= 25 + 1e-9


def test_step_profile_holds_previous_value():
    prof = TemperatureProfile([0, 10, 20], [4, 20, 4], "step")
    assert prof(9.99) == 4 and prof(10.0) == 20 and prof(15) == 20


def test_edge_case_profiles_with_fitted_model():
    df = generate_growth_dataset("Listeria monocytogenes", n=500, seed=4)
    model = MechanisticModel().fit(frame_to_X(df), df["mu_max_per_h"].to_numpy())
    mu_fn = build_mu_lookup(model, 6.5, 0.98)
    # sudden freeze: growth must stop, never go negative
    frozen = simulate(SimulationScenario(organism="x", environmental_log=sudden_freeze()), mu_fn)
    late = frozen.cfu_per_g[frozen.time_hours > 20]
    assert np.all(np.diff(late) >= -1e-9) and abs(late[-1] / late[0] - 1) < 1e-6
    # heat spike breaches sooner than the refrigerated reference
    hot = simulate(
        SimulationScenario(organism="x", environmental_log=PRESETS["heat_spike"].build()), mu_fn
    )
    cold = simulate(
        SimulationScenario(organism="x", environmental_log=PRESETS["constant_fridge"].build()),
        mu_fn,
    )
    assert hot.cfu_per_g[-1] > cold.cfu_per_g[-1]
    out = hot.to_output()
    assert len(out.time_hours) == len(out.predicted_cfu_per_g)


def test_bad_inputs():
    with pytest.raises(SimulationError):
        simulate(_scenario([(0, 5), (10, 5)]), lambda T: 0.1, n_max_cfu_per_g=10.0)
