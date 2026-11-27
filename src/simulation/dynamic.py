"""Dynamic cold-chain simulator (FR-3, FR-5).

Solves the Baranyi-type system from the technical specification

    dN/dt      = alpha(t) * mu_max(T(t), pH, aw) * (1 - N/Nmax) * N
    dalpha/dt  = mu_max(T(t), pH, aw) * alpha * (1 - alpha)

with ``scipy.integrate.solve_ivp`` (RK45).  Implementation notes:

* The population is integrated as ``ln N`` (identical dynamics, but well-scaled over
  many log-cycles): ``d(lnN)/dt = alpha * mu * (1 - N/Nmax)``.
* ``alpha(0) = exp(-h0)`` where ``h0`` is the (dimensionless) "work to be done" of the
  inoculum; for isothermal conditions the lag is ``lambda = h0 / mu_max``.
* Temperature profiles are interpolated ``linear`` (default), ``pchip`` (smooth cubic,
  monotone - no spline overshoot between logger readings) or ``step``
  (sample-and-hold).  Integration is split at telemetry time-stamps so kinks and jumps
  never fall inside a solver step.
* ``t_crit`` comes from a solver *event* on ``ln N - ln N_crit`` (exact crossing time,
  not a grid approximation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from src.models.secondary import GrowthRateModel, make_X
from src.schemas import SimulationOutput, SimulationScenario

SENSOR_MIN_C, SENSOR_MAX_C = -20.0, 50.0
INTERPOLATIONS = ("linear", "pchip", "step")
_MAX_SEGMENTS = 1500


class SimulationError(RuntimeError):
    """Raised when the ODE solver fails or inputs are inconsistent."""


# --------------------------------------------------------------------------- #
# Temperature profile
# --------------------------------------------------------------------------- #
class TemperatureProfile:
    """Continuous ``T(t)`` built from discrete telemetry (held constant outside its span)."""

    def __init__(self, times, temps, kind: str = "linear"):
        if kind not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
        t = np.asarray(times, dtype=float)
        T = np.asarray(temps, dtype=float)
        if t.shape != T.shape or t.ndim != 1 or len(t) < 1:
            raise SimulationError("telemetry needs matching, non-empty 1-D time and temperature")
        order = np.argsort(t, kind="stable")
        t, T = t[order], T[order]
        keep = np.concatenate([[True], np.diff(t) > 0])  # drop duplicate stamps (keep first)
        self.t, self.T, self.kind = t[keep], T[keep], kind
        self._pchip = (
            PchipInterpolator(self.t, self.T) if kind == "pchip" and len(self.t) > 2 else None
        )

    def __call__(self, t: float) -> float:
        if len(self.t) == 1:
            return float(self.T[0])
        if t <= self.t[0]:
            return float(self.T[0])
        if t >= self.t[-1]:
            return float(self.T[-1])
        if self.kind == "step":
            return float(self.T[np.searchsorted(self.t, t, side="right") - 1])
        if self._pchip is not None:
            return float(self._pchip(t))
        return float(np.interp(t, self.t, self.T))

    def evaluate(self, times) -> np.ndarray:
        return np.array([self(float(x)) for x in np.asarray(times, dtype=float)])


# --------------------------------------------------------------------------- #
# Growth-rate lookup
# --------------------------------------------------------------------------- #
def build_mu_lookup(
    model: GrowthRateModel,
    ph: float,
    aw: float,
    t_range=(SENSOR_MIN_C - 5.0, SENSOR_MAX_C + 5.0),
    step: float = 0.05,
) -> Callable[[float], float]:
    """Tabulate ``mu_max(T)`` at fixed pH / aw once; the ODE right-hand side then only
    does a cheap linear interpolation (model-agnostic, works for slow ML predictors)."""
    grid = np.arange(t_range[0], t_range[1] + step, step)
    mu = model.predict(make_X(grid, ph, aw))
    return lambda T: float(np.interp(T, grid, mu))


def build_environment_mu_lookup(
    model: GrowthRateModel, default_ph: float, default_aw: float
) -> Callable[[float, float, float], float]:
    """Evaluate growth for time-varying pH and water activity, reusing the fixed lookup."""
    fixed_lookup = build_mu_lookup(model, default_ph, default_aw)

    def lookup(temperature: float, ph: float, water_activity: float) -> float:
        if ph == default_ph and water_activity == default_aw:
            return fixed_lookup(temperature)
        prediction = model.predict(make_X(temperature, ph, water_activity))
        return float(prediction[0])

    return lookup


def build_environment_mu_interval_lookup(
    model: GrowthRateModel, default_ph: float, default_aw: float
) -> Callable[[float, float, float], tuple[float, float, float]]:
    """Return the model's pointwise 95% growth-rate interval at environment conditions."""
    grid = np.arange(SENSOR_MIN_C - 5.0, SENSOR_MAX_C + 5.0 + 0.05, 0.05)
    fixed_mean, fixed_lower, fixed_upper = model.predict_interval(
        make_X(grid, default_ph, default_aw)
    )

    def lookup(temperature: float, ph: float, water_activity: float) -> tuple[float, float, float]:
        if ph == default_ph and water_activity == default_aw:
            return (
                float(np.interp(temperature, grid, fixed_mean)),
                float(np.interp(temperature, grid, fixed_lower)),
                float(np.interp(temperature, grid, fixed_upper)),
            )
        mean, lower, upper = model.predict_interval(make_X(temperature, ph, water_activity))
        return float(mean[0]), float(lower[0]), float(upper[0])

    return lookup


def _environment_profile(log, field: str, fallback: float, interpolation: str):
    times = np.asarray([point.timestamp_hours for point in log], dtype=float)
    values = np.asarray(
        [np.nan if getattr(point, field) is None else getattr(point, field) for point in log],
        dtype=float,
    )
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    known = np.isfinite(values)
    if not known.any():
        values = np.full(len(times), fallback, dtype=float)
    else:
        values = np.interp(times, times[known], values[known])
    return TemperatureProfile(times, values, interpolation)


# --------------------------------------------------------------------------- #
# Trace container
# --------------------------------------------------------------------------- #
@dataclass
class SimulationTrace:
    time_hours: np.ndarray
    temperature_celsius: np.ndarray
    cfu_per_g: np.ndarray
    alpha: np.ndarray
    mu_max: np.ndarray
    n0: float
    n_crit: float
    spoilage_breached: bool
    time_to_breach_hours: Optional[float]
    ph: np.ndarray
    water_activity: np.ndarray

    @property
    def log10_cfu(self) -> np.ndarray:
        return np.log10(self.cfu_per_g)

    @property
    def doublings(self) -> np.ndarray:
        """Cumulative number of population doublings, log2(N / N0)."""
        return np.log2(self.cfu_per_g / self.n0)

    def to_output(self) -> SimulationOutput:
        return SimulationOutput(
            time_hours=self.time_hours.tolist(),
            temperatures=self.temperature_celsius.tolist(),
            ph=self.ph.tolist(),
            water_activity=self.water_activity.tolist(),
            predicted_cfu_per_g=self.cfu_per_g.tolist(),
            spoilage_breached=self.spoilage_breached,
            time_to_breach_hours=self.time_to_breach_hours,
        )

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "time_hours": self.time_hours,
                "temperature_celsius": self.temperature_celsius,
                "ph": self.ph,
                "water_activity": self.water_activity,
                "mu_max_per_h": self.mu_max,
                "alpha": self.alpha,
                "cfu_per_g": self.cfu_per_g,
                "log10_cfu_per_g": self.log10_cfu,
                "doublings": self.doublings,
            }
        )


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def simulate(
    scenario: SimulationScenario,
    mu_fn: Callable[[float], float],
    *,
    environment_mu_fn: Optional[Callable[[float, float, float], float]] = None,
    n_max_cfu_per_g: float = 1e9,
    h0: float = 1.0,
    interpolation: str = "linear",
    horizon_hours: Optional[float] = None,
    n_points: int = 600,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> SimulationTrace:
    """Run the dynamic growth ODE over the scenario's temperature history.

    Parameters
    ----------
    mu_fn : callable
        ``T (deg C) -> mu_max (1/h)`` at the scenario's pH / aw (see :func:`build_mu_lookup`).
    n_max_cfu_per_g : float
        Carrying capacity ``N_max`` of the food matrix.
    h0 : float
        Physiological "work to be done" of the inoculum (``alpha0 = exp(-h0)``);
        ``h0 = 0`` removes the lag phase.
    horizon_hours : float, optional
        Simulation length; defaults to the last telemetry time-stamp.
    """
    log = scenario.environmental_log
    if len(log) < 1:
        raise SimulationError("environmental_log is empty")
    if h0 < 0:
        raise SimulationError("h0 must be >= 0")
    n0 = float(scenario.initial_cfu_per_g)
    n_crit = float(scenario.critical_threshold_cfu_per_g)
    if n_max_cfu_per_g <= n0:
        raise SimulationError("n_max_cfu_per_g must exceed initial_cfu_per_g")

    times = np.array([p.timestamp_hours for p in log], dtype=float)
    temps = np.array([p.temperature_celsius for p in log], dtype=float)
    profile = TemperatureProfile(times, temps, interpolation)
    ph_profile = _environment_profile(log, "ph", scenario.ph, interpolation)
    aw_profile = _environment_profile(log, "water_activity", scenario.water_activity, interpolation)
    t_end = float(horizon_hours if horizon_hours is not None else profile.t[-1])
    if t_end <= 0:
        raise SimulationError("simulation horizon must be positive")

    grid = np.unique(
        np.concatenate([np.linspace(0.0, t_end, n_points), profile.t[profile.t <= t_end]])
    )
    grid = grid[(grid >= 0.0) & (grid <= t_end)]

    ln_crit = np.log(n_crit)
    ln_nmax = np.log(n_max_cfu_per_g)

    def rhs(_t, s):
        ln_n, alpha = s
        temperature, ph, water_activity = profile(_t), ph_profile(_t), aw_profile(_t)
        mu = (
            environment_mu_fn(temperature, ph, water_activity)
            if environment_mu_fn is not None
            else mu_fn(temperature)
        )
        crowding = 1.0 - np.exp(min(ln_n - ln_nmax, 50.0))
        return [alpha * mu * crowding, mu * alpha * (1.0 - alpha)]

    def crossing(_t, s):
        return s[0] - ln_crit

    crossing.direction = 1.0

    # Segment boundaries at interior telemetry stamps keep kinks/jumps off solver steps.
    inner = profile.t[(profile.t > 0.0) & (profile.t < t_end)]
    max_step = np.inf
    if len(inner) > _MAX_SEGMENTS:
        max_step = float(max(np.median(np.diff(profile.t)), t_end / 5000.0))
        inner = np.array([])
    bounds = np.concatenate([[0.0], inner, [t_end]])

    state = np.array([np.log(n0), np.exp(-h0)])
    t_out = [grid[0]]
    y_out = [state.copy()]
    t_crit: Optional[float] = 0.0 if n0 >= n_crit else None

    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = grid[(grid > a) & (grid <= b)]
        if len(sel) == 0 or sel[-1] < b:
            sel = np.append(sel, b)
        sol = solve_ivp(
            rhs,
            (a, b),
            state,
            method="RK45",
            t_eval=sel,
            events=crossing if t_crit is None else None,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not sol.success:
            raise SimulationError(f"ODE solver failed on [{a:.3f}, {b:.3f}] h: {sol.message}")
        if t_crit is None and sol.t_events and len(sol.t_events[0]):
            t_crit = float(sol.t_events[0][0])
        t_out.extend(sol.t.tolist())
        y_out.extend(sol.y.T.tolist())
        state = sol.y[:, -1]

    t_arr = np.array(t_out)
    y_arr = np.array(y_out)
    # Report only the requested grid (segment ends that are not on it are dropped).
    keep = np.isin(t_arr, grid)
    t_arr, y_arr = t_arr[keep], y_arr[keep]
    temp_arr = profile.evaluate(t_arr)
    ph_arr = ph_profile.evaluate(t_arr)
    aw_arr = aw_profile.evaluate(t_arr)
    return SimulationTrace(
        time_hours=t_arr,
        temperature_celsius=temp_arr,
        cfu_per_g=np.exp(y_arr[:, 0]),
        alpha=y_arr[:, 1],
        mu_max=np.array(
            [
                environment_mu_fn(T, pH, aw) if environment_mu_fn is not None else mu_fn(T)
                for T, pH, aw in zip(temp_arr, ph_arr, aw_arr)
            ]
        ),
        n0=n0,
        n_crit=n_crit,
        spoilage_breached=t_crit is not None,
        time_to_breach_hours=t_crit,
        ph=ph_arr,
        water_activity=aw_arr,
    )
