"""Preset temperature-history builders for the cold-chain simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.schemas import DynamicTelemetryPoint


def from_waypoints(
    waypoints: Sequence[Tuple[float, float]], dt: float = 0.25
) -> List[DynamicTelemetryPoint]:
    """Sample a piecewise-linear ``(hours, deg C)`` waypoint list every ``dt`` hours."""
    w = np.asarray(sorted(waypoints), dtype=float)
    if len(w) < 2:
        raise ValueError("need at least two waypoints")
    t = np.unique(np.concatenate([np.arange(w[0, 0], w[-1, 0], dt), w[:, 0]]))
    T = np.interp(t, w[:, 0], w[:, 1])
    return [
        DynamicTelemetryPoint(timestamp_hours=float(a), temperature_celsius=float(b))
        for a, b in zip(t, T)
    ]


def constant(temp_c: float = 4.0, hours: float = 72.0, dt: float = 1.0):
    return from_waypoints([(0.0, temp_c), (hours, temp_c)], dt)


def step_function(
    t_low: float = 4.0, t_high: float = 20.0, switch_hour: float = 24.0, hours: float = 72.0
):
    """Instantaneous jump from ``t_low`` to ``t_high`` at ``switch_hour`` (use 'step' hold)."""
    eps = 1e-3
    return from_waypoints(
        [(0.0, t_low), (switch_hour, t_low), (switch_hour + eps, t_high), (hours, t_high)], 0.5
    )


def cold_chain_break(
    fridge_c: float = 4.0,
    excursion_c: float = 25.0,
    excursion_start: float = 24.0,
    excursion_hours: float = 8.0,
    hours: float = 96.0,
    ramp_hours: float = 1.0,
):
    """Refrigerated transport with one ambient excursion (door left open, reefer failure)."""
    s, e = excursion_start, excursion_start + excursion_hours
    return from_waypoints(
        [
            (0.0, fridge_c),
            (s, fridge_c),
            (s + ramp_hours, excursion_c),
            (e, excursion_c),
            (e + ramp_hours, fridge_c),
            (hours, fridge_c),
        ],
        0.25,
    )


def heat_spike(
    base_c: float = 4.0, peak_c: float = 40.0, hours: float = 48.0, at_hour: float = 20.0
):
    """Short, sharp heat spike (e.g. vehicle parked in the sun)."""
    return from_waypoints(
        [
            (0.0, base_c),
            (at_hour, base_c),
            (at_hour + 1.0, peak_c),
            (at_hour + 3.0, peak_c),
            (at_hour + 4.0, base_c),
            (hours, base_c),
        ],
        0.25,
    )


def sudden_freeze(
    start_c: float = 4.0, freeze_c: float = -18.0, freeze_hour: float = 12.0, hours: float = 48.0
):
    """Temperature collapses below the growth minimum (growth must stop, not go negative)."""
    return from_waypoints(
        [(0.0, start_c), (freeze_hour, start_c), (freeze_hour + 2.0, freeze_c), (hours, freeze_c)],
        0.25,
    )


@dataclass(frozen=True)
class Preset:
    label: str
    description: str
    build: Callable[[], List[DynamicTelemetryPoint]]
    interpolation: str = "linear"


PRESETS: Dict[str, Preset] = {
    "cold_chain_break": Preset(
        "Cold-chain break (8 h ambient excursion)",
        "4 C reefer, 8 h at 25 C after 24 h, then restored; 96 h total.",
        cold_chain_break,
    ),
    "heat_spike": Preset(
        "Heat spike (3 h at 40 C)",
        "Short vehicle heat spike on a 4 C load; 48 h total.",
        heat_spike,
    ),
    "sudden_freeze": Preset(
        "Sudden freeze (-18 C)", "Cold room fault freezes the load after 12 h.", sudden_freeze
    ),
    "step_function": Preset(
        "Step function (4 C -> 20 C)",
        "Instant switch to 20 C at 24 h (sample-and-hold interpolation).",
        step_function,
        "step",
    ),
    "constant_fridge": Preset(
        "Constant 4 C (reference)", "Isothermal refrigerated reference for 72 h.", constant
    ),
}
