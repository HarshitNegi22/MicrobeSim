"""Statistical and predictive-microbiology performance metrics (FR-4)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _as_arrays(obs, pred):
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if obs.shape != pred.shape:
        raise ValueError("observed and predicted must have the same length")
    if obs.size == 0:
        raise ValueError("empty input")
    return obs, pred


def rmse(obs, pred) -> float:
    obs, pred = _as_arrays(obs, pred)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def r_squared(obs, pred) -> float:
    obs, pred = _as_arrays(obs, pred)
    sst = float(np.sum((obs - obs.mean()) ** 2))
    if sst == 0:
        return float("nan")
    return 1.0 - float(np.sum((obs - pred) ** 2)) / sst


def _log_ratios(obs, pred, min_obs: float = 0.0):
    obs, pred = _as_arrays(obs, pred)
    eligible = obs > max(min_obs, 0.0)
    mask = eligible & (pred > 0)
    return np.log10(pred[mask] / obs[mask]), int(mask.sum()), int(eligible.sum() - mask.sum())


def bias_factor(obs, pred, min_obs: float = 0.0) -> float:
    """Bf = 10 ** mean(log10(pred/obs)).  Bf > 1: fail-safe (over-prediction)."""
    lr, n, _ = _log_ratios(obs, pred, min_obs)
    return float(10 ** lr.mean()) if n else float("nan")


def accuracy_factor(obs, pred, min_obs: float = 0.0) -> float:
    """Af = 10 ** sqrt(mean(log10(pred/obs)**2)).  Af = 1 is a perfect model."""
    lr, n, _ = _log_ratios(obs, pred, min_obs)
    return float(10 ** np.sqrt(np.mean(lr**2))) if n else float("nan")


def all_metrics(obs, pred, min_obs: float = 0.0) -> Dict[str, float]:
    """R2 and RMSE on all points; Af/Bf on pairs with ``obs > min_obs`` and ``pred > 0``.

    Ratio-based factors explode near the growth boundary where observed rates are
    tiny, so ``min_obs`` lets callers restrict them to meaningful growth rates.
    ``n_missed`` counts eligible observations for which the model predicted no growth
    (they cannot enter a log-ratio and are reported separately).
    """
    obs, pred = _as_arrays(obs, pred)
    _, n_used, n_missed = _log_ratios(obs, pred, min_obs)
    return {
        "R2": r_squared(obs, pred),
        "RMSE": rmse(obs, pred),
        "Bf": bias_factor(obs, pred, min_obs),
        "Af": accuracy_factor(obs, pred, min_obs),
        "n": int(obs.size),
        "n_af_bf": n_used,
        "n_missed": n_missed,
    }
