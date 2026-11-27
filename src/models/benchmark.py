"""Benchmarking engine (FR-4): mechanistic equations vs. ML regressors."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from src.models.metrics import all_metrics
from src.models.ml_regressors import EnsembleModel, GPRModel, HybridModel, RandomForestModel
from src.models.secondary import GrowthRateModel, MechanisticModel, frame_to_X

MODEL_FACTORIES: Dict[str, Callable[[], GrowthRateModel]] = {
    "ratkowsky": lambda: MechanisticModel("ratkowsky"),
    "ctm": lambda: MechanisticModel("ctm"),
    "gpr": lambda: GPRModel(),
    "rf": lambda: RandomForestModel(),
    "hybrid": lambda: HybridModel("et"),
    "ensemble": lambda: EnsembleModel(),
}


MODEL_LABELS: Dict[str, str] = {
    "ratkowsky": "Ratkowsky (mechanistic)",
    "ctm": "Rosso CTM (mechanistic)",
    "gpr": "Gaussian Process (ML)",
    "rf": "Random Forest (ML)",
    "hybrid": "Hybrid: Ratkowsky + Extra Trees",
    "ensemble": "Ensemble (mechanistic + ML)",
}


def available_model_keys() -> List[str]:
    return list(MODEL_FACTORIES)


def build_model(key: str) -> GrowthRateModel:
    if key not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model '{key}'. Options: {list(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[key]()


@dataclass
class BenchmarkResult:
    table: pd.DataFrame
    models: Dict[str, GrowthRateModel]
    predictions: pd.DataFrame  # test-set observed vs. predicted mu_max per model
    train_index: np.ndarray
    test_index: np.ndarray
    errors: Dict[str, str] = field(default_factory=dict)
    validation_description: str = "Repeated random holdout"
    validation_splits: List[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    loaded_from_disk: bool = False
    comparison_table: Optional[pd.DataFrame] = None
    real_test_count: int = 0


def _preassigned_benchmark(
    df: pd.DataFrame,
    model_keys: List[str],
    training_source: str,
    min_obs_afbf: float,
) -> BenchmarkResult:
    if training_source not in {"real", "real+synthetic", "synthetic"}:
        raise ValueError("training_source must be 'real', 'real+synthetic', or 'synthetic'.")
    if not pd.api.types.is_bool_dtype(df["is_synthetic"]):
        raise ValueError("is_synthetic must contain boolean values.")
    if df["is_synthetic"].isna().any():
        raise ValueError("is_synthetic cannot contain missing values.")
    valid_splits = {"train", "validation", "test"}
    invalid_split = df["split"].notna() & ~df["split"].isin(valid_splits)
    if invalid_split.any():
        raise ValueError("split values must be train, validation, or test when present.")

    X = frame_to_X(df)
    mu = df["mu_max_per_h"].to_numpy(dtype=float)
    synthetic = df["is_synthetic"].to_numpy(dtype=bool)
    split = df["split"].to_numpy(dtype=object)
    real = ~synthetic
    real_validation = np.flatnonzero((split == "validation") & real)
    real_test = np.flatnonzero((split == "test") & real)
    real_train = (split == "train") & real
    synthetic_train = (split == "train") & synthetic

    source_modes = ["real", "real+synthetic"]
    if training_source == "synthetic":
        source_modes.append("synthetic")

    def source_train_mask(source: str, include_validation: bool) -> np.ndarray:
        mask = real_train.copy()
        if source in {"real+synthetic", "synthetic"}:
            mask = synthetic_train.copy()
            if source == "real+synthetic":
                mask |= real_train
        if include_validation and source != "synthetic":
            mask |= (split == "validation") & real
        return mask

    def score(model: GrowthRateModel, indices: np.ndarray) -> Dict[str, float]:
        if not len(indices):
            return {name: float("nan") for name in ("R2", "RMSE", "Bf", "Af", "n_missed")} | {
                "n": 0.0
            }
        metrics = all_metrics(mu[indices], model.predict(X[indices]), min_obs=min_obs_afbf)
        return {name: float(metrics[name]) for name in ("R2", "RMSE", "Bf", "Af", "n", "n_missed")}

    comparison_rows = []
    selected_rows = []
    selected_models: Dict[str, GrowthRateModel] = {}
    selected_test_predictions: Dict[str, np.ndarray] = {}
    errors: Dict[str, str] = {}
    selected_train_indices = np.flatnonzero(
        source_train_mask(training_source, include_validation=True)
    )

    ordered = sorted(model_keys, key=lambda key: 0 if key == "ratkowsky" else 1)
    for source in source_modes:
        evaluation_indices = np.flatnonzero(source_train_mask(source, include_validation=False))
        final_indices = np.flatnonzero(source_train_mask(source, include_validation=True))
        for key in ordered:
            row: Dict[str, object] = {
                "key": key,
                "training_source": {
                    "real": "Real only",
                    "real+synthetic": "Real + synthetic",
                    "synthetic": "Synthetic only",
                }[source],
                "n_train": int(len(final_indices)),
                "n_validation": int(len(real_validation)),
                "n_test": int(len(real_test)),
            }
            try:
                validation_model = build_model(key).fit(
                    X[evaluation_indices], mu[evaluation_indices]
                )
                validation_metrics = score(validation_model, real_validation)
                test_model = build_model(key).fit(X[final_indices], mu[final_indices])
                test_metrics = score(test_model, real_test)
            except (ValueError, RuntimeError, ImportError) as exc:
                row["Model"] = MODEL_LABELS.get(key, key)
                row["error"] = str(exc)
                comparison_rows.append(row)
                if source == training_source:
                    errors[key] = str(exc)
                continue

            row["Model"] = validation_model.name
            row.update({name: validation_metrics[name] for name in ("R2", "RMSE", "Bf", "Af")})
            row.update({f"test_{name}": test_metrics[name] for name in ("R2", "RMSE", "Bf", "Af")})
            row["n_missed"] = validation_metrics["n_missed"]
            row["test_n_missed"] = test_metrics["n_missed"]
            comparison_rows.append(row)
            if source == training_source:
                selected_rows.append(row.copy())
                selected_models[key] = test_model
                selected_test_predictions[key] = (
                    test_model.predict(X[real_test]) if len(real_test) else np.array([])
                )

    table = pd.DataFrame(selected_rows)
    if not table.empty:
        table = table.sort_values("Af", na_position="last").reset_index(drop=True)
    comparison_table = pd.DataFrame(comparison_rows)
    predictions: Dict[str, object] = {"observed_mu_max": mu[real_test]}
    predictions.update(selected_test_predictions)
    description = (
        "Preassigned curve-level train/validation/test splits; validation and test "
        "scores use real rows only, and synthetic rows are training-only."
    )
    return BenchmarkResult(
        table=table,
        models=selected_models,
        predictions=pd.DataFrame(predictions),
        train_index=selected_train_indices,
        test_index=real_test,
        errors=errors,
        validation_description=description,
        validation_splits=[(selected_train_indices, real_test)],
        comparison_table=comparison_table,
        real_test_count=len(real_test),
    )


def benchmark(
    df: pd.DataFrame,
    model_keys: Optional[List[str]] = None,
    test_size: float = 0.25,
    seed: int = 0,
    min_obs_afbf: float = 0.05,
    groups: Optional[np.ndarray] = None,
    group_name: str = "supplied labels",
    n_repeats: int = 5,
    training_source: Optional[str] = None,
) -> BenchmarkResult:
    """Repeatedly validate models on held-out data, then fit usable models on all rows.

    When supplied, ``groups`` ensures all rows from each group stay on the same side
    of every split. Af / Bf are computed on held-out points with observed
    ``mu_max > min_obs_afbf``. Returned models are refit on all available rows.
    """
    model_keys = model_keys or available_model_keys()
    if "is_synthetic" in df.columns:
        if "split" not in df.columns:
            raise ValueError(
                "Synthetic-flagged benchmark data requires a preassigned split column."
            )
        return _preassigned_benchmark(
            df,
            model_keys,
            training_source or "real",
            min_obs_afbf,
        )
    if training_source is not None:
        raise ValueError("training_source is only supported for split-labeled expanded data.")
    X = frame_to_X(df)
    mu = df["mu_max_per_h"].to_numpy(dtype=float)
    if len(df) < 30:
        raise ValueError("benchmarking needs at least 30 rows")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2")
    split_groups = None if groups is None else np.asarray(groups)
    if split_groups is not None and len(split_groups) != len(df):
        raise ValueError("groups must have one value per dataset row")
    if split_groups is not None and pd.isna(split_groups).any():
        raise ValueError("groups cannot contain missing values")
    if split_groups is not None and pd.Series(split_groups).nunique() < 2:
        split_groups = None

    if split_groups is None:
        splitter = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
        splits = list(splitter.split(X))
        validation_description = f"{n_repeats} repeated random holdouts"
    else:
        splitter = GroupShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
        splits = list(splitter.split(X, groups=split_groups))
        validation_description = f"{n_repeats} repeated holdouts grouped by {group_name}"

    ordered = sorted(model_keys, key=lambda k: 0 if k == "ratkowsky" else 1)
    models: Dict[str, GrowthRateModel] = {}
    scores: Dict[str, List[Dict[str, float]]] = {key: [] for key in ordered}
    fold_predictions: Dict[str, List[tuple[np.ndarray, np.ndarray]]] = {key: [] for key in ordered}
    tested_indices: List[np.ndarray] = []
    errors: Dict[str, str] = {}
    for train_idx, test_idx in splits:
        shared = None
        fold_models: Dict[str, GrowthRateModel] = {}
        for key in ordered:
            fold_model = build_model(key)
            try:
                fold_model.fit(
                    X[train_idx],
                    mu[train_idx],
                    limits=None if key == "ratkowsky" else shared,
                )
            except (ValueError, RuntimeError, ImportError) as exc:
                errors[key] = str(exc)
                continue
            if key == "ratkowsky":
                shared = fold_model.limits
            fold_models[key] = fold_model

        for key, fold_model in fold_models.items():
            pred = fold_model.predict(X[test_idx])
            metrics = all_metrics(mu[test_idx], pred, min_obs=min_obs_afbf)
            scores[key].append(
                {name: float(metrics[name]) for name in ("R2", "RMSE", "Bf", "Af", "n", "n_missed")}
            )
            fold_predictions[key].append((test_idx, pred))
        tested_indices.append(test_idx)

    # Fit the models used for predictions on the full dataset, preserving shared limits.
    shared = None
    rows = []
    for key in ordered:
        model = build_model(key)
        start = time.perf_counter()
        try:
            model.fit(X, mu, limits=None if key == "ratkowsky" else shared)
        except (ValueError, RuntimeError, ImportError) as exc:
            errors[key] = str(exc)
            continue
        elapsed = time.perf_counter() - start
        if key == "ratkowsky":
            shared = model.limits
        if len(scores[key]) != len(splits):
            errors.setdefault(key, "Model could not be scored in every validation split.")
            continue
        models[key] = model
        score_frame = pd.DataFrame(scores[key])
        row = {
            "key": key,
            "Model": model.name,
            **{column: float(score_frame[column].mean()) for column in score_frame},
            **{
                f"{column}_std": float(score_frame[column].std(ddof=1))
                for column in ("R2", "RMSE", "Bf", "Af")
            },
            "fit_seconds": elapsed,
        }
        row["n_test"] = float(score_frame["n"].mean())
        row["n_missed"] = float(score_frame["n_missed"].mean())
        rows.append(row)

    tested = (
        np.unique(np.concatenate(tested_indices)) if tested_indices else np.array([], dtype=int)
    )
    pred_cols: Dict[str, object] = {"observed_mu_max": mu[tested]}
    for key in ordered:
        predictions_for_key = fold_predictions[key]
        if not predictions_for_key:
            continue
        by_index: Dict[int, List[float]] = {}
        for indices, predictions in predictions_for_key:
            for index, prediction in zip(indices, predictions):
                by_index.setdefault(int(index), []).append(prediction)
        pred_cols[key] = [float(np.mean(by_index.get(int(index), [np.nan]))) for index in tested]

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("Af").reset_index(drop=True)
    first_train, _ = splits[0]
    return BenchmarkResult(
        table=table,
        models=models,
        predictions=pd.DataFrame(pred_cols),
        train_index=np.sort(first_train),
        test_index=tested,
        errors=errors,
        validation_description=validation_description,
        validation_splits=[(np.sort(train), np.sort(test)) for train, test in splits],
    )
