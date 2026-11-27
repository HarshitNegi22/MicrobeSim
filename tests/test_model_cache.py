import os

import pandas as pd

from src.datasets import generate_growth_dataset
from src.model_cache import load_or_train


def test_fitted_models_and_benchmark_are_reused_from_disk(tmp_path, monkeypatch):
    import src.model_cache as model_cache

    monkeypatch.setattr(model_cache, "_CACHE_DIR", tmp_path / "trained_models")
    frame = generate_growth_dataset("Salmonella spp.", n=80, seed=7)

    trained = load_or_train(frame, ["ratkowsky"])
    loaded = load_or_train(frame.copy(), ["ratkowsky"])

    assert trained.loaded_from_disk is False
    assert loaded.loaded_from_disk is True
    assert loaded.table.equals(trained.table)
    assert (
        loaded.models["ratkowsky"]
        .predict(frame[["temperature_celsius", "ph", "water_activity"]].to_numpy()[:5])
        .tolist()
        == trained.models["ratkowsky"]
        .predict(frame[["temperature_celsius", "ph", "water_activity"]].to_numpy()[:5])
        .tolist()
    )
    assert len(list((tmp_path / "trained_models").glob("*.joblib"))) == 1


def test_changed_dataset_gets_a_distinct_saved_model(tmp_path, monkeypatch):
    import src.model_cache as model_cache

    monkeypatch.setattr(model_cache, "_CACHE_DIR", tmp_path / "trained_models")
    frame = generate_growth_dataset("Salmonella spp.", n=80, seed=7)
    load_or_train(frame, ["ratkowsky"])

    changed = frame.copy()
    changed.loc[0, "mu_max_per_h"] += 0.01
    result = load_or_train(changed, ["ratkowsky"])

    assert result.loaded_from_disk is False
    assert len(list((tmp_path / "trained_models").glob("*.joblib"))) == 2


def test_force_retrain_replaces_the_matching_cached_model(tmp_path, monkeypatch):
    import src.model_cache as model_cache

    monkeypatch.setattr(model_cache, "_CACHE_DIR", tmp_path / "trained_models")
    frame = generate_growth_dataset("Salmonella spp.", n=80, seed=7)

    first = load_or_train(frame, ["ratkowsky"])
    retrained = load_or_train(frame, ["ratkowsky"], force_retrain=True)
    loaded = load_or_train(frame, ["ratkowsky"])

    assert first.loaded_from_disk is False
    assert retrained.loaded_from_disk is False
    assert loaded.loaded_from_disk is True
    assert len(list((tmp_path / "trained_models").glob("*.joblib"))) == 1


def test_grouping_labels_are_part_of_persistent_cache_key(tmp_path, monkeypatch):
    import src.model_cache as model_cache

    monkeypatch.setattr(model_cache, "_CACHE_DIR", tmp_path / "trained_models")
    frame = generate_growth_dataset("Salmonella spp.", n=80, seed=7)
    groups = pd.Series(["first" if i % 2 else "second" for i in range(len(frame))]).to_numpy()
    load_or_train(frame, ["ratkowsky"], groups=groups, group_name="food category")

    loaded = load_or_train(
        frame.copy(), ["ratkowsky"], groups=groups.copy(), group_name="food category"
    )
    changed_groups = groups.copy()
    changed_groups[0] = "third"
    retrained = load_or_train(
        frame, ["ratkowsky"], groups=changed_groups, group_name="food category"
    )

    assert loaded.loaded_from_disk is True
    assert retrained.loaded_from_disk is False


def test_training_source_is_part_of_persistent_cache_signature():
    import src.model_cache as model_cache

    frame = generate_growth_dataset("Salmonella spp.", n=80, seed=7)

    real_signature = model_cache._signature(frame, ["ratkowsky"], None, "supplied labels", "real")
    augmented_signature = model_cache._signature(
        frame,
        ["ratkowsky"],
        None,
        "supplied labels",
        "real+synthetic",
    )

    assert real_signature != augmented_signature


def test_model_cache_pruning_retains_the_current_artifact(tmp_path, monkeypatch):
    import src.model_cache as model_cache

    cache_dir = tmp_path / "trained_models"
    cache_dir.mkdir()
    monkeypatch.setattr(model_cache, "_CACHE_DIR", cache_dir)
    monkeypatch.setattr(model_cache, "_MAX_CACHE_ENTRIES", 2)
    stale = [cache_dir / f"stale-{index}.joblib" for index in range(3)]
    for index, path in enumerate(stale):
        path.write_text("stale", encoding="utf-8")
        os.utime(path, (index + 1, index + 1))
    current = cache_dir / "current.joblib"
    current.write_text("current", encoding="utf-8")

    model_cache._prune_cache(current)

    remaining = {path.name for path in cache_dir.glob("*.joblib")}
    assert current.name in remaining
    assert len(remaining) == 2
