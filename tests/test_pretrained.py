import json

import pytest

from src import model_cache, pretrained
from src.datasets import (
    expanded_data_available,
    load_real_growth_dataset,
    load_synthetic_growth_dataset,
    real_organisms,
    synthetic_organisms,
)

pytestmark = pytest.mark.usefixtures("real_bundle")


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Private cache + pretrained dirs and a one-model registry so training is fast."""
    cache, store = tmp_path / "cache", tmp_path / "pretrained"
    cache.mkdir(mode=0o700)
    monkeypatch.setattr(model_cache, "_CACHE_DIR", cache)
    monkeypatch.setattr(model_cache, "_PRETRAINED_DIR", store)
    monkeypatch.setattr(pretrained, "PRETRAINED_DIR", store)
    monkeypatch.setattr(pretrained, "available_model_keys", lambda: ["ratkowsky"])
    monkeypatch.setattr(
        model_cache, "_BENCHMARK_OPTIONS", {**model_cache._BENCHMARK_OPTIONS, "n_repeats": 2}
    )
    return cache, store


def test_grouping_options_match_the_ui_rules():
    real = load_real_growth_dataset("Bacillus cereus")
    options, default = pretrained.grouping_options(real)
    assert options[0] is None and "record_id" in options and "food_category" in options
    assert default == "food_category"
    synthetic = load_synthetic_growth_dataset(synthetic_organisms()[0])
    assert pretrained.grouping_options(synthetic)[0] == [None]


def test_grouping_arguments_validates():
    real = load_real_growth_dataset("Bacillus cereus")
    assert pretrained.grouping_arguments(real, None) == {}
    args = pretrained.grouping_arguments(real, "food_category")
    assert args["group_name"] == "food_category" and len(args["groups"]) == len(real)
    with pytest.raises(ValueError):
        pretrained.grouping_arguments(real, "not_a_column")
    with pytest.raises(ValueError):
        pretrained.grouping_arguments(real.assign(one="x"), "one")


def test_jobs_cover_every_dataset_and_grouping_the_ui_offers():
    jobs = pretrained.build_jobs()
    labels = {(j["kind"], j["organism"], j["group_column"]) for j in jobs}
    for organism in real_organisms():
        options, _ = pretrained.grouping_options(load_real_growth_dataset(organism))
        assert all(("real", organism, option) in labels for option in options)
    for organism in synthetic_organisms():
        assert ("synthetic", organism, None) in labels
    if expanded_data_available():
        assert any(
            j["kind"] == "expanded" and j["training_source"] == "real+synthetic" for j in jobs
        )


def test_signature_ignores_line_endings(tmp_path):
    lf = tmp_path / "a.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf = tmp_path / "b.py"
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert model_cache._normalised_bytes(lf) == model_cache._normalised_bytes(crlf)


def test_pretrained_models_load_without_training(isolated, monkeypatch):
    cache, store = isolated
    frame = load_real_growth_dataset("Bacillus cereus")
    keys = ["ratkowsky"]
    signature, status, _ = model_cache.train_and_store(frame, keys, store)
    assert status == "trained" and (store / f"{signature}.joblib").exists()

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not run when a pretrained model exists")

    monkeypatch.setattr(model_cache, "benchmark", forbidden)
    result = model_cache.load_or_train(frame, keys)
    assert result.loaded_from_disk and result.cache_source == "pretrained"
    assert list(cache.glob("*.joblib")) == []  # nothing needed to be written locally


def test_train_and_store_is_resumable(isolated):
    _, store = isolated
    frame = load_real_growth_dataset("Bacillus cereus")
    first = model_cache.train_and_store(frame, ["ratkowsky"], store)
    second = model_cache.train_and_store(frame, ["ratkowsky"], store)
    assert (first[1], second[1]) == ("trained", "exists") and first[0] == second[0]


def test_corrupt_pretrained_file_falls_back_to_training(isolated):
    cache, store = isolated
    frame = load_real_growth_dataset("Bacillus cereus")
    signature = model_cache.signature_for(frame, ["ratkowsky"])
    store.mkdir(parents=True)
    (store / f"{signature}.joblib").write_bytes(b"not a pickle")
    result = model_cache.load_or_train(frame, ["ratkowsky"])
    assert result.cache_source == "trained"


def test_changed_data_does_not_reuse_a_pretrained_model(isolated):
    _, store = isolated
    frame = load_real_growth_dataset("Bacillus cereus")
    model_cache.train_and_store(frame, ["ratkowsky"], store)
    changed = frame.copy()
    changed.loc[0, "mu_max_per_h"] *= 1.5
    assert model_cache.load_or_train(changed, ["ratkowsky"]).cache_source == "trained"


def test_status_and_manifest(isolated, monkeypatch):
    _, store = isolated
    jobs = [
        j
        for j in pretrained.build_jobs(include_expanded=False)
        if j["kind"] == "real" and j["organism"] == "Bacillus cereus" and j["group_column"] is None
    ]
    monkeypatch.setattr(pretrained, "build_jobs", lambda include_expanded=True: jobs)
    assert pretrained.pretrained_status(store)["state"] == "missing"
    records = pretrained.pretrain(store)
    assert [r["status"] for r in records] == ["trained"]
    status = pretrained.pretrained_status(store)
    assert status["state"] == "ready" and status["ready"] == status["total"] == 1
    manifest = json.loads((store / pretrained.MANIFEST_NAME).read_text())
    monkeypatch.setitem(manifest["environment"], "scikit-learn", "0.0.1")
    (store / pretrained.MANIFEST_NAME).write_text(json.dumps(manifest))
    assert pretrained.pretrained_status(store)["state"] == "stale"
    (store / pretrained.MANIFEST_NAME).write_text(
        json.dumps(
            {
                **manifest,
                "environment": model_cache.environment_fingerprint(),
                "source_digest": "other",
            }
        )
    )
    assert pretrained.pretrained_status(store)["state"] == "stale"


def test_second_pretrain_run_trains_nothing(isolated, monkeypatch):
    _, store = isolated
    jobs = [
        j
        for j in pretrained.build_jobs(include_expanded=False)
        if j["kind"] == "real" and j["organism"] == "Bacillus cereus" and j["group_column"] is None
    ]
    monkeypatch.setattr(pretrained, "build_jobs", lambda include_expanded=True: jobs)
    pretrained.pretrain(store)
    assert [r["status"] for r in pretrained.pretrain(store)] == ["exists"]


def test_shipped_pretrained_store_serves_every_dataset_without_training(tmp_path, monkeypatch):
    """Guarantee behind 'no training on visits'. Skipped where the libraries differ from the ones the
    store was built with (run scripts/pretrain_models.py there)."""
    status = pretrained.pretrained_status()
    if status["state"] in ("missing", "stale"):
        pytest.skip(f"pretrained store not built for this environment: {status['detail']}")
    assert status["state"] == "ready"

    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    monkeypatch.setattr(model_cache, "_CACHE_DIR", cache)

    def forbidden(*args, **kwargs):
        raise AssertionError("training attempted although a pretrained model should exist")

    monkeypatch.setattr(model_cache, "benchmark", forbidden)
    for job in pretrained.build_jobs():
        frame = pretrained._load(job)
        result = model_cache.load_or_train(
            frame,
            pretrained.available_model_keys(),
            training_source=job["training_source"],
            **pretrained.grouping_arguments(frame, job["group_column"]),
        )
        assert result.cache_source == "pretrained", job["label"]
