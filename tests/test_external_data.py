import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import external_data as ext
from src.datasets import load_real_growth_dataset
from src.external_data import (
    ExternalDataError,
    load_external_growth_dataset,
    validate_external_frame,
)

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_external_data_is_valid_and_traceable():
    frame = pd.read_csv(ext.EXTERNAL_RATES_CSV)
    sources = ext.load_sources()
    assert validate_external_frame(frame, sources) == []
    assert len(frame) == 39 and set(frame.source_id) == {"Son2024_Foods_vegetables"}
    row = sources.set_index("source_id").loc["Son2024_Foods_vegetables"]
    assert (
        row["doi"] == "10.3390/foods13182972"
        and row["license"] == "cc-by-4.0"
        and row["status"] == "imported"
    )


def test_rates_are_natural_log_conversions_of_the_published_values():
    frame = pd.read_csv(ext.EXTERNAL_RATES_CSV)
    assert np.allclose(frame.mu_max_per_h, frame.rate_log10_per_h_published * np.log(10), atol=1e-5)


def test_duplicate_listeria_cabbage_block_is_excluded_not_imported():
    frame = pd.read_csv(ext.EXTERNAL_RATES_CSV)
    excluded = pd.read_csv(ext.EXCLUDED_ROWS_CSV)
    assert len(excluded) == 5 and excluded.exclusion_reason.str.contains("copy-paste").all()
    cabbage_listeria = (frame.organism == "Listeria monocytogenes") & (frame.food_name == "cabbage")
    assert not cabbage_listeria.any()
    onion = frame[(frame.organism == "Listeria monocytogenes") & (frame.food_name == "onion")]
    shared = set(excluded.rate_log10_per_h_published) & set(onion.rate_log10_per_h_published)
    assert len(shared) == 4  # the 10-36 C rates repeat the onion block; the 4 C rate differs
    assert set(excluded.max_density_log10) == set(onion.max_density_log10)


def test_series_behave_physically():
    frame = pd.read_csv(ext.EXTERNAL_RATES_CSV)
    for _, group in frame.groupby(["organism", "food_name"]):
        group = group.sort_values("temperature_celsius")
        assert group.mu_max_per_h.is_monotonic_increasing
        assert group.lag_h.is_monotonic_decreasing


def test_build_script_reproduces_the_shipped_files(tmp_path):
    before = pd.read_csv(ext.EXTERNAL_RATES_CSV)
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_external_dataset.py")],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    pd.testing.assert_frame_equal(before, pd.read_csv(ext.EXTERNAL_RATES_CSV))


def _row(**overrides):
    base = dict(
        record_id="X1",
        organism="Escherichia coli",
        temperature_celsius=30.0,
        ph=7.0,
        water_activity=0.99,
        mu_max_per_h=1.2,
        lag_h=1.0,
        food_category="Culture medium",
        food_name="broth",
        source_id="S1",
        table_ref="Table 1",
        n_replicates=3,
    )
    base.update(overrides)
    return pd.DataFrame([base])


def _sources(license="cc-by-4.0", doi="10.1/x", status="imported"):
    return pd.DataFrame([{"source_id": "S1", "license": license, "doi": doi, "status": status}])


def test_validation_accepts_a_good_row_and_rejects_bad_ones():
    assert validate_external_frame(_row(), _sources()) == []
    assert any(
        "organism" in p for p in validate_external_frame(_row(organism="Made up"), _sources())
    )
    assert any(
        "temperature" in p
        for p in validate_external_frame(_row(temperature_celsius=500), _sources())
    )
    assert any(
        "water_activity" in p for p in validate_external_frame(_row(water_activity=1.5), _sources())
    )
    assert any(
        "mu_max" in p for p in validate_external_frame(_row(mu_max_per_h=float("nan")), _sources())
    )
    assert any(
        "food_category" in p
        for p in validate_external_frame(_row(food_category="Pizza"), _sources())
    )
    assert any(
        "not in external_sources" in p
        for p in validate_external_frame(_row(source_id="nope"), _sources())
    )
    two = pd.concat([_row(), _row()], ignore_index=True)
    assert any("unique" in p for p in validate_external_frame(two, _sources()))


@pytest.mark.parametrize(
    "license",
    ["cc-by-nc-4.0", "cc-by-nd-4.0", "all-rights-reserved", "combase-terms-nontransferable", ""],
)
def test_licences_that_do_not_permit_bundling_are_refused(license):
    problems = validate_external_frame(_row(), _sources(license=license))
    assert any("does not permit bundling" in p for p in problems)


def test_doi_and_status_rules():
    assert any("DOI" in p for p in validate_external_frame(_row(), _sources(doi="")))
    assert validate_external_frame(_row(), _sources(license="owner-contributed", doi="")) == []
    assert any(
        "not 'imported'" in p
        for p in validate_external_frame(_row(), _sources(status="candidate_not_imported"))
    )


def test_candidate_sources_are_registered_but_not_imported():
    sources = ext.load_sources().set_index("source_id")
    for source_id in ("ArtiSaneFood_Zenodo_8088477", "Wang2024_Toxins_Bcereus", "ComBase_browser"):
        assert sources.loc[source_id, "status"] == "candidate_not_imported"
    assert sources.loc["ComBase_browser", "license"] not in ext.BUNDLE_LICENSES


def test_loader_standard_columns_and_combined_dataset(real_bundle):
    plain = load_real_growth_dataset("Salmonella spp.")
    combined = load_real_growth_dataset("Salmonella spp.", include_external=True)
    assert list(plain.columns) == list(combined.columns)
    assert len(combined) - len(plain) == 21
    assert load_external_growth_dataset("Listeria monocytogenes").shape[0] == 18
    assert (combined.source.str.startswith("published_table")).sum() == 21
    # default behaviour of the real-data loader is unchanged
    assert not plain.source.str.startswith("published_table").any()


def test_loader_refuses_invalid_external_files(tmp_path, monkeypatch):
    bad = pd.read_csv(ext.EXTERNAL_RATES_CSV)
    bad.loc[0, "water_activity"] = 9
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    monkeypatch.setattr(ext, "EXTERNAL_RATES_CSV", path)
    with pytest.raises(ExternalDataError):
        load_external_growth_dataset()


def _run_import(csv, *args, tmp_path):
    import os

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "import_external_data.py"), str(csv), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )


def test_importer_dry_run_refuses_unlicensed_and_changes_nothing(tmp_path):
    rows = _row(record_id="NEW1", mu_max_per_h=0.77).drop(columns="source_id")
    rows.to_csv(tmp_path / "rows.csv", index=False)
    before = ext.EXTERNAL_RATES_CSV.read_bytes()
    # unregistered source id -> refused
    done = _run_import(tmp_path / "rows.csv", "--source-id", "NotRegistered", tmp_path=tmp_path)
    assert done.returncode == 1 and "NOT IMPORTED" in done.stdout
    # registering with a non-bundling licence -> refused
    done = _run_import(
        tmp_path / "rows.csv",
        "--source-id",
        "NC1",
        "--register-source",
        "--citation",
        "c",
        "--obtained-how",
        "h",
        "--license",
        "cc-by-nc-4.0",
        "--doi",
        "10.1/x",
        tmp_path=tmp_path,
    )
    assert done.returncode == 1 and "does not permit bundling" in done.stdout
    # a good source in dry run writes nothing
    done = _run_import(
        tmp_path / "rows.csv",
        "--source-id",
        "Own1",
        "--register-source",
        "--citation",
        "c",
        "--obtained-how",
        "lab run",
        "--license",
        "owner-contributed",
        tmp_path=tmp_path,
    )
    assert done.returncode == 0 and "Dry run" in done.stdout
    assert ext.EXTERNAL_RATES_CSV.read_bytes() == before


def test_importer_refuses_rows_labelled_with_another_source(tmp_path):
    _row(record_id="NEW2").to_csv(tmp_path / "rows.csv", index=False)  # carries source_id S1
    done = _run_import(
        tmp_path / "rows.csv", "--source-id", "Son2024_Foods_vegetables", tmp_path=tmp_path
    )
    assert done.returncode == 1 and "different from --source-id" in done.stdout


def test_importer_rejects_duplicates_of_existing_rows(tmp_path):
    existing = pd.read_csv(ext.EXTERNAL_RATES_CSV).iloc[[0]]
    dup = _row(
        record_id="DUP1",
        organism=existing.organism.iloc[0],
        temperature_celsius=existing.temperature_celsius.iloc[0],
        ph=existing.ph.iloc[0],
        water_activity=existing.water_activity.iloc[0],
        mu_max_per_h=existing.mu_max_per_h.iloc[0],
        food_category="Vegetable or fruit and their products",
        source_id="Son2024_Foods_vegetables",
    )
    dup.to_csv(tmp_path / "dup.csv", index=False)
    done = _run_import(
        tmp_path / "dup.csv", "--source-id", "Son2024_Foods_vegetables", tmp_path=tmp_path
    )
    assert done.returncode == 1 and "duplicate" in done.stdout


def test_ui_offers_the_combined_datasets_and_pretraining_covers_them(real_bundle):
    from src.pretrained import build_jobs

    plus = [j for j in build_jobs(include_expanded=False) if j["kind"] == "real_plus"]
    assert {j["organism"] for j in plus} == {"Salmonella spp.", "Listeria monocytogenes"}
    assert all("published" in j["label"] for j in plus)
