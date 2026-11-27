import io

import numpy as np
import pandas as pd
import pytest

from src.ingestion import (
    IngestionError,
    MAX_CLEANED_POINTS_PER_CURVE,
    MAX_UPLOAD_BYTES,
    hampel,
    load_cold_chain_telemetry,
    load_kinetic_series,
    read_table,
    to_ln_ratio,
)
from src.models.primary import fit_primary_curve, gompertz


def _df(seed=0, n_series=1):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 24.01, 0.5)
    frames = []
    for k in range(n_series):
        y = gompertz(t, 5.0, 0.6, 3.0) + rng.normal(0, 0.05, len(t))
        frames.append(
            pd.DataFrame({"series_id": f"w{k}", "Time": t, "Measurement": 2 + y / np.log(10)})
        )
    return pd.concat(frames, ignore_index=True)


def test_alias_columns_and_defaults():
    series, rep = load_kinetic_series(_df(), default_organism="E. coli", default_temperature_c=30)
    assert series[0].organism_name == "E. coli" and series[0].temperature_celsius == 30
    assert len(series[0].points) == 49


def test_defaults_fill_missing_metadata_and_can_explicitly_override_it():
    df = _df()
    df["organism"] = "Salmonella"
    df["temperature_celsius"] = 37.0
    df["ph"] = 7.0
    df["water_activity"] = 0.99

    kept, _ = load_kinetic_series(
        df,
        default_organism="E. coli",
        default_temperature_c=30.0,
        default_ph=5.5,
        default_aw=0.91,
    )
    replaced, _ = load_kinetic_series(
        df,
        default_organism="E. coli",
        default_temperature_c=30.0,
        default_ph=5.5,
        default_aw=0.91,
        override_metadata=True,
    )

    assert (kept[0].organism_name, kept[0].temperature_celsius, kept[0].ph) == (
        "Salmonella",
        37.0,
        7.0,
    )
    assert (
        replaced[0].organism_name,
        replaced[0].temperature_celsius,
        replaced[0].ph,
        replaced[0].water_activity,
    ) == ("E. coli", 30.0, 5.5, 0.91)


def test_cleaning_options_change_processed_curve():
    df = _df()
    df.loc[12, "Measurement"] += 2.0
    smoothed, report_smoothed = load_kinetic_series(df, smooth=True, sg_window=5, hampel_sigmas=2.0)
    raw, report_raw = load_kinetic_series(df, smooth=False, hampel_sigmas=2.0)

    clean_smoothed = report_smoothed.cleaned[smoothed[0].series_id]
    clean_raw = report_raw.cleaned[raw[0].series_id]
    assert clean_smoothed.outliers[12]
    assert clean_smoothed.outliers.sum() == clean_raw.outliers.sum()
    assert not np.array_equal(clean_smoothed.clean, clean_raw.clean)


def test_gap_fill_nan_and_outlier_are_repaired():
    df = _df()
    df.loc[10, "Measurement"] += 2.0
    df.loc[20, "Measurement"] = np.nan
    df = df.drop(index=range(30, 34))
    series, rep = load_kinetic_series(df)
    joined = " ".join(rep.warnings)
    assert "interpolated" in joined and "outlier" in joined
    curve = rep.cleaned["w0"]
    assert curve.interpolated.sum() >= 3 and curve.outliers[10]
    assert not np.isnan(curve.clean).any()


def test_hampel_flags_spike_only():
    x = np.linspace(0, 5, 40)
    x[17] += 3
    out, mask = hampel(x)
    assert mask[17] and mask.sum() == 1 and abs(out[17] - x[17]) > 2.5


def test_end_to_end_recovery():
    series, rep = load_kinetic_series(_df(3))
    c = rep.cleaned["w0"]
    fit = fit_primary_curve(c.time, to_ln_ratio(c.clean, "log10_cfu"))
    assert abs(fit.mu_max - 0.6) < 0.08 and abs(fit.lam - 3.0) < 0.6


def test_batch_of_100_under_five_seconds():
    import time

    start = time.perf_counter()
    series, rep = load_kinetic_series(_df(1, n_series=100))
    for s in series:
        c = rep.cleaned[s.series_id]
        fit_primary_curve(c.time, to_ln_ratio(c.clean, "log10_cfu"))
    assert len(series) == 100 and time.perf_counter() - start < 5.0


def test_missing_columns_raise():
    with pytest.raises(IngestionError):
        load_kinetic_series(io.StringIO("a,b\n1,2\n"), filename="x.csv")


def test_table_reader_rejects_oversized_file_like_input():
    oversized = io.BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1))
    with pytest.raises(IngestionError, match="25 MB"):
        read_table(oversized, filename="large.csv")


def test_gap_interpolation_is_bounded_for_extreme_time_gaps():
    time = np.array([0.0, 1.0, 2.0, 1_000_000.0])
    measurement = np.arange(len(time), dtype=float)

    with pytest.raises(IngestionError, match="200,000-point"):
        from src.ingestion import clean_curve

        clean_curve(time, measurement)

    assert MAX_CLEANED_POINTS_PER_CURVE == 200_000


def test_negative_measurements_dropped_with_warning():
    df = _df()
    df.loc[0, "Measurement"] = -1.0
    _, rep = load_kinetic_series(df)
    assert any("negative" in w for w in rep.warnings)


def test_telemetry_validation():
    pts, warns = load_cold_chain_telemetry(
        io.StringIO("timestamp_hours,temperature_celsius\n0,4\n1,5\n1,6\n2,7\n"), filename="t.csv"
    )
    assert len(pts) == 3 and warns
    with pytest.raises(IngestionError):
        load_cold_chain_telemetry(
            io.StringIO("timestamp_hours,temperature_celsius\n0,4\n1,80\n"), filename="t.csv"
        )
    with pytest.raises(IngestionError):
        load_cold_chain_telemetry(
            io.StringIO("timestamp_hours,temperature_celsius\n0,4\n"), filename="t.csv"
        )


def test_telemetry_accepts_dynamic_ph_and_water_activity_profiles():
    source = io.StringIO(
        "timestamp_hours,temperature_celsius,ph,aw\n"
        "0,4,6.0,0.98\n"
        "1,5,,0.96\n"
        "2,8,8.0,0.94\n"
    )
    points, warnings = load_cold_chain_telemetry(source, filename="dynamic.csv")

    assert [point.ph for point in points] == [6.0, 7.0, 8.0]
    assert [point.water_activity for point in points] == [0.98, 0.96, 0.94]
    assert any("interpolated" in warning for warning in warnings)
    with pytest.raises(IngestionError, match="water_activity"):
        load_cold_chain_telemetry(
            io.StringIO("timestamp_hours,temperature_celsius,water_activity\n0,4,0\n1,5,0.95\n"),
            filename="invalid.csv",
        )


def test_ln_ratio_conversion():
    y = to_ln_ratio(np.array([3.0, 3.0, 4.0]), "log10_cfu")
    assert abs(y[2] - np.log(10)) < 1e-12
    assert abs(to_ln_ratio(np.array([0.1, 0.1, 0.2]), "od600")[2] - np.log(2)) < 1e-12
