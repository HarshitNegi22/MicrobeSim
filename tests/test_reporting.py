import io

import numpy as np

from src.datasets import generate_growth_dataset
from src.ingestion import load_kinetic_series, to_ln_ratio
from src.models.benchmark import benchmark
from src.models.primary import fit_primary_curve, gompertz
from src.reporting import (
    build_benchmark_report,
    build_fit_report,
    build_simulation_report,
    df_to_csv_bytes,
    frame_fingerprint,
    frames_to_xlsx_bytes,
    fingerprint,
    fit_results_frame,
)
from src.schemas import SimulationScenario
from src.simulation.dynamic import build_mu_lookup, simulate
from src.simulation.scenarios import PRESETS
import pandas as pd
from openpyxl import load_workbook


def _entries():
    t = np.arange(0, 24.01, 0.5)
    df = pd.DataFrame({"time_hours": t, "measurement": 2 + gompertz(t, 5, 0.6, 3) / np.log(10)})
    series, rep = load_kinetic_series(df, filename="a.csv")
    c = rep.cleaned[series[0].series_id]
    y = to_ln_ratio(c.clean, "log10_cfu")
    return [
        dict(
            series=series[0],
            fit=fit_primary_curve(c.time, y),
            curve=c,
            y_raw=to_ln_ratio(c.raw, "log10_cfu"),
            y_clean=y,
        )
    ]


def test_pdf_and_csv_exports():
    e = _entries()
    assert build_fit_report(e, "log10_cfu")[:5] == b"%PDF-"
    frame = fit_results_frame(e)
    assert b"mu_max_per_h" in df_to_csv_bytes(frame)
    res = benchmark(generate_growth_dataset("Salmonella spp.", n=300, seed=1), ["ratkowsky", "rf"])
    assert build_benchmark_report(res.table, "synthetic", 300)[:5] == b"%PDF-"
    mu_fn = build_mu_lookup(res.models["ratkowsky"], 6.5, 0.98)
    sc = SimulationScenario(organism="x", environmental_log=PRESETS["heat_spike"].build())
    trace = simulate(sc, mu_fn)
    assert build_simulation_report("x", "m", trace, {"pH": 6.5})[:5] == b"%PDF-"


def test_fingerprint_is_deterministic():
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    frame = pd.DataFrame({"value": [1, 2]})
    assert frame_fingerprint(frame) == frame_fingerprint(frame.copy())
    assert frames_to_xlsx_bytes({"Data": frame})[:2] == b"PK"


def test_spreadsheet_exports_escape_formula_like_text():
    frame = pd.DataFrame(
        {
            "user_text": ["=1+1", "  +SUM(A1:A2)", '\t@HYPERLINK("x")', "-1", "ordinary"],
            "number": [1, 2, 3, 4, 5],
        }
    )

    csv = df_to_csv_bytes(frame).decode("utf-8")
    assert "'=1+1" in csv
    assert "'  +SUM(A1:A2)" in csv
    assert "'\t@HYPERLINK" in csv
    assert "'-1" in csv
    assert frame.iloc[0, 0] == "=1+1"

    workbook = load_workbook(io.BytesIO(frames_to_xlsx_bytes({"Data": frame})))
    values = [row[0].value for row in workbook["Data"].iter_rows(min_row=2)]
    assert values == [
        "'=1+1",
        "'  +SUM(A1:A2)",
        '\'\t@HYPERLINK("x")',
        "'-1",
        "ordinary",
    ]

    header_frame = pd.DataFrame({"=malicious header": ["safe data"]})
    assert b"'=malicious header" in df_to_csv_bytes(header_frame)
    header_book = load_workbook(io.BytesIO(frames_to_xlsx_bytes({"Data": header_frame})))
    assert header_book["Data"]["A1"].value == "'=malicious header"
