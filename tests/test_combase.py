"""Tests for the ComBase export loader, using small synthetic workbooks so the
suite does not depend on any real (private) export file."""

import numpy as np
import openpyxl
import pytest

from src.combase import load_combase_folder, normalise_organism

LN10 = np.log(10.0)


def _wb(path, records, logcs=None, temperatures=None, phs=None, aws=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data Records"
    headers = [
        "Record ID",
        "Organism",
        "Food category",
        "Food Name",
        "Temperature (C)",
        "Aw",
        "pH",
        "Assumed",
        "Max.rate(logc.conc / h)",
        "Conditions",
        "Logcs",
    ]
    ws.append(headers)
    for r in records:
        ws.append([r.get(h) for h in headers])
    if logcs is not None:
        lc = wb.create_sheet("Logcs")
        lc.append(["Record ID", "Time", "Logc"])
        for row in logcs:
            lc.append(list(row))
    for name, rows in (("Temperatures", temperatures), ("pHs", phs), ("Aws", aws)):
        if rows is not None:
            sheet = wb.create_sheet(name)
            sheet.append(["Record ID", "Time", name[:-1] if name != "pHs" else "pH"])
            for row in rows:
                sheet.append(list(row))
    wb.save(path)


def _growth_curve(rid, mu=0.5, lag=2.0, n=8, hours=20.0):
    """A clean synthetic log10 growth curve (Gompertz-like via simple logistic-ish rise)."""
    t = np.linspace(0, hours, n)
    y = mu * np.maximum(t - lag, 0)  # ln-scale ramp after lag
    log10 = 2.0 + y / LN10
    return [(rid, float(tt), float(ll)) for tt, ll in zip(t, log10)]


def test_organism_normalisation():
    assert normalise_organism("Salmonella spp") == "Salmonella spp."
    assert normalise_organism("listeria monocytogenes/innocua") == "Listeria monocytogenes"
    assert normalise_organism("Some New Organism") == "Some New Organism"


def test_numeric_growth_and_decline_rates(tmp_path):
    records = [
        dict(
            **{
                "Record ID": "G1",
                "Organism": "Escherichia coli",
                "Temperature (C)": 30,
                "Aw": 0.99,
                "pH": 7.0,
                "Max.rate(logc.conc / h)": 0.3,
            }
        ),
        dict(
            **{
                "Record ID": "D1",
                "Organism": "Escherichia coli",
                "Temperature (C)": 60,
                "Aw": 0.99,
                "pH": 7.0,
                "Max.rate(logc.conc / h)": -0.5,
            }
        ),
        dict(
            **{
                "Record ID": "N1",
                "Organism": "Escherichia coli",
                "Temperature (C)": 5,
                "Aw": 0.99,
                "pH": 7.0,
                "Max.rate(logc.conc / h)": "No growth",
            }
        ),
    ]
    _wb(tmp_path / "a.xlsx", records)
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_growth == 1 and rep.n_decline == 1 and rep.n_no_growth_flag == 1
    assert len(growth) == 1
    row = growth.iloc[0]
    assert row["organism"] == "Escherichia coli"
    assert abs(row["mu_max_per_h"] - 0.3 * LN10) < 1e-9


def test_curve_derived_growth_and_kinetic_frame(tmp_path):
    records = [
        dict(
            **{
                "Record ID": "C1",
                "Organism": "Salmonella spp",
                "Temperature (C)": 25,
                "Aw": 0.98,
                "pH": 6.5,
                "Max.rate(logc.conc / h)": "Fit data",
            }
        )
    ]
    logcs = _growth_curve("C1", mu=0.4, lag=1.0, n=10, hours=24)
    _wb(tmp_path / "b.xlsx", records, logcs=logcs)
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_growth == 1
    assert abs(growth.iloc[0]["mu_max_per_h"] - 0.4) < 0.1
    assert growth.iloc[0]["organism"] == "Salmonella spp."
    assert kin["series_id"].nunique() == 1
    assert len(kin) == 10


def test_degenerate_zero_duration_curve_excluded(tmp_path):
    records = [
        dict(
            **{
                "Record ID": "Z1",
                "Organism": "Escherichia coli",
                "Temperature (C)": 70,
                "Aw": 0.99,
                "pH": 7.0,
                "Max.rate(logc.conc / h)": "Fit data",
            }
        )
    ]
    logcs = [("Z1", 0.0, v) for v in [2.8, 3.1, 4.2, 5.3, 6.2, 6.4, 8.9]]  # all Time == 0
    _wb(tmp_path / "c.xlsx", records, logcs=logcs)
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_degenerate_curve == 1
    assert rep.n_growth == 0
    assert len(kin) == 0


def test_sparse_gap_curve_rejected_by_uncertainty_gate(tmp_path):
    """Reproduces the real failure mode found in testing: growth compressed into one
    inter-sample gap, then a noisy plateau -> huge parameter uncertainty -> rejected."""
    records = [
        dict(
            **{
                "Record ID": "S1",
                "Organism": "Bacillus cereus",
                "Temperature (C)": 28,
                "Aw": 0.99,
                "pH": 5.4,
                "Max.rate(logc.conc / h)": "Fit data",
            }
        )
    ]
    logcs = [
        ("S1", 0.0, 3.18),
        ("S1", 22.0, 7.36),
        ("S1", 28.0, 8.32),
        ("S1", 40.0, 7.52),
        ("S1", 46.0, 7.26),
        ("S1", 52.0, 8.03),
    ]
    _wb(tmp_path / "d.xlsx", records, logcs=logcs)
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_low_confidence_fit == 1
    assert rep.n_growth == 0


def test_flat_curve_and_short_curve(tmp_path):
    flat = [("F1", t, 3.0 + 0.01 * t) for t in range(5)]
    short = [("SH1", 0.0, 2.0), ("SH1", 5.0, 4.0), ("SH1", 10.0, 6.0)]
    records = [
        dict(
            **{
                "Record ID": "F1",
                "Organism": "Escherichia coli",
                "Temperature (C)": 4,
                "Aw": 0.99,
                "pH": 7.0,
                "Max.rate(logc.conc / h)": "Fit data",
            }
        ),
        dict(
            **{
                "Record ID": "SH1",
                "Organism": "Escherichia coli",
                "Temperature (C)": 30,
                "Aw": 0.99,
                "pH": 7.0,
                "Max.rate(logc.conc / h)": "Fit data",
            }
        ),
    ]
    _wb(tmp_path / "e.xlsx", records, logcs=flat + short)
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_flat == 1
    assert rep.n_curve_points_short == 1
    assert rep.n_growth == 0


def test_dynamic_profile_set_aside(tmp_path):
    records = [
        dict(
            **{
                "Record ID": "DYN1",
                "Organism": "Campylobacter jejuni",
                "Temperature (C)": "0;4;72;4;96;-20",
                "Aw": None,
                "pH": None,
                "Max.rate(logc.conc / h)": "Fit data",
            }
        )
    ]
    logcs = [("DYN1", 0.0, 7.73), ("DYN1", 72.0, 7.38), ("DYN1", 96.0, 5.78)]
    temps = [("DYN1", 0.0, 4.0), ("DYN1", 72.0, 4.0), ("DYN1", 96.0, -20.0)]
    _wb(tmp_path / "f.xlsx", records, logcs=logcs, temperatures=temps)
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_dynamic_profile == 1
    assert rep.n_growth == 0 and rep.n_unclassifiable == 0
    assert len(dyn) == 3
    assert dyn.iloc[0]["temperature_celsius"] == 4.0


def test_missing_ph_or_aw_excluded(tmp_path):
    records = [
        dict(
            **{
                "Record ID": "M1",
                "Organism": "Escherichia coli",
                "Temperature (C)": 30,
                "Aw": None,
                "pH": 7.0,
                "Max.rate(logc.conc / h)": 0.3,
            }
        )
    ]
    _wb(tmp_path / "g.xlsx", records)
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_unclassifiable == 1 and rep.n_growth == 0


def test_duplicate_record_id_across_files_is_counted(tmp_path):
    r1 = dict(
        **{
            "Record ID": "DUP",
            "Organism": "Escherichia coli",
            "Temperature (C)": 30,
            "Aw": 0.99,
            "pH": 7.0,
            "Max.rate(logc.conc / h)": 0.3,
        }
    )
    r2 = dict(r1, **{"Max.rate(logc.conc / h)": 0.5})
    _wb(tmp_path / "h1.xlsx", [r1])
    _wb(tmp_path / "h2.xlsx", [r2])
    kin, growth, dyn, rep = load_combase_folder(str(tmp_path))
    assert rep.n_duplicate_ids == 1
    assert rep.n_records == 1  # still one logical record


def test_empty_folder_raises(tmp_path):
    with pytest.raises(Exception):
        load_combase_folder(str(tmp_path))
