"""Loader for real ComBase Browser exports (``ComBaseExport*.xlsx``).

Each export workbook has one required sheet and up to four optional ones:

* ``Data Records`` (required) - one row per experiment: organism, food matrix,
  T / pH / Aw, and either a fitted ``Max.rate(logc.conc / h)`` (log10 units),
  the literal string ``'Fit data'`` (only the raw curve is given) or
  ``'No growth'`` (explicit negative result).
* ``Logcs`` - long-format raw points: ``Record ID, Time (h), Logc (log10 count)``.
* ``Temperatures`` / ``pHs`` / ``Aws`` - present only for records where that
  condition *varied during the experiment* (dynamic / cold-chain-style studies);
  for these, ``Data Records`` holds a semicolon-packed ``time;value;time;value...``
  string instead of a single number.

ComBase mixes **growth** studies with **survival / thermal-inactivation** studies
(freezing, curing, heat treatment - the rate is negative or the curve declines),
and constant-condition studies with **dynamic** ones (a changing temperature/pH/Aw
profile). This project's secondary models are isothermal-growth-only (FR-2:
"estimate specific growth rates" at a fixed T / pH / Aw), so this loader classifies
every record and keeps constant-condition growth only. Every excluded record is
counted, not silently dropped - see :class:`CombaseLoadReport`. Dynamic-profile
records are set aside separately: they are exactly the shape of data the cold-chain
*simulator* (not the secondary-model trainer) consumes, and are a natural candidate
for validating it against real measured outcomes (see ``dynamic_frame`` below).

Units: ComBase's rate is log10(count)/h; the rest of MicrobeSim works in natural-log
units (1/h), so it is multiplied by ln(10) on the way in.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import openpyxl
import pandas as pd

from src.ingestion import IngestionError
from src.models.primary import FitError, fit_primary_curve

LN10 = np.log(10.0)
MIN_CURVE_POINTS = 6
FLAT_LOG10_TOL = 0.2  # ln10-count change below which a curve counts as neither growth nor decline

RECORD_COLUMNS = {
    "Record ID": "record_id",
    "Organism": "organism",
    "Food category": "food_category",
    "Food Name": "food_name",
    "Temperature (C)": "temperature_celsius",
    "Aw": "water_activity",
    "pH": "ph",
    "Assumed": "assumed",
    "Max.rate(logc.conc / h)": "rate_log10_per_h",
    "Conditions": "conditions",
}

# ComBase's own free-text "Organism" values, normalised to one label per organism so
# records from differently-worded searches/exports still group together.
ORGANISM_ALIASES = {
    "escherichia coli": "Escherichia coli",
    "salmonella spp": "Salmonella spp.",
    "salmonella": "Salmonella spp.",
    "listeria monocytogenes/innocua": "Listeria monocytogenes",
    "listeria monocytogenes": "Listeria monocytogenes",
    "listeria monocytogens": "Listeria monocytogenes",
    "staphylococcus aureus": "Staphylococcus aureus",
    "campylobacteraceae (c.jejuni and relatives)": "Campylobacter jejuni",
    "campylobacter jejuni": "Campylobacter jejuni",
    "bacillus cereus": "Bacillus cereus",
    "clostridium perfringens": "Clostridium perfringens",
    "clostridium botulinum (prot.)": "Clostridium botulinum (proteolytic)",
    "clostridium botulinum (non-prot.)": "Clostridium botulinum (non-proteolytic)",
    "clostridium botulinum": "Clostridium botulinum (proteolytic)",
    "yersinia enterocolitica": "Yersinia enterocolitica",
}


def normalise_organism(raw: Optional[str]) -> str:
    if raw is None:
        return "Unknown"
    key = str(raw).strip().lower()
    return ORGANISM_ALIASES.get(key, str(raw).strip())


@dataclass
class CombaseLoadReport:
    n_files: int = 0
    n_records: int = 0
    n_duplicate_ids: int = 0
    n_growth: int = 0
    n_decline: int = 0
    n_no_growth_flag: int = 0
    n_flat: int = 0
    n_dynamic_profile: int = 0
    n_unclassifiable: int = 0
    n_curve_points_short: int = 0
    n_degenerate_curve: int = 0
    n_low_confidence_fit: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        rows = [
            ("Records parsed", self.n_records),
            ("Kept: isothermal growth", self.n_growth),
            ("Excluded: decline / inactivation", self.n_decline),
            ("Excluded: explicit 'No growth'", self.n_no_growth_flag),
            ("Excluded: no significant change", self.n_flat),
            ("Set aside: dynamic (time-varying) profile", self.n_dynamic_profile),
            ("Excluded: curve too short (<6 pts)", self.n_curve_points_short),
            ("Excluded: degenerate curve (all points same time)", self.n_degenerate_curve),
            ("Excluded: low-confidence curve fit", self.n_low_confidence_fit),
            ("Excluded: missing/unparseable T, pH or Aw", self.n_unclassifiable),
        ]
        return pd.DataFrame(rows, columns=["Category", "Count"])


def _is_dynamic(value) -> bool:
    return isinstance(value, str) and ";" in value


def _read_workbook(path: str):
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Data Records" not in wb.sheetnames:
        raise IngestionError(f"{os.path.basename(path)}: missing 'Data Records' sheet")
    dr = wb["Data Records"]
    headers = [c.value for c in dr[1]]
    records = [dict(zip(headers, row)) for row in dr.iter_rows(min_row=2, values_only=True)]

    curves: Dict[str, List] = {}
    if "Logcs" in wb.sheetnames:
        for rid, t, logc in wb["Logcs"].iter_rows(min_row=2, values_only=True):
            if rid is None:
                continue
            try:
                curves.setdefault(rid, []).append((float(t), float(logc)))
            except (TypeError, ValueError):
                continue

    dynamic: Dict[str, Dict[str, List]] = {}
    for sheet, key in (("Temperatures", "temperature"), ("pHs", "ph"), ("Aws", "water_activity")):
        if sheet not in wb.sheetnames:
            continue
        for rid, t, v in wb[sheet].iter_rows(min_row=2, values_only=True):
            if rid is None:
                continue
            try:
                dynamic.setdefault(rid, {}).setdefault(key, []).append((float(t), float(v)))
            except (TypeError, ValueError):
                continue
    return records, curves, dynamic


def _find_files(source: Union[str, os.PathLike, Sequence[str]]) -> List[str]:
    if isinstance(source, (list, tuple)):
        out: List[str] = []
        for s in source:
            out.extend(_find_files(s))
        return out
    source = str(source)
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "**", "*.xlsx"), recursive=True))
        if not files:
            raise IngestionError(f"no .xlsx files found under {source}")
        return files
    return [source]


def load_combase_folder(
    source: Union[str, os.PathLike, Sequence[str]],
    min_curve_points: int = MIN_CURVE_POINTS,
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CombaseLoadReport]":
    """Parse and consolidate one or many ComBase export workbooks (or a list of folders).

    Returns
    -------
    kinetic_frame : DataFrame
        One row per point of a usable constant-condition *growth* curve, ready for
        :func:`src.ingestion.load_kinetic_series` (``series_id, time_hours, measurement,
        organism, temperature_celsius, ph, water_activity``; ``measurement`` is log10
        count, i.e. ``measurement_type='log10_cfu'``).
    growth_rate_frame : DataFrame
        One row per growth record's mu_max (from ComBase's own fitted rate where given,
        otherwise fitted from the raw curve here), ready for
        :func:`src.datasets.load_growth_dataset` (``organism, temperature_celsius, ph,
        water_activity, mu_max_per_h``, natural-log units).
    dynamic_frame : DataFrame
        One row per telemetry point for records where T, pH or Aw *varied* during the
        experiment (``record_id, organism, time_hours, temperature_celsius, ph,
        water_activity, log10_count``), set aside rather than used for secondary-model
        training. A natural real-world check for the cold-chain *simulator*, not
        currently wired into the app.
    report : CombaseLoadReport
        Counts and warnings for every record, including why anything was excluded.
    """
    files = _find_files(source)
    all_records: Dict[str, dict] = {}
    all_curves: Dict[str, List] = {}
    all_dynamic: Dict[str, Dict[str, List]] = {}
    report = CombaseLoadReport(n_files=len(files))

    for path in files:
        try:
            records, curves, dynamic = _read_workbook(path)
        except IngestionError as exc:
            report.warnings.append(str(exc))
            continue
        for raw in records:
            rid = raw.get("Record ID")
            if rid is None:
                continue
            row = {RECORD_COLUMNS.get(k, k): v for k, v in raw.items() if k in RECORD_COLUMNS}
            row["organism"] = normalise_organism(row.get("organism"))
            if rid in all_records and all_records[rid] != row:
                report.n_duplicate_ids += 1
            all_records[rid] = row
        for rid, pts in curves.items():
            all_curves.setdefault(rid, pts)
        for rid, d in dynamic.items():
            all_dynamic.setdefault(rid, d)

    report.n_records = len(all_records)

    kinetic_rows: List[dict] = []
    growth_rows: List[dict] = []
    dynamic_rows: List[dict] = []

    for rid, rec in all_records.items():
        organism = rec.get("organism")
        raw_t, raw_ph, raw_aw = (
            rec.get("temperature_celsius"),
            rec.get("ph"),
            rec.get("water_activity"),
        )

        if _is_dynamic(raw_t) or _is_dynamic(raw_ph) or _is_dynamic(raw_aw):
            report.n_dynamic_profile += 1
            d = all_dynamic.get(rid, {})
            curve = sorted(all_curves.get(rid, []))
            temp_series = dict(d.get("temperature", []))
            ph_series = dict(d.get("ph", []))
            aw_series = dict(d.get("water_activity", []))
            for t, logc in curve:
                dynamic_rows.append(
                    {
                        "record_id": rid,
                        "organism": organism,
                        "time_hours": t,
                        "temperature_celsius": temp_series.get(t),
                        "ph": ph_series.get(t),
                        "water_activity": aw_series.get(t),
                        "log10_count": logc,
                        "food_category": rec.get("food_category"),
                    }
                )
            continue

        try:
            temp = float(raw_t)
            ph = float(raw_ph) if raw_ph not in (None, "") else None
            aw = float(raw_aw) if raw_aw not in (None, "") else None
        except (TypeError, ValueError):
            report.n_unclassifiable += 1
            continue
        if ph is None or aw is None:
            report.n_unclassifiable += 1
            continue

        rate = rec.get("rate_log10_per_h")
        pts = sorted(all_curves.get(rid, []))

        mu_max: Optional[float] = None
        if rate == "No growth":
            report.n_no_growth_flag += 1
            continue
        if isinstance(rate, (int, float)):
            if rate > 0:
                mu_max = rate * LN10
            elif rate < 0:
                report.n_decline += 1
                continue
            else:
                report.n_flat += 1
                continue
        elif pts:
            if len(pts) < 3:
                report.n_unclassifiable += 1
                continue
            t_raw = np.array([p[0] for p in pts])
            if float(np.ptp(t_raw)) <= 0:
                # every point stamped at the same time - replicate counts, not a time series
                report.n_degenerate_curve += 1
                continue
            delta = pts[-1][1] - pts[0][1]
            if delta > FLAT_LOG10_TOL:
                if len(pts) < min_curve_points:
                    report.n_curve_points_short += 1
                    continue
                y = LN10 * (np.array([p[1] for p in pts]) - min(p[1] for p in pts[:3]))
                try:
                    fit = fit_primary_curve(t_raw, y)
                except FitError as exc:
                    report.warnings.append(f"{rid}: curve fit failed ({exc})")
                    continue
                # A fit pinned at the optimizer's lower bound, with poor R2, or with huge
                # parameter uncertainty is not trustworthy growth evidence: noisy/near-flat
                # curves that just cleared FLAT_LOG10_TOL, or - the specific failure mode found
                # by testing on real exports - a curve where the entire rise falls in one gap
                # between samples (mu_max is then unconstrained even though R2 looks fine).
                rel_err = fit.param_std[1] / fit.mu_max if fit.mu_max > 0 else np.inf
                if (
                    fit.mu_max <= 1.5e-3
                    or fit.r_squared < 0.8
                    or not np.isfinite(rel_err)
                    or rel_err > 0.5
                ):
                    report.n_low_confidence_fit += 1
                    continue
                mu_max = fit.mu_max
            elif delta < -FLAT_LOG10_TOL:
                report.n_decline += 1
                continue
            else:
                report.n_flat += 1
                continue
        else:
            report.n_unclassifiable += 1
            continue

        report.n_growth += 1
        growth_rows.append(
            {
                "record_id": rid,
                "organism": organism,
                "temperature_celsius": temp,
                "ph": ph,
                "water_activity": aw,
                "mu_max_per_h": mu_max,
                "food_category": rec.get("food_category"),
                "assumed": rec.get("assumed"),
                "source": (
                    "combase_summary_rate"
                    if isinstance(rate, (int, float))
                    else "combase_curve_fit"
                ),
            }
        )
        if len(pts) >= min_curve_points and float(np.ptp([p[0] for p in pts])) > 0:
            for t, logc in pts:
                kinetic_rows.append(
                    {
                        "series_id": rid,
                        "organism": organism,
                        "temperature_celsius": temp,
                        "ph": ph,
                        "water_activity": aw,
                        "time_hours": t,
                        "measurement": logc,
                        "food_category": rec.get("food_category"),
                    }
                )

    if report.n_duplicate_ids:
        report.warnings.append(
            f"{report.n_duplicate_ids} record ID(s) appeared in more than one file with different "
            "values; the first file's values were kept"
        )
    return (
        pd.DataFrame(kinetic_rows),
        pd.DataFrame(growth_rows),
        pd.DataFrame(dynamic_rows),
        report,
    )
