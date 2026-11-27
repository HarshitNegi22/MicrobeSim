"""Ingestion pipeline (FR-1): parse, validate, clean and convert assay / telemetry files.

Steps for kinetic curves
------------------------
1. Read CSV / Excel, standardise column names (``Time``, ``Measurement`` are accepted).
2. Validate against the Pydantic data contracts (``src/schemas.py``).
3. Fill missing values and time gaps by linear interpolation.
4. Remove optical artefacts: Hampel (rolling-median) outlier replacement, then
   Savitzky-Golay smoothing.
5. Convert the response to ``y = ln(N / N0)`` for the primary models.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.schemas import DynamicTelemetryPoint, KineticSeries, RawCurvePoint

SENSOR_MIN_C, SENSOR_MAX_C = -20.0, 50.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_EXPANDED_WORKBOOK_BYTES = 100 * 1024 * 1024
MAX_TABLE_ROWS = 100_000
MAX_TABLE_COLUMNS = 100
MAX_CURVES_PER_FILE = 500
MAX_CLEANED_POINTS_PER_CURVE = 200_000
MEASUREMENT_TYPES = ("log10_cfu", "od600")

Source = Union[str, os.PathLike, io.IOBase, pd.DataFrame]

_TIME_ALIASES = {"time_hours", "time", "time_h", "t", "hours", "hour", "elapsed_h", "time_(h)"}
_MEAS_ALIASES = {
    "measurement",
    "od600",
    "od",
    "od_600",
    "logcfu",
    "log_cfu",
    "log10_cfu",
    "log_cfu/ml",
    "value",
}
_SERIES_ALIASES = {"series_id", "series", "well", "curve", "curve_id", "sample", "sample_id"}
_META_ALIASES = {
    "organism_name": {"organism_name", "organism", "strain", "species"},
    "temperature_celsius": {
        "temperature_celsius",
        "temperature",
        "temp",
        "temp_c",
        "temperature_c",
    },
    "ph": {"ph"},
    "water_activity": {"water_activity", "aw", "wateractivity"},
}
_TELEM_TIME = {"timestamp_hours", "timestamp", "time", "time_hours", "hours", "t", "time_h"}
_TELEM_TEMP = {"temperature_celsius", "temperature", "temp", "temp_c", "temperature_c", "celsius"}
_TELEM_PH = {"ph"}
_TELEM_AW = {"water_activity", "aw", "wateractivity"}


class IngestionError(ValueError):
    """Raised for files that violate the data contract in a way that cannot be repaired."""


def _validate_workbook_size(source: Source) -> None:
    """Reject compressed Excel files whose expanded archive is unexpectedly large."""
    if isinstance(source, (str, os.PathLike)):
        archive_source = source
    elif hasattr(source, "getvalue"):
        archive_source = io.BytesIO(source.getvalue())
    elif hasattr(source, "seek") and hasattr(source, "tell"):
        position = source.tell()
        source.seek(0)
        archive_source = io.BytesIO(source.read())
        source.seek(position)
    else:
        raise IngestionError("Excel input must be seekable.")
    with zipfile.ZipFile(archive_source) as archive:
        expanded_bytes = sum(item.file_size for item in archive.infolist())
    if expanded_bytes > MAX_EXPANDED_WORKBOOK_BYTES:
        raise IngestionError("Excel input expands beyond the 100 MB safety limit.")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _norm(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _source_name(source: Source, filename: Optional[str]) -> str:
    if filename:
        return str(filename)
    if isinstance(source, (str, os.PathLike)):
        return str(source)
    return str(getattr(source, "name", "") or "")


def _inspect_upload_source(source: Source, name: str) -> None:
    """Content-sniff and scan an uploaded file (src/upload_security.py) before any parser touches it."""
    from src import upload_security
    from src.observability import METRICS

    data = upload_security.source_bytes(source)
    if data is None:
        return
    try:
        upload_security.inspect_upload(name, data)
    except upload_security.UploadRejected as exc:
        METRICS.inc("microbesim_uploads_total", outcome="rejected")
        raise IngestionError(str(exc)) from exc
    METRICS.inc("microbesim_uploads_total", outcome="accepted")


def read_table(source: Source, filename: Optional[str] = None) -> pd.DataFrame:
    """Read CSV / TSV / Excel from a path, file-like object or DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        name = _source_name(source, filename)
        if isinstance(source, (str, os.PathLike)):
            try:
                size = os.path.getsize(source)
            except OSError as exc:
                raise IngestionError(f"could not access '{name or 'input'}': {exc}") from exc
            if size > MAX_UPLOAD_BYTES:
                raise IngestionError("input exceeds the 25 MB file-size limit")
        elif hasattr(source, "getvalue") and len(source.getvalue()) > MAX_UPLOAD_BYTES:
            raise IngestionError("input exceeds the 25 MB file-size limit")
        _inspect_upload_source(source, name)
        ext = os.path.splitext(str(name))[1].lower()
        try:
            if ext in (".xlsx", ".xlsm", ".xls"):
                if ext in (".xlsx", ".xlsm"):
                    _validate_workbook_size(source)
                df = pd.read_excel(source, nrows=MAX_TABLE_ROWS + 1)
            elif ext == ".tsv":
                df = pd.read_csv(source, sep="\t", nrows=MAX_TABLE_ROWS + 1)
            else:
                df = pd.read_csv(source, sep=None, engine="python", nrows=MAX_TABLE_ROWS + 1)
        except Exception as exc:  # pandas raises many types for malformed input
            raise IngestionError(f"could not parse '{name or 'input'}': {exc}") from exc
    if len(df) > MAX_TABLE_ROWS:
        raise IngestionError(f"input exceeds the {MAX_TABLE_ROWS:,}-row limit")
    if len(df.columns) > MAX_TABLE_COLUMNS:
        raise IngestionError(f"input exceeds the {MAX_TABLE_COLUMNS}-column limit")
    return df


def _rename(df: pd.DataFrame, mapping: Dict[str, set]) -> pd.DataFrame:
    lookup, used = {}, set()
    for target, aliases in mapping.items():
        for col in df.columns:
            if col in used:
                continue
            if _norm(col) in aliases:
                lookup[col] = target
                used.add(col)
                break
    return df.rename(columns=lookup)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
@dataclass
class CleanedCurve:
    """A cleaned kinetic curve; ``raw_*`` keep the original observations for plotting."""

    time: np.ndarray
    raw: np.ndarray  # measurement after gap filling, before outlier removal/smoothing
    clean: np.ndarray  # after Hampel + Savitzky-Golay
    interpolated: np.ndarray  # bool mask: point was inserted / filled by interpolation
    outliers: np.ndarray  # bool mask: point replaced by the Hampel filter
    notes: List[str] = field(default_factory=list)


def _fill_gaps(
    t: np.ndarray, m: np.ndarray, gap_factor: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert points inside time gaps larger than ``gap_factor`` x the median step."""
    flag = np.zeros(len(t), dtype=bool)
    if len(t) < 3:
        return t, m, flag
    dt = np.diff(t)
    step = float(np.median(dt))
    if step <= 0:
        return t, m, flag
    new_t, new_m, new_f = [t[0]], [m[0]], [False]
    for i, gap in enumerate(dt):
        if gap > gap_factor * step:
            k = int(round(gap / step)) - 1
            if len(new_t) + k + (len(t) - i - 1) > MAX_CLEANED_POINTS_PER_CURVE:
                raise IngestionError(
                    "gap filling would exceed the 200,000-point limit for one curve"
                )
            for j in range(1, k + 1):
                tj = t[i] + gap * j / (k + 1)
                new_t.append(tj)
                new_m.append(float(np.interp(tj, [t[i], t[i + 1]], [m[i], m[i + 1]])))
                new_f.append(True)
        new_t.append(t[i + 1])
        new_m.append(m[i + 1])
        new_f.append(False)
    return np.array(new_t), np.array(new_m), np.array(new_f)


def hampel(x: np.ndarray, window: int = 7, n_sigmas: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Replace points deviating from the local median by > ``n_sigmas`` scaled MADs."""
    n = len(x)
    if n < 5:
        return x.copy(), np.zeros(n, dtype=bool)
    window = max(3, window | 1)
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, window)
    med = np.median(win, axis=1)
    mad = 1.4826 * np.median(np.abs(win - med[:, None]), axis=1)
    floor = max(1e-9, 0.02 * float(np.median(mad[mad > 0])) if np.any(mad > 0) else 1e-9)
    is_out = np.abs(x - med) > n_sigmas * np.maximum(mad, floor)
    out = x.copy()
    out[is_out] = med[is_out]
    return out, is_out


def clean_curve(
    time,
    measurement,
    *,
    fill_gaps: bool = True,
    gap_factor: float = 2.5,
    hampel_window: int = 7,
    hampel_sigmas: float = 3.0,
    smooth: bool = True,
    sg_window: int = 5,
    sg_polyorder: int = 2,
) -> CleanedCurve:
    """Sort, de-duplicate, interpolate, de-spike and smooth a single curve."""
    t = np.asarray(time, dtype=float)
    m = np.asarray(measurement, dtype=float)
    notes: List[str] = []

    ok = np.isfinite(t)
    if (~ok).any():
        notes.append(f"dropped {int((~ok).sum())} row(s) with missing time")
    t, m = t[ok], m[ok]

    valid = np.isfinite(m)
    if (~valid).any() and valid.sum() >= 2:  # interpolate interior NaNs, trim edge NaNs
        first, last = np.flatnonzero(valid)[[0, -1]]
        t, m, valid = t[first : last + 1], m[first : last + 1], valid[first : last + 1]
        n_nan = int((~valid).sum())
        if n_nan:
            m = np.interp(t, t[valid], m[valid])
            notes.append(f"linearly interpolated {n_nan} missing measurement(s)")
    else:
        t, m = t[valid], m[valid]

    if len(t) == 0:
        raise IngestionError("curve has no valid points")
    order = np.argsort(t, kind="stable")
    t, m = t[order], m[order]
    if len(np.unique(t)) != len(t):
        frame = pd.DataFrame({"t": t, "m": m}).groupby("t", as_index=False).mean()
        notes.append(f"averaged {len(t) - len(frame)} duplicate time-stamp(s)")
        t, m = frame["t"].to_numpy(), frame["m"].to_numpy()

    flag = np.zeros(len(t), dtype=bool)
    if fill_gaps:
        t2, m2, flag = _fill_gaps(t, m, gap_factor)
        if flag.any():
            notes.append(f"inserted {int(flag.sum())} interpolated point(s) in time gaps")
        t, m = t2, m2

    despiked, outliers = hampel(m, hampel_window, hampel_sigmas)
    if outliers.any():
        notes.append(f"replaced {int(outliers.sum())} outlier(s) (Hampel filter)")

    clean = despiked
    if smooth and len(clean) >= 5:
        w = min(sg_window | 1, len(clean) if len(clean) % 2 else len(clean) - 1)
        if w > sg_polyorder:
            clean = savgol_filter(despiked, w, sg_polyorder, mode="interp")
    clean = np.maximum(clean, 0.0)
    return CleanedCurve(t, m, clean, flag, outliers, notes)


def to_ln_ratio(clean: np.ndarray, measurement_type: str, baseline_points: int = 3) -> np.ndarray:
    """Convert a cleaned response to ``y = ln(N / N0)``.

    * ``log10_cfu``: ``y = ln(10) * (log10N - log10N0)``
    * ``od600``:     ``y = ln(OD / OD0)``  (OD floored at 1e-3 to avoid log(0))

    ``N0`` / ``OD0`` is the lowest of the first ``baseline_points`` readings.
    """
    if measurement_type not in MEASUREMENT_TYPES:
        raise ValueError(f"measurement_type must be one of {MEASUREMENT_TYPES}")
    clean = np.asarray(clean, dtype=float)
    k = max(1, min(baseline_points, len(clean)))
    if measurement_type == "log10_cfu":
        return np.log(10.0) * (clean - clean[:k].min())
    od = np.maximum(clean, 1e-3)
    return np.log(od / od[:k].min())


# --------------------------------------------------------------------------- #
# Kinetic series loading
# --------------------------------------------------------------------------- #
@dataclass
class IngestionReport:
    warnings: List[str] = field(default_factory=list)
    cleaned: Dict[str, CleanedCurve] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)


def load_kinetic_series(
    source: Source,
    *,
    filename: Optional[str] = None,
    default_organism: str = "Unspecified organism",
    default_temperature_c: float = 37.0,
    default_ph: float = 7.0,
    default_aw: float = 0.99,
    override_metadata: bool = False,
    **clean_kwargs,
) -> Tuple[List[KineticSeries], IngestionReport]:
    """Parse an assay file (one or many curves) into validated :class:`KineticSeries`.

    A ``series_id`` (or ``well`` / ``curve`` ...) column splits the file into curves;
    otherwise the whole file is one curve.  Optional columns ``organism``,
    ``temperature_celsius``, ``ph`` and ``water_activity`` override the defaults per curve,
    unless ``override_metadata`` is true.
    Values are returned *cleaned* (gap-filled, de-spiked, smoothed); the original
    observations stay available in ``IngestionReport.cleaned`` for plotting.
    """
    df = read_table(source, filename)
    df = _rename(
        df,
        {"time_hours": _TIME_ALIASES, "measurement": _MEAS_ALIASES, "series_id": _SERIES_ALIASES},
    )
    df = _rename(df, _META_ALIASES)
    missing = [c for c in ("time_hours", "measurement") if c not in df.columns]
    if missing:
        raise IngestionError(
            f"missing required column(s) {missing}; found {list(map(str, df.columns))}. "
            "Expected 'time_hours' and 'measurement' (aliases: Time, OD600, LogCFU ...)"
        )
    for col in ("time_hours", "measurement"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report = IngestionReport()
    if "series_id" not in df.columns:
        base = os.path.splitext(os.path.basename(_source_name(source, filename)))[0]
        df["series_id"] = base or "series_1"
    df["series_id"] = df["series_id"].astype(str)
    if df["series_id"].nunique() > MAX_CURVES_PER_FILE:
        raise IngestionError(f"input exceeds the {MAX_CURVES_PER_FILE}-curve limit")

    series: List[KineticSeries] = []
    for sid, grp in df.groupby("series_id", sort=False):
        neg = grp["measurement"] < 0
        if neg.any():
            report.warnings.append(f"{sid}: dropped {int(neg.sum())} negative measurement row(s)")
            grp = grp[~neg]
        if (grp["time_hours"] < 0).any():
            report.warnings.append(
                f"{sid}: dropped {int((grp['time_hours'] < 0).sum())} row(s) with negative time"
            )
            grp = grp[grp["time_hours"] >= 0]
        try:
            curve = clean_curve(grp["time_hours"], grp["measurement"], **clean_kwargs)
        except IngestionError as exc:
            report.skipped[sid] = str(exc)
            continue
        report.warnings.extend(f"{sid}: {n}" for n in curve.notes)
        if len(curve.time) < 6:
            report.skipped[sid] = f"only {len(curve.time)} usable points (need >= 6)"
            continue

        def meta(col: str, default):
            if override_metadata:
                return default
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals):
                    return vals.iloc[0]
            return default

        try:
            series.append(
                KineticSeries(
                    series_id=sid,
                    organism_name=str(meta("organism_name", default_organism)),
                    temperature_celsius=float(meta("temperature_celsius", default_temperature_c)),
                    ph=float(meta("ph", default_ph)),
                    water_activity=float(meta("water_activity", default_aw)),
                    points=[
                        RawCurvePoint(time_hours=float(a), measurement=float(b))
                        for a, b in zip(curve.time, curve.clean)
                    ],
                )
            )
        except ValueError as exc:  # pydantic.ValidationError is a ValueError
            report.skipped[sid] = f"schema validation failed: {exc}"
            continue
        report.cleaned[sid] = curve

    if not series:
        detail = "; ".join(f"{k}: {v}" for k, v in report.skipped.items()) or "no rows"
        raise IngestionError(f"no valid kinetic series found ({detail})")
    return series, report


# --------------------------------------------------------------------------- #
# Cold-chain telemetry
# --------------------------------------------------------------------------- #
def load_cold_chain_telemetry(
    source: Source, *, filename: Optional[str] = None
) -> Tuple[List[DynamicTelemetryPoint], List[str]]:
    """Parse a cold-chain log (``timestamp_hours``, ``temperature_celsius``).

    Temperatures must lie within the sensor range (-20 to 50 C); violations are an error
    (a silently clipped log would invalidate a spoilage verdict).  Returns the points and a
    list of non-fatal warnings.
    """
    df = read_table(source, filename)
    df = _rename(
        df,
        {
            "timestamp_hours": _TELEM_TIME,
            "temperature_celsius": _TELEM_TEMP,
            "ph": _TELEM_PH,
            "water_activity": _TELEM_AW,
        },
    )
    missing = [c for c in ("timestamp_hours", "temperature_celsius") if c not in df.columns]
    if missing:
        raise IngestionError(
            f"missing required column(s) {missing}; expected 'timestamp_hours' and 'temperature_celsius'"
        )
    warnings: List[str] = []
    for col in ("timestamp_hours", "temperature_celsius"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("ph", "water_activity"):
        if col in df:
            raw_values = df[col].replace(r"^\s*$", np.nan, regex=True)
            numeric_values = pd.to_numeric(raw_values, errors="coerce")
            invalid = raw_values.notna() & numeric_values.isna()
            if invalid.any():
                raise IngestionError(f"{col} contains non-numeric values")
            df[col] = numeric_values
    bad = df[["timestamp_hours", "temperature_celsius"]].isna().any(axis=1)
    if bad.any():
        warnings.append(f"dropped {int(bad.sum())} row(s) with missing/non-numeric values")
        df = df[~bad]
    if (df["timestamp_hours"] < 0).any():
        raise IngestionError("timestamp_hours must be >= 0")
    out_of_range = (df["temperature_celsius"] < SENSOR_MIN_C) | (
        df["temperature_celsius"] > SENSOR_MAX_C
    )
    if out_of_range.any():
        rows = (df.index[out_of_range][:5] + 2).tolist()
        raise IngestionError(
            f"{int(out_of_range.sum())} temperature reading(s) outside {SENSOR_MIN_C:g} to "
            f"{SENSOR_MAX_C:g} C (first at file row(s) {rows})"
        )
    df = df.sort_values("timestamp_hours", kind="stable")
    dup = df["timestamp_hours"].duplicated(keep="last")
    if dup.any():
        warnings.append(f"kept the last reading for {int(dup.sum())} duplicate time-stamp(s)")
        df = df[~dup]
    if len(df) < 2:
        raise IngestionError("telemetry needs at least two readings")
    profile_fields = {
        column: column in df and bool(df[column].notna().any())
        for column in ("ph", "water_activity")
    }
    for column, limits in (("ph", (0.0, 14.0)), ("water_activity", (0.0, 1.0))):
        if not profile_fields[column]:
            continue
        missing_count = int(df[column].isna().sum())
        if missing_count:
            warnings.append(
                f"interpolated {missing_count} missing {column} value(s) in the environmental profile"
            )
            indexed = pd.Series(
                df[column].to_numpy(), index=df["timestamp_hours"].to_numpy()
            ).interpolate(method="index", limit_direction="both")
            df[column] = indexed.to_numpy()
        lower_invalid = (
            df[column] <= limits[0] if column == "water_activity" else df[column] < limits[0]
        )
        if lower_invalid.any() or (df[column] > limits[1]).any():
            raise IngestionError(f"{column} values must be between {limits[0]:g} and {limits[1]:g}")
    if float(df["timestamp_hours"].iloc[0]) > 0:
        warnings.append(
            f"log starts at {df['timestamp_hours'].iloc[0]:g} h; the first reading is held back to t = 0"
        )
    points = []
    for _, row in df.iterrows():
        points.append(
            DynamicTelemetryPoint(
                timestamp_hours=float(row["timestamp_hours"]),
                temperature_celsius=float(row["temperature_celsius"]),
                ph=float(row["ph"]) if profile_fields["ph"] else None,
                water_activity=(
                    float(row["water_activity"]) if profile_fields["water_activity"] else None
                ),
            )
        )
    return points, warnings
