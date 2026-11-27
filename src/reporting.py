"""Reporting & export (FR-6): CSV tables and audit-ready PDF reports.

Every PDF carries a provenance block (UTC timestamp, software version, SHA-256
fingerprint of the input parameters) so that a result can be traced back to exactly
what produced it.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import __version__  # noqa: E402
from src.models.primary import PrimaryFit  # noqa: E402
from src.simulation.dynamic import SimulationTrace  # noqa: E402

# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return _spreadsheet_safe_frame(df).to_csv(index=False).encode("utf-8")


def _spreadsheet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent untrusted text from being interpreted as formulas by spreadsheet apps."""
    safe = frame.copy()

    def safe_text(value):
        if not isinstance(value, str):
            return value
        probe = value.lstrip(" \t\r\n\v\f\x00")
        if probe.startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    safe.columns = [safe_text(column) for column in safe.columns]
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column].dtype) or pd.api.types.is_string_dtype(
            safe[column].dtype
        ):
            safe[column] = safe[column].map(safe_text)
    return safe


def frames_to_xlsx_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Export named tables into a multi-sheet Excel workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            _spreadsheet_safe_frame(frame).to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def fit_results_frame(entries: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Tabulate fitted primary parameters.  Each entry: ``{'series': KineticSeries, 'fit': PrimaryFit}``."""
    rows = []
    for e in entries:
        s, f = e["series"], e["fit"]
        rows.append(
            {
                "series_id": s.series_id,
                "organism": s.organism_name,
                "temperature_celsius": s.temperature_celsius,
                "ph": s.ph,
                "water_activity": s.water_activity,
                "model": f.model,
                "mu_max_per_h": f.mu_max,
                "doubling_time_h": f.doubling_time_h,
                "lag_time_h": f.lam,
                "y_max_ln": f.A,
                "y_max_standard_error": f.param_std[0],
                "mu_max_standard_error": f.param_std[1],
                "lag_time_standard_error": f.param_std[2],
                "r_squared": f.r_squared,
                "rmse": f.rmse,
                "n_points": f.n_points,
                "optimizer": f.method,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def fingerprint(payload: Dict[str, Any]) -> str:
    """SHA-256 of a canonical JSON rendering of the inputs (first 16 hex chars)."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Stable SHA-256 prefix of a tabular dataset's exported CSV representation."""
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


def _provenance(payload: Dict[str, Any]) -> List[List[str]]:
    return [
        ["Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")],
        ["MicrobeSim version", __version__],
        ["Input fingerprint (SHA-256)", fingerprint(payload)],
    ]


# --------------------------------------------------------------------------- #
# Figures (matplotlib -> PNG bytes)
# --------------------------------------------------------------------------- #
def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fit_figure(
    t: np.ndarray, y_raw: np.ndarray, y_clean: np.ndarray, fit: PrimaryFit, title: str
) -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter(t, y_raw, s=14, color="#888888", label="observed (gap-filled)")
    ax.plot(t, y_clean, color="#1f77b4", lw=1, alpha=0.8, label="cleaned")
    tt = np.linspace(float(t.min()), float(t.max()), 300)
    ax.plot(tt, fit.predict(tt), color="#d62728", lw=2, label=f"{fit.model} fit")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("ln(N/N0)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _png(fig)


def simulation_figure(trace: SimulationTrace, title: str) -> bytes:
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.plot(trace.time_hours, trace.log10_cfu, color="#d62728", lw=2, label="log10 CFU/g")
    ax1.axhline(np.log10(trace.n_crit), color="black", ls="--", lw=1, label="N_crit")
    if trace.time_to_breach_hours is not None:
        ax1.axvline(trace.time_to_breach_hours, color="black", ls=":", lw=1)
    ax1.set_xlabel("time (h)")
    ax1.set_ylabel("log10 CFU/g")
    ax2 = ax1.twinx()
    ax2.plot(trace.time_hours, trace.temperature_celsius, color="#1f77b4", lw=1.2, alpha=0.7)
    ax2.set_ylabel("temperature (C)", color="#1f77b4")
    ax1.set_title(title, fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    return _png(fig)


def benchmark_figure(table: pd.DataFrame) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for ax, col in zip(axes, ["RMSE", "Af", "R2"]):
        ax.barh(table["Model"], table[col], color="#1f7a5a")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    return _png(fig)


# --------------------------------------------------------------------------- #
# PDF builder
# --------------------------------------------------------------------------- #
_styles = getSampleStyleSheet()
_body = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=9, leading=12)
_cell = ParagraphStyle("cell", parent=_body, fontSize=8, leading=10)


def _table(rows: Sequence[Sequence[Any]], header: bool = True, col_widths=None) -> Table:
    data = [[Paragraph(escape(str(c)), _cell) for c in r] for r in rows]
    tbl = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f2ec")))
    tbl.setStyle(TableStyle(style))
    return tbl


def _fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def frame_rows(df: pd.DataFrame) -> List[List[str]]:
    return [list(map(str, df.columns))] + [
        [_fmt(v) for v in row] for row in df.itertuples(index=False)
    ]


def build_pdf(
    title: str,
    sections: Sequence[Dict[str, Any]],
    provenance_payload: Dict[str, Any],
    subtitle: str = "",
) -> bytes:
    """Render sections ``{'heading', 'paragraphs', 'table', 'figure'}`` into a PDF (bytes)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author=f"MicrobeSim {__version__}",
    )
    story: List[Any] = [Paragraph(escape(title), _styles["Title"])]
    if subtitle:
        story.append(Paragraph(escape(subtitle), _body))
    story += [
        Spacer(1, 4 * mm),
        _table(_provenance(provenance_payload), header=False, col_widths=[55 * mm, 110 * mm]),
    ]

    for sec in sections:
        items: List[Any] = [Paragraph(escape(sec["heading"]), _styles["Heading2"])]
        for para in sec.get("paragraphs", []):
            items.append(Paragraph(para if sec.get("raw_markup") else escape(para), _body))
            items.append(Spacer(1, 1.5 * mm))
        if sec.get("table") is not None:
            items.append(_table(sec["table"]))
        if sec.get("figure"):
            img = Image(io.BytesIO(sec["figure"]))
            scale = min(170 * mm / img.imageWidth, 1.0)
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            items.append(img)
        story.append(Spacer(1, 5 * mm))
        # keep a heading together with its figure; long tables are allowed to split
        if sec.get("figure") and sec.get("table") is None:
            story.append(KeepTogether(items))
        else:
            story.extend(items)
    doc.build(story)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Concrete reports
# --------------------------------------------------------------------------- #
def build_fit_report(
    entries: List[Dict[str, Any]],
    measurement_type: str,
    max_figures: int = 6,
    audit_metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """Primary-fit report. Entry keys: series, fit, curve (CleanedCurve), y_raw, y_clean."""
    df = fit_results_frame(entries)
    show = df[["series_id", "model", "mu_max_per_h", "lag_time_h", "y_max_ln", "r_squared", "rmse"]]
    sections: List[Dict[str, Any]] = [
        {
            "heading": "Reproducibility",
            "table": [["Field", "Value"]]
            + [[str(key), _fmt(value)] for key, value in (audit_metadata or {}).items()],
            "paragraphs": [
                "Primary fit outputs are screening estimates based on the listed assay data "
                "and cleaning settings; inspect the data-quality notes before interpretation."
            ],
        },
        {
            "heading": "Method",
            "paragraphs": [
                "Curves were gap-filled by linear interpolation, de-spiked with a Hampel filter and "
                "smoothed with a Savitzky-Golay filter. The response was converted to ln(N/N0) "
                f"(input type: {measurement_type}) and fitted by non-linear least squares "
                "(Levenberg-Marquardt, derivative-seeded; bounded trust-region fallback).",
            ],
        },
        {"heading": f"Fitted kinetic parameters ({len(df)} curve(s))", "table": frame_rows(show)},
    ]
    for e in entries[:max_figures]:
        s, f = e["series"], e["fit"]
        sections.append(
            {
                "heading": f"Curve {s.series_id}",
                "paragraphs": [
                    f"{s.organism_name}, {s.temperature_celsius:g} C, pH {s.ph:g}, aw {s.water_activity:g}. "
                    f"mu_max = {f.mu_max:.4g} 1/h, lag = {f.lam:.4g} h, y_max = {f.A:.4g}, R2 = {f.r_squared:.4f}."
                ],
                "figure": fit_figure(
                    e["curve"].time, e["y_raw"], e["y_clean"], f, f"{s.series_id} ({f.model})"
                ),
            }
        )
    if len(entries) > max_figures:
        sections.append(
            {
                "heading": "Note",
                "paragraphs": [
                    f"Figures shown for the first {max_figures} of {len(entries)} curves; see CSV export for all."
                ],
            }
        )
    payload = {
        "kind": "fit",
        "measurement_type": measurement_type,
        "results": df.round(8).to_dict("records"),
    }
    payload["audit_metadata"] = audit_metadata or {}
    return build_pdf("MicrobeSim - Primary Growth Curve Fit Report", sections, payload)


def build_benchmark_report(
    table: pd.DataFrame,
    dataset_label: str,
    n_rows: int,
    validation_description: str = "Repeated random holdouts",
    audit_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    show = table.drop(columns=[c for c in ("key",) if c in table]).copy()
    sections = [
        {
            "heading": "Setup",
            "paragraphs": [
                f"Dataset: {dataset_label} ({n_rows} rows). Validation: {validation_description}. "
                "Bf > 1 indicates over-prediction (fail-safe); Af = 1 is a perfect model. "
                "Reported scores are means across validation splits; SD columns show split-to-split variation."
            ],
        },
        {
            "heading": "Reproducibility",
            "table": [["Field", "Value"]]
            + [[str(key), _fmt(value)] for key, value in (audit_metadata or {}).items()],
        },
        {"heading": "Scorecard", "table": frame_rows(show), "figure": benchmark_figure(table)},
    ]
    return build_pdf(
        "MicrobeSim - Secondary Model Benchmark Report",
        sections,
        {
            "kind": "benchmark",
            "dataset": dataset_label,
            "validation": validation_description,
            "audit_metadata": audit_metadata or {},
            "table": show.round(8).to_dict("records"),
        },
    )


def format_duration(hours: Optional[float]) -> str:
    if hours is None:
        return "n/a"
    d, h = divmod(hours, 24.0)
    return f"{hours:.2f} h ({int(d)} d {h:.1f} h)" if d >= 1 else f"{hours:.2f} h"


def build_simulation_report(
    organism: str,
    model_name: str,
    trace: SimulationTrace,
    settings: Dict[str, Any],
    audit_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    breach = (
        f"BREACH: N reaches the critical threshold at t_crit = {format_duration(trace.time_to_breach_hours)}."
        if trace.spoilage_breached
        else "No breach: the population stays below the critical threshold for the whole simulated period."
    )
    inputs = [["Parameter", "Value"]] + [[k, _fmt(v)] for k, v in settings.items()]
    summary = [
        ["Metric", "Value"],
        ["Initial load (CFU/g)", _fmt(trace.n0)],
        ["Critical threshold (CFU/g)", _fmt(trace.n_crit)],
        ["Final load (CFU/g)", _fmt(float(trace.cfu_per_g[-1]))],
        ["Population doublings", _fmt(float(trace.doublings[-1]))],
        ["Peak temperature (C)", _fmt(float(trace.temperature_celsius.max()))],
        ["t_crit", format_duration(trace.time_to_breach_hours)],
    ]
    sections = [
        {"heading": "Verdict", "paragraphs": [breach]},
        {
            "heading": "Reproducibility and interpretation",
            "table": [["Field", "Value"]]
            + [[str(key), _fmt(value)] for key, value in (audit_metadata or {}).items()],
            "paragraphs": [
                "This report is a screening estimate. It is not a validated food-safety "
                "decision, regulatory determination, or substitute for product-specific review."
            ],
        },
        {"heading": "Simulation settings", "table": inputs},
        {"heading": "Summary", "table": summary},
        {"heading": "Trajectory", "figure": simulation_figure(trace, f"{organism} - {model_name}")},
    ]
    payload = {
        "kind": "simulation",
        "organism": organism,
        "model": model_name,
        "settings": settings,
        "t_crit": trace.time_to_breach_hours,
        "final": float(trace.cfu_per_g[-1]),
        "audit_metadata": audit_metadata or {},
    }
    return build_pdf(
        "MicrobeSim - Cold-Chain Spoilage Report",
        sections,
        payload,
        subtitle=f"{organism} | model: {model_name}",
    )
