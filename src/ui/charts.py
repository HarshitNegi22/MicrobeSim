"""Plotly figure builders (kept free of Streamlit so they are easy to test)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.primary import PrimaryFit
from src.simulation.dynamic import SimulationTrace

INK, LINE, MUTED = "#EEF3FC", "#293750", "#7187A8"
GREEN, BLUE, CORAL, AMBER = "#00D6C2", "#55C8FF", "#FF63C7", "#FFC400"
PALETTE = ["#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def apply_theme(fig: go.Figure) -> go.Figure:
    """One consistent look for every chart: quiet grid, legend on top, left-aligned title."""
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=64 if has_title else 44, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(
            bgcolor="#101A2C", font=dict(family=FONT, size=13, color=INK), bordercolor=LINE
        ),
    )
    if has_title:
        fig.update_layout(title=dict(x=0, xanchor="left", font=dict(size=15, color=INK)))
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE
    )
    fig.update_yaxes(showgrid=True, gridcolor="#1B2638", zeroline=False, linecolor=LINE)
    fig.update_yaxes(showgrid=False, selector=dict(overlaying="y"))  # no second grid on twin axes
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def fit_chart(
    curve, y_raw: np.ndarray, y_clean: np.ndarray, fit: PrimaryFit, title: str
) -> go.Figure:
    """Observed points vs. cleaned data vs. fitted primary model (ln N/N0)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve.time,
            y=y_raw,
            mode="markers",
            name="observed",
            marker=dict(color="#91A4C0", size=6),
        )
    )
    if curve.outliers.any():
        fig.add_trace(
            go.Scatter(
                x=curve.time[curve.outliers],
                y=y_raw[curve.outliers],
                mode="markers",
                name="outlier (replaced)",
                marker=dict(color=AMBER, size=10, symbol="x"),
            )
        )
    if curve.interpolated.any():
        fig.add_trace(
            go.Scatter(
                x=curve.time[curve.interpolated],
                y=y_raw[curve.interpolated],
                mode="markers",
                name="interpolated",
                marker=dict(color=BLUE, size=7, symbol="circle-open"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=curve.time,
            y=y_clean,
            mode="lines",
            name="cleaned",
            line=dict(color=BLUE, width=1, dash="dot"),
        )
    )
    tt = np.linspace(float(curve.time.min()), float(curve.time.max()), 400)
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=fit.predict(tt),
            mode="lines",
            name=f"{fit.model} fit",
            line=dict(color=CORAL, width=3),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="time (hours)",
        yaxis_title="growth, ln(N/N0)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fit_comparison_chart(entries: list, selected_ids: list[str]) -> go.Figure:
    """Compare normalized observed and fitted curves on a shared time axis."""
    fig = go.Figure()
    for index, entry in enumerate(entries):
        series = entry["series"]
        if series.series_id not in selected_ids:
            continue
        color = PALETTE[index % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=entry["curve"].time,
                y=entry["y_raw"],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.45),
                name=f"{series.series_id} observed",
                legendgroup=series.series_id,
            )
        )
        time_grid = np.linspace(
            float(entry["curve"].time.min()), float(entry["curve"].time.max()), 250
        )
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=entry["fit"].predict(time_grid),
                mode="lines",
                line=dict(width=2.5, color=color),
                name=f"{series.series_id} fit",
                legendgroup=series.series_id,
            )
        )
    fig.update_layout(
        title="Selected assay curves",
        xaxis_title="time (hours)",
        yaxis_title="normalized growth, ln(N/N0)",
    )
    return fig


def mu_ribbon_chart(
    x: np.ndarray,
    curves: Dict[str, Dict[str, np.ndarray]],
    x_label: str,
    training: Optional[pd.DataFrame] = None,
    x_col: Optional[str] = None,
) -> go.Figure:
    """mu_max vs. a swept variable with nominal 95 % ribbons, one colour per model.

    ``curves[name] = {'mean', 'lo', 'hi'}``; optional training points overlaid.
    """
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([c["hi"], c["lo"][::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.15),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} nominal 95% model range",
            )
        )
        fig.add_trace(
            go.Scatter(x=x, y=c["mean"], mode="lines", name=name, line=dict(color=color, width=2.5))
        )
    if training is not None and x_col is not None and len(training):
        fig.add_trace(
            go.Scatter(
                x=training[x_col],
                y=training["mu_max_per_h"],
                mode="markers",
                name="training data (nearby conditions)",
                marker=dict(color="#91A4C0", size=5, opacity=0.5),
            )
        )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="growth rate μmax (1/h)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def benchmark_bars(table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=["RMSE (lower is better)", "Af (1 is perfect)", "Bf (1 is unbiased)", "R²"],
    )
    for j, col in enumerate(["RMSE", "Af", "Bf", "R2"], start=1):
        fig.add_trace(
            go.Bar(
                x=table["Model"], y=table[col], marker_color=PALETTE[: len(table)], showlegend=False
            ),
            row=1,
            col=j,
        )
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def parity_chart(predictions: pd.DataFrame, labels: Dict[str, str]) -> go.Figure:
    fig = go.Figure()
    obs = predictions["observed_mu_max"]
    hi = float(max(obs.max(), predictions.drop(columns="observed_mu_max").max().max())) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, hi], y=[0, hi], mode="lines", line=dict(color="#7187A8", dash="dash"), name="1:1"
        )
    )
    for i, col in enumerate([c for c in predictions.columns if c != "observed_mu_max"]):
        fig.add_trace(
            go.Scatter(
                x=obs,
                y=predictions[col],
                mode="markers",
                name=labels.get(col, col),
                marker=dict(color=PALETTE[i % len(PALETTE)], size=5, opacity=0.6),
            )
        )
    fig.update_layout(
        xaxis_title="observed μmax (1/h)",
        yaxis_title="predicted μmax (1/h)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def simulation_chart(
    trace: SimulationTrace,
    lower_trace: Optional[SimulationTrace] = None,
    upper_trace: Optional[SimulationTrace] = None,
    comparison_trace: Optional[SimulationTrace] = None,
    comparison_label: str = "Comparison scenario",
    focus_hour: Optional[float] = None,
    event_hours: Optional[list[tuple[float, str]]] = None,
) -> go.Figure:
    """Dual-axis chart: log10 CFU/g (left) and temperature (right)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if lower_trace is not None and upper_trace is not None:
        fig.add_trace(
            go.Scatter(
                x=upper_trace.time_hours,
                y=upper_trace.log10_cfu,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=lower_trace.time_hours,
                y=lower_trace.log10_cfu,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_rgba(CORAL, 0.14),
                name="pointwise model interval envelope",
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=trace.time_hours,
            y=trace.log10_cfu,
            name="log10 CFU/g",
            line=dict(color=CORAL, width=3),
        ),
        secondary_y=False,
    )
    if comparison_trace is not None:
        fig.add_trace(
            go.Scatter(
                x=comparison_trace.time_hours,
                y=comparison_trace.log10_cfu,
                name=comparison_label,
                line=dict(color=GREEN, width=2, dash="dash"),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=trace.time_hours,
            y=trace.temperature_celsius,
            name="temperature (°C)",
            line=dict(color=BLUE, width=1.5, shape="linear"),
            opacity=0.75,
        ),
        secondary_y=True,
    )
    t0, t1 = float(trace.time_hours[0]), float(trace.time_hours[-1])
    y_crit = float(np.log10(trace.n_crit))
    fig.add_trace(
        go.Scatter(
            x=[t0, t1],
            y=[y_crit, y_crit],
            mode="lines",
            name="N_crit",
            line=dict(color="#FFC400", dash="dash", width=1.5),
        ),
        secondary_y=False,
    )
    if trace.time_to_breach_hours is not None:
        y_lo = float(min(trace.log10_cfu.min(), y_crit))
        y_hi = float(max(trace.log10_cfu.max(), y_crit))
        fig.add_trace(
            go.Scatter(
                x=[trace.time_to_breach_hours] * 2,
                y=[y_lo, y_hi],
                mode="lines",
                name=f"t_crit = {trace.time_to_breach_hours:.1f} h",
                line=dict(color="#FFC400", dash="dot", width=1.5),
            ),
            secondary_y=False,
        )
    if focus_hour is not None:
        fig.add_vline(x=focus_hour, line_color="#FFFFFF", line_dash="dot", opacity=0.65)
    for event_hour, event_label in event_hours or []:
        fig.add_vline(
            x=event_hour,
            line_color=AMBER,
            line_dash="dash",
            opacity=0.45,
            annotation_text=event_label,
            annotation_position="top",
        )
    fig.update_xaxes(title_text="time (hours)")
    fig.update_yaxes(title_text="log10 CFU/g", secondary_y=False)
    fig.update_yaxes(title_text="temperature (°C)", secondary_y=True)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig
