"""Shared Streamlit helpers: paths, cached model training, dataset selector, plot wrapper."""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.datasets import (
    load_growth_dataset,
    load_real_growth_dataset,
    load_synthetic_growth_dataset,
    published_plus_organisms,
    real_organisms,
    suspect_record_count,
    synthetic_organisms,
)
from src import guardrails, registry
from src.model_cache import signature_for
from src.observability import METRICS, timed
from src.pretrained import grouping_arguments, grouping_options
from src.ui.charts import apply_theme
from src.ui.theme import provenance
from src.models.benchmark import MODEL_LABELS, BenchmarkResult, available_model_keys
from src.model_cache import load_or_train

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REAL_SUFFIX = " (installed real)"
SYNTHETIC_SUFFIX = " (synthetic)"
PUBLISHED_SUFFIX = " (local real + published)"

_PLOTLY_NEW_API = "width" in inspect.signature(st.plotly_chart).parameters


def show_plot(fig, key=None) -> None:
    """Apply the shared chart theme, then ``st.plotly_chart`` across Streamlit versions."""
    apply_theme(fig)
    if _PLOTLY_NEW_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


# --------------------------------------------------------------------------- #
# Cached training
# --------------------------------------------------------------------------- #
@st.cache_resource(
    show_spinner="Loading saved models or training them on the selected data...",
    ttl=3600,
    max_entries=8,
)
def train_frame(
    df: pd.DataFrame, keys: Tuple[str, ...], group_column: Optional[str]
) -> Tuple[pd.DataFrame, BenchmarkResult]:
    return df, _benchmark_dataset(df, keys, group_column)


def _benchmark_dataset(
    df: pd.DataFrame, keys: Tuple[str, ...], group_column: Optional[str]
) -> BenchmarkResult:
    # Same helper the pretraining script uses, so saved models always match what the UI asks for.
    extra = grouping_arguments(df, group_column)
    with timed("model_load", rows=len(df), grouping=group_column or "random"):
        result = load_or_train(df, list(keys), **extra)
    METRICS.inc("microbesim_model_loads_total", source=getattr(result, "cache_source", "unknown"))
    try:  # identifies the artifact in the model registry; tagging must never break model loading
        result.signature = signature_for(df, list(keys), **extra)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return result


def dataset_selector(prefix: str) -> Tuple[str, pd.DataFrame, BenchmarkResult]:
    """Let the user choose an installed real, synthetic, or uploaded training dataset.

    Shows where the data came from right under the picker. Returns
    ``(label, dataframe, benchmark_result)``; stops the page on unrecoverable errors.
    """
    keys = tuple(available_model_keys())
    uploaded_bytes = st.session_state.get("uploaded_growth_csv")
    real_options = [f"{o}{REAL_SUFFIX}" for o in real_organisms()]
    published_options = [f"{o}{PUBLISHED_SUFFIX}" for o in published_plus_organisms()]
    synthetic_options = [f"{o}{SYNTHETIC_SUFFIX}" for o in synthetic_organisms()]
    options = (
        real_options
        + published_options
        + synthetic_options
        + (["Uploaded dataset"] if uploaded_bytes else [])
    )
    choice = st.selectbox(
        "Organism and data source",
        options,
        key=f"{prefix}_dataset",
        help="Installed real entries come from the local dataset bundle. "
        "Synthetic entries are generated from illustrative parameters for demonstration.",
    )
    if choice == "Uploaded dataset":
        kind = "user"
        try:
            if len(uploaded_bytes) > 25 * 1024 * 1024:
                raise ValueError("Dataset exceeds the 25 MB file-size limit.")
            df = load_growth_dataset(io.BytesIO(uploaded_bytes))
        except ValueError as exc:
            st.error(f"Could not use the uploaded dataset: {exc}")
            st.stop()
    elif choice.endswith(PUBLISHED_SUFFIX):
        kind = "real"
        df = load_real_growth_dataset(choice[: -len(PUBLISHED_SUFFIX)], include_external=True)
    elif choice.endswith(REAL_SUFFIX):
        kind = "real"
        df = load_real_growth_dataset(choice[: -len(REAL_SUFFIX)])
    else:
        kind = "synthetic"
        df = load_synthetic_growth_dataset(choice[: -len(SYNTHETIC_SUFFIX)])

    group_options, default_group = grouping_options(df)
    selected_group = st.selectbox(
        "Validation grouping",
        group_options,
        index=group_options.index(default_group),
        key=f"{prefix}_validation_group",
        format_func=lambda column: "Random row-level split" if column is None else str(column),
        help=(
            "Keep records from the same study, batch or other selected group together "
            "during validation. Choose a real experiment identifier when available."
        ),
    )
    df, bench = train_frame(df, keys, selected_group)

    t = df["temperature_celsius"]
    provenance(kind, len(df), (float(t.min()), float(t.max())), int(t.nunique()))
    if choice.endswith(PUBLISHED_SUFFIX):
        n_pub = int((~df["source"].astype(str).str.startswith("combase")).sum())
        st.caption(
            f"{n_pub} rows come from other published or contributed studies (see data/external/external_sources.csv). "
            "Models trained here are not validated outside the conditions of those studies; see docs/EXTERNAL_DATA.md."
        )
    if kind == "real":
        set_aside = suspect_record_count(choice.split(" (real,")[0])
        if set_aside:
            st.caption(
                f"{set_aside} records that contradict the rest of this organism's data were set aside "
                "before training (listed with reasons in data/suspect_records.csv)."
            )
    if getattr(bench, "cache_source", None) == "pretrained":
        st.caption("Loaded pretrained models and benchmark for this dataset (no training needed).")
    elif bench.loaded_from_disk:
        st.caption("Loaded the saved model training and benchmark for this dataset.")
    else:
        st.caption("Trained and saved these models for reuse with this dataset.")
    if bench.errors:
        st.warning(
            "Some models could not be trained: "
            + "; ".join(f"{MODEL_LABELS[k]}: {v}" for k, v in bench.errors.items())
        )
    if not bench.models:
        st.error("No model could be trained on this dataset.")
        st.stop()
    return choice, df, bench


def conditions_outside_training_data(
    frame: pd.DataFrame, conditions: Dict[str, Tuple[float, float]]
) -> List[str]:
    """Describe requested environmental values outside observed feature ranges."""
    messages = []
    for column, (minimum, maximum) in conditions.items():
        if column not in frame or frame[column].empty:
            continue
        observed_min, observed_max = float(frame[column].min()), float(frame[column].max())
        if minimum < observed_min or maximum > observed_max:
            messages.append(
                f"{column}: requested {minimum:g}–{maximum:g}; observed data "
                f"{observed_min:g}–{observed_max:g}"
            )
    return messages


def fmt_hours(hours) -> str:
    if hours is None:
        return "n/a"
    d, h = divmod(float(hours), 24.0)
    return f"{hours:.1f} h ({int(d)} d {h:.1f} h)" if d >= 1 else f"{hours:.1f} h"


def report_button(label: str, build, file_name: str, key: str, signature: str) -> None:
    """Build-then-download flow that survives Streamlit reruns.

    The rendered PDF is kept in session state together with a signature of the inputs
    that produced it, and the download button is only offered while the inputs still
    match - so a report can never be downloaded for settings that have since changed.
    """
    stored = st.session_state.get(key)
    if st.button(f"Build {label}", key=f"{key}_build"):
        with st.spinner("Rendering report..."):
            stored = (signature, build())
            st.session_state[key] = stored
    if stored and stored[0] == signature:
        st.download_button(
            f"Download {label}", stored[1], file_name, "application/pdf", key=f"{key}_download"
        )
    elif stored:
        st.caption(
            "Inputs changed since the last report. Build it again to download an up-to-date copy."
        )


def model_status(bench: BenchmarkResult) -> str:
    """Registry review status of the loaded models ('draft' when unknown or not yet registered)."""
    try:
        return registry.status_of(getattr(bench, "signature", "")) or "draft"
    except Exception:  # the registry must never stop a page from rendering
        return "draft"


def render_assessment(
    df: pd.DataFrame,
    bench: BenchmarkResult,
    best_key: str,
    temperature: float,
    ph: float,
    water_activity: float,
    mean: float,
    lower: float,
    upper: float,
) -> guardrails.Assessment:
    """Show whether this prediction is an exploratory estimate, a validated one, or out of scope."""
    row = bench.table.loc[bench.table["key"] == best_key]
    r2 = (
        float(row["R2"].iloc[0])
        if len(row) and "R2" in row and pd.notna(row["R2"].iloc[0])
        else None
    )
    n_train = int((df["mu_max_per_h"] > 0).sum()) if "mu_max_per_h" in df else None
    assessment = guardrails.assess(
        guardrails.training_envelope(df),
        temperature,
        ph,
        water_activity,
        mean,
        lower,
        upper,
        model_status=model_status(bench),
        validation_r2=r2,
        n_train=n_train,
    )
    render = {"validated": st.success, "exploratory": st.info, "out_of_scope": st.error}[
        assessment.tier
    ]
    render(
        f"**{assessment.label}.** "
        + (
            "; ".join(assessment.reasons).capitalize() + "."
            if assessment.reasons
            else "Inside the validated range of an approved model."
        )
    )
    st.caption(guardrails.REVIEW_NOTICE)
    return assessment
