"""Page 1: fit primary growth models to measured curves."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.datasets import load_real_kinetic_curves, real_kinetic_organisms
from src.ingestion import MEASUREMENT_TYPES, IngestionError, load_kinetic_series, to_ln_ratio
from src.ledger import save_run_record
from src.models.primary import PRIMARY_MODELS, FitError, fit_primary_curve
from src.reporting import (
    build_fit_report,
    df_to_csv_bytes,
    fit_results_frame,
    frame_fingerprint,
    frames_to_xlsx_bytes,
)
from src.ui.charts import fit_chart, fit_comparison_chart
from src.ui.common import DATA_DIR, report_button, show_plot
from src.ui.theme import group, hint, page_header, readout

MEAS_LABELS = {"log10_cfu": "Log10 CFU per mL (plate counts)", "od600": "Optical density (OD600)"}
SRC_UPLOAD, SRC_SAMPLE, SRC_REAL = (
    "Upload my own file",
    "Bundled synthetic curves (11 organisms)",
    "Installed reference curves",
)
TABLE_NAMES = {
    "series_id": "Curve",
    "organism": "Organism",
    "temperature_celsius": "Temp (°C)",
    "mu_max_per_h": "μmax (1/h)",
    "doubling_time_h": "Doubling time (h)",
    "lag_time_h": "Lag (h)",
    "y_max_ln": "Max growth (ln)",
    "y_max_standard_error": "Max growth SE",
    "mu_max_standard_error": "μmax SE (1/h)",
    "lag_time_standard_error": "Lag SE (h)",
    "r_squared": "R²",
    "rmse": "RMSE",
}


def render() -> None:
    user_id = st.session_state["user_id"]
    page_header(
        "Fit growth curves",
        "Load measured growth curves and get the growth rate, lag time and maximum population "
        "for each one.",
        "MODULE 01 / PRIMARY EXTRACTION",
    )
    left, right = st.columns([1, 2.3], gap="large")

    with left, st.container(border=True):
        group("Curves", first=True)
        real_orgs = real_kinetic_organisms()
        curve_source = st.selectbox(
            "Where do the curves come from?",
            [SRC_UPLOAD, SRC_SAMPLE] + ([SRC_REAL] if real_orgs else []),
            index=1,
        )
        uploaded, real_organism = None, None
        if curve_source == SRC_UPLOAD:
            uploaded = st.file_uploader(
                "Curve file (CSV or Excel)",
                type=["csv", "tsv", "xlsx", "xls"],
                help="Needs time_hours and measurement columns. Optional: series_id, organism, "
                "temperature_celsius, ph, water_activity.",
            )
        elif curve_source == SRC_REAL:
            real_organism = st.selectbox("Organism", real_orgs)
            hint(
                "Isothermal reference curves included in the local installation. "
                "Sparse or low-quality curves are filtered before fitting."
            )

        group("Fitting")
        if curve_source == SRC_REAL:
            meas_type = "log10_cfu"
        else:
            meas_type = st.radio(
                "What was measured?", MEASUREMENT_TYPES, format_func=MEAS_LABELS.get
            )
        model = st.selectbox(
            "Growth model",
            list(PRIMARY_MODELS),
            format_func=str.capitalize,
            help="Both describe lag, exponential growth and a plateau. Gompertz is the "
            "classic choice; Baranyi is more mechanistic.",
        )

        with st.expander("Cleaning options and defaults"):
            hint(
                "Defaults fill missing metadata. Existing file metadata is kept unless overridden."
            )
            organism = st.text_input("Organism", "Unspecified organism")
            c1, c2, c3 = st.columns(3)
            temp = c1.number_input("Temp (°C)", -20.0, 60.0, 37.0, 0.5)
            ph = c2.number_input("pH", 0.0, 14.0, 7.0, 0.1)
            aw = c3.number_input("Water activity", 0.0, 1.0, 0.99, 0.01)
            override_metadata = st.checkbox(
                "Use these defaults even when the file provides metadata",
                value=False,
                help="When selected, replace organism, temperature, pH and water-activity "
                "metadata found in the input file with the values above.",
            )
            smooth = st.checkbox("Smooth noise (Savitzky-Golay)", value=True)
            sg_window = st.slider("Smoothing window (points)", 3, 15, 5, 2)
            sigmas = st.slider(
                "Outlier threshold (sigmas)",
                2.0,
                6.0,
                3.0,
                0.5,
                help="Lower removes more points as outliers.",
            )

    with right:
        if curve_source == SRC_UPLOAD:
            if uploaded is None:
                st.info(
                    "Upload a curve file to begin, or pick a bundled sample from the panel on the left."
                )
                return
            file_source, name = io.BytesIO(uploaded.getvalue()), uploaded.name
        elif curve_source == SRC_REAL:
            file_source, name = (
                load_real_kinetic_curves(real_organism),
                f"reference_{real_organism}",
            )
        else:
            file_source, name = DATA_DIR / "kinetic_curve_sample.csv", "kinetic_curve_sample.csv"

        try:
            series, report = load_kinetic_series(
                file_source,
                filename=name,
                default_organism=organism,
                default_temperature_c=temp,
                default_ph=ph,
                default_aw=aw,
                override_metadata=override_metadata,
                smooth=smooth,
                sg_window=sg_window,
                hampel_sigmas=sigmas,
            )
        except IngestionError as exc:
            st.error(f"That file couldn't be read. {exc}")
            return

        entries: List[Dict[str, Any]] = []
        failures: Dict[str, str] = {}
        for s in series:
            curve = report.cleaned[s.series_id]
            y_clean, y_raw = to_ln_ratio(curve.clean, meas_type), to_ln_ratio(curve.raw, meas_type)
            try:
                fit = fit_primary_curve(curve.time, y_clean, model)
            except FitError as exc:
                failures[s.series_id] = str(exc)
                continue
            entries.append(
                {"series": s, "fit": fit, "curve": curve, "y_raw": y_raw, "y_clean": y_clean}
            )
        if not entries:
            st.error(
                "None of the curves could be fitted. Check that the file holds growth over time."
            )
            for sid, why in {**report.skipped, **failures}.items():
                st.caption(f"{sid}: {why}")
            return

        table = fit_results_frame(entries)
        save_signature = (
            f"{curve_source}|{name}|{model}|{meas_type}|{smooth}|{sg_window}|"
            f"{sigmas}|{override_metadata}|{frame_fingerprint(table)}"
        )
        previous_save = st.session_state.get("assay_saved_record")
        run_ids = (
            previous_save[1]
            if isinstance(previous_save, tuple) and previous_save[0] == save_signature
            else {}
        )
        skipped = len(series) - len(entries) + len(report.skipped)
        st.markdown(
            f"**{len(entries)} curves fitted**" + (f", {skipped} skipped." if skipped else ".")
        )
        if run_ids:
            st.success(f"Saved {len(run_ids)} fitted curve(s) to your analysis history.")
        elif st.button(
            f"Save {len(entries)} fitted curve(s) to history",
            key=f"save_assay_{frame_fingerprint(table)}",
        ):
            run_ids = {}
            for entry in entries:
                fitted_series = entry["series"]
                fitted = entry["fit"]
                run_ids[fitted_series.series_id] = save_run_record(
                    user_id,
                    "assay",
                    f"{fitted_series.organism_name} · primary fit",
                    f"{fitted.model.capitalize()} · {fitted_series.temperature_celsius:g} °C · "
                    f"R² {fitted.r_squared:.3f}",
                    "μmax",
                    f"{fitted.mu_max:.3f} / h",
                    {
                        "source": curve_source,
                        "file": name,
                        "curve": fitted_series.series_id,
                        "organism": fitted_series.organism_name,
                        "temperature_celsius": fitted_series.temperature_celsius,
                        "pH": fitted_series.ph,
                        "water_activity": fitted_series.water_activity,
                        "measurement_type": meas_type,
                        "model": fitted.model,
                        "cleaning": {
                            "smooth": smooth,
                            "smoothing_window": sg_window,
                            "outlier_threshold_sigmas": sigmas,
                            "override_metadata": override_metadata,
                        },
                        "results": {
                            "mu_max_per_h": fitted.mu_max,
                            "mu_max_standard_error": fitted.param_std[1],
                            "doubling_time_h": fitted.doubling_time_h,
                            "lag_time_h": fitted.lam,
                            "lag_time_standard_error": fitted.param_std[2],
                            "y_max_ln": fitted.A,
                            "y_max_standard_error": fitted.param_std[0],
                            "r_squared": fitted.r_squared,
                            "rmse": fitted.rmse,
                            "n_points": fitted.n_points,
                        },
                    },
                )[1]
            st.session_state["assay_saved_record"] = (save_signature, run_ids)
            st.success(f"Saved {len(run_ids)} fitted curve(s) to your analysis history.")
        cleaned_points = sum(len(entry["curve"].clean) for entry in entries)
        replaced_points = sum(int(entry["curve"].outliers.sum()) for entry in entries)
        interpolated_points = sum(int(entry["curve"].interpolated.sum()) for entry in entries)
        st.caption(
            f"Cleaning applied: {cleaned_points} fitted points; "
            f"{replaced_points} outlier(s) replaced and "
            f"{interpolated_points} gap point(s) inserted. "
            f"Smoothing {'on' if smooth else 'off'} "
            f"(window {sg_window}); outlier threshold {sigmas:g}σ."
        )

        tab_fit, tab_all, tab_export = st.tabs(["Selected curve", "All curves", "Export"])

        with tab_fit:
            ids = [e["series"].series_id for e in entries]
            lookup = {e["series"].series_id: e for e in entries}

            def label(i: str) -> str:
                s = lookup[i]["series"]
                return f"{i} ({s.organism_name}, {s.temperature_celsius:g} °C)"

            pick = st.selectbox("Curve", ids, format_func=label)
            e = lookup[pick]
            f = e["fit"]
            mu_standard_error = f.param_std[1]
            mu_standard_error_label = (
                f"{mu_standard_error:.3g}" if np.isfinite(mu_standard_error) else "n/a"
            )
            readout(
                [
                    ("Growth rate μmax", (f"{f.mu_max:.3f}", "1/h"), None),
                    ("Doubling time", (f"{f.doubling_time_h:.2f}", "h"), None),
                    ("Lag phase", (f"{f.lam:.2f}", "h"), None),
                    (
                        "μmax standard error",
                        mu_standard_error_label,
                        None,
                    ),
                    (
                        "Goodness of fit R²",
                        f"{f.r_squared:.4f}",
                        "good" if f.r_squared >= 0.98 else None,
                    ),
                ]
            )
            if f.r_squared < 0.8:
                st.warning(
                    "This fit explains the observed curve poorly (R² below 0.8). "
                    "Inspect the data-quality notes and avoid relying on its parameters without review."
                )
            if not np.isfinite(mu_standard_error) or mu_standard_error > 0.5 * f.mu_max:
                st.warning(
                    "The fitted growth rate has high or unavailable parameter uncertainty. "
                    "Treat its point estimate cautiously."
                )
            st.caption(
                "Parameter standard errors describe local fit uncertainty; they are not "
                "calibrated prediction intervals."
            )
            show_plot(
                fit_chart(e["curve"], e["y_raw"], e["y_clean"], f, f"{pick}, {f.model} fit"),
                key="fit_chart",
            )
            notes = report.warnings + [
                f"{k}: {v}" for k, v in {**report.skipped, **failures}.items()
            ]
            if notes:
                with st.expander(f"Data-quality notes ({len(notes)})"):
                    for w in notes:
                        st.write("- " + w)

        with tab_all:
            view = table[list(TABLE_NAMES)].rename(columns=TABLE_NAMES)
            cfg = {
                c: st.column_config.NumberColumn(format="%.3f")
                for c in view.columns
                if c not in ("Curve", "Organism")
            }
            cfg["Temp (°C)"] = st.column_config.NumberColumn(format="%.1f")
            for column in ("μmax SE (1/h)", "Lag SE (h)"):
                cfg[column] = st.column_config.NumberColumn(format="%.3g")
            st.dataframe(view, hide_index=True, column_config=cfg)
            selected_curves = st.multiselect(
                "Compare curves",
                [entry["series"].series_id for entry in entries],
                default=[entry["series"].series_id for entry in entries[: min(5, len(entries))]],
                help="Observed points and fitted curves share the normalized ln(N/N0) scale.",
            )
            if selected_curves:
                show_plot(fit_comparison_chart(entries, selected_curves), key="fit_comparison")
                selected_fits = [
                    entry["fit"]
                    for entry in entries
                    if entry["series"].series_id in selected_curves
                ]
                st.caption(
                    f"Across these curves: μmax {min(f.mu_max for f in selected_fits):.3g}–"
                    f"{max(f.mu_max for f in selected_fits):.3g} 1/h; "
                    f"median R² {sorted(f.r_squared for f in selected_fits)[len(selected_fits)//2]:.3f}."
                )

        with tab_export:
            st.download_button(
                "Download parameters (CSV)",
                df_to_csv_bytes(table),
                "primary_fit_parameters.csv",
                "text/csv",
            )
            st.download_button(
                "Download analysis workbook (Excel)",
                frames_to_xlsx_bytes(
                    {
                        "Fit parameters": table,
                        "Data quality": pd.DataFrame(
                            [{"note": warning} for warning in report.warnings]
                            + [
                                {"note": f"{key}: {value}"}
                                for key, value in {**report.skipped, **failures}.items()
                            ]
                        ),
                    }
                ),
                "primary_fit_analysis.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            sig = f"{curve_source}|{real_organism}|{name}|{model}|{meas_type}|{len(entries)}|{smooth}|{sg_window}|{sigmas}"  # noqa: E501
            report_button(
                "PDF report",
                lambda: build_fit_report(
                    entries,
                    meas_type,
                    audit_metadata={
                        "source": curve_source,
                        "file": name,
                        "model": model,
                        "saved run IDs": ", ".join(run_ids.values()),
                    },
                ),
                "primary_fit_report.pdf",
                "pdf_fit",
                sig,
            )
