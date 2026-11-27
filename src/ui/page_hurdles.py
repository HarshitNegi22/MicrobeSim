"""Page 2: predict growth rate from temperature, pH and water activity, and compare models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.ledger import save_run_record
from src.models.benchmark import MODEL_LABELS
from src.models.secondary import make_X
from src.reporting import (
    build_benchmark_report,
    df_to_csv_bytes,
    frame_fingerprint,
    frames_to_xlsx_bytes,
)
from src.ui.charts import benchmark_bars, mu_ribbon_chart, parity_chart
from src.ui.common import (
    conditions_outside_training_data,
    dataset_selector,
    render_assessment,
    report_button,
    show_plot,
)
from src.ui.theme import group, hint, page_header, readout, verdict

SWEEPS = {
    "Temperature": ("temperature_celsius", -10.0, 55.0, 0.5, "Temperature (°C)"),
    "pH": ("ph", 3.0, 10.5, 0.05, "pH"),
    "Water activity": ("water_activity", 0.85, 1.0, 0.001, "Water activity"),
}
BENCH_NAMES = {
    "R2": "R²",
    "R2_std": "R² SD",
    "RMSE_std": "RMSE SD",
    "Bf_std": "Bf SD",
    "Af_std": "Af SD",
    "n_test": "Test rows",
    "n_missed": "Missed growth",
    "fit_seconds": "Fit time (s)",
}


def render() -> None:
    user_id = st.session_state["user_id"]
    page_header(
        "Predict growth rate",
        "See how fast an organism grows at a given temperature, acidity and moisture, "
        "and how much the different models agree.",
        "MODULE 02 / SECONDARY MODELING",
    )
    left, right = st.columns([1, 2.3], gap="large")

    with left, st.container(border=True):
        group("Organism and data", first=True)
        with st.expander("Use my own dataset instead"):
            hint("CSV with temperature_celsius, ph, water_activity and mu_max_per_h columns.")
            up = st.file_uploader("Dataset (CSV)", type=["csv"], label_visibility="collapsed")
            if up is not None:
                st.session_state["uploaded_growth_csv"] = up.getvalue()
            elif "uploaded_growth_csv" in st.session_state and st.button("Remove uploaded dataset"):
                del st.session_state["uploaded_growth_csv"]
                st.rerun()
        label, df, bench = dataset_selector("hurdles")

        group("Models")
        keys = list(bench.models)
        chosen = st.multiselect(
            "Models to compare",
            keys,
            default=keys,
            format_func=MODEL_LABELS.get,
            label_visibility="collapsed",
        )

        group("Conditions")
        T = st.slider("Temperature (°C)", -10.0, 55.0, 25.0, 0.5)
        ph = st.slider("pH", 3.0, 10.5, 6.5, 0.1)
        aw = st.slider("Water activity", 0.85, 1.0, 0.98, 0.005)

    with right:
        for warning in conditions_outside_training_data(
            df,
            {
                "temperature_celsius": (T, T),
                "ph": (ph, ph),
                "water_activity": (aw, aw),
            },
        ):
            st.warning(
                f"Outside this dataset's observed range ({warning}). Treat the prediction as extrapolation."
            )
        if not chosen:
            st.info("Pick at least one model in the panel on the left.")
            return
        tab_pred, tab_bench = st.tabs(["Prediction", "How accurate are the models?"])

        with tab_pred:
            best = next(k for k in bench.table["key"] if k in chosen)
            X = make_X(T, ph, aw)
            mean, lo, hi = bench.models[best].predict_interval(X)
            if mean[0] <= 0:
                verdict(
                    "warn",
                    "No growth expected",
                    f"At {T:g} °C, pH {ph:g} and water activity {aw:g} the most accurate model "
                    f"({MODEL_LABELS[best]}) predicts the organism cannot grow.",
                )
            else:
                readout(
                    [
                        ("Growth rate μmax", (f"{mean[0]:.3f}", "1/h"), None),
                        ("Doubling time", (f"{np.log(2) / mean[0]:.1f}", "h"), None),
                        ("Nominal 95% model range", (f"{lo[0]:.3f} to {hi[0]:.3f}", "1/h"), None),
                    ]
                )
            st.caption(
                f"Shown for {MODEL_LABELS[best]}, the most accurate of the selected models on held-out data."
            )
            render_assessment(
                df, bench, best, T, ph, aw, float(mean[0]), float(lo[0]), float(hi[0])
            )
            with st.expander("Why this prediction?"):
                st.write(
                    f"{MODEL_LABELS[best]} was selected using {bench.validation_description}; "
                    "the scorecard shows average performance and variability across splits."
                )
                limits = bench.models[best].limits
                st.write(
                    "Fitted growth envelope: "
                    f"{limits.tmin:.2f}–{limits.tmax:.2f} °C, "
                    f"pH {limits.phmin:.2f}–{limits.phmax:.2f}, "
                    f"water activity > {limits.awmin:.3f} (no fitted upper cutoff)."
                )
                st.caption(
                    "An empirical training-range warning is separate from the model's fitted "
                    "growth envelope; neither is a guarantee of real-world performance."
                )

            rows = []
            for k in chosen:
                m, l, h = bench.models[k].predict_interval(X)
                rows.append(
                    {
                        "Model": MODEL_LABELS[k],
                        "μmax (1/h)": float(m[0]),
                        "Low (nominal 95%)": float(l[0]),
                        "High (nominal 95%)": float(h[0]),
                        "Doubling time (h)": float(np.log(2) / m[0]) if m[0] > 0 else None,
                    }
                )
            row_frame = pd.DataFrame(rows)
            save_signature = (
                f"{label}|{T}|{ph}|{aw}|{','.join(chosen)}|{frame_fingerprint(row_frame)}"
            )
            previous_save = st.session_state.get("prediction_saved_record")
            prediction_run_id = (
                previous_save[1]
                if isinstance(previous_save, tuple) and previous_save[0] == save_signature
                else None
            )
            if prediction_run_id:
                st.success("This prediction is saved to your analysis history.")
            elif st.button("Save this prediction to history", key="save_hurdle_prediction"):
                prediction_run_id = save_run_record(
                    user_id,
                    "prediction",
                    f"Hurdle vector · {T:g}°C",
                    f"{label} · pH {ph:g} · aw {aw:.3f} · {len(chosen)} model(s)",
                    "μmax",
                    f"{mean[0]:.3f} / h",
                    {
                        "dataset": label,
                        "validation_method": bench.validation_description,
                        "temperature_celsius": T,
                        "pH": ph,
                        "water_activity": aw,
                        "selected_models": [MODEL_LABELS[key] for key in chosen],
                        "reported_model": MODEL_LABELS[best],
                        "predictions": rows,
                    },
                )[1]
                st.session_state["prediction_saved_record"] = (
                    save_signature,
                    prediction_run_id,
                )
                st.success("Prediction saved to your analysis history.")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={
                    c: st.column_config.NumberColumn(format="%.3f")
                    for c in (
                        "μmax (1/h)",
                        "Low (nominal 95%)",
                        "High (nominal 95%)",
                    )
                }
                | {"Doubling time (h)": st.column_config.NumberColumn(format="%.1f")},
            )

            st.markdown("**How growth rate changes**")
            sweep = st.radio("Vary", list(SWEEPS), horizontal=True, label_visibility="collapsed")
            col, lo_, hi_, step, axis_label = SWEEPS[sweep]
            x = np.arange(lo_, hi_ + step, step)
            fixed = {"temperature_celsius": T, "ph": ph, "water_activity": aw}
            fixed[col] = x
            Xs = make_X(fixed["temperature_celsius"], fixed["ph"], fixed["water_activity"])
            curves = {}
            for k in chosen:
                m, l, h = bench.models[k].predict_interval(Xs)
                curves[MODEL_LABELS[k]] = {"mean": m, "lo": l, "hi": h}
            train_view = None
            if st.checkbox("Show measured data near these conditions", value=True):
                near = df
                tol = {"temperature_celsius": 3.0, "ph": 0.5, "water_activity": 0.02}
                for other in fixed:
                    if other != col:
                        near = near[(near[other] - fixed[other]).abs() <= tol[other]]
                train_view = near
            show_plot(mu_ribbon_chart(x, curves, axis_label, train_view, col), key="ribbon")
            st.caption(
                "Shaded bands are nominal 95% model ranges, not calibrated probability intervals. "
                "Every model is forced to zero growth outside its estimated biological envelope."
            )

        with tab_bench:
            table = bench.table[bench.table["key"].isin(chosen)].reset_index(drop=True)
            st.markdown(
                f"Models were compared across {bench.validation_description}. "
                f"{len(bench.test_index)} distinct rows were held out at least once. "
                "Models shown in the prediction tab are refit on all available rows."
            )
            if bench.validation_description.endswith("random holdouts"):
                st.warning(
                    "No useful grouping field was available, so repeated row-level splits were used. "
                    "Scores may be optimistic when related experiment records occur in both partitions."
                )
            hint(
                "Af is the typical error as a multiplier: 1.0 is perfect, 1.5 means predictions are usually "
                "within 50% of the truth. Bf above 1 means the model over-predicts growth (the safe side), "
                "below 1 means it under-predicts."
            )
            weak = table[(table["R2"] < 0) | (table["Af"] > 5)]
            if len(weak):
                st.warning(
                    "Treat these with caution, they scored poorly here: "
                    + ", ".join(weak["Model"])
                    + ". Another model is a better choice for this dataset."
                )
            view = table.drop(columns="key").rename(columns=BENCH_NAMES)
            st.dataframe(
                view,
                hide_index=True,
                column_config={
                    c: st.column_config.NumberColumn(format="%.3f")
                    for c in (
                        "R²",
                        "R² SD",
                        "RMSE",
                        "RMSE SD",
                        "Bf",
                        "Bf SD",
                        "Af",
                        "Af SD",
                        "Fit time (s)",
                    )
                },
            )
            show_plot(benchmark_bars(table), key="bench_bars")
            show_plot(
                parity_chart(bench.predictions[["observed_mu_max"] + chosen], MODEL_LABELS),
                key="parity",
            )
            st.caption("Points on the dashed line are perfect predictions.")

            st.download_button(
                "Download scorecard (CSV)",
                df_to_csv_bytes(table.drop(columns="key")),
                "benchmark_scorecard.csv",
                "text/csv",
            )
            st.download_button(
                "Download scorecard (Excel)",
                frames_to_xlsx_bytes(
                    {
                        "Model scorecard": table.drop(columns="key"),
                        "Dataset coverage": pd.DataFrame(
                            [
                                {
                                    "feature": column,
                                    "minimum": float(df[column].min()),
                                    "maximum": float(df[column].max()),
                                    "distinct_values": int(df[column].nunique()),
                                }
                                for column in (
                                    "temperature_celsius",
                                    "ph",
                                    "water_activity",
                                    "mu_max_per_h",
                                )
                            ]
                        ),
                    }
                ),
                "benchmark_scorecard.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            report_button(
                "PDF report",
                lambda: build_benchmark_report(
                    table,
                    label,
                    len(df),
                    bench.validation_description,
                    audit_metadata={
                        "Associated prediction run ID": prediction_run_id,
                        "Training row count": len(df),
                        "Training data fingerprint": frame_fingerprint(df),
                        "validation seed": 0,
                        "holdout fraction": 0.25,
                        "splits": len(bench.validation_splits),
                        "grouping labels": bench.validation_description,
                    },
                ),
                "benchmark_report.pdf",
                "pdf_bench",
                f"{label}|{','.join(chosen)}|{len(df)}",
            )
