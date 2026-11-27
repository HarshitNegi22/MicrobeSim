"""Page 3: replay a temperature history and find when the population reaches a limit."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from src.ingestion import IngestionError, load_cold_chain_telemetry
from src.ledger import (
    delete_user_profile,
    list_user_profiles,
    save_run_record,
    save_user_profile,
)
from src.models.benchmark import MODEL_LABELS
from src.reporting import (
    build_simulation_report,
    df_to_csv_bytes,
    frame_fingerprint,
    frames_to_xlsx_bytes,
)
from src.schemas import DynamicTelemetryPoint, SimulationScenario
from src.simulation.dynamic import (
    INTERPOLATIONS,
    SimulationError,
    build_environment_mu_lookup,
    build_environment_mu_interval_lookup,
    build_mu_lookup,
    simulate,
)
from src.simulation.scenarios import PRESETS, cold_chain_break, step_function
from src.ui.charts import simulation_chart
from src.ui.common import (
    DATA_DIR,
    conditions_outside_training_data,
    dataset_selector,
    fmt_hours,
    report_button,
    show_plot,
)
from src.ui.theme import group, hint, page_header, readout, verdict

SRC_PRESET, SRC_STEP, SRC_EXCURSION, SRC_UPLOAD = (
    "Ready-made failure",
    "Step change",
    "Custom warm spell",
    "Upload sensor log",
)


def _telemetry(scenario_profiles):
    """Draw the temperature-history inputs; return ``(points, interpolation, label)`` or ``None``."""
    source_options = [SRC_PRESET, SRC_EXCURSION, SRC_STEP, SRC_UPLOAD]
    if scenario_profiles:
        source_options.append("Saved scenario profile")
    source = st.selectbox(
        "Temperature history",
        source_options,
        key="sim_temperature_source",
    )
    if source == SRC_PRESET:
        key = st.selectbox(
            "Failure",
            list(PRESETS),
            format_func=lambda k: PRESETS[k].label,
            key="sim_preset_key",
        )
        hint(PRESETS[key].description)
        return PRESETS[key].build(), PRESETS[key].interpolation, PRESETS[key].label
    if source == "Saved scenario profile":
        profile = st.selectbox(
            "Scenario profile",
            [item["profile_id"] for item in scenario_profiles],
            format_func=lambda profile_id: next(
                item["name"] for item in scenario_profiles if item["profile_id"] == profile_id
            ),
            key="sim_scenario_profile",
        )
        profile = next(item for item in scenario_profiles if item["profile_id"] == profile)
        points = [
            DynamicTelemetryPoint.model_validate(point)
            for point in profile["payload"]["environmental_log"]
        ]
        return points, profile["payload"]["interpolation"], profile["name"]
    if source == SRC_STEP:
        a, b = st.columns(2)
        lo = a.number_input("Before (°C)", -20.0, 50.0, 4.0, 0.5)
        hi = b.number_input("After (°C)", -20.0, 50.0, 20.0, 0.5)
        sw = a.number_input("Switch at (h)", 1.0, 500.0, 24.0, 1.0)
        dur = b.number_input("Total (h)", 2.0, 1000.0, 72.0, 1.0)
        if sw >= dur:
            st.error("The switch has to happen before the end of the history.")
            return None
        return step_function(lo, hi, sw, dur), "step", f"Step {lo:g} to {hi:g} °C at {sw:g} h"
    if source == SRC_EXCURSION:
        a, b = st.columns(2)
        fr = a.number_input("Normal (°C)", -20.0, 20.0, 4.0, 0.5)
        ex = b.number_input("Warm spell (°C)", -20.0, 50.0, 25.0, 0.5)
        st_ = a.number_input("Starts at (h)", 0.0, 500.0, 24.0, 1.0)
        du = b.number_input("Lasts (h)", 0.5, 200.0, 8.0, 0.5)
        tot = a.number_input("Total (h)", 2.0, 1000.0, 96.0, 1.0)
        if st_ + du + 1.0 >= tot:
            st.error("The warm spell (plus a 1 h cool-down) has to end before the total time.")
            return None
        return (
            cold_chain_break(fr, ex, st_, du, tot),
            "linear",
            f"Warm spell at {ex:g} °C for {du:g} h",
        )
    up = st.file_uploader(
        "Sensor log (CSV)",
        type=["csv", "xlsx"],
        help=(
            "Required: timestamp_hours and temperature_celsius. Optional: ph (0–14) and "
            "water_activity (>0–1). Missing values in a supplied profile are interpolated."
        ),
    )
    if up is None:
        if not st.checkbox("Use the bundled sample log", value=True):
            st.info("Upload a sensor log to continue.")
            return None
        src, nm = DATA_DIR / "cold_chain_sample.csv", "cold_chain_sample.csv"
    else:
        src, nm = io.BytesIO(up.getvalue()), up.name
    try:
        pts, warns = load_cold_chain_telemetry(src, filename=nm)
    except IngestionError as exc:
        st.error(f"That sensor log couldn't be used. {exc}")
        return None
    for w in warns:
        st.warning(w)
    return pts, "linear", f"Sensor log ({nm})"


def _profile_events(points):
    events = []
    slopes = []
    for first, second in zip(points[:-1], points[1:]):
        duration = second.timestamp_hours - first.timestamp_hours
        slopes.append(
            (second.temperature_celsius - first.temperature_celsius) / duration
            if duration > 0
            else 0.0
        )
    for index in range(1, len(slopes)):
        before, after = slopes[index - 1], slopes[index]
        if before > 0.5 and after <= 0:
            events.append((points[index].timestamp_hours, "Warm peak"))
        elif before < -0.5 and after >= 0:
            events.append((points[index].timestamp_hours, "Cooling begins"))
    return events


def render() -> None:
    user_id = st.session_state["user_id"]
    product_profiles = list_user_profiles(user_id, "product")
    scenario_profiles = list_user_profiles(user_id, "scenario")

    def apply_product_profile() -> None:
        selected_id = st.session_state.get("sim_product_profile")
        selected = next(
            (profile for profile in product_profiles if profile["profile_id"] == selected_id),
            None,
        )
        if selected is None:
            return
        for key, value in selected["payload"].items():
            st.session_state[f"sim_{key}"] = value

    page_header(
        "Simulate a cold-chain failure",
        "Replay a temperature history for a product and see exactly when the bacteria "
        "reach your safety or spoilage limit.",
        "MODULE 03 / DYNAMIC SIMULATION",
    )
    st.markdown("### Start with a scenario")
    st.caption("Explore a ready-made cold-chain event or customize its readings below.")
    gallery_cols = st.columns(len(PRESETS), gap="small")
    for column, (key, preset) in zip(gallery_cols, PRESETS.items()):
        with column, st.container(border=True):
            st.markdown(f"**{preset.label}**")
            st.caption(preset.description)
            if st.button("Use scenario", key=f"gallery_{key}", width="stretch"):
                st.session_state["sim_temperature_source"] = SRC_PRESET
                st.session_state["sim_preset_key"] = key
                st.rerun()

    left, right = st.columns([1, 2.3], gap="large")

    with left, st.container(border=True):
        st.selectbox(
            "Saved product profile",
            [None] + [profile["profile_id"] for profile in product_profiles],
            format_func=lambda profile_id: (
                "Manual conditions"
                if profile_id is None
                else next(p["name"] for p in product_profiles if p["profile_id"] == profile_id)
            ),
            key="sim_product_profile",
            on_change=apply_product_profile,
        )
        group("Product", first=True)
        a, b = st.columns(2)
        log_n0 = a.number_input("Starting load (log10 CFU/g)", 0.0, 8.0, 2.0, 0.1, key="sim_log_n0")
        log_nc = b.number_input(
            "Limit (log10 CFU/g)",
            0.5,
            9.0,
            5.0,
            0.1,
            help="The count at which the product is spoiled or unsafe.",
            key="sim_log_nc",
        )
        ph = a.number_input("pH", 3.0, 10.0, 6.5, 0.1, key="sim_ph")
        aw = b.number_input("Water activity", 0.85, 1.0, 0.98, 0.005, key="sim_aw")

        group("Temperature history")
        scenario_data = _telemetry(scenario_profiles)

        group("Growth model")
        label, sim_df, bench = dataset_selector("sim")
        keys = list(bench.models)
        model_key = st.selectbox(
            "Model",
            keys,
            format_func=MODEL_LABELS.get,
            index=keys.index("ratkowsky") if "ratkowsky" in keys else 0,
        )
        with st.expander("Advanced settings"):
            log_nmax = st.number_input(
                "Maximum possible load (log10 CFU/g)",
                6.0,
                12.0,
                9.0,
                0.5,
                help="Growth levels off as the population approaches this.",
                key="sim_log_nmax",
            )
            h0 = st.number_input(
                "Lag (0 means none)",
                0.0,
                10.0,
                1.0,
                0.1,
                help="Starting physiological state h0. Lag time at a constant temperature is h0 divided by the growth rate.",  # noqa: E501
                key="sim_h0",
            )
            default_interp = scenario_data[1] if scenario_data else "linear"
            interp = st.selectbox(
                "Between sensor readings",
                INTERPOLATIONS,
                index=INTERPOLATIONS.index(default_interp),
                help="linear: straight lines. pchip: smooth curve without overshoot. step: hold each reading until the next.",  # noqa: E501
            )
        with st.expander("Save reusable profiles"):
            with st.form("save_product_profile"):
                product_name = st.text_input("Product profile name", max_chars=80)
                save_product = st.form_submit_button("Save current product conditions")
            if save_product:
                try:
                    save_user_profile(
                        user_id,
                        "product",
                        product_name,
                        {
                            "log_n0": log_n0,
                            "log_nc": log_nc,
                            "ph": ph,
                            "aw": aw,
                            "log_nmax": log_nmax,
                            "h0": h0,
                        },
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Product profile saved.")
                    st.rerun()
            if scenario_data is not None:
                with st.form("save_scenario_profile"):
                    scenario_name = st.text_input("Scenario profile name", max_chars=80)
                    save_scenario = st.form_submit_button("Save current environmental history")
                if save_scenario:
                    points_to_save = scenario_data[0]
                    try:
                        save_user_profile(
                            user_id,
                            "scenario",
                            scenario_name,
                            {
                                "interpolation": interp,
                                "environmental_log": [
                                    point.model_dump() for point in points_to_save
                                ],
                            },
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Scenario profile saved.")
                        st.rerun()
            if product_profiles or scenario_profiles:
                profile_options = [
                    (profile["profile_id"], f"Product · {profile['name']}")
                    for profile in product_profiles
                ] + [
                    (profile["profile_id"], f"Scenario · {profile['name']}")
                    for profile in scenario_profiles
                ]
                with st.form("delete_profile_form"):
                    profile_id = st.selectbox(
                        "Remove a saved profile",
                        [item[0] for item in profile_options],
                        format_func=lambda selected_id: next(
                            label
                            for profile_id_, label in profile_options
                            if profile_id_ == selected_id
                        ),
                    )
                    delete_profile = st.form_submit_button("Delete profile")
                if delete_profile:
                    delete_user_profile(user_id, profile_id)
                    st.rerun()

    with right:
        if scenario_data is None:
            st.info("Finish the temperature history in the panel on the left.")
            return
        points, _, scenario_label = scenario_data
        for warning in conditions_outside_training_data(
            sim_df,
            {
                "temperature_celsius": (
                    min(point.temperature_celsius for point in points),
                    max(point.temperature_celsius for point in points),
                ),
                "ph": (
                    min(point.ph if point.ph is not None else ph for point in points),
                    max(point.ph if point.ph is not None else ph for point in points),
                ),
                "water_activity": (
                    min(
                        point.water_activity if point.water_activity is not None else aw
                        for point in points
                    ),
                    max(
                        point.water_activity if point.water_activity is not None else aw
                        for point in points
                    ),
                ),
            },
        ):
            st.warning(
                f"Environmental history extends outside the training-data range ({warning}). "
                "Treat the simulation as extrapolation."
            )
        if log_nc <= log_n0:
            st.error("The limit has to be higher than the starting load.")
            return

        model = bench.models[model_key]
        mu_fn = build_mu_lookup(model, ph, aw)
        has_dynamic_conditions = any(
            point.ph is not None or point.water_activity is not None for point in points
        )
        environment_mu_fn = (
            build_environment_mu_lookup(model, ph, aw) if has_dynamic_conditions else None
        )
        interval_mu_fn = build_environment_mu_interval_lookup(model, ph, aw)
        if has_dynamic_conditions:
            rates = [
                environment_mu_fn(
                    point.temperature_celsius,
                    point.ph if point.ph is not None else ph,
                    point.water_activity if point.water_activity is not None else aw,
                )
                for point in points
            ]
        else:
            rates = [mu_fn(T) for T in np.arange(-20, 50, 1.0)]
        if float(np.max(rates)) == 0.0:
            st.warning(
                "At this pH and water activity the selected model allows no growth at any temperature."
            )

        scenario = SimulationScenario(
            organism=label,
            initial_cfu_per_g=10.0**log_n0,
            critical_threshold_cfu_per_g=10.0**log_nc,
            environmental_log=points,
            ph=ph,
            water_activity=aw,
        )
        compare_keys = [None] + list(PRESETS)
        compare_key = st.selectbox(
            "Compare against a reference scenario",
            compare_keys,
            format_func=lambda key: (
                "No comparison"
                if key is None
                else f"{PRESETS[key].label} · {key.replace('_', ' ')}"
            ),
        )

        def lower_mu(temperature):
            return interval_mu_fn(temperature, ph, aw)[1]

        def upper_mu(temperature):
            return interval_mu_fn(temperature, ph, aw)[2]

        def lower_environment_mu(temperature, pH, activity):
            return interval_mu_fn(temperature, pH, activity)[1]

        def upper_environment_mu(temperature, pH, activity):
            return interval_mu_fn(temperature, pH, activity)[2]

        lower_environment_fn = lower_environment_mu if has_dynamic_conditions else None
        upper_environment_fn = upper_environment_mu if has_dynamic_conditions else None
        try:
            trace = simulate(
                scenario,
                mu_fn,
                environment_mu_fn=environment_mu_fn,
                n_max_cfu_per_g=10.0**log_nmax,
                h0=h0,
                interpolation=interp,
            )
            lower_trace = simulate(
                scenario,
                lower_mu,
                environment_mu_fn=lower_environment_fn,
                n_max_cfu_per_g=10.0**log_nmax,
                h0=h0,
                interpolation=interp,
            )
            upper_trace = simulate(
                scenario,
                upper_mu,
                environment_mu_fn=upper_environment_fn,
                n_max_cfu_per_g=10.0**log_nmax,
                h0=h0,
                interpolation=interp,
            )
            comparison_trace = None
            if compare_key is not None:
                comparison_preset = PRESETS[compare_key]
                comparison_scenario = SimulationScenario(
                    organism=label,
                    initial_cfu_per_g=10.0**log_n0,
                    critical_threshold_cfu_per_g=10.0**log_nc,
                    environmental_log=comparison_preset.build(),
                    ph=ph,
                    water_activity=aw,
                )
                comparison_trace = simulate(
                    comparison_scenario,
                    mu_fn,
                    n_max_cfu_per_g=10.0**log_nmax,
                    h0=h0,
                    interpolation=comparison_preset.interpolation,
                )
        except SimulationError as exc:
            st.error(f"The simulation failed. {exc}")
            return

        final, peak = float(np.log10(trace.cfu_per_g[-1])), float(np.log10(trace.cfu_per_g.max()))
        end_h = float(trace.time_hours[-1])
        fastest_growth_index = int(np.argmax(trace.mu_max))
        fastest_growth = float(trace.mu_max[fastest_growth_index])
        if lower_trace.spoilage_breached:
            verdict(
                "bad",
                "Limit reached across the model interval",
                f"Even the lower growth-rate trajectory crosses {log_nc:g} log10 CFU/g. "
                f"The central estimate reaches it after {fmt_hours(trace.time_to_breach_hours)}.",
            )
        elif upper_trace.spoilage_breached:
            verdict(
                "warn",
                "Model interval includes a possible limit crossing",
                (
                    f"The central estimate reaches the limit after {fmt_hours(trace.time_to_breach_hours)}."
                    if trace.spoilage_breached
                    else "The central estimate stays below the limit, but the upper growth-rate trajectory crosses it."
                ),
            )
        else:
            verdict(
                "good",
                "No crossing in the model interval",
                f"The upper growth-rate trajectory peaks at "
                f"{np.log10(upper_trace.cfu_per_g.max()):.2f} log10 CFU/g, below your "
                f"{log_nc:g} limit over {end_h:g} h.",
            )
        lower_final = float(np.log10(lower_trace.cfu_per_g[-1]))
        upper_final = float(np.log10(upper_trace.cfu_per_g[-1]))
        compare_delta = (
            float(np.log10(comparison_trace.cfu_per_g[-1]) - final)
            if comparison_trace is not None
            else None
        )
        breach_range = (
            f"{fmt_hours(upper_trace.time_to_breach_hours)} to "
            f"{fmt_hours(lower_trace.time_to_breach_hours)}"
            if upper_trace.spoilage_breached and lower_trace.spoilage_breached
            else (
                f"{fmt_hours(upper_trace.time_to_breach_hours)} to not reached"
                if upper_trace.spoilage_breached
                else "Not reached"
            )
        )
        readout(
            [
                ("Final load (central)", (f"{final:.2f}", "log10 CFU/g"), None),
                ("Final load (model range)", f"{lower_final:.2f}–{upper_final:.2f}", None),
                ("Population doublings", f"{trace.doublings[-1]:.1f}", None),
                ("Peak temperature", (f"{trace.temperature_celsius.max():.1f}", "°C"), None),
                ("Fastest modeled growth", (f"{fastest_growth:.3f}", "1/h"), None),
                (
                    "Time-to-limit model range",
                    breach_range,
                    "bad" if upper_trace.spoilage_breached else None,
                ),
            ]
        )
        st.caption(
            f"Main modeled growth driver: the highest predicted μmax is {fastest_growth:.3f} 1/h "
            f"at {trace.time_hours[fastest_growth_index]:.1f} h "
            f"({trace.temperature_celsius[fastest_growth_index]:.1f} °C, "
            f"pH {trace.ph[fastest_growth_index]:.2f}, "
            f"aw {trace.water_activity[fastest_growth_index]:.3f})."
        )
        st.caption(
            "Shading and ranges propagate the model's pointwise 95% growth-rate interval through "
            "the simulator. They are not calibrated probability bounds and do not include input uncertainty."
        )
        if comparison_trace is not None:
            st.info(
                f"Compared with {PRESETS[compare_key].label}, final load is "
                f"{abs(compare_delta):.2f} log10 CFU/g "
                f"{'higher' if compare_delta > 0 else 'lower'} in the reference scenario."
            )
        timeline_hour = st.slider(
            "Explore the simulation timeline (hours)",
            min_value=float(trace.time_hours[0]),
            max_value=float(trace.time_hours[-1]),
            value=float(trace.time_hours[-1]),
            step=max(float(trace.time_hours[-1] - trace.time_hours[0]) / 500, 0.1),
            key="sim_timeline_hour",
            help="Scrub the slider to inspect temperature, product conditions and predicted load.",
        )
        timeline_index = int(np.abs(trace.time_hours - timeline_hour).argmin())
        readout(
            [
                ("At time", (f"{trace.time_hours[timeline_index]:.1f}", "h"), None),
                ("Temperature", (f"{trace.temperature_celsius[timeline_index]:.1f}", "°C"), None),
                ("Predicted load", (f"{trace.log10_cfu[timeline_index]:.2f}", "log10 CFU/g"), None),
                (
                    "pH / aw",
                    f"{trace.ph[timeline_index]:.2f} / {trace.water_activity[timeline_index]:.3f}",
                    None,
                ),
            ]
        )
        events = _profile_events(points)
        if events:
            st.caption(
                "Detected temperature events: "
                + ", ".join(f"{name} at {hour:g} h" for hour, name in events)
            )
        settings = {
            "scenario": scenario_label,
            "model": MODEL_LABELS[model_key],
            "training dataset": label,
            "validation method": bench.validation_description,
            "pH": ph,
            "water activity": aw,
            "N0 (CFU/g)": 10.0**log_n0,
            "N_crit (CFU/g)": 10.0**log_nc,
            "N_max (CFU/g)": 10.0**log_nmax,
            "h0": h0,
            "comparison scenario": PRESETS[compare_key].label if compare_key else "None",
            "interpolation": interp,
            "telemetry points": len(points),
            "time_varying_pH": has_dynamic_conditions and any(p.ph is not None for p in points),
            "time_varying_water_activity": has_dynamic_conditions
            and any(p.water_activity is not None for p in points),
        }
        breach_value = (
            f"Breach at {trace.time_to_breach_hours:.1f} h"
            if trace.spoilage_breached
            else "No breach"
        )
        run_payload = {
            **settings,
            "environmental_log": [
                {
                    "timestamp_hours": point.timestamp_hours,
                    "temperature_celsius": point.temperature_celsius,
                    "ph": point.ph,
                    "water_activity": point.water_activity,
                }
                for point in points
            ],
            "results": {
                "spoilage_breached": trace.spoilage_breached,
                "time_to_breach_hours": trace.time_to_breach_hours,
                "final_log10_cfu_per_g": final,
                "peak_log10_cfu_per_g": peak,
                "doublings": float(trace.doublings[-1]),
                "peak_temperature_celsius": float(trace.temperature_celsius.max()),
                "final_log10_cfu_per_g_model_range": [lower_final, upper_final],
                "time_to_breach_model_range_hours": [
                    upper_trace.time_to_breach_hours,
                    lower_trace.time_to_breach_hours,
                ],
            },
            "trajectory": trace.to_frame().to_dict(orient="records"),
        }
        save_signature = repr(sorted(settings.items())) + frame_fingerprint(trace.to_frame())
        previous_save = st.session_state.get("simulation_saved_record")
        saved_run_id = (
            previous_save[1]
            if isinstance(previous_save, tuple) and previous_save[0] == save_signature
            else None
        )
        if saved_run_id:
            st.success("This simulation is saved to your analysis history.")
        elif st.button(
            "Save this simulation to history",
            key=f"save_simulation_{frame_fingerprint(trace.to_frame())}",
        ):
            saved_run_id = save_run_record(
                user_id,
                "simulation",
                f"{scenario_label} · {label}",
                (
                    f"{'Threshold reached' if trace.spoilage_breached else 'Within threshold'} · "
                    f"{end_h:g} h · final {final:.2f} log10 CFU/g"
                ),
                "Threshold",
                breach_value,
                run_payload,
            )[1]
            st.session_state["simulation_saved_record"] = (save_signature, saved_run_id)
            st.success("Simulation saved to your analysis history.")
        show_plot(
            simulation_chart(
                trace,
                lower_trace,
                upper_trace,
                comparison_trace,
                (
                    f"Reference: {PRESETS[compare_key].label}"
                    if compare_key is not None
                    else "Reference scenario"
                ),
                timeline_hour,
                events,
            ),
            key="sim_chart",
        )

        frame = trace.to_frame()
        tab_data, tab_export = st.tabs(["Trajectory data", "Export"])
        with tab_data:
            st.dataframe(frame, hide_index=True)
        with tab_export:
            st.download_button(
                "Download trajectory (CSV)",
                df_to_csv_bytes(frame),
                "cold_chain_simulation.csv",
                "text/csv",
            )
            st.download_button(
                "Download reproducible analysis (Excel)",
                frames_to_xlsx_bytes(
                    {
                        "Trajectory": frame,
                        "Settings": pd.DataFrame(
                            [{"parameter": key, "value": value} for key, value in settings.items()]
                        ),
                        "Model envelope": pd.DataFrame(
                            [
                                {
                                    "trajectory": "lower",
                                    **lower_trace.to_frame().iloc[-1].to_dict(),
                                },
                                {"trajectory": "central", **trace.to_frame().iloc[-1].to_dict()},
                                {
                                    "trajectory": "upper",
                                    **upper_trace.to_frame().iloc[-1].to_dict(),
                                },
                            ]
                        ),
                    }
                ),
                "cold_chain_analysis.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            report_button(
                "PDF report",
                lambda: build_simulation_report(
                    label,
                    MODEL_LABELS[model_key],
                    trace,
                    settings,
                    audit_metadata={
                        "saved_run_id": saved_run_id,
                        "model_range_final_log10_cfu_per_g": [lower_final, upper_final],
                        "model_range_time_to_breach_hours": [
                            upper_trace.time_to_breach_hours,
                            lower_trace.time_to_breach_hours,
                        ],
                        "screening_notice": "Model interval envelope; not a calibrated probability or regulatory assessment.",  # noqa: E501
                    },
                ),
                "cold_chain_report.pdf",
                "pdf_sim",
                repr(sorted(settings.items()))
                + str(hash(tuple(round(p.temperature_celsius, 3) for p in points))),
            )
