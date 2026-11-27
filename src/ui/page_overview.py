"""Landing page: model context, data provenance and links to the analysis workspaces."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.datasets import REAL_GROWTH_CSV, real_data_available

MIN_ROWS, MIN_TEMPS = 20, 4
ORGANISMS = [
    "Listeria monocytogenes",
    "Salmonella spp.",
    "Escherichia coli",
    "Staphylococcus aureus",
    "Bacillus cereus",
    "Campylobacter jejuni",
]


def _coverage() -> pd.DataFrame:
    df = pd.read_csv(REAL_GROWTH_CSV)
    g = df.groupby("organism").agg(
        rows=("mu_max_per_h", "size"),
        temps=("temperature_celsius", "nunique"),
        t_min=("temperature_celsius", "min"),
        t_max=("temperature_celsius", "max"),
        ph_values=("ph", "count"),
        aw_values=("water_activity", "count"),
    )
    out = pd.DataFrame(
        {
            "Organism": g.index,
            "Growth records": g["rows"].to_numpy(),
            "Temperatures": g["temps"].to_numpy(),
            "Range (°C)": [f"{a:g} to {b:g}" for a, b in zip(g["t_min"], g["t_max"])],
            "pH records": g["ph_values"].to_numpy(),
            "aw records": g["aw_values"].to_numpy(),
            "Status": [
                "Limited" if r < MIN_ROWS or t < MIN_TEMPS else "Screening dataset"
                for r, t in zip(g["rows"], g["temps"])
            ],
        }
    )
    return out.sort_values("Growth records", ascending=False).reset_index(drop=True)


def _workspace_card(page, icon: str, title: str, body: str, label: str) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="ms-card-icon">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f"### {title}")
        st.markdown(f'<p class="ms-hint">{body}</p>', unsafe_allow_html=True)
        st.page_link(page, label=label, width="content")


def render(pages: dict) -> None:
    coverage = _coverage() if real_data_available() else pd.DataFrame()
    record_count = int(coverage["Growth records"].sum()) if not coverage.empty else 0
    profile_count = len(ORGANISMS)

    with st.container(key="ms-hero"):
        hero_copy, hero_stats = st.columns([2.5, 1], gap="medium", vertical_alignment="center")
        with hero_copy:
            st.markdown(
                '<div class="ms-section-kicker">MODEL CONTEXT / TRANSPARENT MVP</div>'
                "<h2>Make microbial growth risk legible before it becomes a line item.</h2>"
                "<p>MicrobeSim connects assay curves, environmental hurdles, and cold-chain telemetry "
                "in one reviewable workspace. Explore what the current build knows, what it assumes, "
                "and where your data enters the analysis.</p>",
                unsafe_allow_html=True,
            )
            st.page_link(
                pages["assays"], label="Open assay workspace", icon=":material/chevron_right:"
            )
        with hero_stats:
            st.markdown(
                f'<div class="ms-hero-metrics"><div class="ms-hero-stat"><strong>3</strong>'
                f'<span>ANALYSIS LANES</span></div><div class="ms-hero-stat"><strong>RK45</strong>'
                f'<span>DYNAMIC SOLVER</span></div><div class="ms-hero-stat"><strong>{profile_count}</strong>'
                f"<span>ORGANISM PROFILES</span></div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    with st.expander("New to MicrobeSim? Start with this 3-minute tour", expanded=False):
        st.markdown("""
            1. **Explore a measured curve** in *Assays & Primary Extraction* and compare fitted growth rates.
            2. **Choose a training source** in *Predict growth rate*. Real, synthetic and uploaded data
               have different evidence levels; the validation tab shows performance and split variability.
            3. **Replay a cold-chain scenario** in *Cold-Chain Simulator*, inspect the model range, and
               compare the result with a reference history.
            4. **Save and organize results** in *Saved runs*, where you can add project folders and notes.

            Start with bundled examples if you do not have data. Every result is a screening estimate;
            validate against product-specific evidence before making operational decisions.
            """)
        st.page_link(pages["assays"], label="Open the sample assay workflow")
        st.page_link(pages["simulator"], label="Explore ready-made cold-chain scenarios")
    if not st.session_state.get("onboarding_seen"):
        st.info(
            "First time here? Open the short guided tour below, then try the bundled assay or "
            "a ready-made cold-chain scenario. No upload is needed."
        )
        if st.button("Got it", key="dismiss_onboarding"):
            st.session_state["onboarding_seen"] = True
            st.rerun()

    st.markdown('<div class="ms-section-kicker">PROVENANCE LEDGER</div>', unsafe_allow_html=True)
    title_col, note_col = st.columns([2, 1], vertical_alignment="center")
    with title_col:
        st.markdown("## What is behind each prediction?")
    with note_col:
        st.markdown(
            '<div style="text-align:right;color:#FFC400;font-size:.65rem;font-weight:800;'
            'letter-spacing:.13em">SCREENING USE · NOT REGULATORY EVIDENCE</div>',
            unsafe_allow_html=True,
        )
    provenance_cards = st.columns(3, gap="medium")
    card_info = [
        (
            provenance_cards[0],
            "▤",
            "Your assay data",
            "Primary curves are fitted directly from the CSV you upload or the bundled sample curve. "
            "No assay rows are assumed to be organism-specific unless your data says so.",
        ),
        (
            provenance_cards[1],
            "◉",
            "Mechanistic boundaries",
            "Hurdle and cold-chain rates use temperature, pH and water-activity limits estimated "
            "from the selected training data.",
        ),
        (
            provenance_cards[2],
            "▤",
            "Benchmark layer",
            "Every growth-rate option identifies its source. Synthetic demo data is illustrative, "
            "not evidence of real-world model accuracy.",
        ),
    ]
    for column, icon, title, body in card_info:
        with column, st.container(border=True):
            st.markdown(f'<div class="ms-card-icon">{icon}</div>', unsafe_allow_html=True)
            st.markdown(f"### {title}")
            st.markdown(f'<p class="ms-hint">{body}</p>', unsafe_allow_html=True)

    organs_col, contracts_col = st.columns([3.25, 1], gap="medium")
    with organs_col, st.container(border=True):
        st.markdown("### Organisms you can check")
        st.caption("Curated food-safety profiles for organizing review runs")
        for organism in ORGANISMS:
            st.markdown(
                f'<span class="ms-organism-chip">{organism}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="ms-hint" style="margin-top:1rem">Organism names organize your assays and reports. '
            "They do not imply species-specific calibration; validate results against organism-specific data.</p>",
            unsafe_allow_html=True,
        )
    with contracts_col, st.container(border=True):
        st.markdown("### Data contracts")
        for name, columns in (
            ("kinetic_curve.csv", "time_hours, measurement"),
            ("cold_chain.csv", "timestamp_hours, temperature_celsius"),
            ("validation", "Non-negative values, monotonic time, bounded pH / aw"),
        ):
            st.markdown(
                f'<div class="ms-data-contract"><strong>{name}</strong>'
                f"<span>{columns}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown('<div class="ms-section-kicker">CHOOSE A WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown("## Move from evidence to decision")
    workspace_cols = st.columns(3, gap="medium")
    with workspace_cols[0]:
        _workspace_card(
            pages["assays"],
            "▤",
            "Assay extraction",
            "Fit one curve or rank a batch of isolates by μmax.",
            "Open workspace ↗",
        )
    with workspace_cols[1]:
        _workspace_card(
            pages["hurdles"],
            "◉",
            "Hurdle benchmarking",
            "Compare environmental growth-rate estimates and biological boundaries.",
            "Open workspace ↗",
        )
    with workspace_cols[2]:
        _workspace_card(
            pages["simulator"],
            "❄",
            "Cold-chain risk",
            "Replay telemetry and locate the first critical threshold breach.",
            "Open workspace ↗",
        )

    if not coverage.empty:
        with st.expander(f"Installed real data coverage · {record_count:,} growth records"):
            st.dataframe(coverage, hide_index=True, use_container_width=True)
            st.caption(
                "Coverage is descriptive, not a measure of model accuracy. Records may mix "
                "food matrices; sparse pH or water-activity coverage limits transfer to new products."
            )
    else:
        st.info(
            "No real dataset is installed, so the app is using synthetic demonstration data. "
            "No additional local real-data bundle is installed. The app is using synthetic "
            "demonstration data."
        )

    with st.expander("Read before relying on a number"):
        st.markdown(
            "- Real growth records are collected across many foods and can be scattered; check held-out model accuracy.\n"  # noqa: E501
            "- Synthetic entries are labelled and are for exploring the app, not safety decisions.\n"
            "- Campylobacter jejuni may have sparse usable growth records because some source data "
            "primarily describes survival rather than growth.\n"
            "- Read docs/REAL_DATA.md for data cleaning details and known organism-specific limitations."
        )
    st.markdown(
        '<div class="ms-footer">MICROBESIM / ANALYTICAL WORKSPACE'
        '<span style="float:right">LOCAL-FIRST ANALYSIS LEDGER</span></div>',
        unsafe_allow_html=True,
    )
