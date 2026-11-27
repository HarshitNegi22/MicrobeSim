"""Persistent history for completed fitting, prediction and simulation workflows."""

from __future__ import annotations

import json
from html import escape
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.ledger import (
    clear_runs,
    count_run_summaries,
    count_runs_by_type,
    create_project,
    delete_run,
    get_run,
    list_projects,
    list_run_summaries,
    update_run_metadata,
)
from src.reporting import df_to_csv_bytes
from src.ui.theme import page_header, readout

TYPE_LABELS = {
    "all": "All analyses",
    "assay": "Assay fits",
    "prediction": "Hurdle predictions",
    "simulation": "Cold-chain simulations",
}
TYPE_ICONS = {"assay": "▤", "prediction": "◉", "simulation": "❄"}
MAX_LEDGER_EXPORT_ROWS = 10_000


def _run_label(run: Dict[str, object]) -> str:
    return f"{run['title']}  ·  {run['created_at']}  ·  {str(run['run_id'])[:8]}"


def render() -> None:
    user_id = st.session_state["user_id"]
    page_header(
        "Saved analysis runs",
        "Only your completed fits, hurdle predictions and cold-chain simulations appear in this "
        "account's private analysis ledger.",
        "WORKSPACE / AUDIT HISTORY",
    )
    counts = count_runs_by_type(user_id)
    readout(
        [
            ("Saved records", str(sum(counts.values())), None),
            ("Assay fits", str(counts["assay"]), None),
            ("Hurdle predictions", str(counts["prediction"]), None),
            ("Simulations", str(counts["simulation"]), None),
        ]
    )
    st.caption("Account-scoped SQLite storage · identical reruns are deduplicated")

    if not sum(counts.values()):
        st.info(
            "No completed analyses have been saved yet. Finish an assay fit, hurdle prediction or simulation to start the ledger."  # noqa: E501
        )
        return

    filter_col, search_col = st.columns([1, 2], gap="medium")
    with filter_col:
        kind = st.selectbox("Filter by workflow", list(TYPE_LABELS), format_func=TYPE_LABELS.get)
    with search_col:
        query = st.text_input(
            "Search saved runs",
            placeholder="Search titles, summaries or result values",
        )

    projects = list_projects(user_id)
    project_options = {"All projects": None, "Unassigned": "unassigned"}
    project_options.update({project["name"]: project["project_id"] for project in projects})
    project_name = st.selectbox("Project folder", list(project_options))
    selected_project_id = project_options[project_name]
    if selected_project_id == "unassigned":
        selected_project_id = ""
    with st.expander("Create a project folder"):
        with st.form("new_project_form"):
            new_project_name = st.text_input("Project name", max_chars=80)
            create_project_submitted = st.form_submit_button("Create project")
        if create_project_submitted:
            try:
                create_project(user_id, new_project_name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    selected_type = None if kind == "all" else kind
    project_filter = selected_project_id if isinstance(selected_project_id, str) else None
    total = count_run_summaries(user_id, selected_type, query, project_filter)
    if not total:
        st.info("No saved analyses match those filters.")
        return

    page_size = 50
    page_count = (total + page_size - 1) // page_size
    current_page = 1
    if page_count > 1:
        current_page = st.selectbox(
            "History page",
            list(range(1, page_count + 1)),
            format_func=lambda page: f"Page {page} of {page_count}",
            key=f"run-history-page-{kind}-{query}",
        )
    visible = list_run_summaries(
        user_id,
        selected_type,
        query,
        limit=page_size,
        offset=(current_page - 1) * page_size,
        project_id=project_filter,
    )

    st.markdown("### Analysis ledger")
    with st.container(border=True):
        first_record = (current_page - 1) * page_size + 1
        last_record = first_record + len(visible) - 1
        st.caption(
            f"Showing {first_record}–{last_record} of {total} matching records · newest first"
        )
        for run in visible:
            st.markdown(
                f'<div class="ms-run-row">'
                f'<div><div class="ms-run-title">{escape(TYPE_ICONS[run["run_type"]])} '
                f'&nbsp;{escape(run["title"])}</div>'
                f'<div class="ms-run-sub">{escape(run["created_at"])} · '
                f'{escape(run["summary"])}</div></div>'
                f'<div class="ms-run-metric">{escape(run["metric_value"])}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    records_by_label = {_run_label(run): run for run in visible}
    selected_label = st.selectbox("Inspect a saved run", list(records_by_label))
    selected_summary = records_by_label[selected_label]
    selected = get_run(user_id, selected_summary["run_id"])
    if selected is None:
        st.warning("That saved record is no longer available. Refresh the history to continue.")
        return
    with st.container(border=True):
        st.markdown(f"### {selected['title']}")
        st.caption(
            f"{TYPE_LABELS[selected['run_type']]} · {selected['created_at']} · "
            f"{selected['metric_label']}: {selected['metric_value']}"
        )
        st.write(selected["summary"])
        with st.expander("Inputs and recorded results"):
            st.json(selected["payload"], expanded=True)

        project_map = {project["name"]: project["project_id"] for project in projects}
        project_map = {"Unassigned": None, **project_map}
        current_project_name = next(
            (
                name
                for name, project_id in project_map.items()
                if project_id == selected.get("project_id")
            ),
            "Unassigned",
        )
        with st.form(f"metadata_{selected['run_id']}"):
            chosen_project_name = st.selectbox(
                "Add to project",
                list(project_map),
                index=list(project_map).index(current_project_name),
            )
            saved_notes = st.text_area(
                "Project notes",
                value=selected.get("notes", ""),
                max_chars=2000,
                help="Private to this account; notes are included in search and CSV export.",
            )
            metadata_submitted = st.form_submit_button("Save project and notes")
        if metadata_submitted:
            update_run_metadata(
                user_id,
                selected["run_id"],
                project_map[chosen_project_name],
                saved_notes,
            )
            st.success("Project and notes saved.")

        download_json = json.dumps(selected, indent=2, sort_keys=True, default=str).encode("utf-8")
        st.download_button(
            "Download this run (JSON)",
            download_json,
            file_name=f"microbesim_run_{selected['run_id'][:8]}.json",
            mime="application/json",
            key=f"download_{selected['run_id']}",
        )
        confirm_delete = st.checkbox(
            "Confirm deletion of this saved run", key=f"confirm_{selected['run_id']}"
        )
        if st.button(
            "Delete selected run", disabled=not confirm_delete, key=f"delete_{selected['run_id']}"
        ):
            delete_run(user_id, selected["run_id"])
            st.rerun()

    with st.expander("Export or clear the local ledger"):
        export_limit = min(total, MAX_LEDGER_EXPORT_ROWS)
        if total > MAX_LEDGER_EXPORT_ROWS:
            st.warning(
                f"Ledger export is limited to the newest {MAX_LEDGER_EXPORT_ROWS:,} "
                "matching records to protect memory. Apply filters to export a smaller subset."
            )
        export_rows: List[Dict[str, object]] = list_run_summaries(
            user_id,
            selected_type,
            query,
            limit=export_limit,
            project_id=project_filter,
        )
        rows: List[Dict[str, object]] = [
            {
                "run_id": run["run_id"],
                "run_type": run["run_type"],
                "title": run["title"],
                "summary": run["summary"],
                "metric_label": run["metric_label"],
                "metric_value": run["metric_value"],
                "created_at": run["created_at"],
                "project": next(
                    (
                        project["name"]
                        for project in projects
                        if project["project_id"] == run.get("project_id")
                    ),
                    "",
                ),
                "notes": run.get("notes", ""),
            }
            for run in export_rows
        ]
        csv_data = df_to_csv_bytes(pd.DataFrame(rows))
        st.download_button(
            "Export filtered ledger (CSV)",
            csv_data,
            file_name="microbesim_analysis_ledger.csv",
            mime="text/csv",
        )
        confirm_clear = st.checkbox("Confirm deletion of every saved run")
        if st.button("Clear my local history", disabled=not confirm_clear):
            clear_runs(user_id)
            st.rerun()
