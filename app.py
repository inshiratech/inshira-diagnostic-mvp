from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import (
    ALL_FIELDS, DEFAULT_ASSUMPTIONS, FIELD_LABELS, REQUIRED_FIELDS,
    build_html_report, build_improvement_plan, diagnose,
    quantify_opportunities, standardize, suggest_mapping,
)


st.set_page_config(page_title="Inshira Diagnostic", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1320px;}
[data-testid="stMetric"] {background: #f3f8f6; border: 1px solid #dcebe6; padding: 14px; border-radius: 10px;}
.stTabs [data-baseweb="tab-list"] {gap: 1.2rem;}
.stTabs [aria-selected="true"] {color:#0b6b5f!important; border-bottom-color:#0b6b5f!important;}
.small-note {color:#58636b;font-size:.9rem}
.eyebrow {color:#0b6b5f;letter-spacing:.09em;font-weight:700;font-size:.78rem;text-transform:uppercase}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='eyebrow'>Inshira Technologies · Operational Intelligence</div>", unsafe_allow_html=True)
st.title("Manufacturing Improvement Engine")
st.caption("Turn approximately 12 weeks of existing factory data into quantified priorities, owned actions and a repeatable improvement cycle.")

with st.sidebar:
    st.header("Engagement setup")
    factory_name = st.text_input("Factory name", "Bata Bangladesh — demonstration")
    scope_name = st.text_input("Line / process / product family", "Selected production line")
    source = st.radio("Data source", ["Use demonstration data", "Upload client data"])
    st.markdown("<div class='small-note'>Client files are processed for this session. Do not use the public prototype for confidential data.</div>", unsafe_allow_html=True)

if source == "Use demonstration data":
    raw = pd.read_csv(Path(__file__).parent / "sample_factory_data.csv")
else:
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if not uploaded:
        st.info("Upload a client file to continue, or choose demonstration data from the sidebar.")
        st.stop()
    raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)

st.subheader("1. Connect the data")
suggested = suggest_mapping(raw.columns)
options = ["— Not supplied —"] + list(raw.columns)
mapping = {}
cols = st.columns(3)
for idx, field in enumerate(ALL_FIELDS):
    default = options.index(suggested[field]) if suggested[field] in options else 0
    selected = cols[idx % 3].selectbox(
        FIELD_LABELS[field] + (" *" if field in REQUIRED_FIELDS else ""),
        options,
        index=default,
        key=f"map_{field}",
    )
    mapping[field] = None if selected == "— Not supplied —" else selected

missing = [FIELD_LABELS[f] for f in REQUIRED_FIELDS if not mapping[f]]
if missing:
    st.warning("Map all required fields before running the diagnostic: " + ", ".join(missing))
    st.stop()

try:
    result = diagnose(standardize(raw, mapping))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

tabs = st.tabs(["Executive view", "Loss & value register", "Improvement engine", "Data confidence"])

with tabs[0]:
    st.subheader("Executive snapshot")
    m = result.metrics

    def p(value):
        return "N/A" if np.isnan(value) else f"{value:.1%}"

    def n(value, suffix=""):
        return "N/A" if np.isnan(value) else f"{value:,.1f}{suffix}"

    metrics = st.columns(4)
    metrics[0].metric("Plan attainment", p(m["plan_attainment"]))
    metrics[1].metric("First Time Through", p(m["ftt"]), delta=None if np.isnan(m["ftt"]) else f"{(m['ftt']-.91)*100:+.1f} pp vs 91%")
    metrics[2].metric("Rejection rate", p(m["rejection_rate"]))
    metrics[3].metric("Average changeover", n(m["avg_changeover_minutes"], " min"))

    metrics2 = st.columns(4)
    metrics2[0].metric("Produced units", f"{m['produced_units']:,.0f}")
    metrics2[1].metric("Rework rate", p(m["rework_rate"]))
    metrics2[2].metric("Downtime rate", p(m["downtime_rate"]))
    metrics2[3].metric("Energy per unit", n(m["energy_per_unit"], " kWh"))

    daily = result.data.groupby("date", as_index=False)[["planned_units", "produced_units", "good_first_pass_units"]].sum()
    fig = px.line(daily, x="date", y=["planned_units", "produced_units", "good_first_pass_units"], labels={"value":"Units", "variable":"Measure", "date":"Date"})
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Priority investigation areas")
    display = result.opportunities.copy()
    if display.empty:
        st.success("No rule-based performance gaps were identified in the supplied fields.")
    else:
        display = display[display["Priority score"] > 0].head(5)
        if display.empty:
            st.success("No rule-based performance gaps were identified in the supplied fields.")
        else:
            display["Current"] = display.apply(lambda r: f"{r['Current']:.1f} {r['Unit']}", axis=1)
            display["Reference"] = display.apply(lambda r: f"{r['Reference']:.1f} {r['Unit']}", axis=1)
            st.dataframe(display[["Area", "Metric", "Current", "Reference", "Why investigate"]], hide_index=True, width="stretch")

with tabs[1]:
    st.subheader("Quantify the case for action")
    st.caption("These figures show indicative annual loss exposure, not guaranteed savings. Validate every assumption with the client.")
    a1, a2, a3, a4 = st.columns(4)
    assumptions = {
        "working_weeks": a1.number_input("Operating weeks / year", 1, 52, int(DEFAULT_ASSUMPTIONS["working_weeks"])),
        "unit_value": a2.number_input("Value per unit (£)", 0.0, value=DEFAULT_ASSUMPTIONS["unit_value"], step=0.5),
        "labour_cost_per_hour": a3.number_input("Labour cost / hour (£)", 0.0, value=DEFAULT_ASSUMPTIONS["labour_cost_per_hour"], step=0.5),
        "material_cost_per_unit": a4.number_input("Material cost / unit (£)", 0.0, value=DEFAULT_ASSUMPTIONS["material_cost_per_unit"], step=0.25),
    }
    recovery = st.slider("Realistically recoverable share", 5, 50, 20, 5, help="Applied only to the management opportunity figure; the register retains gross loss exposure.")
    register = quantify_opportunities(result, assumptions)
    if register.empty:
        st.info("The supplied data does not support a quantified value register yet.")
    else:
        gross = float(register["Annual loss exposure (£)"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual loss exposure", f"£{gross:,.0f}")
        c2.metric("Recoverable opportunity", f"£{gross * recovery / 100:,.0f}")
        c3.metric("Opportunities quantified", str(len(register)))
        st.dataframe(register, hide_index=True, width="stretch")

with tabs[2]:
    st.subheader("Continuous Improvement Engine")
    st.caption("Assign each action, update the actual KPI, record learning and set the next review. Download the register to carry the cycle into weekly operations reviews.")
    if "improvement_plan" not in st.session_state or st.session_state.get("plan_scope") != (factory_name, scope_name):
        st.session_state.improvement_plan = build_improvement_plan(register)
        st.session_state.plan_scope = (factory_name, scope_name)
    plan = st.data_editor(
        st.session_state.improvement_plan,
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "Status": st.column_config.SelectboxColumn(options=["Backlog", "Planned", "In progress", "Blocked", "Verify", "Complete"]),
            "Due date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Next review": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Actual": st.column_config.NumberColumn(format="%.2f"),
        },
        key="ci_editor",
    )
    st.session_state.improvement_plan = plan
    active = int(plan["Status"].isin(["Planned", "In progress", "Verify"]).sum()) if not plan.empty else 0
    complete = int((plan["Status"] == "Complete").sum()) if not plan.empty else 0
    blocked = int((plan["Status"] == "Blocked").sum()) if not plan.empty else 0
    k1, k2, k3 = st.columns(3)
    k1.metric("Active actions", active)
    k2.metric("Completed", complete)
    k3.metric("Blocked", blocked)
    st.download_button("Download improvement register (CSV)", plan.to_csv(index=False), "inshira_continuous_improvement_register.csv", "text/csv")

with tabs[3]:
    st.subheader("Data confidence and traceability")
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    else:
        st.success("No material data-quality warnings were detected.")
    with st.expander("Review standardized data"):
        st.dataframe(result.data, width="stretch")

st.divider()
report = build_html_report(result, factory_name, register, st.session_state.improvement_plan)
st.download_button("Download management report", report, file_name="inshira_manufacturing_improvement_report.html", mime="text/html", type="primary")
st.caption(f"Scope: {scope_name}. Indicators are decision-support outputs and must be validated with operational and financial owners.")
