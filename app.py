from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import ALL_FIELDS, FIELD_LABELS, REQUIRED_FIELDS, build_html_report, diagnose, standardize, suggest_mapping


st.set_page_config(page_title="Inshira Diagnostic", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stMetric"] {background: #f3f8f6; border: 1px solid #dcebe6; padding: 14px; border-radius: 10px;}
.small-note {color:#58636b;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

st.title("Inshira Manufacturing Diagnostic")
st.caption("Turn approximately 12 weeks of factory data into prioritised operational decisions.")

with st.sidebar:
    st.header("Diagnostic setup")
    factory_name = st.text_input("Factory name", "Bata Bangladesh — demonstration")
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

st.subheader("1. Confirm the data mapping")
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

st.subheader("2. Executive snapshot")
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

st.subheader("3. Priority investigation areas")
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

st.subheader("4. Data confidence")
if result.warnings:
    for warning in result.warnings:
        st.warning(warning)
else:
    st.success("No material data-quality warnings were detected.")

report = build_html_report(result, factory_name)
st.download_button("Download management report", report, file_name="inshira_diagnostic_report.html", mime="text/html", type="primary")

with st.expander("Review standardized data"):
    st.dataframe(result.data, width="stretch")

