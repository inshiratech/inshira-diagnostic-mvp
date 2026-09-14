from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import DEFAULT_ASSUMPTIONS, REQUIRED_FIELDS, build_html_report, build_improvement_plan, diagnose, quantify_opportunities
from ingestion import build_semantic_layer, demo_source_tables, read_uploaded_files
from knowledge import actionable_frontline_insights, add_frontline_actions, demo_frontline_knowledge, structure_observation


st.set_page_config(page_title="Inshira Intelligence Engine", page_icon="⚙️", layout="wide")
st.markdown("""
<style>
.stApp {background:#ffffff;color:#14213d}
[data-testid="stSidebar"] {background:#f3f8f6}
[data-testid="stSidebar"] * {color:#14213d}
.stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] {color:#14213d}
.block-container {padding-top:1.4rem;max-width:1350px}
[data-testid="stMetric"] {background:#f3f8f6;border:1px solid #dcebe6;padding:14px;border-radius:10px}
[data-testid="stMetric"] * {color:#14213d}
.stTabs [data-baseweb="tab-list"] {gap:1.1rem}
.stTabs [aria-selected="true"] {color:#0b6b5f!important;border-bottom-color:#0b6b5f!important}
.eyebrow {color:#0b6b5f;letter-spacing:.09em;font-weight:700;font-size:.78rem;text-transform:uppercase}
.flow {background:#eef6f3;border-left:4px solid #0b6b5f;padding:12px 16px;border-radius:6px;margin:.5rem 0 1.2rem}
.small-note {color:#58636b;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='eyebrow'>Inshira Technologies · Operational Intelligence</div>", unsafe_allow_html=True)
st.title("Manufacturing Intelligence & Improvement Engine")
st.caption("Turn fragmented existing factory records into verified operational intelligence and a repeatable improvement cycle.")
st.markdown("<div class='flow'><b>Raw evidence</b> → Ingestion layer → Semantic layer → Human approval → Loss & value intelligence → Continuous improvement</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Engagement setup")
    factory_name = st.text_input("Factory name", "Bata Bangladesh — demonstration")
    scope_name = st.text_input("Line / process / product family", "Selected production line")
    source_mode = st.radio("Data source", ["Interactive demo", "Upload factory data"])
    st.markdown("<div class='small-note'>Public prototype: use synthetic or non-confidential data only.</div>", unsafe_allow_html=True)

if source_mode == "Interactive demo":
    launch_col, copy_col = st.columns([1, 2])
    with launch_col:
        if st.button("Launch interactive demo", type="primary", width="stretch"):
            st.session_state.demo_active = True
            st.session_state.frontline_knowledge = demo_frontline_knowledge()
            st.session_state.data_approved = False
    with copy_col:
        st.info("Loads synthetic factory records and operator observations. Nothing confidential is required.")
    if not st.session_state.get("demo_active", False):
        st.stop()

st.header("1. Factory data ingestion")
st.write("Drop the existing exports and evidence here. Files do not need matching column names, sheet names or header positions.")
if source_mode == "Interactive demo":
    source_tables, evidence, ingestion_warnings = demo_source_tables(Path(__file__).parent / "sample_factory_data.csv")
    signature = "interactive-demo-v3"
else:
    uploaded = st.file_uploader(
        "Upload CSV, Excel, PDF, text or image files", type=["csv", "xlsx", "pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="CSV and Excel records are parsed now. PDFs and images are retained in the evidence register for human/AI extraction.",
    )
    if not uploaded:
        st.info("Upload the raw factory data pack to begin, or launch the interactive demo.")
        st.stop()
    source_tables, evidence, ingestion_warnings = read_uploaded_files(uploaded)
    signature = "|".join(f"{file.name}:{file.size}" for file in uploaded)
    if st.session_state.get("knowledge_signature") != signature:
        st.session_state.frontline_knowledge = demo_frontline_knowledge().iloc[0:0].copy()
        st.session_state.knowledge_signature = signature

ingestion = build_semantic_layer(source_tables, evidence, ingestion_warnings)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Files / tables read", len(ingestion.sources))
s2.metric("Fields interpreted", len(ingestion.field_register))
s3.metric("Semantic records", len(ingestion.semantic_data))
s4.metric("Evidence items pending", len(ingestion.evidence_register))

ingest_tabs = st.tabs(["Source inventory", "Interpreted fields", "Evidence register"])
with ingest_tabs[0]:
    inventory = pd.DataFrame([{
        "Source file": table.source, "Sheet / table": table.sheet, "Detected header row": table.header_row,
        "Rows ingested": len(table.data), "Fields recognised": sum(value is not None for value in table.mapping.values()),
    } for table in ingestion.sources])
    st.dataframe(inventory, hide_index=True, width="stretch")
with ingest_tabs[1]:
    st.dataframe(ingestion.field_register, hide_index=True, width="stretch")
with ingest_tabs[2]:
    if ingestion.evidence_register.empty:
        st.info("No document or image evidence was supplied.")
    else:
        st.dataframe(ingestion.evidence_register, hide_index=True, width="stretch")

st.header("2. Semantic layer")
st.write("The semantic layer translates different source labels into one operational language. Review and correct the interpreted records before releasing them for analysis.")
if st.session_state.get("semantic_signature") != signature:
    st.session_state.semantic_data = ingestion.semantic_data.copy()
    st.session_state.semantic_signature = signature
    st.session_state.data_approved = False

semantic_data = st.data_editor(
    st.session_state.semantic_data, hide_index=True, width="stretch", num_rows="dynamic",
    column_config={"date": st.column_config.DateColumn("Date", format="DD MMM YYYY")}, key="semantic_editor",
)
st.session_state.semantic_data = semantic_data
with st.expander("Validation findings", expanded=bool(ingestion.warnings)):
    if ingestion.warnings:
        for warning in ingestion.warnings:
            st.warning(warning)
    else:
        st.success("All minimum diagnostic fields were found in the semantic dataset.")

st.header("3. Frontline knowledge and human review")
st.write("Operators and supervisors add what the systems do not capture. Each observation stays pending until a human validates it.")

with st.form("frontline_observation", clear_on_submit=True):
    f1, f2, f3 = st.columns(3)
    process_stage = f1.selectbox("Process stage", ["Changeover", "Start-up quality", "Production", "Maintenance", "Inspection", "Other"])
    asset = f2.text_input("Line, machine or mould", placeholder="Line 1 / Mould A17")
    category = f3.selectbox("Category", ["Changeover", "Quality", "Downtime", "Materials", "Maintenance", "Other"])
    observation = st.text_area("Frontline observation", placeholder="What happened, what was different, and what did you have to do?")
    reporter = st.text_input("Reported by", placeholder="Operator, technician or shift lead")
    evidence_file = st.file_uploader("Optional photo or voice note", type=["png", "jpg", "jpeg", "wav", "mp3", "m4a"])
    add_observation = st.form_submit_button("Add frontline observation", type="primary")

if add_observation:
    if not observation.strip():
        st.warning("Add an observation before submitting.")
    else:
        evidence_label = f"Attached: {evidence_file.name}" if evidence_file else "Text observation"
        st.session_state.frontline_knowledge = structure_observation(
            st.session_state.frontline_knowledge, process_stage, asset, category,
            observation, reporter, evidence_label,
        )
        st.success("Observation added for manager review.")

knowledge = st.data_editor(
    st.session_state.frontline_knowledge,
    hide_index=True,
    width="stretch",
    num_rows="dynamic",
    column_config={
        "Observed at": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
        "Validation": st.column_config.SelectboxColumn(options=["Pending review", "Validated", "Rejected"]),
    },
    key="frontline_editor",
)
st.session_state.frontline_knowledge = knowledge
validated_count = int((knowledge["Validation"] == "Validated").sum()) if not knowledge.empty else 0
pending_count = int((knowledge["Validation"] == "Pending review").sum()) if not knowledge.empty else 0
k1, k2 = st.columns(2)
k1.metric("Validated observations", validated_count)
k2.metric("Awaiting review", pending_count)

reviewer = st.text_input("Reviewed by", placeholder="Manager or process owner validating the data and frontline knowledge")
approved = st.checkbox(
    "I have checked the semantic data, frontline observations and known limitations. Release the actionable insights.",
    value=st.session_state.get("data_approved", False),
)
st.session_state.data_approved = approved
missing_required = [field for field in REQUIRED_FIELDS if field not in semantic_data or semantic_data[field].isna().all()]
if missing_required:
    st.error("The dataset cannot be passed because minimum fields are missing: " + ", ".join(missing_required))
    st.stop()
if not approved or not reviewer.strip():
    st.info("Human approval is required before actionable insights, the loss register and the improvement engine are released.")
    st.stop()

try:
    result = diagnose(semantic_data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

frontline_insights = actionable_frontline_insights(knowledge)
st.success(f"Data and frontline knowledge approved by {reviewer.strip()}. Actionable insights are now released.")
st.header("4. Actionable insights and continuous improvement")
tabs = st.tabs(["Actionable insights", "Loss & value register", "Improvement engine", "Data confidence"])

with tabs[0]:
    m = result.metrics

    def p(value):
        return "N/A" if np.isnan(value) else f"{value:.1%}"

    def n(value, suffix=""):
        return "N/A" if np.isnan(value) else f"{value:,.1f}{suffix}"

    metrics = st.columns(4)
    metrics[0].metric("Plan attainment", p(m["plan_attainment"]))
    metrics[1].metric("First Time Through", p(m["ftt"]), delta=None if np.isnan(m["ftt"]) else f"{(m['ftt'] - .91) * 100:+.1f} pp vs 91%")
    metrics[2].metric("Rejection rate", p(m["rejection_rate"]))
    metrics[3].metric("Average changeover", n(m["avg_changeover_minutes"], " min"))
    metrics2 = st.columns(4)
    metrics2[0].metric("Produced units", f"{m['produced_units']:,.0f}")
    metrics2[1].metric("Rework rate", p(m["rework_rate"]))
    metrics2[2].metric("Downtime rate", p(m["downtime_rate"]))
    metrics2[3].metric("Energy per unit", n(m["energy_per_unit"], " kWh"))
    daily = result.data.groupby("date", as_index=False)[["planned_units", "produced_units", "good_first_pass_units"]].sum()
    fig = px.line(daily, x="date", y=["planned_units", "produced_units", "good_first_pass_units"], labels={"value": "Units", "variable": "Measure", "date": "Date"})
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    display = result.opportunities[result.opportunities["Priority score"] > 0].head(5).copy()
    st.subheader("Priority investigation areas")
    if display.empty:
        st.success("No rule-based performance gaps were identified.")
    else:
        display["Current"] = display.apply(lambda row: f"{row['Current']:.1f} {row['Unit']}", axis=1)
        display["Reference"] = display.apply(lambda row: f"{row['Reference']:.1f} {row['Unit']}", axis=1)
        st.dataframe(display[["Area", "Metric", "Current", "Reference", "Why investigate"]], hide_index=True, width="stretch")
    st.subheader("What frontline knowledge adds")
    if frontline_insights.empty:
        st.info("No validated frontline observations are available yet.")
    else:
        st.dataframe(frontline_insights, hide_index=True, width="stretch")

with tabs[1]:
    st.subheader("Quantify the case for action")
    st.caption("Indicative annual loss exposure, not guaranteed savings. Every assumption remains visible and client-validated.")
    a1, a2, a3, a4 = st.columns(4)
    assumptions = {
        "working_weeks": a1.number_input("Operating weeks / year", 1, 52, int(DEFAULT_ASSUMPTIONS["working_weeks"])),
        "unit_value": a2.number_input("Value per unit (£)", 0.0, value=DEFAULT_ASSUMPTIONS["unit_value"], step=0.5),
        "labour_cost_per_hour": a3.number_input("Labour cost / hour (£)", 0.0, value=DEFAULT_ASSUMPTIONS["labour_cost_per_hour"], step=0.5),
        "material_cost_per_unit": a4.number_input("Material cost / unit (£)", 0.0, value=DEFAULT_ASSUMPTIONS["material_cost_per_unit"], step=0.25),
    }
    recovery = st.slider("Realistically recoverable share", 5, 50, 20, 5)
    register = quantify_opportunities(result, assumptions)
    if register.empty:
        st.info("The approved data does not support a quantified value register yet.")
    else:
        gross = float(register["Annual loss exposure (£)"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual loss exposure", f"£{gross:,.0f}")
        c2.metric("Recoverable opportunity", f"£{gross * recovery / 100:,.0f}")
        c3.metric("Opportunities quantified", str(len(register)))
        st.dataframe(register, hide_index=True, width="stretch")

with tabs[2]:
    st.subheader("Continuous Improvement Engine")
    st.caption("Assign actions, update actual KPI results, record learning and set the next review.")
    knowledge_state = tuple(zip(knowledge["ID"].astype(str), knowledge["Validation"].astype(str))) if not knowledge.empty else ()
    plan_signature = (signature, factory_name, scope_name, knowledge_state)
    if "improvement_plan" not in st.session_state or st.session_state.get("plan_scope") != plan_signature:
        st.session_state.improvement_plan = add_frontline_actions(build_improvement_plan(register), frontline_insights)
        st.session_state.plan_scope = plan_signature
    plan = st.data_editor(
        st.session_state.improvement_plan, hide_index=True, width="stretch", num_rows="dynamic",
        column_config={
            "Status": st.column_config.SelectboxColumn(options=["Backlog", "Planned", "In progress", "Blocked", "Verify", "Complete"]),
            "Due date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Next review": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Actual": st.column_config.NumberColumn(format="%.2f"),
        }, key="ci_editor",
    )
    st.session_state.improvement_plan = plan
    k1, k2, k3 = st.columns(3)
    k1.metric("Active actions", int(plan["Status"].isin(["Planned", "In progress", "Verify"]).sum()) if not plan.empty else 0)
    k2.metric("Completed", int((plan["Status"] == "Complete").sum()) if not plan.empty else 0)
    k3.metric("Blocked", int((plan["Status"] == "Blocked").sum()) if not plan.empty else 0)
    st.download_button("Download improvement register (CSV)", plan.to_csv(index=False), "inshira_continuous_improvement_register.csv", "text/csv")

with tabs[3]:
    st.subheader("Data confidence and traceability")
    st.write(f"Approved by: **{reviewer.strip()}**")
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    else:
        st.success("No additional diagnostic data-quality warnings were detected.")
    with st.expander("Review approved semantic data"):
        st.dataframe(result.data, width="stretch")

st.divider()
report = build_html_report(result, factory_name, register, st.session_state.improvement_plan)
st.download_button("Download management report", report, "inshira_manufacturing_intelligence_report.html", "text/html", type="primary")
st.caption(f"Scope: {scope_name}. Approved by {reviewer.strip()}. Outputs are decision support and require operational and financial validation.")
