from __future__ import annotations

import numpy as np
import pandas as pd


KNOWLEDGE_COLUMNS = [
    "ID", "Observed at", "Process stage", "Asset / mould", "Category",
    "Observation", "Evidence", "Reporter", "Validation",
]


def demo_frontline_knowledge() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ID": "FK-001", "Observed at": pd.Timestamp("2026-01-06 08:10"),
            "Process stage": "Changeover", "Asset / mould": "Line 1 / Mould A17",
            "Category": "Changeover", "Observation": "The next mould was not ready when the previous run ended.",
            "Evidence": "Voice note", "Reporter": "Operator 1", "Validation": "Validated",
        },
        {
            "ID": "FK-002", "Observed at": pd.Timestamp("2026-01-13 10:35"),
            "Process stage": "Start-up quality", "Asset / mould": "Line 1 / Mould A17",
            "Category": "Quality", "Observation": "Temperature and alignment were adjusted twice before the first acceptable batch.",
            "Evidence": "Photo and note", "Reporter": "Operator 2", "Validation": "Validated",
        },
        {
            "ID": "FK-003", "Observed at": pd.Timestamp("2026-01-20 14:20"),
            "Process stage": "Production", "Asset / mould": "Line 1",
            "Category": "Downtime", "Observation": "Several short stops were resolved locally and were not entered in the downtime log.",
            "Evidence": "Shift note", "Reporter": "Shift lead", "Validation": "Pending review",
        },
    ])[KNOWLEDGE_COLUMNS]


def structure_observation(
    existing: pd.DataFrame,
    process_stage: str,
    asset: str,
    category: str,
    observation: str,
    reporter: str,
    evidence: str,
) -> pd.DataFrame:
    next_id = f"FK-{len(existing) + 1:03d}"
    row = pd.DataFrame([{
        "ID": next_id,
        "Observed at": pd.Timestamp.now().floor("min"),
        "Process stage": process_stage,
        "Asset / mould": asset or "Not supplied",
        "Category": category,
        "Observation": observation.strip(),
        "Evidence": evidence,
        "Reporter": reporter.strip() or "Anonymous operator",
        "Validation": "Pending review",
    }])
    return pd.concat([existing, row], ignore_index=True)[KNOWLEDGE_COLUMNS]


def actionable_frontline_insights(knowledge: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "Changeover": ("Setup readiness constraint", "Pre-stage the next mould, tools and settings before the current run ends.", "Average changeover"),
        "Quality": ("Start-up process instability", "Record the approved start-up settings and check first-off quality before releasing the run.", "First Time Through"),
        "Downtime": ("Downtime may be under-recorded", "Add a simple micro-stop reason code and compare operator observations with downtime logs.", "Downtime rate"),
        "Materials": ("Material availability constraint", "Check material readiness at the production-release gate.", "Plan attainment"),
        "Maintenance": ("Recurring equipment condition signal", "Link the observation to maintenance history and define an inspection trigger.", "Downtime rate"),
        "Other": ("Frontline operational signal", "Review the observation with the process owner and define the next evidence check.", "Plan attainment"),
    }
    if knowledge.empty:
        return pd.DataFrame(columns=["Signal", "Frontline evidence", "Linked KPI", "Recommended action", "Source IDs"])
    valid = knowledge[knowledge["Validation"] == "Validated"].copy()
    rows = []
    for category, group in valid.groupby("Category", dropna=False):
        signal, action, kpi = rules.get(str(category), rules["Other"])
        rows.append({
            "Signal": signal,
            "Frontline evidence": " | ".join(group["Observation"].astype(str).tolist()),
            "Linked KPI": kpi,
            "Recommended action": action,
            "Source IDs": ", ".join(group["ID"].astype(str).tolist()),
        })
    return pd.DataFrame(rows)


def add_frontline_actions(plan: pd.DataFrame, insights: pd.DataFrame) -> pd.DataFrame:
    if insights.empty:
        return plan
    existing_actions = set(plan["Action"].astype(str)) if not plan.empty else set()
    additions = []
    for _, insight in insights.iterrows():
        if insight["Recommended action"] in existing_actions:
            continue
        additions.append({
            "ID": "", "Opportunity": insight["Signal"], "KPI": insight["Linked KPI"],
            "Baseline": np.nan, "Target": np.nan, "Unit": "", "Owner": "Unassigned",
            "Action": insight["Recommended action"], "Status": "Backlog", "Due date": pd.NaT,
            "Actual": np.nan, "Next review": pd.NaT,
            "Evidence / learning": f"Frontline evidence: {insight['Source IDs']}",
        })
    combined = pd.concat([plan, pd.DataFrame(additions)], ignore_index=True) if additions else plan.copy()
    combined["ID"] = [f"CI-{i + 1:03d}" for i in range(len(combined))]
    return combined
