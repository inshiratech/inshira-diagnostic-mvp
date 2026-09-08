from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_FIELDS = [
    "date",
    "planned_units",
    "produced_units",
    "good_first_pass_units",
    "rejected_units",
]

OPTIONAL_FIELDS = [
    "reworked_units",
    "b_grade_units",
    "planned_minutes",
    "downtime_minutes",
    "changeover_minutes",
    "changeovers",
    "energy_kwh",
    "machine",
    "product",
    "shift",
]

ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

FIELD_LABELS = {
    "date": "Date",
    "planned_units": "Planned units",
    "produced_units": "Produced units",
    "good_first_pass_units": "Good first-pass units",
    "rejected_units": "Rejected units",
    "reworked_units": "Reworked units",
    "b_grade_units": "B-grade units",
    "planned_minutes": "Planned production minutes",
    "downtime_minutes": "Downtime minutes",
    "changeover_minutes": "Total changeover minutes",
    "changeovers": "Number of changeovers",
    "energy_kwh": "Energy consumption (kWh)",
    "machine": "Machine / line",
    "product": "Product / SKU",
    "shift": "Shift",
}

ALIASES = {
    "date": ["date", "day", "production_date"],
    "planned_units": ["planned_units", "plan", "target", "target_units", "planned_qty"],
    "produced_units": ["produced_units", "production", "actual", "actual_units", "output"],
    "good_first_pass_units": ["good_first_pass_units", "first_pass_good", "ftt_good", "right_first_time"],
    "rejected_units": ["rejected_units", "rejects", "scrap", "scrap_units"],
    "reworked_units": ["reworked_units", "rework", "rework_units"],
    "b_grade_units": ["b_grade_units", "b_grade", "seconds"],
    "planned_minutes": ["planned_minutes", "available_minutes", "scheduled_minutes"],
    "downtime_minutes": ["downtime_minutes", "downtime", "lost_minutes"],
    "changeover_minutes": ["changeover_minutes", "changeover_time", "setup_minutes"],
    "changeovers": ["changeovers", "number_of_changeovers", "setups"],
    "energy_kwh": ["energy_kwh", "kwh", "electricity_kwh"],
    "machine": ["machine", "line", "machine_id", "line_id"],
    "product": ["product", "sku", "style", "product_code"],
    "shift": ["shift", "shift_name", "crew"],
}


@dataclass
class DiagnosticResult:
    data: pd.DataFrame
    metrics: dict[str, float]
    opportunities: pd.DataFrame
    warnings: list[str]


DEFAULT_ASSUMPTIONS = {
    "working_weeks": 48.0,
    "unit_value": 8.0,
    "labour_cost_per_hour": 12.0,
    "material_cost_per_unit": 3.0,
    "energy_cost_per_kwh": 0.24,
}


def suggest_mapping(columns: Iterable[str]) -> dict[str, str | None]:
    normalized = {str(c).strip().lower().replace(" ", "_"): str(c) for c in columns}
    result: dict[str, str | None] = {}
    for field in ALL_FIELDS:
        result[field] = next((normalized[a] for a in ALIASES[field] if a in normalized), None)
    return result


def standardize(raw: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    selected = {source: target for target, source in mapping.items() if source}
    df = raw[list(selected)].rename(columns=selected).copy()
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = [f for f in ALL_FIELDS if f not in {"date", "machine", "product", "shift"}]
    for col in set(numeric).intersection(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and not np.isnan(denominator) else np.nan


def diagnose(df: pd.DataFrame) -> DiagnosticResult:
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))

    warnings: list[str] = []
    working = df.copy()
    invalid_dates = int(working["date"].isna().sum())
    if invalid_dates:
        warnings.append(f"{invalid_dates} row(s) have invalid dates and were excluded.")
        working = working.dropna(subset=["date"])

    numeric_cols = working.select_dtypes(include="number").columns
    negatives = int((working[numeric_cols] < 0).sum().sum()) if len(numeric_cols) else 0
    if negatives:
        warnings.append(f"{negatives} negative numeric value(s) need client validation.")

    start, end = working["date"].min(), working["date"].max()
    if pd.notna(start) and pd.notna(end):
        span = (end - start).days + 1
        if span > 92:
            warnings.append(f"The uploaded period spans {span} days; the agreed diagnostic window is approximately 12 weeks.")

    sums = working.sum(numeric_only=True)
    planned = float(sums.get("planned_units", 0))
    produced = float(sums.get("produced_units", 0))
    good = float(sums.get("good_first_pass_units", 0))
    rejected = float(sums.get("rejected_units", 0))
    reworked = float(sums.get("reworked_units", 0))
    b_grade = float(sums.get("b_grade_units", 0))
    planned_minutes = float(sums.get("planned_minutes", 0))
    downtime = float(sums.get("downtime_minutes", 0))
    changeover_minutes = float(sums.get("changeover_minutes", 0))
    changeovers = float(sums.get("changeovers", 0))
    energy = float(sums.get("energy_kwh", 0))

    metrics = {
        "plan_attainment": _safe_ratio(produced, planned),
        "ftt": _safe_ratio(good, produced),
        "rework_rate": _safe_ratio(reworked, produced),
        "rejection_rate": _safe_ratio(rejected, produced),
        "b_grade_rate": _safe_ratio(b_grade, produced),
        "downtime_rate": _safe_ratio(downtime, planned_minutes),
        "avg_changeover_minutes": _safe_ratio(changeover_minutes, changeovers),
        "energy_per_unit": _safe_ratio(energy, produced),
        "planned_units": planned,
        "produced_units": produced,
    }

    opportunity_rows = []

    def add(area: str, metric: str, current: float, reference: float, unit: str, rationale: str, priority: int):
        if np.isnan(current):
            return
        gap = max(0.0, reference - current) if unit == "%" else max(0.0, current - reference)
        opportunity_rows.append({
            "Priority score": priority if gap > 0 else 0,
            "Area": area,
            "Metric": metric,
            "Current": current,
            "Reference": reference,
            "Gap": gap,
            "Unit": unit,
            "Why investigate": rationale,
        })

    add("Quality", "First Time Through", metrics["ftt"] * 100, 91.0, "%", "Poor first-pass yield consumes labour and capacity through rework.", 5)
    add("Delivery", "Plan attainment", metrics["plan_attainment"] * 100, 95.0, "%", "A persistent plan gap can indicate constraints in scheduling, materials or process stability.", 4)
    add("Quality", "Rejection rate", metrics["rejection_rate"] * 100, 1.0, "lower %", "Rejected output creates material loss and may reveal recurring process defects.", 4)
    if not np.isnan(metrics["rework_rate"]):
        add("Quality", "Rework rate", metrics["rework_rate"] * 100, 5.0, "lower %", "Rework can hide lost capacity even when final recovery is high.", 5)
    if not np.isnan(metrics["downtime_rate"]):
        add("Reliability", "Downtime rate", metrics["downtime_rate"] * 100, 2.0, "lower %", "Validate whether reported downtime captures micro-stops and all unplanned losses.", 3)
    if not np.isnan(metrics["avg_changeover_minutes"]):
        add("Flow", "Average changeover", metrics["avg_changeover_minutes"], 30.0, "minutes", "Long or variable setups reduce available capacity and schedule flexibility.", 5)

    opportunities = pd.DataFrame(opportunity_rows)
    if not opportunities.empty:
        lower_is_better = opportunities["Unit"].isin(["lower %", "minutes"])
        opportunities.loc[lower_is_better, "Gap"] = (
            opportunities.loc[lower_is_better, "Current"] - opportunities.loc[lower_is_better, "Reference"]
        ).clip(lower=0)
        opportunities["Priority score"] = opportunities["Priority score"] * (opportunities["Gap"] > 0)
        opportunities = opportunities.sort_values(["Priority score", "Gap"], ascending=False).reset_index(drop=True)

    if "downtime_minutes" not in working:
        warnings.append("Downtime data was not supplied, so reliability losses cannot be assessed.")
    if "changeover_minutes" not in working or "changeovers" not in working:
        warnings.append("Changeover duration and count were not both supplied.")
    if "energy_kwh" not in working:
        warnings.append("Energy data was not supplied; energy intensity is excluded.")

    return DiagnosticResult(working, metrics, opportunities, warnings)


def quantify_opportunities(
    result: DiagnosticResult,
    assumptions: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Create an indicative annual value register from transparent assumptions.

    The calculation deliberately avoids claiming that every theoretical loss is
    recoverable. Users choose a recovery percentage in the app and validate the
    assumptions with the manufacturer before using the figures commercially.
    """
    a = DEFAULT_ASSUMPTIONS | (assumptions or {})
    data = result.data
    days = max(int(data["date"].dt.normalize().nunique()), 1)
    annual_factor = a["working_weeks"] * 5 / days
    m = result.metrics
    rows: list[dict[str, object]] = []

    def add(area: str, loss: str, annual_loss: float, basis: str, action: str, metric: str, baseline: float, target: float, unit: str):
        if np.isnan(annual_loss) or annual_loss <= 0:
            return
        rows.append({
            "Area": area,
            "Opportunity": loss,
            "Annual loss exposure (£)": round(annual_loss, 0),
            "Calculation basis": basis,
            "Recommended first action": action,
            "KPI": metric,
            "Baseline": round(baseline, 2),
            "Target": round(target, 2),
            "Unit": unit,
        })

    produced = m["produced_units"]
    ftt_gap_units = max(0.0, produced * (0.91 - m["ftt"])) if not np.isnan(m["ftt"]) else 0.0
    add("Quality", "FTT gap", ftt_gap_units * a["material_cost_per_unit"] * annual_factor,
        "Units below 91% FTT × material cost", "Run a Pareto of first-pass defect reasons and validate the top three causes.",
        "First Time Through", m["ftt"] * 100, 91.0, "%")

    if not np.isnan(m["rejection_rate"]):
        excess_rejects = max(0.0, produced * (m["rejection_rate"] - 0.01))
        add("Quality", "Excess rejection", excess_rejects * (a["material_cost_per_unit"] + a["unit_value"]) * annual_factor,
            "Rejects above 1% × material and lost unit value", "Stratify rejection by defect, product, line and shift.",
            "Rejection rate", m["rejection_rate"] * 100, 1.0, "%")

    if not np.isnan(m["avg_changeover_minutes"]):
        total_changeovers = float(data.get("changeovers", pd.Series(dtype=float)).sum())
        excess_hours = max(0.0, m["avg_changeover_minutes"] - 30.0) * total_changeovers / 60
        add("Flow", "Excess changeover time", excess_hours * a["labour_cost_per_hour"] * annual_factor,
            "Minutes above 30-minute reference × changeovers × labour rate", "Observe and separate internal from external setup tasks using SMED.",
            "Average changeover", m["avg_changeover_minutes"], 30.0, "min")

    if not np.isnan(m["downtime_rate"]):
        planned_minutes = float(data.get("planned_minutes", pd.Series(dtype=float)).sum())
        excess_hours = max(0.0, m["downtime_rate"] - 0.02) * planned_minutes / 60
        add("Reliability", "Excess recorded downtime", excess_hours * a["labour_cost_per_hour"] * annual_factor,
            "Downtime above 2% × labour rate", "Validate reason-code coverage, including micro-stops, before selecting countermeasures.",
            "Downtime rate", m["downtime_rate"] * 100, 2.0, "%")

    register = pd.DataFrame(rows)
    if register.empty:
        return register
    return register.sort_values("Annual loss exposure (£)", ascending=False).reset_index(drop=True)


def build_improvement_plan(register: pd.DataFrame) -> pd.DataFrame:
    """Seed an editable PDCA-style action register from the value register."""
    columns = ["ID", "Opportunity", "KPI", "Baseline", "Target", "Unit", "Owner", "Action", "Status", "Due date", "Actual", "Next review", "Evidence / learning"]
    if register.empty:
        return pd.DataFrame(columns=columns)
    plan = pd.DataFrame({
        "ID": [f"CI-{i + 1:03d}" for i in range(len(register))],
        "Opportunity": register["Opportunity"],
        "KPI": register["KPI"],
        "Baseline": register["Baseline"],
        "Target": register["Target"],
        "Unit": register["Unit"],
        "Owner": "Unassigned",
        "Action": register["Recommended first action"],
        "Status": "Backlog",
        "Due date": pd.NaT,
        "Actual": np.nan,
        "Next review": pd.NaT,
        "Evidence / learning": "",
    })
    return plan[columns]


def build_html_report(result: DiagnosticResult, factory_name: str, value_register: pd.DataFrame | None = None, improvement_plan: pd.DataFrame | None = None) -> str:
    m = result.metrics

    def pct(value: float) -> str:
        return "Not available" if np.isnan(value) else f"{value:.1%}"

    rows = "".join(
        f"<tr><td>{escape(str(row['Area']))}</td><td>{escape(str(row['Metric']))}</td>"
        f"<td>{row['Current']:.1f} {escape(str(row['Unit']))}</td>"
        f"<td>{escape(str(row['Why investigate']))}</td></tr>"
        for _, row in result.opportunities.head(5).iterrows()
    ) or "<tr><td colspan='4'>No rule-based gaps were identified in the supplied data.</td></tr>"
    warnings = "".join(f"<li>{escape(w)}</li>" for w in result.warnings) or "<li>No material data-quality warnings.</li>"
    value_rows = ""
    if value_register is not None and not value_register.empty:
        value_rows = "".join(
            f"<tr><td>{escape(str(r['Opportunity']))}</td><td>£{float(r['Annual loss exposure (£)']):,.0f}</td><td>{escape(str(r['Calculation basis']))}</td></tr>"
            for _, r in value_register.iterrows()
        )
    else:
        value_rows = "<tr><td colspan='3'>Insufficient data to quantify an indicative value.</td></tr>"
    plan_rows = ""
    if improvement_plan is not None and not improvement_plan.empty:
        plan_rows = "".join(
            f"<tr><td>{escape(str(r['ID']))}</td><td>{escape(str(r['Action']))}</td><td>{escape(str(r['Owner']))}</td><td>{escape(str(r['Status']))}</td><td>{escape(str(r['Next review']))}</td></tr>"
            for _, r in improvement_plan.iterrows()
        )
    else:
        plan_rows = "<tr><td colspan='5'>No improvement actions have been defined.</td></tr>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Inshira Diagnostic</title>
<style>body{{font-family:Arial,sans-serif;max-width:920px;margin:40px auto;color:#14213d;line-height:1.45}}h1,h2{{color:#0b6b5f}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:#f1f7f5;padding:16px;border-radius:8px}}.value{{font-size:24px;font-weight:bold}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}small{{color:#555}}</style></head>
<body><h1>Manufacturing Diagnostic</h1><p><strong>{escape(factory_name)}</strong></p>
<div class='grid'><div class='card'>Plan attainment<div class='value'>{pct(m['plan_attainment'])}</div></div><div class='card'>First Time Through<div class='value'>{pct(m['ftt'])}</div></div><div class='card'>Rejection rate<div class='value'>{pct(m['rejection_rate'])}</div></div></div>
<h2>Priority investigation areas</h2><table><tr><th>Area</th><th>Metric</th><th>Current</th><th>Why investigate</th></tr>{rows}</table>
<h2>Indicative annual loss exposure</h2><table><tr><th>Opportunity</th><th>Exposure</th><th>Basis</th></tr>{value_rows}</table>
<h2>Continuous improvement plan</h2><table><tr><th>ID</th><th>Action</th><th>Owner</th><th>Status</th><th>Next review</th></tr>{plan_rows}</table>
<h2>Data limitations</h2><ul>{warnings}</ul>
<p><small>Generated by Inshira Technologies. The findings are diagnostic indicators, not guaranteed financial savings. Validate operational and financial assumptions with the client before investment decisions.</small></p></body></html>"""
