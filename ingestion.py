from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from analytics import ALL_FIELDS, ALIASES, FIELD_LABELS, REQUIRED_FIELDS, suggest_mapping


DIMENSIONS = ["date", "machine", "product", "shift"]
NUMERIC_FIELDS = [field for field in ALL_FIELDS if field not in DIMENSIONS]


@dataclass
class SourceTable:
    source: str
    sheet: str
    data: pd.DataFrame
    mapping: dict[str, str | None]
    header_row: int


@dataclass
class IngestionResult:
    sources: list[SourceTable]
    semantic_data: pd.DataFrame
    field_register: pd.DataFrame
    evidence_register: pd.DataFrame
    warnings: list[str]


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _header_score(row: Iterable[object]) -> int:
    known = {alias for aliases in ALIASES.values() for alias in aliases}
    return sum(_normalise(value) in known for value in row if pd.notna(value))


def clean_unformatted_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Find a likely header row and remove empty/noise rows and columns."""
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        return raw, 0
    search_rows = min(15, len(raw))
    scores = [_header_score(raw.iloc[i].tolist()) for i in range(search_rows)]
    header_pos = int(np.argmax(scores)) if max(scores, default=0) > 0 else 0
    headers = []
    seen: dict[str, int] = {}
    for i, value in enumerate(raw.iloc[header_pos].tolist()):
        base = str(value).strip() if pd.notna(value) and str(value).strip() else f"unnamed_{i + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        headers.append(base if count == 0 else f"{base}_{count + 1}")
    data = raw.iloc[header_pos + 1:].copy()
    data.columns = headers
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return data, header_pos + 1


def _read_csv(payload: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(BytesIO(payload), header=None, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The CSV encoding could not be read.")


def read_uploaded_files(files: Iterable[object]) -> tuple[list[SourceTable], pd.DataFrame, list[str]]:
    tables: list[SourceTable] = []
    evidence: list[dict[str, object]] = []
    warnings: list[str] = []
    for uploaded in files:
        name = Path(uploaded.name).name
        suffix = Path(name).suffix.lower()
        payload = uploaded.getvalue()
        if suffix == ".csv":
            workbooks = {"CSV": _read_csv(payload)}
        elif suffix in {".xlsx", ".xls"}:
            workbooks = pd.read_excel(BytesIO(payload), sheet_name=None, header=None)
        else:
            evidence.append({"File": name, "Type": suffix.lstrip(".").upper(), "Status": "Retained as supporting evidence", "Extraction": "Human/AI extraction required"})
            continue
        for sheet, raw in workbooks.items():
            cleaned, header_row = clean_unformatted_table(raw)
            if cleaned.empty:
                warnings.append(f"{name} · {sheet}: no usable rows were found.")
                continue
            mapping = suggest_mapping(cleaned.columns)
            tables.append(SourceTable(name, str(sheet), cleaned, mapping, header_row))
    return tables, pd.DataFrame(evidence), warnings


def demo_source_tables(sample_path: Path) -> tuple[list[SourceTable], pd.DataFrame, list[str]]:
    sample = pd.read_csv(sample_path)
    production = sample[["date", "machine", "product", "shift", "planned_units", "produced_units", "planned_minutes"]].rename(
        columns={"date": "Production Date", "machine": "Line ID", "product": "Style", "planned_units": "Target Units", "produced_units": "Actual Units", "planned_minutes": "Scheduled Minutes"}
    )
    quality = sample[["date", "machine", "good_first_pass_units", "reworked_units", "rejected_units", "b_grade_units"]].rename(
        columns={"date": "Day", "machine": "Machine", "good_first_pass_units": "First Pass Good", "reworked_units": "Rework Units", "rejected_units": "Scrap Units", "b_grade_units": "Seconds"}
    )
    losses = sample[["date", "machine", "downtime_minutes", "changeover_minutes", "changeovers", "energy_kwh"]].rename(
        columns={"date": "Date", "machine": "Line", "downtime_minutes": "Lost Minutes", "changeover_minutes": "Setup Minutes", "changeovers": "Setups", "energy_kwh": "Electricity kWh"}
    )
    tables = []
    for source, sheet, data, header in [
        ("production_export.xlsx", "Output", production, 3),
        ("quality_log.csv", "CSV", quality, 1),
        ("losses_and_energy.xlsx", "Weekly Log", losses, 5),
    ]:
        tables.append(SourceTable(source, sheet, data, suggest_mapping(data.columns), header))
    evidence = pd.DataFrame([{"File": "operator_notes_scan.pdf", "Type": "PDF", "Status": "Retained as supporting evidence", "Extraction": "Human/AI extraction required"}])
    return tables, evidence, []


def build_semantic_layer(tables: list[SourceTable], evidence: pd.DataFrame | None = None, initial_warnings: list[str] | None = None) -> IngestionResult:
    fragments: list[pd.DataFrame] = []
    fields: list[dict[str, object]] = []
    warnings = list(initial_warnings or [])
    for table in tables:
        selected = {source: canonical for canonical, source in table.mapping.items() if source}
        if not selected:
            warnings.append(f"{table.source} · {table.sheet}: no operational fields were recognised.")
            continue
        fragment = table.data[list(selected)].rename(columns=selected).copy()
        if "date" not in fragment:
            warnings.append(f"{table.source} · {table.sheet}: excluded because no date field was recognised.")
            continue
        fragment["date"] = pd.to_datetime(fragment["date"], errors="coerce")
        for column in set(NUMERIC_FIELDS).intersection(fragment.columns):
            fragment[column] = pd.to_numeric(fragment[column], errors="coerce")
        fragment["_source"] = f"{table.source} · {table.sheet}"
        fragments.append(fragment)
        for canonical, source in table.mapping.items():
            if source:
                fields.append({
                    "Source": table.source,
                    "Sheet / table": table.sheet,
                    "Source field": source,
                    "Meaning": FIELD_LABELS[canonical],
                    "Canonical field": canonical,
                    "Confidence": "High — alias match",
                    "Header row": table.header_row,
                })

    if not fragments:
        semantic = pd.DataFrame(columns=ALL_FIELDS)
    else:
        combined = pd.concat(fragments, ignore_index=True, sort=False)
        combined = combined.dropna(subset=["date"])
        available_dims = [d for d in DIMENSIONS if d in combined and combined[d].notna().any()]
        keys = ["date"] + [d for d in available_dims if d != "date"]
        for dim in keys:
            if dim != "date":
                combined[dim] = combined[dim].fillna("Not supplied")
        available_numeric = [n for n in NUMERIC_FIELDS if n in combined]
        semantic = combined.groupby(keys, as_index=False, dropna=False)[available_numeric].sum(min_count=1)
        ordered = [c for c in ALL_FIELDS if c in semantic.columns]
        semantic = semantic[ordered].sort_values("date").reset_index(drop=True)

    missing = [FIELD_LABELS[field] for field in REQUIRED_FIELDS if field not in semantic or semantic[field].isna().all()]
    if missing:
        warnings.append("Required semantic fields still missing: " + ", ".join(missing))
    return IngestionResult(
        sources=tables,
        semantic_data=semantic,
        field_register=pd.DataFrame(fields),
        evidence_register=evidence if evidence is not None else pd.DataFrame(),
        warnings=warnings,
    )

