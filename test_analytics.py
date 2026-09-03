import pandas as pd

from analytics import diagnose, standardize, suggest_mapping


def test_sample_data_matches_expected_diagnostic_profile():
    raw = pd.read_csv("sample_factory_data.csv")
    result = diagnose(standardize(raw, suggest_mapping(raw.columns)))

    assert round(result.metrics["ftt"], 3) == 0.851
    assert round(result.metrics["avg_changeover_minutes"], 1) == 62.5
    assert result.opportunities.iloc[0]["Metric"] == "Average changeover"


def test_alias_mapping_accepts_common_client_headers():
    raw = pd.DataFrame({
        "Day": ["2026-01-01"],
        "Target": [100],
        "Actual": [90],
        "First Pass Good": [80],
        "Rejects": [2],
    })
    mapping = suggest_mapping(raw.columns)
    standardized = standardize(raw, mapping)

    assert set(["date", "planned_units", "produced_units", "good_first_pass_units", "rejected_units"]).issubset(standardized.columns)
