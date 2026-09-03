# Inshira Manufacturing Diagnostic MVP

An internal Streamlit application for turning 12 weeks of factory data into a concise operational diagnostic.

## What it does

- Loads CSV or XLSX files, or the included sample dataset
- Maps client column names to a standard diagnostic schema
- Calculates production, quality, downtime and changeover KPIs
- Flags data-quality limitations
- Ranks improvement opportunities using transparent rules
- Exports a management-ready HTML diagnostic report

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Minimum useful data

The tool only requires `date`, `planned_units`, `produced_units`, `good_first_pass_units`, and `rejected_units`. Downtime, changeover, rework and energy fields are optional but improve the diagnosis.

## Important limitation

Opportunity values are indicators, not guaranteed financial savings. Financial impact should only be shown after labour, material, energy and capacity assumptions have been validated with the client.

