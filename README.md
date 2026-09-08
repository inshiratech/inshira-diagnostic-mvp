# Inshira Manufacturing Diagnostic MVP

A Streamlit application for turning 8–12 weeks of existing factory data into a quantified diagnostic and a repeatable continuous-improvement cycle.

## What it does

- Loads CSV or XLSX files, or the included sample dataset
- Maps client column names to a standard diagnostic schema
- Calculates production, quality, downtime and changeover KPIs
- Flags data-quality limitations
- Ranks improvement opportunities using transparent rules
- Builds an indicative annual loss and recoverable-value register from client-validated assumptions
- Seeds an editable continuous-improvement register with owners, status, targets, actuals, reviews and learning
- Exports the action register and a management-ready HTML report

## MVP workflow

`Connect data → Diagnose → Quantify → Prioritise → Improve → Review`

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
