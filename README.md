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

`Raw evidence → Ingest and classify → Semantic layer → Human approval → Diagnose → Quantify → Improve → Review`

## Ingestion and semantic layer

- Accepts multiple CSV and XLSX tables in one raw data pack
- Detects buried header rows and recognises common ERP, production, quality, downtime, changeover and energy labels
- Preserves file, sheet and field lineage in an interpretation register
- Consolidates the recognised fields into a canonical operational dataset
- Retains PDF and image files in an evidence register for later human/AI extraction
- Requires a named human reviewer to edit and approve the semantic dataset before downstream intelligence is released

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
