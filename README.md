# Pro Macro Stock Scorer

This is a Python web app version of the Google Sheet scorer. It keeps the scoring logic in Python and replaces Google Sheets fetches with:

- SEC EDGAR `submissions` and `companyfacts` APIs for filing-backed fundamentals
- SEC inline filing parsing for facts that are in the filing but not exposed cleanly in `companyfacts`
- Yahoo Finance direct HTML parsing and `yfinance` as the default no-key fallback for current price, market-derived fields, and company profile data
- Massive, Financial Modeling Prep, and Alpha Vantage as optional extra fallbacks if you choose to configure them

## What is implemented

- Company profile block
- Fundamental raw metrics
- Valuation, profitability, growth, financial strength, cash-flow, and composite scores
- Risk rating and allocation recommendation
- Browser UI instead of a terminal-only script

## Important note

The original sheet referenced custom functions such as `EP_SPREAD()` and REIT helpers that were not fully defined in the CSV export. This app ports the visible sheet formulas closely, but:

- `EP_SPREAD()` is approximated as `ROIC - 10.84%`
- REIT occupancy is only shown when a stable source actually returns it
- The macro section is currently represented by a manual `macro score` input, defaulting to `50` for neutral conditions

## Run it

```bash

python3 -m pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your@email.com"
python3 app.py
```

Optional only, not required:

```bash

export FMP_API_KEY="your_fmp_key"
export ALPHAVANTAGE_API_KEY="your_alpha_vantage_key"
```

Then open:

`http://127.0.0.1:5055`

## Verify the formula logic

This checks the Python scorer against the exported MSFT sheet values and confirms the formula outputs match the sheet's score cells:

```bash
python3 scripts/verify_sheet_logic.py
```

## Verify the live data pipeline

This runs a focused cross-sector sanity check against live data and also verifies the Flask route still renders the sheet-style output:

```bash
python3 scripts/verify_live_pipeline.py
```

## Accuracy hierarchy

For the metrics that are hardest to keep reliable, the app now follows this source order:

1. SEC EDGAR standardized facts
2. SEC inline filing parsing
3. Yahoo Finance public HTML parsing
4. yfinance fallback
5. Massive market-data or ratios fallback, if configured
6. Financial Modeling Prep estimates or TTM ratio fallback, if configured
7. Alpha Vantage quote, overview, or earnings-estimate fallback, if configured

For ratios like `Price/Book`, `Price/Sales`, `EV/Sales`, and `EV/EBITDA`, the app prefers self-computed values from raw ingredients whenever those ingredients are available and only falls back to vendor-calculated ratios when the raw path is incomplete.
