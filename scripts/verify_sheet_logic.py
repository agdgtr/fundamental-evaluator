from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fundamental_scorer.scoring import (
    build_risk_rating,
    cash_flow_score,
    composite_score,
    financial_strength_score,
    growth_score,
    profitability_score,
    valuation_score,
)


CSV_PATH = Path("/Users/30andgarcia/Desktop/calcualtor bien - Sheet1.csv")


def load_rows() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip():
                rows[row[0].strip()] = row
    return rows


def cell_value(rows: dict[str, list[str]], label: str, column: int = 1) -> float | None:
    raw = rows.get(label, ["", "", "", ""])[column]
    if raw in {"", "N/A", "#NUM!", "#ERROR!", "#VALUE!", "Metric not found"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> None:
    rows = load_rows()
    metrics = {
        "forward_pe": cell_value(rows, "Forward P/E"),
        "peg_ratio": cell_value(rows, "PEG Ratio"),
        "price_to_book": cell_value(rows, "Price to Book (P/B)"),
        "price_to_sales": cell_value(rows, "Price to Sales (P/S)"),
        "ev_ebitda": cell_value(rows, "EV / EBITDA"),
        "ev_sales": cell_value(rows, "EV / Sales"),
        "pe_ratio": cell_value(rows, "Pe"),
        "price_fcf_ratio": cell_value(rows, "Price/FCF Ratio"),
        "earnings_yield": cell_value(rows, "Earnings Yield"),
        "gross_margin": cell_value(rows, "Gross Margin %"),
        "operating_margin": cell_value(rows, "Operating Margin %"),
        "net_margin": cell_value(rows, "Net Margin %"),
        "roe": cell_value(rows, "ROE %"),
        "roa": cell_value(rows, "ROA %"),
        "roic": cell_value(rows, "ROIC %"),
        "fcf_margin": cell_value(rows, "Free Cash Flow Margin"),
        "ebit_margin": cell_value(rows, "EBIT Margin"),
        "economic_profit_spread": cell_value(rows, "Economic Profit Spread"),
        "revenue_growth": cell_value(rows, "Revenue Growth (YoY)"),
        "earnings_growth": cell_value(rows, "Earnings Growth (YoY)"),
        "free_cash_flow_growth": cell_value(rows, "Free Cash Flow Growth (YoY)"),
        "dividend_growth": cell_value(rows, "Dividend Growth (YoY)"),
        "eps_growth": cell_value(rows, "EPS Growth"),
        "debt_to_equity": cell_value(rows, "Debt / Equity"),
        "interest_coverage": cell_value(rows, "Interest Coverage"),
        "current_ratio": cell_value(rows, "Current Ratio"),
        "debt_ebitda": cell_value(rows, "Debt / EBITDA"),
        "cash_ratio": cell_value(rows, "Cash Ratio"),
        "debt_fcf_ratio": cell_value(rows, "Debt / FCF Ratio"),
        "quick_ratio": cell_value(rows, "Quick Ratio"),
        "altman_z": cell_value(rows, 'Altman Z"'),
        "free_cash_flow": cell_value(rows, "Free Cash Flow"),
        "operating_cash_flow_margin": cell_value(rows, "Operating Cash Flow Margin"),
        "fcf_yield": cell_value(rows, "FCF Yield"),
        "capex_to_sales": cell_value(rows, "CapEx to Sales"),
        "cfo_net_income": cell_value(rows, "CFO / Net Income"),
        "capex_cfo": cell_value(rows, "CapEx / CFO"),
        "accrual_ratio": cell_value(rows, "Accrual Ratio"),
    }

    scores = {
        "valuation": valuation_score(metrics),
        "profitability": profitability_score(metrics),
        "growth": growth_score(metrics),
        "financial_strength": financial_strength_score(metrics),
        "cash_flow": cash_flow_score(metrics),
    }
    scores["composite"] = composite_score(scores)

    expected = {
        "valuation": 57.87,
        "profitability": 90.68,
        "growth": 67.71,
        "financial_strength": 40.03,
        "cash_flow": 78.04,
        "composite": 68.882,
    }

    for key, target in expected.items():
        actual = scores[key]
        assert actual is not None, f"{key} returned None"
        assert round(actual, 3) == round(target, 3), f"{key}: expected {target}, got {actual}"

    sector = rows["sector"][1].strip("[]")
    risk = build_risk_rating(metrics, sector)
    assert risk == "Excellent (Quality Override)", risk

    print("Sheet logic regression passed.")
    for key, value in scores.items():
        print(key, value)
    print("risk", risk)


if __name__ == "__main__":
    main()
