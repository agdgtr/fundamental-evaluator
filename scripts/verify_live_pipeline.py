from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fundamental_scorer.app import create_app
from fundamental_scorer.data_sources import FactStore, duration_days, parse_date, safe_float, to_millions
from fundamental_scorer.service import FundamentalScorerService

RECENT_QUARTER_MAX_AGE_DAYS = 500

REVENUE_CONCEPTS = ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"]
DIRECT_COST_CONCEPTS = [
    "CostOfRevenue",
    "CostOfSales",
    "CostOfGoodsSold",
    "CostOfGoodsAndServicesSold",
    "CostOfServices",
    "DirectOperatingCosts",
]
GROSS_PROFIT_CONCEPTS = ["GrossProfit"]
NET_INCOME_CONCEPTS = ["NetIncomeLoss"]
OCF_CONCEPTS = ["NetCashProvidedByUsedInOperatingActivities"]
EPS_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]
CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]
SHORT_TERM_INVESTMENT_CONCEPTS = ["ShortTermInvestments"]
CURRENT_ASSET_CONCEPTS = ["AssetsCurrent"]
CURRENT_LIABILITY_CONCEPTS = ["LiabilitiesCurrent"]


def annual_growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / abs(previous)


def assert_close(actual: float | None, expected: float | None, *, tolerance: float, label: str) -> None:
    assert actual is not None, f"{label} missing actual value"
    assert expected is not None, f"{label} missing expected value"
    delta = abs(actual - expected)
    assert delta <= tolerance, f"{label} mismatch: actual={actual} expected={expected} delta={delta}"


def quarterly_entries(facts: FactStore, concepts: list[str], units: list[str]) -> list[dict]:
    for concept in concepts:
        entries = facts._quarterly_series_with_derived_q4(
            [concept],
            unit_preferences=units,
            limit=8,
            max_age_days=RECENT_QUARTER_MAX_AGE_DAYS,
        )
        if entries:
            return entries
    return []


def comparable_quarter_pair(entries: list[dict], transform) -> tuple[float | None, float | None]:
    if not entries:
        return None, None
    current = entries[0]
    current_end = parse_date(current.get("end"))
    current_duration = duration_days(current)
    current_value = transform(current.get("val"))
    if current_end is None:
        return current_value, None

    match_value = None
    best_gap: int | None = None
    for entry in entries[1:]:
        end = parse_date(entry.get("end"))
        if end is None:
            continue
        delta = (current_end - end).days
        if delta < 330 or delta > 400:
            continue
        entry_duration = duration_days(entry)
        if current_duration is not None and entry_duration is not None and abs(current_duration - entry_duration) > 25:
            continue
        value = transform(entry.get("val"))
        if value is None:
            continue
        gap = abs(delta - 365)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            match_value = value
    return current_value, match_value


def latest_instant_millions(facts: FactStore, concepts: list[str]) -> float | None:
    for concept in concepts:
        value = facts.latest_instant_value([concept], unit_preferences=["USD"], max_age_days=RECENT_QUARTER_MAX_AGE_DAYS)
        if value is not None:
            return to_millions(value)
    return None


def sec_expected_quarter_metrics(service: FundamentalScorerService, symbol: str) -> dict[str, float | None]:
    cik = service.sec.lookup_cik(symbol)
    facts = FactStore(service.sec.fetch_companyfacts(cik))

    revenue_entries = quarterly_entries(facts, REVENUE_CONCEPTS, ["USD"])
    quarter_revenue, prior_revenue = comparable_quarter_pair(revenue_entries, to_millions)

    direct_cost_entries = quarterly_entries(facts, DIRECT_COST_CONCEPTS, ["USD"])
    quarter_direct_costs, _ = comparable_quarter_pair(direct_cost_entries, to_millions)

    gross_profit_entries = quarterly_entries(facts, GROSS_PROFIT_CONCEPTS, ["USD"])
    quarter_gross_profit, _ = comparable_quarter_pair(gross_profit_entries, to_millions)
    if quarter_gross_profit is None and quarter_revenue is not None and quarter_direct_costs is not None:
        quarter_gross_profit = quarter_revenue - quarter_direct_costs

    net_income_entries = quarterly_entries(facts, NET_INCOME_CONCEPTS, ["USD"])
    quarter_net_income, prior_net_income = comparable_quarter_pair(net_income_entries, to_millions)

    eps_entries = quarterly_entries(facts, EPS_CONCEPTS, ["USD/shares"])
    quarter_eps, prior_eps = comparable_quarter_pair(eps_entries, safe_float)

    ocf_entries = quarterly_entries(facts, OCF_CONCEPTS, ["USD"])
    quarter_ocf, _ = comparable_quarter_pair(ocf_entries, to_millions)

    current_assets = latest_instant_millions(facts, CURRENT_ASSET_CONCEPTS)
    current_liabilities = latest_instant_millions(facts, CURRENT_LIABILITY_CONCEPTS)
    cash = latest_instant_millions(facts, CASH_CONCEPTS)
    short_term_investments = latest_instant_millions(facts, SHORT_TERM_INVESTMENT_CONCEPTS)

    gross_margin = (quarter_gross_profit / quarter_revenue) if quarter_gross_profit is not None and quarter_revenue not in (None, 0) else None
    net_margin = (quarter_net_income / quarter_revenue) if quarter_net_income is not None and quarter_revenue not in (None, 0) else None
    revenue_growth = annual_growth(quarter_revenue, prior_revenue)
    earnings_growth = annual_growth(quarter_net_income, prior_net_income)
    eps_growth = annual_growth(quarter_eps, prior_eps)
    current_ratio = (current_assets / current_liabilities) if current_assets is not None and current_liabilities not in (None, 0) else None
    cash_ratio = ((cash or 0) + (short_term_investments or 0)) / current_liabilities if current_liabilities not in (None, 0) and (cash is not None or short_term_investments is not None) else None
    operating_cash_flow_margin = (quarter_ocf / quarter_revenue) if quarter_ocf is not None and quarter_revenue not in (None, 0) else None

    return {
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "eps_growth": eps_growth,
        "current_ratio": current_ratio,
        "cash_ratio": cash_ratio,
        "operating_cash_flow_margin": operating_cash_flow_margin,
    }


def main() -> None:
    service = FundamentalScorerService()
    tickers = ["MSFT", "GOOG", "JPM", "TDW", "XOM", "CAT", "PFE", "NEE", "O", "PLD", "AMT", "EQIX", "BABA"]
    expected_ceo_fragments = {
        "MSFT": "Satya Nadella",
        "GOOG": "Sundar Pichai",
        "JPM": "James Dimon",
        "TDW": "Quintin V. Kneen",
    }

    results: dict[str, dict] = {}
    for symbol in tickers:
        result = service.analyze(symbol)
        results[symbol] = result
        metrics = result["metrics"]
        profile = result["profile"]

        assert profile["symbol"] == symbol
        assert profile["sector"], f"{symbol} missing sector"
        assert result["scores"]["composite"] is not None, f"{symbol} missing composite score"
        assert metrics["current_price"] is not None and metrics["current_price"] > 0, f"{symbol} missing price"
        assert metrics["market_cap"] is not None and metrics["market_cap"] > 0, f"{symbol} missing market cap"
        assert profile["ceo"], f"{symbol} missing CEO"
        assert "  " not in profile["ceo"], f"{symbol} CEO contains double spaces: {profile['ceo']!r}"

        if symbol in expected_ceo_fragments:
            assert expected_ceo_fragments[symbol] in profile["ceo"], f"{symbol} CEO mismatch: {profile['ceo']}"

        if symbol in {"MSFT", "GOOG", "CAT", "PFE", "TDW", "XOM", "AMT", "BABA"}:
            expected = sec_expected_quarter_metrics(service, symbol)
            for field in ["gross_margin", "net_margin", "revenue_growth", "earnings_growth", "eps_growth"]:
                if expected[field] is not None:
                    assert_close(metrics[field], expected[field], tolerance=0.0005, label=f"{symbol} {field}")

            if symbol == "AMT":
                if expected["current_ratio"] is not None:
                    assert_close(metrics["current_ratio"], expected["current_ratio"], tolerance=0.0005, label="AMT current_ratio")
                if expected["cash_ratio"] is not None:
                    assert_close(metrics["cash_ratio"], expected["cash_ratio"], tolerance=0.0005, label="AMT cash_ratio")

            if symbol == "TDW":
                assert metrics["cash_ratio"] is not None and 1.5 <= metrics["cash_ratio"] <= 2.5, f"TDW cash ratio still looks wrong: {metrics['cash_ratio']}"
                assert metrics["operating_cash_flow_margin"] is not None and metrics["operating_cash_flow_margin"] > 0.1, f"TDW OCF margin looks wrong: {metrics['operating_cash_flow_margin']}"

            if symbol == "BABA":
                if expected["current_ratio"] is not None:
                    assert_close(metrics["current_ratio"], expected["current_ratio"], tolerance=0.0005, label="BABA current_ratio")
                for field in ["total_assets", "current_assets", "current_liabilities", "total_liabilities", "equity", "retained_earnings", "altman_z"]:
                    assert metrics[field] is not None, f"BABA {field} missing after SEC instant fix"
                assert metrics["altman_z"] > 0, f"BABA altman_z should be positive: {metrics['altman_z']}"

        if symbol == "O":
            assert metrics["current_ratio"] is not None and abs(metrics["current_ratio"] - 2.062) < 0.01, f"O current ratio fallback regressed: {metrics['current_ratio']}"
            assert metrics["occupancy"] is not None and 0.85 <= metrics["occupancy"] <= 1.0, f"O occupancy missing or invalid: {metrics['occupancy']}"
            assert metrics["ffo"] is not None and metrics["ffo"] > 0, "O missing FFO"
            assert metrics["p_ffo"] is not None and metrics["p_ffo"] > 0, "O missing P/FFO"

        if symbol == "PLD":
            assert metrics["occupancy"] is not None and 0.85 <= metrics["occupancy"] <= 1.0, f"PLD occupancy missing or invalid: {metrics['occupancy']}"
            assert metrics["ffo"] is not None and metrics["ffo"] > 0, "PLD missing FFO"

        if symbol == "EQIX":
            assert metrics["current_price"] > 100, f"EQIX price fallback still looks broken: {metrics['current_price']}"
            assert metrics["price_to_sales"] is None or metrics["price_to_sales"] > 0, f"EQIX missing price/sales"

        if symbol == "PFE":
            assert 10 <= metrics["current_price"] <= 100, f"PFE price fallback still looks broken: {metrics['current_price']}"

        if symbol == "JPM":
            assert profile["sector"] == "Financials", f"JPM sector mapping regressed: {profile['sector']}"

    app = create_app()
    client = app.test_client()
    for symbol in ["TDW", "O", "GOOG", "AMT", "PFE", "BABA"]:
        response = client.post(
            "/",
            data={
                "symbol": symbol,
                "current_hold": "0",
                "macro_score": "50",
                "sec_name": "Jane Doe",
                "sec_email": "jane@example.com",
            },
        )
        body = response.get_data(as_text=True)
        assert response.status_code == 200, f"{symbol} route failed with {response.status_code}"
        assert "Literal Formula Outputs" in body, f"{symbol} route missing sheet output block"
        assert "Final Recommendation" in body, f"{symbol} route missing recommendation block"

    print("Live pipeline verification passed.")
    for symbol in tickers:
        metrics = results[symbol]["metrics"]
        profile = results[symbol]["profile"]
        scores = results[symbol]["scores"]
        print(
            "|".join(
                [
                    symbol,
                    profile["sector"] or "NA",
                    profile["ceo"] or "NA",
                    str(metrics["gross_margin"]),
                    str(metrics["net_margin"]),
                    str(metrics["revenue_growth"]),
                    str(metrics["earnings_growth"]),
                    str(metrics["eps_growth"]),
                    str(metrics["current_ratio"]),
                    str(metrics["cash_ratio"]),
                    str(metrics["debt_ebitda"]),
                    str(metrics["debt_to_equity"]),
                    str(metrics["roic"]),
                    str(scores["composite"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
