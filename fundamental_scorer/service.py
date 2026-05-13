from __future__ import annotations

import datetime as dt
import re
from typing import Any

from .data_sources import (
    AlphaVantageClient,
    FactStore,
    FMPClient,
    MassiveClient,
    SecEdgarClient,
    YahooFinanceClient,
    YahooWebClient,
    duration_days,
    first_item,
    first_non_null,
    parse_date,
    safe_float,
    to_millions,
)
from .scoring import (
    build_recommendation,
    build_risk_rating,
    cash_flow_score,
    clamp,
    composite_score,
    financial_strength_score,
    growth_score,
    has_value,
    profitability_score,
    safe_div,
    valuation_score,
)

DEFAULT_WACC = 0.1084
RECENT_QUARTER_MAX_AGE_DAYS = 500
RECENT_ANNUAL_MAX_AGE_DAYS = 800


def format_ipo_date(history: Any) -> str | None:
    if history is None or getattr(history, "empty", True):
        return None
    first = history.index[0]
    if hasattr(first, "to_pydatetime"):
        first = first.to_pydatetime()
    return first.strftime("%b %d, %Y")


def annual_growth(current: float | None, previous: float | None) -> float | None:
    if not has_value(current) or not has_value(previous) or previous == 0:
        return None
    return (current - previous) / abs(previous)


def officer_name(info: dict[str, Any]) -> str | None:
    officers = info.get("companyOfficers") or []
    if not officers:
        return None
    ranked: list[tuple[int, str]] = []
    for officer in officers:
        if not isinstance(officer, dict):
            continue
        name = officer.get("name")
        title = (officer.get("title") or "").lower()
        if not name:
            continue
        is_ceo = "chief executive officer" in title or re.search(r"\bceo\b", title)
        if not is_ceo:
            continue
        score = 100 if "chief executive officer" in title else 80
        if "president" in title:
            score += 20
        if "interim" in title:
            score -= 10
        if " of " in title:
            score -= 60
        if score:
            ranked.append((score, name))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][1]
    return None


def normalize_fraction(value: float | None) -> float | None:
    if not has_value(value):
        return None
    if -5 <= value <= 5:
        return value
    if -100 <= value <= 100:
        return value / 100
    return value


def normalize_sector_name(value: str | None) -> str:
    if not value:
        return "Unknown"
    mapping = {
        "Financial Services": "Financials",
        "Consumer Defensive": "Consumer Staples",
        "Consumer Cyclical": "Consumer Discretionary",
        "Basic Materials": "Materials",
    }
    return mapping.get(value, value)


def pick_number(*values: Any) -> float | None:
    for value in values:
        number = safe_float(value)
        if number is not None:
            return number
    return None


def subtract_series(left: list[float], right: list[float]) -> list[float]:
    values: list[float] = []
    for lhs, rhs in zip(left, right):
        if has_value(lhs) and has_value(rhs):
            values.append(lhs - rhs)
    return values


def sane_margin(value: float | None) -> float | None:
    if not has_value(value):
        return None
    if -1 <= value <= 1:
        return value
    return None


def nested_number(payload: dict[str, Any] | None, *path_options: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for path in path_options:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        number = safe_float(current)
        if number is not None:
            return number
    return None


def parse_year(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text)


def estimate_eps_value(row: dict[str, Any]) -> float | None:
    priority = (
        "estimatedEpsAvg",
        "estimatedEps",
        "epsAvg",
        "epsEstimated",
        "estimated_eps_avg",
        "estimated_eps",
        "eps_estimate",
    )
    value = pick_number(*(row.get(key) for key in priority))
    if value is not None:
        return value
    for key, raw in row.items():
        lowered = key.lower()
        if "eps" in lowered and ("avg" in lowered or "estimate" in lowered):
            value = safe_float(raw)
            if value is not None:
                return value
    return None


def estimate_series(rows: list[dict[str, Any]]) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for row in rows:
        year = first_non_null(
            parse_year(row.get("date")),
            parse_year(row.get("fiscalDateEnding")),
            parse_year(row.get("calendarYear")),
            parse_year(row.get("year")),
        )
        eps = estimate_eps_value(row)
        if year is None or eps is None:
            continue
        values.append((year, eps))
    deduped: dict[int, float] = {}
    for year, eps in sorted(values):
        deduped.setdefault(year, eps)
    return sorted(deduped.items())


def format_number(value: float | None, decimals: int = 2) -> str:
    if not has_value(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 1) -> str:
    if not has_value(value):
        return "N/A"
    return f"{value * 100:,.{decimals}f}%"


def format_multiple(value: float | None, decimals: int = 2) -> str:
    if not has_value(value):
        return "N/A"
    return f"{value:,.{decimals}f}x"


def format_money_millions(value: float | None, decimals: int = 0) -> str:
    if not has_value(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def formula_value(value: float | None, decimals: int = 2) -> str:
    if not has_value(value):
        return "N/A"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def formula_percent(value: float | None, decimals: int = 4) -> str:
    if not has_value(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%".rstrip("0").rstrip(".")


def make_sheet_row(
    label: str,
    value: str,
    side_label: str = "",
    side_value: str = "",
) -> dict[str, str]:
    return {
        "label": label,
        "value": value,
        "side_label": side_label,
        "side_value": side_value,
    }


class FundamentalScorerService:
    def __init__(self) -> None:
        self.sec = SecEdgarClient()
        self.yahoo = YahooFinanceClient()
        self.yahoo_web = YahooWebClient()
        self.massive = MassiveClient()
        self.fmp = FMPClient()
        self.alpha_vantage = AlphaVantageClient()

    def analyze(self, symbol: str, current_hold: float = 0, macro_score: float = 50) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        cik = self.sec.lookup_cik(symbol)
        submissions = self.sec.fetch_submissions(cik)
        companyfacts = self.sec.fetch_companyfacts(cik)
        yahoo = self.yahoo.fetch_snapshot(symbol)
        yahoo_web = self.yahoo_web.fetch_metrics(symbol)
        massive_snapshot = self.massive.fetch_single_ticker_snapshot(symbol)
        massive_overview = self.massive.fetch_ticker_overview(symbol)
        massive_ratios = self.massive.fetch_ratios(symbol)
        fmp_quote = self.fmp.fetch_quote(symbol)
        fmp_key_metrics = self.fmp.fetch_key_metrics_ttm(symbol)
        fmp_ratios = self.fmp.fetch_ratios_ttm(symbol)
        fmp_estimates = self.fmp.fetch_analyst_estimates(symbol)
        alpha_quote = self.alpha_vantage.fetch_global_quote(symbol)
        alpha_overview = self.alpha_vantage.fetch_company_overview(symbol)
        alpha_estimates_payload = self.alpha_vantage.fetch_earnings_estimates(symbol)
        info = yahoo["info"]
        history = yahoo["history"]
        facts = FactStore(companyfacts)
        sector = normalize_sector_name(info.get("sector") or yahoo_web.get("sector") or submissions.get("sicDescription"))
        is_reit = sector == "Real Estate"
        alpha_annual_estimates = (
            alpha_estimates_payload.get("annualEstimates")
            if isinstance(alpha_estimates_payload, dict) and isinstance(alpha_estimates_payload.get("annualEstimates"), list)
            else []
        )

        def ttm_series_millions(concepts: list[str]) -> list[float]:
            for concept in concepts:
                values = [
                    to_millions(value)
                    for value in facts.trailing_twelve_month_values(
                        [concept],
                        unit_preferences=["USD"],
                        count=2,
                        max_age_days=RECENT_QUARTER_MAX_AGE_DAYS,
                    )
                    if to_millions(value) is not None
                ]
                if values:
                    return values
            return []

        def annual_series_millions(concepts: list[str]) -> list[float]:
            for concept in concepts:
                values = [
                    to_millions(value)
                    for value in facts.annual_values(
                        [concept],
                        unit_preferences=["USD"],
                        count=2,
                        max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS,
                    )
                    if to_millions(value) is not None
                ]
                if values:
                    return values
            return []

        def quarterly_entries(concepts: list[str], unit_preferences: list[str], limit: int = 8) -> list[dict[str, Any]]:
            for concept in concepts:
                entries = facts._quarterly_series_with_derived_q4(
                    [concept],
                    unit_preferences=unit_preferences,
                    limit=limit,
                    max_age_days=RECENT_QUARTER_MAX_AGE_DAYS,
                )
                if entries:
                    return entries
            return []

        def comparable_quarter_pair(
            entries: list[dict[str, Any]],
            *,
            transform,
        ) -> tuple[float | None, float | None]:
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
                gap = abs(delta - 365)
                value = transform(entry.get("val"))
                if value is None:
                    continue
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    match_value = value
            return current_value, match_value

        def latest_instant_millions(concepts: list[str], filing_concepts: list[str] | None = None) -> float | None:
            for concept in concepts:
                value = facts.latest_instant_value([concept], unit_preferences=["USD"], max_age_days=RECENT_QUARTER_MAX_AGE_DAYS)
                if value is not None:
                    return to_millions(value)
            if filing_concepts:
                filing_value = self.sec.recent_filing_value(cik, filing_concepts, kind="instant", unit_preferences=["usd"])
                if filing_value is not None:
                    return to_millions(filing_value)
            return None

        revenue_ttm_series = ttm_series_millions(["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"])
        revenue_annual_series = annual_series_millions(["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"])
        revenue_series = revenue_ttm_series or revenue_annual_series
        revenue_quarter_entries = quarterly_entries(["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"], ["USD"])
        quarter_revenue, prior_year_quarter_revenue = comparable_quarter_pair(revenue_quarter_entries, transform=to_millions)

        direct_cost_ttm_series = ttm_series_millions(
            [
                "CostOfRevenue",
                "CostOfSales",
                "CostOfGoodsSold",
                "CostOfGoodsAndServicesSold",
                "CostOfServices",
                "DirectOperatingCosts",
            ]
        )
        direct_cost_annual_series = annual_series_millions(
            [
                "CostOfRevenue",
                "CostOfSales",
                "CostOfGoodsSold",
                "CostOfGoodsAndServicesSold",
                "CostOfServices",
                "DirectOperatingCosts",
            ]
        )
        direct_cost_quarter_entries = quarterly_entries(
            [
                "CostOfRevenue",
                "CostOfSales",
                "CostOfGoodsSold",
                "CostOfGoodsAndServicesSold",
                "CostOfServices",
                "DirectOperatingCosts",
            ],
            ["USD"],
        )
        quarter_direct_costs, _ = comparable_quarter_pair(direct_cost_quarter_entries, transform=to_millions)
        gross_profit_ttm_series = ttm_series_millions(["GrossProfit"])
        gross_profit_annual_series = annual_series_millions(["GrossProfit"])
        gross_profit_series = (
            gross_profit_ttm_series
            or gross_profit_annual_series
            or subtract_series(revenue_ttm_series, direct_cost_ttm_series)
            or subtract_series(revenue_annual_series, direct_cost_annual_series)
        )
        gross_profit_quarter_entries = quarterly_entries(["GrossProfit"], ["USD"])
        quarter_gross_profit, _ = comparable_quarter_pair(gross_profit_quarter_entries, transform=to_millions)
        if quarter_gross_profit is None and has_value(quarter_revenue) and has_value(quarter_direct_costs):
            quarter_gross_profit = quarter_revenue - quarter_direct_costs
        operating_income_ttm_series = ttm_series_millions(["OperatingIncomeLoss"])
        operating_income_annual_series = annual_series_millions(["OperatingIncomeLoss"])
        operating_income_series = operating_income_ttm_series or operating_income_annual_series
        operating_income_quarter_entries = quarterly_entries(["OperatingIncomeLoss"], ["USD"])
        quarter_operating_income, prior_year_quarter_operating_income = comparable_quarter_pair(operating_income_quarter_entries, transform=to_millions)
        net_income_ttm_series = ttm_series_millions(["NetIncomeLoss"])
        net_income_annual_series = annual_series_millions(["NetIncomeLoss"])
        net_income_series = net_income_ttm_series or net_income_annual_series
        net_income_quarter_entries = quarterly_entries(["NetIncomeLoss"], ["USD"])
        quarter_net_income, prior_year_quarter_net_income = comparable_quarter_pair(net_income_quarter_entries, transform=to_millions)
        pretax_ttm_series = ttm_series_millions(["IncomeBeforeTaxExpenseBenefit", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"])
        pretax_annual_series = annual_series_millions(["IncomeBeforeTaxExpenseBenefit", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"])
        pretax_series = pretax_ttm_series or pretax_annual_series
        income_tax_ttm_series = ttm_series_millions(["IncomeTaxExpenseBenefit"])
        income_tax_annual_series = annual_series_millions(["IncomeTaxExpenseBenefit"])
        income_tax_series = income_tax_ttm_series or income_tax_annual_series
        ocf_ttm_series = ttm_series_millions(["NetCashProvidedByUsedInOperatingActivities"])
        ocf_annual_series = annual_series_millions(["NetCashProvidedByUsedInOperatingActivities"])
        ocf_series = ocf_ttm_series or ocf_annual_series
        ocf_quarter_entries = quarterly_entries(["NetCashProvidedByUsedInOperatingActivities"], ["USD"])
        quarter_ocf, prior_year_quarter_ocf = comparable_quarter_pair(ocf_quarter_entries, transform=to_millions)
        capex_ttm_series = [
            -abs(value)
            for value in ttm_series_millions(
                [
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "PropertyPlantAndEquipmentAdditions",
                    "CapitalExpendituresIncurredButNotYetPaid",
                    "PaymentsForCapitalImprovements",
                ]
            )
        ]
        capex_quarter_entries = quarterly_entries(
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PropertyPlantAndEquipmentAdditions",
                "CapitalExpendituresIncurredButNotYetPaid",
                "PaymentsForCapitalImprovements",
            ],
            ["USD"],
        )
        quarter_capex, prior_year_quarter_capex = comparable_quarter_pair(
            capex_quarter_entries,
            transform=lambda value: -abs(to_millions(value)) if to_millions(value) is not None else None,
        )
        capex_annual_series = [
            -abs(value)
            for value in annual_series_millions(
                [
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "PropertyPlantAndEquipmentAdditions",
                    "CapitalExpendituresIncurredButNotYetPaid",
                    "PaymentsForCapitalImprovements",
                ]
            )
        ]
        capex_series = capex_ttm_series or capex_annual_series

        depreciation_series = ttm_series_millions(
            ["Depreciation", "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"]
        ) or annual_series_millions(
            ["Depreciation", "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"]
        )
        amortization_series = ttm_series_millions(["AmortizationOfIntangibleAssets"]) or annual_series_millions(["AmortizationOfIntangibleAssets"])
        eps_quarter_entries = quarterly_entries(["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"], ["USD/shares"], limit=8)
        quarter_eps, prior_year_quarter_eps = comparable_quarter_pair(eps_quarter_entries, transform=safe_float)
        depreciation = first_item(depreciation_series)
        amortization = first_item(amortization_series)
        interest_expense = to_millions(
            first_non_null(
                facts.latest_annual_value(["InterestExpense", "InterestExpenseNonoperating"], unit_preferences=["USD"], max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS),
                facts.latest_annual_value(["InterestPaid"], unit_preferences=["USD"], max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS),
            )
        )

        assets_series = [
            to_millions(item.get("val"))
            for item in facts.instant_series(["Assets"], unit_preferences=["USD"], limit=2, max_age_days=RECENT_QUARTER_MAX_AGE_DAYS)
            if safe_float(item.get("val")) is not None
        ]
        total_assets = first_non_null(
            latest_instant_millions(["Assets"], ["us-gaap:Assets", "ifrs-full:Assets"]),
            to_millions(info.get("totalAssets")),
        )
        prev_assets = assets_series[1] if len(assets_series) > 1 else None
        current_assets = first_non_null(
            latest_instant_millions(["AssetsCurrent"], ["us-gaap:AssetsCurrent", "ifrs-full:CurrentAssets"]),
        )
        current_liabilities = first_non_null(
            latest_instant_millions(["LiabilitiesCurrent"], ["us-gaap:LiabilitiesCurrent", "ifrs-full:CurrentLiabilities"]),
        )
        total_liabilities = latest_instant_millions(
            ["Liabilities", "LiabilitiesAndPartnersCapital"],
            ["us-gaap:Liabilities", "ifrs-full:Liabilities"],
        )
        equity = first_non_null(
            latest_instant_millions(
                [
                    "StockholdersEquity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                    "CommonStockholdersEquity",
                    "PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest",
                ],
                [
                    "us-gaap:StockholdersEquity",
                    "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                    "us-gaap:CommonStockholdersEquity",
                    "ifrs-full:Equity",
                ],
            ),
            (total_assets - total_liabilities) if has_value(total_assets) and has_value(total_liabilities) else None,
        )
        if total_liabilities is None and has_value(total_assets) and has_value(equity):
            total_liabilities = total_assets - equity
        retained_earnings = first_non_null(
            latest_instant_millions(["RetainedEarningsAccumulatedDeficit"], ["us-gaap:RetainedEarningsAccumulatedDeficit"]),
        )
        cash = first_non_null(
            latest_instant_millions(
                [
                    "CashAndCashEquivalentsAtCarryingValue",
                    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                ],
                [
                    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
                    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                    "ifrs-full:CashAndCashEquivalents",
                ],
            ),
            to_millions(info.get("totalCash")),
        )
        short_term_investments = first_non_null(
            latest_instant_millions(["ShortTermInvestments"], ["us-gaap:ShortTermInvestments"]),
            None,
        )
        if short_term_investments is None:
            cash_and_st = latest_instant_millions(["CashCashEquivalentsAndShortTermInvestments"], ["us-gaap:CashCashEquivalentsAndShortTermInvestments"])
            if has_value(cash_and_st) and has_value(cash):
                short_term_investments = max(0.0, cash_and_st - cash)

        debt_components = [
            latest_instant_millions(
                ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
                ["us-gaap:LongTermDebtCurrent", "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent"],
            ),
            latest_instant_millions(
                ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"],
                ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebtAndCapitalLeaseObligations", "us-gaap:LongTermDebt"],
            ),
            latest_instant_millions(
                ["ShortTermBorrowings", "CommercialPaper"],
                ["us-gaap:ShortTermBorrowings", "us-gaap:CommercialPaper"],
            ),
        ]
        sec_debt = (
            sum(component or 0 for component in debt_components)
            if any(has_value(component) and component != 0 for component in debt_components)
            else None
        )
        info_debt = to_millions(safe_float(info.get("totalDebt")))
        if has_value(sec_debt) and has_value(info_debt):
            debt = max(sec_debt, info_debt)
        else:
            debt = first_non_null(sec_debt, info_debt)
        inventory = first_non_null(
            latest_instant_millions(["InventoryNet"], ["us-gaap:InventoryNet"]),
        )

        shares_outstanding = first_non_null(
            facts.latest_instant_value(["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"], unit_preferences=["shares"], max_age_days=RECENT_QUARTER_MAX_AGE_DAYS),
            pick_number(
                nested_number(massive_overview, ("weighted_shares_outstanding",), ("share_class_shares_outstanding",)),
                fmp_quote.get("sharesOutstanding") if isinstance(fmp_quote, dict) else None,
                alpha_overview.get("SharesOutstanding") if isinstance(alpha_overview, dict) else None,
                info.get("sharesOutstanding"),
                yahoo_web.get("sharesOutstanding"),
            ),
        )
        shares_outstanding_m = shares_outstanding / 1_000_000 if shares_outstanding else None
        diluted_shares_m = to_millions(
            facts.latest_annual_value(["WeightedAverageNumberOfDilutedSharesOutstanding"], unit_preferences=["shares"], max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS)
        ) or shares_outstanding_m

        dividends_per_share_series = facts.annual_values(
            ["CommonStockDividendsPerShareDeclared"], unit_preferences=["USD/shares"], count=2, max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS
        )
        dividends_paid_series = annual_series_millions(
            ["PaymentsOfDividendsCommonStock", "DividendsCommonStockCash"]
        )
        dividend_per_share = first_non_null(
            safe_float(info.get("trailingAnnualDividendRate")),
            safe_float(info.get("dividendRate")),
            dividends_per_share_series[0] if dividends_per_share_series else None,
        )
        dividends_paid = to_millions(
            first_non_null(
                first_item(dividends_paid_series),
                (dividend_per_share or 0) * (diluted_shares_m or 0) * 1_000_000,
            )
        )

        price = first_non_null(
            nested_number(
                massive_snapshot,
                ("ticker", "lastTrade", "p"),
                ("ticker", "min", "c"),
                ("ticker", "day", "c"),
                ("ticker", "prevDay", "c"),
            ),
            pick_number(
                fmp_quote.get("price") if isinstance(fmp_quote, dict) else None,
                alpha_quote.get("05. price") if isinstance(alpha_quote, dict) else None,
                info.get("currentPrice"),
                info.get("regularMarketPrice"),
                yahoo["fast_info"].get("lastPrice"),
                yahoo["fast_info"].get("regularMarketPreviousClose"),
                info.get("regularMarketPreviousClose"),
                yahoo_web.get("regularMarketPrice"),
                yahoo_web.get("regularMarketPreviousClose"),
            ),
        )
        if price is None and history is not None and not history.empty:
            price = safe_float(history["Close"].dropna().iloc[-1])

        computed_market_cap = ((price or 0) * (shares_outstanding or 0)) if has_value(price) and has_value(shares_outstanding) else None
        market_cap = to_millions(
            first_non_null(
                computed_market_cap,
                nested_number(massive_ratios, ("market_cap",)),
                nested_number(massive_overview, ("market_cap",)),
                pick_number(
                    fmp_quote.get("marketCap") if isinstance(fmp_quote, dict) else None,
                    alpha_overview.get("MarketCapitalization") if isinstance(alpha_overview, dict) else None,
                    yahoo["fast_info"].get("marketCap"),
                    info.get("marketCap"),
                    yahoo_web.get("marketCap"),
                ),
            )
        )
        enterprise_value = to_millions(
            first_non_null(
                ((market_cap or 0) + (debt or 0) - (cash or 0)) * 1_000_000 if has_value(market_cap) else None,
                nested_number(massive_ratios, ("enterprise_value",)),
                pick_number(
                    fmp_key_metrics.get("enterpriseValue") if isinstance(fmp_key_metrics, dict) else None,
                    info.get("enterpriseValue"),
                    yahoo_web.get("enterpriseValue"),
                ),
            )
        )
        revenue = first_non_null(first_item(revenue_series), to_millions(info.get("totalRevenue")), to_millions(yahoo_web.get("totalRevenue")))
        gross_profit = first_non_null(first_item(gross_profit_series), to_millions(info.get("grossProfits")))
        operating_income = first_non_null(
            first_item(operating_income_series),
            (normalize_fraction(pick_number(info.get("operatingMargins"), yahoo_web.get("operatingMargins"))) or 0) * revenue if has_value(revenue) and has_value(normalize_fraction(pick_number(info.get("operatingMargins"), yahoo_web.get("operatingMargins")))) else None,
        )
        previous_operating_income = first_item(operating_income_series, 1)
        net_income = first_non_null(first_item(net_income_series), to_millions(info.get("netIncomeToCommon")))
        ocf = first_non_null(first_item(ocf_series), to_millions(info.get("operatingCashflow")), to_millions(yahoo_web.get("operatingCashflow")))
        yahoo_fcf = first_non_null(to_millions(info.get("freeCashflow")), to_millions(yahoo_web.get("freeCashflow")))
        derived_capex_from_yahoo = (yahoo_fcf - ocf) if has_value(yahoo_fcf) and has_value(ocf) else None
        filing_quarter_capex = to_millions(
            self.sec.recent_filing_value(
                cik,
                [
                    "us-gaap:PaymentsForCapitalImprovements",
                    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
                    "us-gaap:PropertyPlantAndEquipmentAdditions",
                    "o:ReLeasingCosts",
                    "o:RecurringCapitalExpendituresRealEstateImprovements",
                ],
                kind="quarterly",
                unit_preferences=["usd"],
            )
        )
        annualized_filing_capex = -abs(filing_quarter_capex * 4) if has_value(filing_quarter_capex) else None
        capex = first_non_null(first_item(capex_series), annualized_filing_capex if is_reit else None, derived_capex_from_yahoo)
        ttm_fcf_series = subtract_series(ocf_ttm_series, [-value for value in capex_ttm_series]) if ocf_ttm_series and capex_ttm_series else []
        annual_fcf_series = subtract_series(ocf_annual_series, [-value for value in capex_annual_series]) if ocf_annual_series and capex_annual_series else []
        annual_fcf = first_item(ttm_fcf_series) if ttm_fcf_series else first_item(annual_fcf_series)
        quarter_fcf = (quarter_ocf + quarter_capex) if has_value(quarter_ocf) and has_value(quarter_capex) else None
        prior_year_quarter_fcf = (prior_year_quarter_ocf + prior_year_quarter_capex) if has_value(prior_year_quarter_ocf) and has_value(prior_year_quarter_capex) else None
        fcf = first_non_null(
            annual_fcf if has_value(first_item(capex_series)) else None,
            nested_number(massive_ratios, ("free_cash_flow",)),
            pick_number(
                fmp_key_metrics.get("freeCashFlowTTM") if isinstance(fmp_key_metrics, dict) else None,
                yahoo_fcf,
            ),
            annual_fcf,
            (ocf or 0) + (capex or 0),
        )
        ebitda = first_non_null(
            (operating_income + (depreciation or 0) + (amortization or 0)) if has_value(operating_income) else None,
            to_millions(info.get("ebitda")),
        )

        margin_revenue = first_non_null(quarter_revenue, revenue)
        margin_gross_profit = first_non_null(quarter_gross_profit, gross_profit)
        margin_operating_income = first_non_null(quarter_operating_income, operating_income)
        margin_net_income = first_non_null(quarter_net_income, net_income)

        gross_margin = first_non_null(sane_margin(safe_div(margin_gross_profit, margin_revenue)), normalize_fraction(pick_number(info.get("grossMargins"), yahoo_web.get("grossMargins"))))
        operating_margin = first_non_null(sane_margin(safe_div(margin_operating_income, margin_revenue)), normalize_fraction(pick_number(info.get("operatingMargins"), yahoo_web.get("operatingMargins"))))
        net_margin = first_non_null(sane_margin(safe_div(margin_net_income, margin_revenue)), normalize_fraction(pick_number(info.get("profitMargins"), yahoo_web.get("profitMargins"))))

        avg_equity = first_non_null(equity, None)
        avg_assets = first_non_null((total_assets + prev_assets) / 2 if has_value(total_assets) and has_value(prev_assets) else total_assets, None)
        roe = first_non_null(safe_div(net_income, avg_equity), normalize_fraction(pick_number(info.get("returnOnEquity"), yahoo_web.get("returnOnEquity"))))
        roa = first_non_null(safe_div(net_income, avg_assets), normalize_fraction(pick_number(info.get("returnOnAssets"), yahoo_web.get("returnOnAssets"))))

        pretax_income = first_item(pretax_series)
        income_tax = first_item(income_tax_series)
        tax_rate = clamp((income_tax or 0) / pretax_income, 0, 0.35) if has_value(pretax_income) and pretax_income not in (0, None) else 0.21
        ebit = first_non_null(operating_income, previous_operating_income)
        nopat = ebit * (1 - tax_rate) if has_value(ebit) else None
        invested_capital = None
        if has_value(debt) or has_value(equity):
            invested_capital = (debt or 0) + (equity or 0) - (cash or 0) - (short_term_investments or 0)
        roic = safe_div(nopat, invested_capital)
        economic_profit_spread = roic - DEFAULT_WACC if has_value(roic) else None

        eps_series = facts.annual_values(
            ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"], unit_preferences=["USD/shares"], count=2, max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS
        )
        diluted_eps = first_item(eps_series) if eps_series else safe_div(net_income, diluted_shares_m)
        previous_eps = first_item(eps_series, 1)

        pe_ratio = first_non_null(
            safe_div(price, diluted_eps),
            safe_div(market_cap, net_income),
            nested_number(massive_ratios, ("price_to_earnings",)),
            pick_number(
                fmp_key_metrics.get("peRatioTTM") if isinstance(fmp_key_metrics, dict) else None,
                info.get("trailingPE"),
                yahoo_web.get("trailingEps") and safe_div(price, yahoo_web.get("trailingEps")),
            ),
        )

        estimate_points = estimate_series(fmp_estimates)
        if not estimate_points:
            estimate_points = estimate_series(alpha_annual_estimates)
        latest_net_income_item = first_item(facts.annual_series(["NetIncomeLoss"], unit_preferences=["USD"], limit=1, max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS))
        trailing_year = parse_year(latest_net_income_item.get("end")) if isinstance(latest_net_income_item, dict) else None
        current_year = dt.date.today().year
        estimate_cutoff = max(current_year, trailing_year or current_year)
        future_estimates = [(year, eps) for year, eps in estimate_points if year >= estimate_cutoff]
        forward_eps = first_item([eps for _, eps in future_estimates])
        second_forward_eps = first_item([eps for _, eps in future_estimates], 1)

        forward_pe = first_non_null(
            safe_div(price, forward_eps) if has_value(price) and has_value(forward_eps) and forward_eps > 0 else None,
            pick_number(
                fmp_key_metrics.get("forwardPE") if isinstance(fmp_key_metrics, dict) else None,
                info.get("forwardPE"),
                yahoo_web.get("forwardPE"),
            ),
        )
        peg_growth = None
        if has_value(second_forward_eps) and has_value(forward_eps) and forward_eps not in (0, None):
            peg_growth = (second_forward_eps - forward_eps) / abs(forward_eps)
        elif has_value(forward_eps) and has_value(diluted_eps) and diluted_eps not in (0, None):
            peg_growth = (forward_eps - diluted_eps) / abs(diluted_eps)

        peg_ratio = first_non_null(
            safe_div(forward_pe, peg_growth * 100) if has_value(forward_pe) and has_value(peg_growth) and peg_growth > 0 else None,
            alpha_overview.get("PEGRatio") if isinstance(alpha_overview, dict) else None,
            safe_float(info.get("pegRatio")),
            yahoo_web.get("pegRatio"),
        )
        if has_value(peg_ratio) and peg_ratio < 0:
            peg_ratio = None
        price_to_book = first_non_null(
            safe_div(market_cap, equity),
            nested_number(massive_ratios, ("price_to_book",)),
            pick_number(
                fmp_key_metrics.get("pbRatioTTM") if isinstance(fmp_key_metrics, dict) else None,
                alpha_overview.get("PriceToBookRatio") if isinstance(alpha_overview, dict) else None,
                info.get("priceToBook"),
                yahoo_web.get("priceToBook"),
            ),
        )
        price_to_sales = first_non_null(
            safe_div(market_cap, revenue),
            nested_number(massive_ratios, ("price_to_sales",)),
            pick_number(
                fmp_key_metrics.get("priceToSalesRatioTTM") if isinstance(fmp_key_metrics, dict) else None,
                info.get("priceToSalesTrailing12Months"),
                yahoo_web.get("priceToSalesTrailing12Months"),
            ),
        )
        ev_sales = first_non_null(
            safe_div(enterprise_value, revenue),
            nested_number(massive_ratios, ("ev_to_sales",)),
            pick_number(
                fmp_key_metrics.get("enterpriseValueOverSalesTTM") if isinstance(fmp_key_metrics, dict) else None,
                alpha_overview.get("EVToRevenue") if isinstance(alpha_overview, dict) else None,
                info.get("enterpriseToRevenue"),
                yahoo_web.get("enterpriseToRevenue"),
            ),
        )
        ev_ebitda = first_non_null(
            safe_div(enterprise_value, ebitda),
            nested_number(massive_ratios, ("ev_to_ebitda",)),
            pick_number(
                alpha_overview.get("EVToEBITDA") if isinstance(alpha_overview, dict) else None,
                info.get("enterpriseToEbitda"),
                yahoo_web.get("enterpriseToEbitda"),
            ),
        )
        price_fcf_ratio = first_non_null(
            safe_div(market_cap, fcf),
            nested_number(massive_ratios, ("price_to_free_cash_flow",)),
            pick_number(
                fmp_key_metrics.get("pfcfRatioTTM") if isinstance(fmp_key_metrics, dict) else None,
                fmp_key_metrics.get("pocfratioTTM") if isinstance(fmp_key_metrics, dict) else None,
            ),
        )
        earnings_yield = safe_div(1, pe_ratio)

        fcf_margin = safe_div(fcf, revenue)
        ebit_margin = safe_div(ebit, revenue)
        revenue_growth = first_non_null(
            annual_growth(quarter_revenue, prior_year_quarter_revenue),
            annual_growth(first_item(revenue_annual_series), first_item(revenue_annual_series, 1)),
            annual_growth(first_item(revenue_ttm_series), first_item(revenue_ttm_series, 1)),
            normalize_fraction(pick_number(info.get("revenueGrowth"))),
        )
        earnings_growth = first_non_null(
            annual_growth(quarter_net_income, prior_year_quarter_net_income),
            annual_growth(first_item(net_income_annual_series), first_item(net_income_annual_series, 1)),
            annual_growth(first_item(net_income_ttm_series), first_item(net_income_ttm_series, 1)),
            normalize_fraction(pick_number(info.get("earningsGrowth"))),
        )
        free_cash_flow_growth = first_non_null(
            annual_growth(quarter_fcf, prior_year_quarter_fcf),
            annual_growth(first_item(annual_fcf_series), first_item(annual_fcf_series, 1)),
            annual_growth(first_item(ttm_fcf_series), first_item(ttm_fcf_series, 1)),
        )
        dividend_growth = first_non_null(
            annual_growth(dividend_per_share, dividends_per_share_series[1] if len(dividends_per_share_series) > 1 else None),
            annual_growth(first_item(dividends_paid_series), first_item(dividends_paid_series, 1)),
        )
        eps_growth = first_non_null(
            annual_growth(quarter_eps, prior_year_quarter_eps),
            annual_growth(diluted_eps, previous_eps),
            normalize_fraction(safe_float(info.get("earningsQuarterlyGrowth"))),
            normalize_fraction(safe_float(info.get("earningsGrowth"))),
        )

        debt_to_equity = first_non_null(
            safe_div(debt, equity),
            nested_number(massive_ratios, ("debt_to_equity",)),
            normalize_fraction(
                pick_number(
                    fmp_ratios.get("debtEquityRatioTTM") if isinstance(fmp_ratios, dict) else None,
                    info.get("debtToEquity"),
                )
            ),
        )
        interest_coverage = safe_div(ebit, abs(interest_expense) if has_value(interest_expense) else None)
        current_ratio = first_non_null(
            safe_div(current_assets, current_liabilities),
            nested_number(massive_ratios, ("current",)),
            pick_number(
                fmp_ratios.get("currentRatioTTM") if isinstance(fmp_ratios, dict) else None,
                alpha_overview.get("CurrentRatio") if isinstance(alpha_overview, dict) else None,
                info.get("currentRatio"),
                yahoo_web.get("currentRatio"),
            ),
        )
        debt_ebitda = safe_div(debt, ebitda)
        cash_ratio = first_non_null(
            safe_div((cash or 0) + (short_term_investments or 0), current_liabilities),
            nested_number(massive_ratios, ("cash",)),
            pick_number(fmp_ratios.get("cashRatioTTM") if isinstance(fmp_ratios, dict) else None),
        )
        debt_fcf_ratio = safe_div(debt, fcf)
        quick_assets = (current_assets - (inventory or 0)) if has_value(current_assets) else None
        quick_ratio = first_non_null(
            safe_div(quick_assets, current_liabilities),
            nested_number(massive_ratios, ("quick",)),
            pick_number(
                fmp_ratios.get("quickRatioTTM") if isinstance(fmp_ratios, dict) else None,
                alpha_overview.get("QuickRatio") if isinstance(alpha_overview, dict) else None,
                info.get("quickRatio"),
                yahoo_web.get("quickRatio"),
            ),
        )

        altman_z = None
        if all(has_value(v) for v in [current_assets, current_liabilities, total_assets, retained_earnings, ebit, equity, total_liabilities, revenue]):
            altman_z = (
                0.717 * (((current_assets - current_liabilities) / total_assets))
                + 0.847 * (retained_earnings / total_assets)
                + 3.107 * (ebit / total_assets)
                + 0.42 * (equity / total_liabilities if total_liabilities else 0)
                + 0.998 * (revenue / total_assets)
            )

        operating_cash_flow_margin = safe_div(ocf, revenue)
        fcf_yield = first_non_null(
            safe_div(fcf, market_cap),
            pick_number(fmp_key_metrics.get("freeCashFlowYieldTTM") if isinstance(fmp_key_metrics, dict) else None),
        )
        capex_to_sales = safe_div(abs(capex) if has_value(capex) else None, revenue)
        cfo_net_income = safe_div(ocf, net_income)
        capex_cfo = safe_div(capex, ocf)
        accrual_ratio = safe_div((net_income or 0) - (ocf or 0), total_assets)

        ffo = None
        affo = None
        p_ffo = None
        p_affo = None
        ffo_yield = None
        payout_ratio_ffo = None
        noi = None
        occupancy = None
        affo_yield = None
        if is_reit:
            lease_income = to_millions(
                facts.latest_annual_value(
                    ["LeaseIncome", "OperatingLeaseLeaseIncome", "OperatingLeasesIncomeStatementLeaseRevenue", "LessorOperatingLeaseIncome"],
                    unit_preferences=["USD"],
                    max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS,
                )
            )
            direct_costs = to_millions(
                facts.latest_annual_value(
                    ["DirectCostsOfLeasedAndRentedPropertyOrEquipment", "PropertyOperatingExpense"],
                    unit_preferences=["USD"],
                    max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS,
                )
            )
            ffo = (net_income or 0) + (depreciation or 0) + (amortization or 0)
            affo = ffo + (capex or 0)
            p_ffo = safe_div(market_cap, ffo)
            p_affo = safe_div(market_cap, affo)
            ffo_yield = safe_div(ffo, market_cap)
            payout_ratio_ffo = safe_div(dividends_paid, ffo)
            noi = first_non_null(
                (lease_income - direct_costs) if has_value(lease_income) and has_value(direct_costs) else None,
                operating_income,
            )
            occupancy = self.sec.extract_recent_occupancy_rate(cik)
            affo_yield = safe_div(affo, market_cap)

        metrics = {
            "forward_pe": forward_pe,
            "peg_ratio": peg_ratio,
            "price_to_book": price_to_book,
            "price_to_sales": price_to_sales,
            "ev_ebitda": ev_ebitda,
            "ev_sales": ev_sales,
            "pe_ratio": pe_ratio,
            "price_fcf_ratio": price_fcf_ratio,
            "earnings_yield": earnings_yield,
            "ffo": ffo,
            "affo": affo,
            "p_ffo": p_ffo,
            "p_affo": p_affo,
            "ffo_yield": ffo_yield,
            "payout_ratio_ffo": payout_ratio_ffo,
            "noi": noi,
            "occupancy": occupancy,
            "affo_yield": affo_yield,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "net_margin": net_margin,
            "roe": roe,
            "roa": roa,
            "roic": roic,
            "fcf_margin": fcf_margin,
            "ebit_margin": ebit_margin,
            "economic_profit_spread": economic_profit_spread,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "free_cash_flow_growth": free_cash_flow_growth,
            "dividend_growth": dividend_growth,
            "eps_growth": eps_growth,
            "debt_to_equity": debt_to_equity,
            "interest_coverage": interest_coverage,
            "current_ratio": current_ratio,
            "debt_ebitda": debt_ebitda,
            "cash_ratio": cash_ratio,
            "debt_fcf_ratio": debt_fcf_ratio,
            "quick_ratio": quick_ratio,
            "altman_z": altman_z,
            "free_cash_flow": fcf,
            "operating_cash_flow_margin": operating_cash_flow_margin,
            "fcf_yield": fcf_yield,
            "capex_to_sales": capex_to_sales,
            "cfo_net_income": cfo_net_income,
            "capex_cfo": capex_cfo,
            "accrual_ratio": accrual_ratio,
            "revenue": revenue,
            "current_price": price,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
        }

        scores = {
            "valuation": valuation_score(metrics),
            "profitability": profitability_score(metrics),
            "growth": growth_score(metrics),
            "financial_strength": financial_strength_score(metrics),
            "cash_flow": cash_flow_score(metrics),
        }
        scores["composite"] = composite_score(scores)

        profile = {
            "symbol": symbol,
            "name": clean_text(submissions.get("name") or info.get("longName") or symbol),
            "sector": sector,
            "industry": clean_text(info.get("industry") or yahoo_web.get("industry") or submissions.get("sicDescription")),
            "ipo_date": format_ipo_date(history),
            "ceo": clean_text(first_non_null(
                officer_name(info),
                self.sec.extract_named_executive(cik),
            )),
            "employees": info.get("fullTimeEmployees"),
            "country": clean_text(info.get("country")),
            "exchange": clean_text(first_non_null((submissions.get("exchanges") or [None])[0], info.get("exchange"), yahoo_web.get("exchange"))),
            "cik": f"{cik:010d}",
        }

        risk_rating = build_risk_rating(metrics, sector)
        recommendation = build_recommendation(
            metrics,
            scores,
            profile,
            current_hold=current_hold,
            macro_score=macro_score,
            risk_rating=risk_rating,
        )

        warnings: list[str] = []
        if is_reit and occupancy is None:
            warnings.append("REIT occupancy was not found in the latest SEC filing text, so occupancy-driven REIT logic may be incomplete.")
        if metrics["forward_pe"] is None:
            warnings.append("Forward P/E was unavailable from the market data source.")
        if metrics["peg_ratio"] is None:
            warnings.append("PEG ratio was unavailable or invalid from the market data source.")
        if is_reit:
            warnings.append("REIT FFO and AFFO are derived from filing-backed net income plus depreciation and amortization when company-reported FFO or AFFO are not exposed as standardized SEC facts.")

        score_cards = [
            {"label": "Valuation", "value": format_number(scores["valuation"])},
            {"label": "Profitability", "value": format_number(scores["profitability"])},
            {"label": "Growth", "value": format_number(scores["growth"])},
            {"label": "Financial Strength", "value": format_number(scores["financial_strength"])},
            {"label": "Cash Flow", "value": format_number(scores["cash_flow"])},
            {"label": "Composite", "value": format_number(scores["composite"], 4)},
        ]

        sections = [
            {
                "title": "Valuation",
                "items": [
                    {"label": "Forward P/E", "value": format_multiple(metrics["forward_pe"])},
                    {"label": "PEG Ratio", "value": format_number(metrics["peg_ratio"], 3)},
                    {"label": "Price to Book (P/B)", "value": format_multiple(metrics["price_to_book"])},
                    {"label": "Price to Sales (P/S)", "value": format_multiple(metrics["price_to_sales"])},
                    {"label": "EV / EBITDA", "value": format_multiple(metrics["ev_ebitda"])},
                    {"label": "EV / Sales", "value": format_multiple(metrics["ev_sales"])},
                    {"label": "P/E", "value": format_multiple(metrics["pe_ratio"])},
                    {"label": "Price / FCF", "value": format_multiple(metrics["price_fcf_ratio"])},
                    {"label": "Earnings Yield", "value": format_percent(metrics["earnings_yield"])},
                ],
            },
            {
                "title": "Profitability",
                "items": [
                    {"label": "Gross Margin", "value": format_percent(metrics["gross_margin"])},
                    {"label": "Operating Margin", "value": format_percent(metrics["operating_margin"])},
                    {"label": "Net Margin", "value": format_percent(metrics["net_margin"])},
                    {"label": "ROE", "value": format_percent(metrics["roe"])},
                    {"label": "ROA", "value": format_percent(metrics["roa"])},
                    {"label": "ROIC", "value": format_percent(metrics["roic"])},
                    {"label": "Free Cash Flow Margin", "value": format_percent(metrics["fcf_margin"])},
                    {"label": "EBIT Margin", "value": format_percent(metrics["ebit_margin"])},
                    {"label": "Economic Profit Spread", "value": format_percent(metrics["economic_profit_spread"])},
                ],
            },
            {
                "title": "Growth",
                "items": [
                    {"label": "Revenue Growth (YoY)", "value": format_percent(metrics["revenue_growth"])},
                    {"label": "Earnings Growth (YoY)", "value": format_percent(metrics["earnings_growth"])},
                    {"label": "Free Cash Flow Growth (YoY)", "value": format_percent(metrics["free_cash_flow_growth"])},
                    {"label": "Dividend Growth (YoY)", "value": format_percent(metrics["dividend_growth"])},
                    {"label": "EPS Growth", "value": format_percent(metrics["eps_growth"])},
                ],
            },
            {
                "title": "Financial Strength",
                "items": [
                    {"label": "Debt / Equity", "value": format_number(metrics["debt_to_equity"])},
                    {"label": "Interest Coverage", "value": format_number(metrics["interest_coverage"])},
                    {"label": "Current Ratio", "value": format_number(metrics["current_ratio"])},
                    {"label": "Debt / EBITDA", "value": format_number(metrics["debt_ebitda"])},
                    {"label": "Cash Ratio", "value": format_number(metrics["cash_ratio"])},
                    {"label": "Debt / FCF Ratio", "value": format_number(metrics["debt_fcf_ratio"])},
                    {"label": "Quick Ratio", "value": format_number(metrics["quick_ratio"])},
                    {"label": "Altman Z", "value": format_number(metrics["altman_z"])},
                ],
            },
            {
                "title": "Cash Flow Quality",
                "items": [
                    {"label": "Free Cash Flow (USD mm)", "value": format_money_millions(metrics["free_cash_flow"])},
                    {"label": "Operating Cash Flow Margin", "value": format_percent(metrics["operating_cash_flow_margin"])},
                    {"label": "FCF Yield", "value": format_percent(metrics["fcf_yield"])},
                    {"label": "CapEx to Sales", "value": format_percent(metrics["capex_to_sales"])},
                    {"label": "CFO / Net Income", "value": format_number(metrics["cfo_net_income"])},
                    {"label": "CapEx / CFO", "value": format_number(metrics["capex_cfo"])},
                    {"label": "Accrual Ratio", "value": format_percent(metrics["accrual_ratio"])},
                ],
            },
        ]
        if profile["sector"] == "Real Estate":
            sections.insert(
                1,
                {
                    "title": "REIT Metrics",
                    "items": [
                        {"label": "FFO (USD mm)", "value": format_money_millions(metrics["ffo"])},
                        {"label": "AFFO (USD mm)", "value": format_money_millions(metrics["affo"])},
                        {"label": "P / FFO", "value": format_multiple(metrics["p_ffo"])},
                        {"label": "P / AFFO", "value": format_multiple(metrics["p_affo"])},
                        {"label": "FFO Yield", "value": format_percent(metrics["ffo_yield"])},
                        {"label": "AFFO Yield", "value": format_percent(metrics["affo_yield"])},
                        {"label": "Payout Ratio (FFO)", "value": format_percent(metrics["payout_ratio_ffo"])},
                        {"label": "NOI (USD mm)", "value": format_money_millions(metrics["noi"])},
                        {"label": "Occupancy", "value": format_percent(metrics["occupancy"])},
                    ],
                },
            )

        recommendation_lines = [
            f"{recommendation['label']} — Make this about {recommendation['allocation']}% of your portfolio — Score {recommendation['final_score']}",
            f"Risk Rating: {risk_rating}",
            f"Risk Score: {recommendation['risk_score']} | Valuation Stretch: {recommendation['valuation_stretch']}x",
            f"Macro Score Used: {macro_score:.0f}",
        ]
        if profile["sector"] == "Real Estate":
            recommendation_lines.extend(
                [
                    f"P/FFO: {format_multiple(metrics['p_ffo'])}",
                    f"P/AFFO: {format_multiple(metrics['p_affo'])}",
                    f"FFO Yield: {format_percent(metrics['ffo_yield'])}",
                    f"AFFO Yield: {format_percent(metrics['affo_yield'])}",
                ]
            )
        else:
            recommendation_lines.extend(
                [
                    f"Forward P/E: {format_multiple(metrics['forward_pe'])}",
                    f"PEG Ratio: {format_number(metrics['peg_ratio'], 3)}",
                    f"ROIC: {format_percent(metrics['roic'])}",
                    f"FCF Margin: {format_percent(metrics['fcf_margin'])}",
                ]
            )

        formula_output_text = "\n".join(recommendation_lines)

        sheet_sections = [
            {
                "title": "Sheet Header",
                "rows": [
                    make_sheet_row("symbol", profile["symbol"]),
                    make_sheet_row("sector", profile["sector"] or "N/A"),
                    make_sheet_row("Industry", profile["industry"] or "N/A"),
                    make_sheet_row("IPO Date", profile["ipo_date"] or "N/A"),
                    make_sheet_row("CEO", profile["ceo"] or "N/A"),
                    make_sheet_row(
                        "Employees",
                        f"{profile['employees']:,}" if profile["employees"] else "N/A",
                    ),
                    make_sheet_row("Country", profile["country"] or "N/A"),
                    make_sheet_row("Exchange", profile["exchange"] or "N/A"),
                ],
            },
            {
                "title": "Final Fundamentals Layout",
                "rows": [
                    make_sheet_row("Forward P/E", formula_value(metrics["forward_pe"], 2), "FFO", formula_value(metrics["ffo"], 2)),
                    make_sheet_row("PEG Ratio", formula_value(metrics["peg_ratio"], 3), "AFFO", formula_value(metrics["affo"], 2)),
                    make_sheet_row("Price to Book (P/B)", formula_value(metrics["price_to_book"], 2), "P/FFO", formula_value(metrics["p_ffo"], 2)),
                    make_sheet_row("Price to Sales (P/S)", formula_value(metrics["price_to_sales"], 2), "P/AFFO", formula_value(metrics["p_affo"], 2)),
                    make_sheet_row("EV / EBITDA", formula_value(metrics["ev_ebitda"], 2), "FFO Yield", formula_value(metrics["ffo_yield"], 4)),
                    make_sheet_row("EV / Sales", formula_value(metrics["ev_sales"], 2), "Payout Ratio (FFO)", formula_value(metrics["payout_ratio_ffo"], 4)),
                    make_sheet_row("Pe", formula_value(metrics["pe_ratio"], 2), "NOI", formula_value(metrics["noi"], 2)),
                    make_sheet_row("Price/FCF Ratio", formula_value(metrics["price_fcf_ratio"], 2), "Occupancy", formula_value(metrics["occupancy"], 4)),
                    make_sheet_row("Earnings Yield", formula_value(metrics["earnings_yield"], 11), "AFFO Yield", formula_value(metrics["affo_yield"], 4)),
                    make_sheet_row("Valuation Score (avg)", formula_value(scores["valuation"], 2)),
                ],
            },
            {
                "title": "Profitability",
                "rows": [
                    make_sheet_row("Gross Margin %", formula_value(metrics["gross_margin"], 4)),
                    make_sheet_row("Operating Margin %", formula_value(metrics["operating_margin"], 4)),
                    make_sheet_row("Net Margin %", formula_value(metrics["net_margin"], 4)),
                    make_sheet_row("ROE %", formula_value(metrics["roe"], 4)),
                    make_sheet_row("ROA %", formula_value(metrics["roa"], 4)),
                    make_sheet_row("ROIC %", formula_value(metrics["roic"], 4)),
                    make_sheet_row("Free Cash Flow Margin", formula_value(metrics["fcf_margin"], 4)),
                    make_sheet_row("EBIT Margin", formula_value(metrics["ebit_margin"], 4)),
                    make_sheet_row("Economic Profit Spread", formula_value(metrics["economic_profit_spread"], 4)),
                    make_sheet_row("Profitability Score (avg)", formula_value(scores["profitability"], 2)),
                ],
            },
            {
                "title": "Growth",
                "rows": [
                    make_sheet_row("Revenue Growth (YoY)", formula_value(metrics["revenue_growth"], 4)),
                    make_sheet_row("Earnings Growth (YoY)", formula_value(metrics["earnings_growth"], 4)),
                    make_sheet_row("Free Cash Flow Growth (YoY)", formula_value(metrics["free_cash_flow_growth"], 4)),
                    make_sheet_row("Dividend Growth (YoY)", formula_value(metrics["dividend_growth"], 4)),
                    make_sheet_row("EPS Growth", formula_value(metrics["eps_growth"], 4)),
                    make_sheet_row("Growth Score (avg)", formula_value(scores["growth"], 2)),
                ],
            },
            {
                "title": "Financial Strength / Solvency",
                "rows": [
                    make_sheet_row("Debt / Equity", formula_value(metrics["debt_to_equity"], 4)),
                    make_sheet_row("Interest Coverage", formula_value(metrics["interest_coverage"], 11)),
                    make_sheet_row("Current Ratio", formula_value(metrics["current_ratio"], 4)),
                    make_sheet_row("Debt / EBITDA", formula_value(metrics["debt_ebitda"], 4)),
                    make_sheet_row("Cash Ratio", formula_value(metrics["cash_ratio"], 4)),
                    make_sheet_row("Debt / FCF Ratio", formula_value(metrics["debt_fcf_ratio"], 4)),
                    make_sheet_row("Quick Ratio", formula_value(metrics["quick_ratio"], 4)),
                    make_sheet_row('Altman Z"', formula_value(metrics["altman_z"], 9)),
                    make_sheet_row("Financial Strength Score (avg)", formula_value(scores["financial_strength"], 2)),
                ],
            },
            {
                "title": "Cash Flow Quality",
                "rows": [
                    make_sheet_row("Free Cash Flow", formula_value(metrics["free_cash_flow"], 4)),
                    make_sheet_row("Operating Cash Flow Margin", formula_value(metrics["operating_cash_flow_margin"], 9)),
                    make_sheet_row("FCF Yield", formula_value(metrics["fcf_yield"], 4)),
                    make_sheet_row("CapEx to Sales", formula_value(metrics["capex_to_sales"], 9)),
                    make_sheet_row("CFO / Net Income", formula_value(metrics["cfo_net_income"], 9)),
                    make_sheet_row("CapEx / CFO", formula_value(metrics["capex_cfo"], 10)),
                    make_sheet_row("Accrual Ratio", formula_value(metrics["accrual_ratio"], 11)),
                    make_sheet_row("Cash Flow Score (avg)", formula_value(scores["cash_flow"], 2)),
                    make_sheet_row("Fundamental Composite Score (0-100)", formula_value(scores["composite"], 4)),
                    make_sheet_row(risk_rating, ""),
                ],
            },
            {
                "title": "Final Formula Output",
                "rows": [
                    make_sheet_row("already holding? if not put 0 if yes put percentage of how much you are holding", formula_value(current_hold, 1)),
                    make_sheet_row("Regime Score (0-100)", formula_value(macro_score, 2)),
                    make_sheet_row("Final Recommendation", formula_output_text),
                ],
            },
        ]

        return {
            "profile": profile,
            "metrics": metrics,
            "scores": scores,
            "risk_rating": risk_rating,
            "recommendation": recommendation,
            "recommendation_lines": recommendation_lines,
            "formula_output_text": formula_output_text,
            "sheet_sections": sheet_sections,
            "score_cards": score_cards,
            "sections": sections,
            "current_hold": current_hold,
            "macro_score": macro_score,
            "warnings": warnings,
            "source_notes": [
            "SEC EDGAR company submissions, companyfacts, and inline filing parsing are used first for filing-backed raw fundamentals.",
            "Yahoo Finance direct HTML parsing and yfinance are the default no-key fallbacks for price, profile, and market-derived fields.",
            "Growth metrics prefer current quarter versus prior-year quarter comparisons from SEC filing data, then fall back to annual and TTM filing comparisons.",
            "REIT FFO and AFFO are derived from filing-backed components unless the latest filing exposes cleaner company-reported values.",
            "Massive, Financial Modeling Prep, and Alpha Vantage remain optional extra fallbacks when their API keys are configured.",
            "The original sheet's custom EP_SPREAD() function was approximated as ROIC minus a 10.84% cost of capital because the underlying Sheets function logic was not provided.",
        ],
        }
