from __future__ import annotations

import datetime as dt
import re
from typing import Any

from .data_sources import (
    FactStore,
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
    asset_manager_scores,
    bank_scores,
    bdc_scores,
    build_recommendation,
    build_risk_rating,
    cash_flow_score,
    clamp,
    capital_markets_scores,
    composite_score,
    financial_strength_score,
    growth_score,
    has_value,
    insurance_scores,
    midstream_scores,
    profitability_score,
    safe_div,
    utility_scores,
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


def classify_company_type(
    *,
    sector: str,
    industry: str | None,
    sic_description: str | None,
    company_name: str | None,
    fact_concepts: set[str] | None = None,
) -> str:
    text = " ".join([industry or "", sic_description or "", company_name or ""]).lower()
    concepts = {concept.lower() for concept in (fact_concepts or set())}

    if sector == "Real Estate":
        return "reit"
    if sector == "Utilities":
        return "utility"
    if sector == "Energy" and ("midstream" in text or "natural gas transmission" in text or "pipeline" in text or "partners l.p." in text):
        return "midstream"
    if sector != "Financials":
        return "general"
    if "insurance" in text or "property & casualty" in text:
        return "insurance"
    has_investment_company_facts = any(concept.startswith("investmentcompany") for concept in concepts)
    looks_like_bdc = (
        "business development" in text
        or "bdc" in text
        or ("asset management" in text and ("capital" in text or "finance corp" in text))
        or (has_investment_company_facts and ("asset management" in text or "capital" in text))
    )
    if looks_like_bdc:
        return "bdc"
    if "bank" in text or "savings institution" in text or "commercial banks" in text:
        return "bank"
    if "asset management" in text or "investment advice" in text or "investment advisory" in text:
        return "asset_manager"
    if "capital markets" in text or "security brokers" in text or "broker" in text or "dealers" in text:
        return "capital_markets"
    return "general"


def classify_industry_group(*, sector: str, industry: str | None, sic_description: str | None, company_name: str | None) -> str:
    text = " ".join([industry or "", sic_description or "", company_name or ""]).lower()
    if "telecom" in text or "telecommunications" in text:
        return "telecom"
    if "auto manufacturer" in text or "automobile" in text or "motor vehicle" in text:
        return "autos"
    if "airline" in text or "air transport" in text:
        return "airlines"
    if "semiconductor" in text:
        return "semiconductors"
    if "software" in text or "saas" in text or "cloud" in text:
        return "saas"
    if "pharma" in text or "drug" in text or "biotechnology" in text:
        return "pharma"
    if "restaurant" in text or "coffee" in text:
        return "restaurants"
    if "retail" in text or "department store" in text or "grocery" in text or "apparel" in text:
        return "retail"
    if "aerospace" in text or "defense" in text or "aircraft" in text:
        return "aerospace_defense"
    return "general"


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

    def analyze(
        self,
        symbol: str,
        current_hold: float = 0,
        macro_score: float = 50,
        sec_name: str | None = None,
        sec_email: str | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        sec = self.sec.for_identity(sec_name, sec_email)
        cik = sec.lookup_cik(symbol)
        submissions = sec.fetch_submissions(cik)
        companyfacts = sec.fetch_companyfacts(cik)
        yahoo = self.yahoo.fetch_snapshot(symbol)
        yahoo_web = self.yahoo_web.fetch_metrics(symbol)
        fmp_quote: dict[str, Any] = {}
        fmp_key_metrics: dict[str, Any] = {}
        fmp_ratios: dict[str, Any] = {}
        fmp_estimates: list[dict[str, Any]] = []
        alpha_quote: dict[str, Any] = {}
        alpha_overview: dict[str, Any] = {}
        alpha_estimates_payload: dict[str, Any] = {}
        info = yahoo["info"]
        history = yahoo["history"]
        facts = FactStore(companyfacts)
        industry_raw = clean_text(info.get("industry") or yahoo_web.get("industry") or submissions.get("sicDescription"))
        sic_description = clean_text(submissions.get("sicDescription"))
        company_name = clean_text(submissions.get("name") or info.get("longName") or symbol)
        fact_concepts = {
            concept
            for taxonomy in companyfacts.get("facts", {}).values()
            if isinstance(taxonomy, dict)
            for concept in taxonomy
        }
        sector = normalize_sector_name(info.get("sector") or yahoo_web.get("sector") or sic_description)
        company_type = classify_company_type(
            sector=sector,
            industry=industry_raw,
            sic_description=sic_description,
            company_name=company_name,
            fact_concepts=fact_concepts,
        )
        industry_group = classify_industry_group(
            sector=sector,
            industry=industry_raw,
            sic_description=sic_description,
            company_name=company_name,
        )
        is_reit = company_type == "reit"
        is_bank = company_type in {"bank", "capital_markets"}
        is_insurance = company_type == "insurance"
        is_bdc = company_type == "bdc"
        is_utility = company_type == "utility"
        is_midstream = company_type == "midstream"
        is_asset_manager = company_type == "asset_manager"
        is_capital_markets = company_type == "capital_markets"
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
                filing_value = sec.recent_filing_value(cik, filing_concepts, kind="instant", unit_preferences=["usd"])
                if filing_value is not None:
                    return to_millions(filing_value)
            return None

        def sum_latest_instant_millions(
            concepts: list[str],
            filing_concepts: list[str] | None = None,
        ) -> float | None:
            values: list[float] = []
            for concept in concepts:
                value = facts.latest_instant_value([concept], unit_preferences=["USD"], max_age_days=RECENT_QUARTER_MAX_AGE_DAYS)
                if value is not None:
                    values.append(to_millions(value))
            if values:
                return sum(values)
            if filing_concepts:
                filing_value = sec.recent_filing_value(cik, filing_concepts, kind="instant", unit_preferences=["usd"])
                if filing_value is not None:
                    return to_millions(filing_value)
            return None

        def latest_instant_ratio(concepts: list[str], filing_concepts: list[str] | None = None) -> float | None:
            for concept in concepts:
                value = facts.latest_instant_value([concept], unit_preferences=["pure"], max_age_days=RECENT_QUARTER_MAX_AGE_DAYS)
                normalized = normalize_fraction(value)
                if normalized is not None:
                    return normalized
            if filing_concepts:
                filing_value = sec.recent_filing_value(cik, filing_concepts, kind="instant", unit_preferences=["pure"])
                normalized = normalize_fraction(filing_value)
                if normalized is not None:
                    return normalized
            return None

        def instant_growth_millions(concepts: list[str]) -> float | None:
            for concept in concepts:
                entries = facts.instant_series([concept], unit_preferences=["USD"], limit=8, max_age_days=RECENT_ANNUAL_MAX_AGE_DAYS)
                if not entries:
                    continue
                current = entries[0]
                current_date = parse_date(current.get("instant") or current.get("end"))
                current_value = to_millions(current.get("val"))
                if current_date is None or current_value is None:
                    continue
                best_value = None
                best_gap = None
                for entry in entries[1:]:
                    entry_date = parse_date(entry.get("instant") or entry.get("end"))
                    entry_value = to_millions(entry.get("val"))
                    if entry_date is None or entry_value in (None, 0):
                        continue
                    delta = (current_date - entry_date).days
                    if delta < 330 or delta > 400:
                        continue
                    gap = abs(delta - 365)
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_value = entry_value
                growth = annual_growth(current_value, best_value)
                if growth is not None:
                    return growth
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
            sum_latest_instant_millions(
                ["RetainedEarningsAppropriated", "RetainedEarningsUnappropriated"],
                [
                    "us-gaap:RetainedEarningsAppropriated",
                    "us-gaap:RetainedEarningsUnappropriated",
                ],
            ),
            latest_instant_millions(
                [
                    "RetainedEarningsAccumulatedDeficit",
                    "RetainedEarningsUnappropriated",
                ],
                [
                    "us-gaap:RetainedEarningsAccumulatedDeficit",
                    "us-gaap:RetainedEarningsUnappropriated",
                ],
            ),
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

        sec_shares_outstanding = facts.latest_instant_value(
            ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
            unit_preferences=["shares"],
            max_age_days=RECENT_QUARTER_MAX_AGE_DAYS,
        )
        massive_snapshot = None
        massive_overview = None
        massive_ratios = None

        shares_outstanding = first_non_null(
            sec_shares_outstanding,
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
        reported_market_cap = pick_number(
            yahoo["fast_info"].get("marketCap"),
            info.get("marketCap"),
            yahoo_web.get("marketCap"),
        )
        if has_value(computed_market_cap) and has_value(reported_market_cap) and reported_market_cap:
            cap_ratio = computed_market_cap / reported_market_cap
            if cap_ratio < 0.4 or cap_ratio > 2.5:
                computed_market_cap = None
        market_cap = to_millions(
            first_non_null(
                computed_market_cap,
                nested_number(massive_ratios, ("market_cap",)),
                nested_number(massive_overview, ("market_cap",)),
                pick_number(
                    reported_market_cap,
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
            sec.recent_filing_value(
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
        if is_bank and not (gross_profit_ttm_series or gross_profit_annual_series or gross_profit_quarter_entries):
            gross_margin = None

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

        pe_by_share = safe_div(price, diluted_eps)
        pe_by_market_cap = safe_div(market_cap, net_income)
        if has_value(pe_by_share) and has_value(pe_by_market_cap) and pe_by_market_cap:
            pe_ratio_gap = pe_by_share / pe_by_market_cap
            if pe_ratio_gap < 0.4 or pe_ratio_gap > 2.5:
                pe_by_share = None
        pe_ratio = first_non_null(
            pe_by_share,
            pe_by_market_cap,
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
            occupancy = sec.extract_recent_occupancy_rate(cik)
            affo_yield = safe_div(affo, market_cap)

        bank_deposits = None
        bank_loans = None
        bank_loans_to_deposits = None
        bank_deposits_to_assets = None
        bank_equity_to_assets = None
        bank_assets_to_equity = None
        bank_tangible_equity = None
        bank_tangible_assets = None
        bank_tangible_equity_to_tangible_assets = None
        bank_tangible_book_value_per_share = None
        bank_price_to_tangible_book = None
        bank_allowance_for_credit_losses = None
        bank_allowance_to_loans = None
        bank_provision_for_credit_losses = None
        bank_provision_to_loans = None
        bank_net_interest_income = None
        bank_net_interest_income_to_assets = None
        bank_net_interest_income_growth = None
        bank_noninterest_income = None
        bank_noninterest_expense = None
        bank_efficiency_ratio = None
        bank_capital_to_rwa = None
        bank_tier1_capital_ratio = None
        bank_tier1_leverage_ratio = None
        bank_risk_weighted_assets = None
        bank_tier1_capital = None
        bank_deposit_growth = None
        bank_loan_growth = None
        if is_bank:
            deposit_concepts = ["Deposits"]
            loan_concepts = [
                "LoansAndLeasesReceivableNetReportedAmount",
                "LoansAndLeasesReceivableNetOfDeferredIncome",
                "LoansReceivableNet",
                "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
                "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
            ]
            allowance_concepts = [
                "FinancingReceivableAllowanceForCreditLosses",
                "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
                "LoansAndLeasesReceivableAllowance",
                "AllowanceForLoanAndLeaseLosses",
            ]
            net_interest_concepts = ["InterestIncomeExpenseNet", "NetInterestIncome", "InterestIncomeExpenseAfterProvisionForLoanLoss"]
            noninterest_income_concepts = ["NoninterestIncome", "NoninterestIncomeOtherOperatingIncome", "NoninterestIncomeOther"]
            noninterest_expense_concepts = ["NoninterestExpense", "OtherNoninterestExpense"]
            provision_concepts = [
                "FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal",
                "ProvisionForLoanLeaseAndOtherLosses",
                "ProvisionForLoanAndLeaseLosses",
                "ProvisionForLoanLossesExpensed",
                "SegmentReportingInformationProvisionForCreditLosses",
            ]
            bank_capital_filing_metrics = sec.extract_recent_bank_capital_metrics(cik)

            bank_deposits = first_non_null(
                latest_instant_millions(deposit_concepts, ["us-gaap:Deposits"]),
                sum_latest_instant_millions(
                    ["DepositsDomestic", "DepositsForeign"],
                    ["us-gaap:DepositsDomestic", "us-gaap:DepositsForeign"],
                ),
                sum_latest_instant_millions(
                    ["InterestBearingDepositLiabilities", "NoninterestBearingDepositLiabilities"],
                    ["us-gaap:InterestBearingDepositLiabilities", "us-gaap:NoninterestBearingDepositLiabilities"],
                ),
            )
            bank_loans = latest_instant_millions(
                loan_concepts,
                [
                    "us-gaap:LoansAndLeasesReceivableNetReportedAmount",
                    "us-gaap:LoansAndLeasesReceivableNetOfDeferredIncome",
                    "us-gaap:LoansReceivableNet",
                    "us-gaap:FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
                    "us-gaap:FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
                ],
            )
            bank_allowance_for_credit_losses = latest_instant_millions(
                allowance_concepts,
                [
                    "us-gaap:FinancingReceivableAllowanceForCreditLosses",
                    "us-gaap:FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
                    "us-gaap:LoansAndLeasesReceivableAllowance",
                    "us-gaap:AllowanceForLoanAndLeaseLosses",
                ],
            )
            bank_net_interest_income_series = ttm_series_millions(net_interest_concepts) or annual_series_millions(net_interest_concepts)
            bank_noninterest_income_series = ttm_series_millions(noninterest_income_concepts) or annual_series_millions(noninterest_income_concepts)
            bank_noninterest_expense_series = ttm_series_millions(noninterest_expense_concepts) or annual_series_millions(noninterest_expense_concepts)
            bank_provision_series = ttm_series_millions(provision_concepts) or annual_series_millions(provision_concepts)
            bank_net_interest_income = first_item(bank_net_interest_income_series)
            bank_noninterest_income = first_item(bank_noninterest_income_series)
            bank_noninterest_expense = first_item(bank_noninterest_expense_series)
            bank_provision_for_credit_losses = first_item(bank_provision_series)
            bank_net_interest_income_growth = annual_growth(first_item(bank_net_interest_income_series), first_item(bank_net_interest_income_series, 1))

            goodwill = latest_instant_millions(["Goodwill"], ["us-gaap:Goodwill"])
            finite_intangibles = latest_instant_millions(
                ["FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNetExcludingGoodwill", "OtherIntangibleAssetsNet"],
                [
                    "us-gaap:FiniteLivedIntangibleAssetsNet",
                    "us-gaap:IntangibleAssetsNetExcludingGoodwill",
                    "us-gaap:OtherIntangibleAssetsNet",
                ],
            )
            intangible_adjustment = (goodwill or 0) + (finite_intangibles or 0)
            bank_tangible_equity = equity - intangible_adjustment if has_value(equity) else None
            bank_tangible_assets = total_assets - intangible_adjustment if has_value(total_assets) else None
            bank_tangible_equity_to_tangible_assets = safe_div(bank_tangible_equity, bank_tangible_assets)
            bank_tangible_book_value_per_share = safe_div((bank_tangible_equity or 0) * 1_000_000, shares_outstanding) if has_value(bank_tangible_equity) else None
            bank_price_to_tangible_book = first_non_null(
                safe_div(market_cap, bank_tangible_equity),
                safe_div(price, bank_tangible_book_value_per_share),
            )

            bank_loans_to_deposits = safe_div(bank_loans, bank_deposits)
            bank_deposits_to_assets = safe_div(bank_deposits, total_assets)
            bank_equity_to_assets = safe_div(equity, total_assets)
            bank_assets_to_equity = safe_div(total_assets, equity)
            bank_allowance_to_loans = safe_div(abs(bank_allowance_for_credit_losses) if has_value(bank_allowance_for_credit_losses) else None, bank_loans)
            bank_provision_to_loans = safe_div(abs(bank_provision_for_credit_losses) if has_value(bank_provision_for_credit_losses) else None, bank_loans)
            bank_net_interest_income_to_assets = safe_div(bank_net_interest_income, avg_assets)
            bank_efficiency_ratio = safe_div(bank_noninterest_expense, (bank_net_interest_income or 0) + (bank_noninterest_income or 0))
            bank_capital_to_rwa = first_non_null(
                bank_capital_filing_metrics.get("cet1_capital_ratio"),
                latest_instant_ratio(
                    ["CapitalToRiskWeightedAssets", "CommonEquityTierOneCapitalRatio"],
                    ["us-gaap:CapitalToRiskWeightedAssets", "us-gaap:CommonEquityTierOneCapitalRatio"],
                ),
            )
            bank_tier1_capital_ratio = first_non_null(
                latest_instant_ratio(
                    ["TierOneRiskBasedCapitalToRiskWeightedAssets"],
                    ["us-gaap:TierOneRiskBasedCapitalToRiskWeightedAssets"],
                ),
                bank_capital_filing_metrics.get("tier1_capital_ratio"),
            )
            bank_tier1_leverage_ratio = first_non_null(
                latest_instant_ratio(
                    ["TierOneLeverageCapitalToAverageAssets"],
                    ["us-gaap:TierOneLeverageCapitalToAverageAssets"],
                ),
                bank_capital_filing_metrics.get("tier1_leverage_ratio"),
            )
            bank_risk_weighted_assets = first_non_null(
                latest_instant_millions(["RiskWeightedAssets"], ["us-gaap:RiskWeightedAssets"]),
                bank_capital_filing_metrics.get("risk_weighted_assets"),
            )
            bank_tier1_capital = first_non_null(
                latest_instant_millions(["TierOneRiskBasedCapital"], ["us-gaap:TierOneRiskBasedCapital"]),
                bank_capital_filing_metrics.get("tier1_capital"),
            )
            bank_deposit_growth = instant_growth_millions(deposit_concepts)
            bank_loan_growth = instant_growth_millions(loan_concepts)

        def current_series_value(concepts: list[str]) -> float | None:
            return first_item(ttm_series_millions(concepts) or annual_series_millions(concepts))

        def series_growth_value(concepts: list[str]) -> float | None:
            series = ttm_series_millions(concepts) or annual_series_millions(concepts)
            return annual_growth(first_item(series), first_item(series, 1))

        def sum_current_series(concept_groups: list[list[str]]) -> float | None:
            values = [current_series_value(group) for group in concept_groups]
            return sum(value for value in values if has_value(value)) if any(has_value(value) for value in values) else None

        cash_dividends_paid_m = first_non_null(
            abs(first_item(dividends_paid_series)) if has_value(first_item(dividends_paid_series)) else None,
            (dividend_per_share or 0) * (diluted_shares_m or 0) if has_value(dividend_per_share) and has_value(diluted_shares_m) else None,
        )
        dividend_payout_ratio = safe_div(cash_dividends_paid_m, net_income)

        insurance_premiums = None
        insurance_claims = None
        insurance_underwriting_expenses = None
        insurance_loss_ratio = None
        insurance_expense_ratio = None
        insurance_combined_ratio = None
        insurance_investment_income = None
        insurance_investment_income_to_revenue = None
        insurance_reserves = None
        insurance_reserves_to_premiums = None
        insurance_equity_to_assets = None
        insurance_premium_growth = None
        insurance_book_value_growth = None
        if is_insurance:
            premium_concepts = ["PremiumsEarnedNet", "PremiumsEarned", "DirectPremiumsEarned", "AssumedPremiumsEarned", "InsuranceRevenue"]
            claims_concepts = [
                "IncurredClaimsPropertyCasualtyAndLiability",
                "LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseIncurredClaims",
                "ShortdurationInsuranceContractsIncurredClaimsAndAllocatedClaimAdjustmentExpenseNet",
                "SupplementalInformationForPropertyCasualtyInsuranceUnderwritersCurrentYearClaimsAndClaimsAdjustmentExpense",
                "SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense",
                "LossesAndLossAdjustmentExpense",
                "ClaimsAndClaimsAdjustmentExpense",
                "IncurredClaims",
                "BenefitsClaimsLossesAndSettlementExpenses",
            ]
            benefits_and_expense_concepts = ["BenefitsLossesAndExpenses", "SupplementaryInsuranceInformationBenefitsClaimsLossesAndSettlementExpense"]
            expense_concepts = [
                "OtherUnderwritingExpense",
                "ExpenseRelatedToDistributionOrServicingAndUnderwritingFees",
                "PolicyAcquisitionCosts",
                "UnderwritingExpenses",
                "InsuranceCommissionsExpense",
                "DeferredPolicyAcquisitionCostAmortizationExpense",
                "SupplementaryInsuranceInformationAmortizationOfDeferredPolicyAcquisitionCosts",
            ]
            reserve_concepts = [
                "LiabilityForClaimsAndClaimsAdjustmentExpense",
                "LiabilityForFuturePolicyBenefitsAndUnpaidClaimsAndClaimsAdjustmentExpense",
                "FuturePolicyBenefits",
                "PolicyholderContractDeposits",
                "UnearnedPremiums",
            ]
            insurance_premiums = current_series_value(premium_concepts)
            insurance_claims = current_series_value(claims_concepts)
            insurance_benefits_and_expenses = current_series_value(benefits_and_expense_concepts)
            insurance_underwriting_expenses = sum_current_series([expense_concepts])
            insurance_loss_ratio = safe_div(abs(insurance_claims) if has_value(insurance_claims) else None, insurance_premiums)
            insurance_expense_ratio = safe_div(abs(insurance_underwriting_expenses) if has_value(insurance_underwriting_expenses) else None, insurance_premiums)
            insurance_combined_ratio = first_non_null(
                safe_div(abs(insurance_benefits_and_expenses) if has_value(insurance_benefits_and_expenses) else None, insurance_premiums),
                (insurance_loss_ratio or 0) + (insurance_expense_ratio or 0)
                if has_value(insurance_loss_ratio) and has_value(insurance_expense_ratio)
                else None,
            )
            insurance_investment_income = current_series_value(["InvestmentIncomeNet", "NetInvestmentIncome", "InvestmentIncomeInvestmentExpense", "GrossInvestmentIncomeOperating"])
            insurance_investment_income_to_revenue = safe_div(insurance_investment_income, revenue)
            insurance_reserves = sum_latest_instant_millions(reserve_concepts)
            insurance_reserves_to_premiums = safe_div(insurance_reserves, insurance_premiums)
            insurance_equity_to_assets = safe_div(equity, total_assets)
            insurance_premium_growth = series_growth_value(premium_concepts)
            insurance_book_value_growth = instant_growth_millions(["StockholdersEquity", "CommonStockholdersEquity"])

        bdc_net_investment_income = None
        bdc_investment_income = None
        bdc_nav = None
        bdc_nav_per_share = None
        bdc_price_to_nav = None
        bdc_nii_yield = None
        bdc_nii_margin = None
        bdc_dividend_coverage = None
        bdc_asset_coverage_ratio = None
        bdc_debt_to_equity = None
        bdc_equity_to_assets = None
        bdc_investment_income_to_assets = None
        bdc_net_investment_income_growth = None
        bdc_investment_income_growth = None
        bdc_nav_growth = None
        if is_bdc:
            bdc_net_investment_income = current_series_value(["NetInvestmentIncome", "InvestmentIncomeOperatingAfterExpenseAndTax"])
            bdc_investment_income = current_series_value(["GrossInvestmentIncomeOperating", "InvestmentIncomeInterest", "InvestmentIncomeInvestmentExpense"])
            bdc_nav = equity
            bdc_nav_per_share = safe_div((bdc_nav or 0) * 1_000_000, shares_outstanding) if has_value(bdc_nav) else None
            bdc_price_to_nav = first_non_null(safe_div(price, bdc_nav_per_share), safe_div(market_cap, bdc_nav))
            bdc_nii_yield = safe_div(bdc_net_investment_income, market_cap)
            bdc_nii_margin = safe_div(bdc_net_investment_income, bdc_investment_income)
            bdc_dividend_coverage = safe_div(bdc_net_investment_income, cash_dividends_paid_m)
            bdc_asset_coverage_ratio = normalize_fraction(
                facts.latest_instant_value(["InvestmentCompanySeniorSecurityIndebtednessAssetCoverageRatio"], unit_preferences=["pure"], max_age_days=RECENT_QUARTER_MAX_AGE_DAYS)
            )
            bdc_debt_to_equity = safe_div(debt, equity)
            bdc_equity_to_assets = safe_div(equity, total_assets)
            bdc_investment_income_to_assets = safe_div(bdc_investment_income, avg_assets)
            bdc_net_investment_income_growth = series_growth_value(["NetInvestmentIncome", "InvestmentIncomeOperatingAfterExpenseAndTax"])
            bdc_investment_income_growth = series_growth_value(["GrossInvestmentIncomeOperating", "InvestmentIncomeInterest", "InvestmentIncomeInvestmentExpense"])
            bdc_nav_growth = instant_growth_millions(["StockholdersEquity", "CommonStockholdersEquity"])

        utility_ffo_to_debt = None
        utility_debt_to_capital = None
        utility_capex_to_ocf = None
        utility_asset_growth = None
        if is_utility:
            utility_ffo_to_debt = safe_div(ocf, debt)
            utility_debt_to_capital = safe_div(debt, (debt or 0) + (equity or 0) if has_value(debt) or has_value(equity) else None)
            utility_capex_to_ocf = safe_div(abs(capex) if has_value(capex) else None, ocf)
            utility_asset_growth = instant_growth_millions(["Assets"])

        midstream_dcf = None
        midstream_dcf_yield = None
        midstream_distribution_coverage = None
        if is_midstream:
            midstream_dcf = first_non_null(
                current_series_value(["DistributableCashFlow", "AvailableCash"]),
                max(fcf or 0, (ocf or 0) * 0.75) if has_value(ocf) or has_value(fcf) else None,
                (ocf or 0) + (capex or 0) if has_value(ocf) and has_value(capex) else None,
            )
            midstream_dcf_yield = safe_div(midstream_dcf, market_cap)
            midstream_distribution_coverage = safe_div(midstream_dcf, cash_dividends_paid_m)

        asset_manager_fee_revenue = None
        asset_manager_fee_margin = None
        asset_manager_fee_growth = None
        if is_asset_manager:
            asset_manager_fee_revenue = sum_current_series(
                [
                    ["InvestmentAdvisoryFees", "InvestmentManagementFees", "ManagementFees", "TtlFeeAmt", "NetFeeAmt"],
                    ["PerformanceFees", "IncentiveFeeExpense"],
                ]
            )
            asset_manager_fee_margin = safe_div(operating_income, asset_manager_fee_revenue)
            asset_manager_fee_growth = series_growth_value(["InvestmentAdvisoryFees", "InvestmentManagementFees", "ManagementFees", "TtlFeeAmt", "NetFeeAmt"])

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
            "total_assets": total_assets,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "total_liabilities": total_liabilities,
            "equity": equity,
            "retained_earnings": retained_earnings,
            "revenue": revenue,
            "current_price": price,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "dividend_payout_ratio": dividend_payout_ratio,
            "bank_deposits": bank_deposits,
            "bank_loans": bank_loans,
            "bank_loans_to_deposits": bank_loans_to_deposits,
            "bank_deposits_to_assets": bank_deposits_to_assets,
            "bank_equity_to_assets": bank_equity_to_assets,
            "bank_assets_to_equity": bank_assets_to_equity,
            "bank_tangible_equity": bank_tangible_equity,
            "bank_tangible_assets": bank_tangible_assets,
            "bank_tangible_equity_to_tangible_assets": bank_tangible_equity_to_tangible_assets,
            "bank_tangible_book_value_per_share": bank_tangible_book_value_per_share,
            "bank_price_to_tangible_book": bank_price_to_tangible_book,
            "bank_allowance_for_credit_losses": bank_allowance_for_credit_losses,
            "bank_allowance_to_loans": bank_allowance_to_loans,
            "bank_provision_for_credit_losses": bank_provision_for_credit_losses,
            "bank_provision_to_loans": bank_provision_to_loans,
            "bank_net_interest_income": bank_net_interest_income,
            "bank_net_interest_income_to_assets": bank_net_interest_income_to_assets,
            "bank_net_interest_income_growth": bank_net_interest_income_growth,
            "bank_noninterest_income": bank_noninterest_income,
            "bank_noninterest_expense": bank_noninterest_expense,
            "bank_efficiency_ratio": bank_efficiency_ratio,
            "bank_capital_to_rwa": bank_capital_to_rwa,
            "bank_tier1_capital_ratio": bank_tier1_capital_ratio,
            "bank_tier1_leverage_ratio": bank_tier1_leverage_ratio,
            "bank_risk_weighted_assets": bank_risk_weighted_assets,
            "bank_tier1_capital": bank_tier1_capital,
            "bank_deposit_growth": bank_deposit_growth,
            "bank_loan_growth": bank_loan_growth,
            "insurance_premiums": insurance_premiums,
            "insurance_claims": insurance_claims,
            "insurance_underwriting_expenses": insurance_underwriting_expenses,
            "insurance_loss_ratio": insurance_loss_ratio,
            "insurance_expense_ratio": insurance_expense_ratio,
            "insurance_combined_ratio": insurance_combined_ratio,
            "insurance_investment_income": insurance_investment_income,
            "insurance_investment_income_to_revenue": insurance_investment_income_to_revenue,
            "insurance_reserves": insurance_reserves,
            "insurance_reserves_to_premiums": insurance_reserves_to_premiums,
            "insurance_equity_to_assets": insurance_equity_to_assets,
            "insurance_premium_growth": insurance_premium_growth,
            "insurance_book_value_growth": insurance_book_value_growth,
            "bdc_net_investment_income": bdc_net_investment_income,
            "bdc_investment_income": bdc_investment_income,
            "bdc_nav": bdc_nav,
            "bdc_nav_per_share": bdc_nav_per_share,
            "bdc_price_to_nav": bdc_price_to_nav,
            "bdc_nii_yield": bdc_nii_yield,
            "bdc_nii_margin": bdc_nii_margin,
            "bdc_dividend_coverage": bdc_dividend_coverage,
            "bdc_asset_coverage_ratio": bdc_asset_coverage_ratio,
            "bdc_debt_to_equity": bdc_debt_to_equity,
            "bdc_equity_to_assets": bdc_equity_to_assets,
            "bdc_investment_income_to_assets": bdc_investment_income_to_assets,
            "bdc_net_investment_income_growth": bdc_net_investment_income_growth,
            "bdc_investment_income_growth": bdc_investment_income_growth,
            "bdc_nav_growth": bdc_nav_growth,
            "utility_ffo_to_debt": utility_ffo_to_debt,
            "utility_debt_to_capital": utility_debt_to_capital,
            "utility_capex_to_ocf": utility_capex_to_ocf,
            "utility_asset_growth": utility_asset_growth,
            "midstream_dcf": midstream_dcf,
            "midstream_dcf_yield": midstream_dcf_yield,
            "midstream_distribution_coverage": midstream_distribution_coverage,
            "asset_manager_fee_revenue": asset_manager_fee_revenue,
            "asset_manager_fee_margin": asset_manager_fee_margin,
            "asset_manager_fee_growth": asset_manager_fee_growth,
        }

        has_bank_fundamentals = company_type == "bank" and any(
            has_value(value)
            for value in [
                bank_deposits,
                bank_loans,
                bank_net_interest_income,
            ]
        )
        has_insurance_fundamentals = is_insurance and any(has_value(metrics[key]) for key in ["insurance_premiums", "insurance_combined_ratio", "insurance_reserves"])
        has_bdc_fundamentals = is_bdc and any(has_value(metrics[key]) for key in ["bdc_price_to_nav", "bdc_net_investment_income", "bdc_asset_coverage_ratio"])
        has_utility_fundamentals = is_utility and any(has_value(metrics[key]) for key in ["utility_ffo_to_debt", "utility_debt_to_capital", "utility_asset_growth"])
        has_midstream_fundamentals = is_midstream and any(has_value(metrics[key]) for key in ["midstream_dcf", "midstream_distribution_coverage", "debt_ebitda"])
        has_asset_manager_fundamentals = is_asset_manager and any(has_value(metrics[key]) for key in ["asset_manager_fee_margin", "operating_margin", "fcf_margin"])
        has_capital_markets_fundamentals = is_capital_markets and any(has_value(metrics[key]) for key in ["bank_price_to_tangible_book", "bank_equity_to_assets", "roe"])

        scores = {
            "valuation": valuation_score(metrics),
            "profitability": profitability_score(metrics),
            "growth": growth_score(metrics),
            "financial_strength": financial_strength_score(metrics),
            "cash_flow": cash_flow_score(metrics),
        }
        scores["composite"] = composite_score(scores)
        if has_bank_fundamentals:
            scores.update(bank_scores(metrics))
            metrics["bank_score"] = scores.get("bank")
        else:
            metrics["bank_score"] = None
        if has_insurance_fundamentals:
            scores.update(insurance_scores(metrics))
            metrics["insurance_score"] = scores.get("insurance")
        else:
            metrics["insurance_score"] = None
        if has_bdc_fundamentals:
            scores.update(bdc_scores(metrics))
            metrics["bdc_score"] = scores.get("bdc")
        else:
            metrics["bdc_score"] = None
        if has_utility_fundamentals:
            scores.update(utility_scores(metrics))
            metrics["utility_score"] = scores.get("utility")
        else:
            metrics["utility_score"] = None
        if has_midstream_fundamentals:
            scores.update(midstream_scores(metrics))
            metrics["midstream_score"] = scores.get("midstream")
        else:
            metrics["midstream_score"] = None
        if has_asset_manager_fundamentals:
            scores.update(asset_manager_scores(metrics))
            metrics["asset_manager_score"] = scores.get("asset_manager")
        else:
            metrics["asset_manager_score"] = None
        if has_capital_markets_fundamentals:
            scores.update(capital_markets_scores(metrics))
            metrics["capital_markets_score"] = scores.get("capital_markets")
        else:
            metrics["capital_markets_score"] = None

        profile = {
            "symbol": symbol,
            "name": company_name,
            "sector": sector,
            "industry": industry_raw,
            "company_type": company_type,
            "industry_group": industry_group,
            "ipo_date": format_ipo_date(history),
            "ceo": clean_text(first_non_null(
                officer_name(info),
                sec.extract_named_executive(cik),
            )),
            "employees": info.get("fullTimeEmployees"),
            "country": clean_text(info.get("country")),
            "exchange": clean_text(first_non_null((submissions.get("exchanges") or [None])[0], info.get("exchange"), yahoo_web.get("exchange"))),
            "cik": f"{cik:010d}",
        }

        risk_rating = build_risk_rating(metrics, sector, industry_group)
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
        if has_bank_fundamentals:
            warnings.append("Financial-sector companies are also scored with a bank-specific model because industrial ratios like gross margin, current ratio, and Altman Z are not reliable bank measures.")
        if has_insurance_fundamentals:
            warnings.append("Insurance companies are scored with underwriting, reserves, premium growth, and balance-sheet strength because bank and industrial ratios can mislead for insurers.")
        if has_bdc_fundamentals:
            warnings.append("BDC and specialty-finance companies are scored with NAV, net investment income, dividend coverage, and asset coverage instead of ordinary industrial cash-flow rules.")
        if has_utility_fundamentals:
            warnings.append("Utilities are scored with FFO/debt, debt/capital, payout safety, and rate-base proxy growth because regulated utilities normally carry more debt and lower ROA.")
        if has_midstream_fundamentals:
            warnings.append("Midstream and MLP-style energy companies are scored with DCF yield, distribution coverage, leverage, and fee-like cash-flow quality.")
        if has_asset_manager_fundamentals:
            warnings.append("Asset managers are scored with fee/margin durability, FCF conversion, and capital-light profitability.")
        if has_capital_markets_fundamentals:
            warnings.append("Capital-markets firms are scored with a broker/dealer hybrid model using tangible book, ROE, capital, and efficiency metrics.")

        score_cards = [
            {"label": "Valuation", "value": format_number(scores["valuation"])},
            {"label": "Profitability", "value": format_number(scores["profitability"])},
            {"label": "Growth", "value": format_number(scores["growth"])},
            {"label": "Financial Strength", "value": format_number(scores["financial_strength"])},
            {"label": "Cash Flow", "value": format_number(scores["cash_flow"])},
            {"label": "Composite", "value": format_number(scores["composite"], 4)},
        ]
        if has_bank_fundamentals:
            score_cards.append({"label": "Bank Score", "value": format_number(scores.get("bank"))})
        if has_insurance_fundamentals:
            score_cards.append({"label": "Insurance Score", "value": format_number(scores.get("insurance"))})
        if has_bdc_fundamentals:
            score_cards.append({"label": "BDC Score", "value": format_number(scores.get("bdc"))})
        if has_utility_fundamentals:
            score_cards.append({"label": "Utility Score", "value": format_number(scores.get("utility"))})
        if has_midstream_fundamentals:
            score_cards.append({"label": "Midstream Score", "value": format_number(scores.get("midstream"))})
        if has_asset_manager_fundamentals:
            score_cards.append({"label": "Asset Manager Score", "value": format_number(scores.get("asset_manager"))})
        if has_capital_markets_fundamentals:
            score_cards.append({"label": "Capital Markets Score", "value": format_number(scores.get("capital_markets"))})

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
        if has_bank_fundamentals:
            sections.insert(
                1,
                {
                    "title": "Bank Metrics",
                    "items": [
                        {"label": "Bank Score", "value": format_number(scores.get("bank"))},
                        {"label": "Bank Valuation", "value": format_number(scores.get("bank_valuation"))},
                        {"label": "Bank Profitability", "value": format_number(scores.get("bank_profitability"))},
                        {"label": "Bank Capital", "value": format_number(scores.get("bank_capital"))},
                        {"label": "Bank Growth", "value": format_number(scores.get("bank_growth"))},
                        {"label": "Bank Credit Quality", "value": format_number(scores.get("bank_credit_quality"))},
                        {"label": "P / Tangible Book", "value": format_multiple(metrics["bank_price_to_tangible_book"])},
                        {"label": "Tangible Book / Share", "value": format_number(metrics["bank_tangible_book_value_per_share"])},
                        {"label": "Equity / Assets", "value": format_percent(metrics["bank_equity_to_assets"])},
                        {"label": "Tangible Equity / Tangible Assets", "value": format_percent(metrics["bank_tangible_equity_to_tangible_assets"])},
                        {"label": "Assets / Equity", "value": format_number(metrics["bank_assets_to_equity"])},
                        {"label": "Deposits (USD mm)", "value": format_money_millions(metrics["bank_deposits"])},
                        {"label": "Deposit Growth (YoY)", "value": format_percent(metrics["bank_deposit_growth"])},
                        {"label": "Loans (USD mm)", "value": format_money_millions(metrics["bank_loans"])},
                        {"label": "Loan Growth (YoY)", "value": format_percent(metrics["bank_loan_growth"])},
                        {"label": "Loans / Deposits", "value": format_percent(metrics["bank_loans_to_deposits"])},
                        {"label": "Deposits / Assets", "value": format_percent(metrics["bank_deposits_to_assets"])},
                        {"label": "Allowance / Loans", "value": format_percent(metrics["bank_allowance_to_loans"])},
                        {"label": "Provision / Loans", "value": format_percent(metrics["bank_provision_to_loans"])},
                        {"label": "Net Interest Income (USD mm)", "value": format_money_millions(metrics["bank_net_interest_income"])},
                        {"label": "Net Interest Income / Assets", "value": format_percent(metrics["bank_net_interest_income_to_assets"])},
                        {"label": "Net Interest Income Growth", "value": format_percent(metrics["bank_net_interest_income_growth"])},
                        {"label": "Noninterest Income (USD mm)", "value": format_money_millions(metrics["bank_noninterest_income"])},
                        {"label": "Noninterest Expense (USD mm)", "value": format_money_millions(metrics["bank_noninterest_expense"])},
                        {"label": "Efficiency Ratio", "value": format_percent(metrics["bank_efficiency_ratio"])},
                        {"label": "Capital / RWA", "value": format_percent(metrics["bank_capital_to_rwa"])},
                        {"label": "Tier 1 Capital Ratio", "value": format_percent(metrics["bank_tier1_capital_ratio"])},
                        {"label": "Tier 1 Leverage Ratio", "value": format_percent(metrics["bank_tier1_leverage_ratio"])},
                    ],
                },
            )
        if has_insurance_fundamentals:
            sections.insert(
                1,
                {
                    "title": "Insurance Metrics",
                    "items": [
                        {"label": "Insurance Score", "value": format_number(scores.get("insurance"))},
                        {"label": "Insurance Valuation", "value": format_number(scores.get("insurance_valuation"))},
                        {"label": "Underwriting Score", "value": format_number(scores.get("insurance_underwriting"))},
                        {"label": "Insurance Profitability", "value": format_number(scores.get("insurance_profitability"))},
                        {"label": "Insurance Balance", "value": format_number(scores.get("insurance_balance"))},
                        {"label": "Premiums (USD mm)", "value": format_money_millions(metrics["insurance_premiums"])},
                        {"label": "Premium Growth", "value": format_percent(metrics["insurance_premium_growth"])},
                        {"label": "Combined Ratio", "value": format_percent(metrics["insurance_combined_ratio"])},
                        {"label": "Loss Ratio", "value": format_percent(metrics["insurance_loss_ratio"])},
                        {"label": "Expense Ratio", "value": format_percent(metrics["insurance_expense_ratio"])},
                        {"label": "Investment Income (USD mm)", "value": format_money_millions(metrics["insurance_investment_income"])},
                        {"label": "Investment Income / Revenue", "value": format_percent(metrics["insurance_investment_income_to_revenue"])},
                        {"label": "Reserves / Premiums", "value": format_number(metrics["insurance_reserves_to_premiums"])},
                        {"label": "Equity / Assets", "value": format_percent(metrics["insurance_equity_to_assets"])},
                        {"label": "Book Value Growth", "value": format_percent(metrics["insurance_book_value_growth"])},
                    ],
                },
            )
        if has_bdc_fundamentals:
            sections.insert(
                1,
                {
                    "title": "BDC / Specialty Finance Metrics",
                    "items": [
                        {"label": "BDC Score", "value": format_number(scores.get("bdc"))},
                        {"label": "BDC Valuation", "value": format_number(scores.get("bdc_valuation"))},
                        {"label": "BDC Income", "value": format_number(scores.get("bdc_income"))},
                        {"label": "BDC Capital", "value": format_number(scores.get("bdc_capital"))},
                        {"label": "NAV / Share", "value": format_number(metrics["bdc_nav_per_share"])},
                        {"label": "Price / NAV", "value": format_multiple(metrics["bdc_price_to_nav"])},
                        {"label": "Net Investment Income (USD mm)", "value": format_money_millions(metrics["bdc_net_investment_income"])},
                        {"label": "NII Yield", "value": format_percent(metrics["bdc_nii_yield"])},
                        {"label": "NII Margin", "value": format_percent(metrics["bdc_nii_margin"])},
                        {"label": "Dividend Coverage", "value": format_number(metrics["bdc_dividend_coverage"])},
                        {"label": "Asset Coverage Ratio", "value": format_number(metrics["bdc_asset_coverage_ratio"])},
                        {"label": "Debt / Equity", "value": format_number(metrics["bdc_debt_to_equity"])},
                        {"label": "Equity / Assets", "value": format_percent(metrics["bdc_equity_to_assets"])},
                        {"label": "NII Growth", "value": format_percent(metrics["bdc_net_investment_income_growth"])},
                        {"label": "NAV Growth", "value": format_percent(metrics["bdc_nav_growth"])},
                    ],
                },
            )
        if has_utility_fundamentals:
            sections.insert(
                1,
                {
                    "title": "Utility Metrics",
                    "items": [
                        {"label": "Utility Score", "value": format_number(scores.get("utility"))},
                        {"label": "Utility Valuation", "value": format_number(scores.get("utility_valuation"))},
                        {"label": "Utility Safety", "value": format_number(scores.get("utility_safety"))},
                        {"label": "Utility Profitability", "value": format_number(scores.get("utility_profitability"))},
                        {"label": "FFO / Debt", "value": format_percent(metrics["utility_ffo_to_debt"])},
                        {"label": "Debt / Capital", "value": format_percent(metrics["utility_debt_to_capital"])},
                        {"label": "Interest Coverage", "value": format_number(metrics["interest_coverage"])},
                        {"label": "Dividend Payout", "value": format_percent(metrics["dividend_payout_ratio"])},
                        {"label": "CapEx / OCF", "value": format_percent(metrics["utility_capex_to_ocf"])},
                        {"label": "Asset / Rate-Base Proxy Growth", "value": format_percent(metrics["utility_asset_growth"])},
                    ],
                },
            )
        if has_midstream_fundamentals:
            sections.insert(
                1,
                {
                    "title": "Midstream / MLP Metrics",
                    "items": [
                        {"label": "Midstream Score", "value": format_number(scores.get("midstream"))},
                        {"label": "Midstream Valuation", "value": format_number(scores.get("midstream_valuation"))},
                        {"label": "Distribution Score", "value": format_number(scores.get("midstream_distribution"))},
                        {"label": "Leverage Score", "value": format_number(scores.get("midstream_leverage"))},
                        {"label": "DCF Proxy (USD mm)", "value": format_money_millions(metrics["midstream_dcf"])},
                        {"label": "DCF Yield", "value": format_percent(metrics["midstream_dcf_yield"])},
                        {"label": "Distribution Coverage", "value": format_number(metrics["midstream_distribution_coverage"])},
                        {"label": "Debt / EBITDA", "value": format_number(metrics["debt_ebitda"])},
                        {"label": "EV / EBITDA", "value": format_multiple(metrics["ev_ebitda"])},
                        {"label": "OCF Margin", "value": format_percent(metrics["operating_cash_flow_margin"])},
                    ],
                },
            )
        if has_asset_manager_fundamentals:
            sections.insert(
                1,
                {
                    "title": "Asset Manager Metrics",
                    "items": [
                        {"label": "Asset Manager Score", "value": format_number(scores.get("asset_manager"))},
                        {"label": "Asset Manager Valuation", "value": format_number(scores.get("asset_manager_valuation"))},
                        {"label": "Asset Manager Profitability", "value": format_number(scores.get("asset_manager_profitability"))},
                        {"label": "Asset Manager Stability", "value": format_number(scores.get("asset_manager_stability"))},
                        {"label": "Fee Revenue (USD mm)", "value": format_money_millions(metrics["asset_manager_fee_revenue"])},
                        {"label": "Fee Margin", "value": format_percent(metrics["asset_manager_fee_margin"])},
                        {"label": "Fee Growth", "value": format_percent(metrics["asset_manager_fee_growth"])},
                        {"label": "Operating Margin", "value": format_percent(metrics["operating_margin"])},
                        {"label": "FCF Margin", "value": format_percent(metrics["fcf_margin"])},
                        {"label": "ROE", "value": format_percent(metrics["roe"])},
                    ],
                },
            )
        if has_capital_markets_fundamentals:
            sections.insert(
                1,
                {
                    "title": "Capital Markets Metrics",
                    "items": [
                        {"label": "Capital Markets Score", "value": format_number(scores.get("capital_markets"))},
                        {"label": "Capital Markets Valuation", "value": format_number(scores.get("capital_markets_valuation"))},
                        {"label": "Capital Markets Profitability", "value": format_number(scores.get("capital_markets_profitability"))},
                        {"label": "Capital Markets Capital", "value": format_number(scores.get("capital_markets_capital"))},
                        {"label": "P / Tangible Book", "value": format_multiple(metrics["bank_price_to_tangible_book"])},
                        {"label": "Tangible Equity / Tangible Assets", "value": format_percent(metrics["bank_tangible_equity_to_tangible_assets"])},
                        {"label": "Equity / Assets", "value": format_percent(metrics["bank_equity_to_assets"])},
                        {"label": "Tier 1 Capital Ratio", "value": format_percent(metrics["bank_tier1_capital_ratio"])},
                        {"label": "Tier 1 Leverage Ratio", "value": format_percent(metrics["bank_tier1_leverage_ratio"])},
                        {"label": "Efficiency Ratio", "value": format_percent(metrics["bank_efficiency_ratio"])},
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
        elif has_bank_fundamentals:
            recommendation_lines.extend(
                [
                    f"Bank Score: {format_number(scores.get('bank'))}",
                    f"P/B: {format_multiple(metrics['price_to_book'])}",
                    f"P/Tangible Book: {format_multiple(metrics['bank_price_to_tangible_book'])}",
                    f"ROE: {format_percent(metrics['roe'])}",
                    f"ROA: {format_percent(metrics['roa'])}",
                    f"Equity/Assets: {format_percent(metrics['bank_equity_to_assets'])}",
                    f"Loans/Deposits: {format_percent(metrics['bank_loans_to_deposits'])}",
                    f"Efficiency Ratio: {format_percent(metrics['bank_efficiency_ratio'])}",
                    f"Capital/RWA: {format_percent(metrics['bank_capital_to_rwa'])}",
                ]
            )
        elif has_insurance_fundamentals:
            recommendation_lines.extend(
                [
                    f"Insurance Score: {format_number(scores.get('insurance'))}",
                    f"Combined Ratio: {format_percent(metrics['insurance_combined_ratio'])}",
                    f"Loss Ratio: {format_percent(metrics['insurance_loss_ratio'])}",
                    f"ROE: {format_percent(metrics['roe'])}",
                    f"Equity/Assets: {format_percent(metrics['insurance_equity_to_assets'])}",
                    f"Premium Growth: {format_percent(metrics['insurance_premium_growth'])}",
                ]
            )
        elif has_bdc_fundamentals:
            recommendation_lines.extend(
                [
                    f"BDC Score: {format_number(scores.get('bdc'))}",
                    f"Price/NAV: {format_multiple(metrics['bdc_price_to_nav'])}",
                    f"NII Yield: {format_percent(metrics['bdc_nii_yield'])}",
                    f"Dividend Coverage: {format_number(metrics['bdc_dividend_coverage'])}",
                    f"Asset Coverage: {format_number(metrics['bdc_asset_coverage_ratio'])}",
                    f"Debt/Equity: {format_number(metrics['bdc_debt_to_equity'])}",
                ]
            )
        elif has_utility_fundamentals:
            recommendation_lines.extend(
                [
                    f"Utility Score: {format_number(scores.get('utility'))}",
                    f"FFO/Debt: {format_percent(metrics['utility_ffo_to_debt'])}",
                    f"Debt/Capital: {format_percent(metrics['utility_debt_to_capital'])}",
                    f"Interest Coverage: {format_number(metrics['interest_coverage'])}",
                    f"Dividend Payout: {format_percent(metrics['dividend_payout_ratio'])}",
                ]
            )
        elif has_midstream_fundamentals:
            recommendation_lines.extend(
                [
                    f"Midstream Score: {format_number(scores.get('midstream'))}",
                    f"DCF Yield: {format_percent(metrics['midstream_dcf_yield'])}",
                    f"Distribution Coverage: {format_number(metrics['midstream_distribution_coverage'])}",
                    f"Debt/EBITDA: {format_number(metrics['debt_ebitda'])}",
                    f"EV/EBITDA: {format_multiple(metrics['ev_ebitda'])}",
                ]
            )
        elif has_asset_manager_fundamentals:
            recommendation_lines.extend(
                [
                    f"Asset Manager Score: {format_number(scores.get('asset_manager'))}",
                    f"Fee Margin: {format_percent(metrics['asset_manager_fee_margin'])}",
                    f"Operating Margin: {format_percent(metrics['operating_margin'])}",
                    f"FCF Margin: {format_percent(metrics['fcf_margin'])}",
                    f"ROE: {format_percent(metrics['roe'])}",
                ]
            )
        elif has_capital_markets_fundamentals:
            recommendation_lines.extend(
                [
                    f"Capital Markets Score: {format_number(scores.get('capital_markets'))}",
                    f"P/Tangible Book: {format_multiple(metrics['bank_price_to_tangible_book'])}",
                    f"ROE: {format_percent(metrics['roe'])}",
                    f"Equity/Assets: {format_percent(metrics['bank_equity_to_assets'])}",
                    f"Efficiency Ratio: {format_percent(metrics['bank_efficiency_ratio'])}",
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
        if has_bank_fundamentals:
            sheet_sections.insert(
                2,
                {
                    "title": "Bank Fundamentals",
                    "rows": [
                        make_sheet_row("Bank Score", formula_value(scores.get("bank"), 2), "P/Tangible Book", formula_value(metrics["bank_price_to_tangible_book"], 2)),
                        make_sheet_row("Bank Valuation Score", formula_value(scores.get("bank_valuation"), 2), "Tangible Book / Share", formula_value(metrics["bank_tangible_book_value_per_share"], 2)),
                        make_sheet_row("Bank Profitability Score", formula_value(scores.get("bank_profitability"), 2), "Equity / Assets", formula_value(metrics["bank_equity_to_assets"], 4)),
                        make_sheet_row("Bank Capital Score", formula_value(scores.get("bank_capital"), 2), "Tangible Equity / Tangible Assets", formula_value(metrics["bank_tangible_equity_to_tangible_assets"], 4)),
                        make_sheet_row("Bank Growth Score", formula_value(scores.get("bank_growth"), 2), "Assets / Equity", formula_value(metrics["bank_assets_to_equity"], 4)),
                        make_sheet_row("Bank Credit Quality Score", formula_value(scores.get("bank_credit_quality"), 2), "Deposits / Assets", formula_value(metrics["bank_deposits_to_assets"], 4)),
                        make_sheet_row("Deposits", formula_value(metrics["bank_deposits"], 2), "Deposit Growth (YoY)", formula_value(metrics["bank_deposit_growth"], 4)),
                        make_sheet_row("Loans", formula_value(metrics["bank_loans"], 2), "Loan Growth (YoY)", formula_value(metrics["bank_loan_growth"], 4)),
                        make_sheet_row("Loans / Deposits", formula_value(metrics["bank_loans_to_deposits"], 4), "Allowance / Loans", formula_value(metrics["bank_allowance_to_loans"], 4)),
                        make_sheet_row("Provision / Loans", formula_value(metrics["bank_provision_to_loans"], 4), "Efficiency Ratio", formula_value(metrics["bank_efficiency_ratio"], 4)),
                        make_sheet_row("Net Interest Income", formula_value(metrics["bank_net_interest_income"], 2), "Net Interest Income / Assets", formula_value(metrics["bank_net_interest_income_to_assets"], 4)),
                        make_sheet_row("Net Interest Income Growth", formula_value(metrics["bank_net_interest_income_growth"], 4), "Capital / RWA", formula_value(metrics["bank_capital_to_rwa"], 4)),
                        make_sheet_row("Tier 1 Capital Ratio", formula_value(metrics["bank_tier1_capital_ratio"], 4), "Tier 1 Leverage Ratio", formula_value(metrics["bank_tier1_leverage_ratio"], 4)),
                    ],
                },
            )
        if has_insurance_fundamentals:
            sheet_sections.insert(
                2,
                {
                    "title": "Insurance Fundamentals",
                    "rows": [
                        make_sheet_row("Insurance Score", formula_value(scores.get("insurance"), 2), "Combined Ratio", formula_value(metrics["insurance_combined_ratio"], 4)),
                        make_sheet_row("Insurance Valuation Score", formula_value(scores.get("insurance_valuation"), 2), "Loss Ratio", formula_value(metrics["insurance_loss_ratio"], 4)),
                        make_sheet_row("Underwriting Score", formula_value(scores.get("insurance_underwriting"), 2), "Expense Ratio", formula_value(metrics["insurance_expense_ratio"], 4)),
                        make_sheet_row("Insurance Profitability Score", formula_value(scores.get("insurance_profitability"), 2), "Premiums", formula_value(metrics["insurance_premiums"], 2)),
                        make_sheet_row("Insurance Balance Score", formula_value(scores.get("insurance_balance"), 2), "Reserves / Premiums", formula_value(metrics["insurance_reserves_to_premiums"], 4)),
                        make_sheet_row("Premium Growth", formula_value(metrics["insurance_premium_growth"], 4), "Book Value Growth", formula_value(metrics["insurance_book_value_growth"], 4)),
                    ],
                },
            )
        if has_bdc_fundamentals:
            sheet_sections.insert(
                2,
                {
                    "title": "BDC / Specialty Finance Fundamentals",
                    "rows": [
                        make_sheet_row("BDC Score", formula_value(scores.get("bdc"), 2), "Price / NAV", formula_value(metrics["bdc_price_to_nav"], 4)),
                        make_sheet_row("BDC Valuation Score", formula_value(scores.get("bdc_valuation"), 2), "NAV / Share", formula_value(metrics["bdc_nav_per_share"], 2)),
                        make_sheet_row("BDC Income Score", formula_value(scores.get("bdc_income"), 2), "NII Yield", formula_value(metrics["bdc_nii_yield"], 4)),
                        make_sheet_row("BDC Capital Score", formula_value(scores.get("bdc_capital"), 2), "Asset Coverage Ratio", formula_value(metrics["bdc_asset_coverage_ratio"], 4)),
                        make_sheet_row("Dividend Coverage", formula_value(metrics["bdc_dividend_coverage"], 4), "Debt / Equity", formula_value(metrics["bdc_debt_to_equity"], 4)),
                        make_sheet_row("NII Growth", formula_value(metrics["bdc_net_investment_income_growth"], 4), "NAV Growth", formula_value(metrics["bdc_nav_growth"], 4)),
                    ],
                },
            )
        if has_utility_fundamentals:
            sheet_sections.insert(
                2,
                {
                    "title": "Utility Fundamentals",
                    "rows": [
                        make_sheet_row("Utility Score", formula_value(scores.get("utility"), 2), "FFO / Debt", formula_value(metrics["utility_ffo_to_debt"], 4)),
                        make_sheet_row("Utility Valuation Score", formula_value(scores.get("utility_valuation"), 2), "Debt / Capital", formula_value(metrics["utility_debt_to_capital"], 4)),
                        make_sheet_row("Utility Safety Score", formula_value(scores.get("utility_safety"), 2), "Dividend Payout", formula_value(metrics["dividend_payout_ratio"], 4)),
                        make_sheet_row("Utility Profitability Score", formula_value(scores.get("utility_profitability"), 2), "CapEx / OCF", formula_value(metrics["utility_capex_to_ocf"], 4)),
                        make_sheet_row("Utility Growth Score", formula_value(scores.get("utility_growth"), 2), "Asset / Rate-Base Proxy Growth", formula_value(metrics["utility_asset_growth"], 4)),
                    ],
                },
            )
        if has_midstream_fundamentals:
            sheet_sections.insert(
                2,
                {
                    "title": "Midstream / MLP Fundamentals",
                    "rows": [
                        make_sheet_row("Midstream Score", formula_value(scores.get("midstream"), 2), "DCF Yield", formula_value(metrics["midstream_dcf_yield"], 4)),
                        make_sheet_row("Midstream Valuation Score", formula_value(scores.get("midstream_valuation"), 2), "DCF Proxy", formula_value(metrics["midstream_dcf"], 2)),
                        make_sheet_row("Distribution Score", formula_value(scores.get("midstream_distribution"), 2), "Distribution Coverage", formula_value(metrics["midstream_distribution_coverage"], 4)),
                        make_sheet_row("Leverage Score", formula_value(scores.get("midstream_leverage"), 2), "Debt / EBITDA", formula_value(metrics["debt_ebitda"], 4)),
                        make_sheet_row("Quality Score", formula_value(scores.get("midstream_quality"), 2), "OCF Margin", formula_value(metrics["operating_cash_flow_margin"], 4)),
                    ],
                },
            )
        if has_asset_manager_fundamentals:
            sheet_sections.insert(
                2,
                {
                    "title": "Asset Manager Fundamentals",
                    "rows": [
                        make_sheet_row("Asset Manager Score", formula_value(scores.get("asset_manager"), 2), "Fee Margin", formula_value(metrics["asset_manager_fee_margin"], 4)),
                        make_sheet_row("Asset Manager Valuation Score", formula_value(scores.get("asset_manager_valuation"), 2), "Fee Revenue", formula_value(metrics["asset_manager_fee_revenue"], 2)),
                        make_sheet_row("Asset Manager Profitability Score", formula_value(scores.get("asset_manager_profitability"), 2), "Fee Growth", formula_value(metrics["asset_manager_fee_growth"], 4)),
                        make_sheet_row("Asset Manager Stability Score", formula_value(scores.get("asset_manager_stability"), 2), "FCF Margin", formula_value(metrics["fcf_margin"], 4)),
                        make_sheet_row("Asset Manager Growth Score", formula_value(scores.get("asset_manager_growth"), 2), "ROE", formula_value(metrics["roe"], 4)),
                    ],
                },
            )
        if has_capital_markets_fundamentals:
            sheet_sections.insert(
                2,
                {
                    "title": "Capital Markets Fundamentals",
                    "rows": [
                        make_sheet_row("Capital Markets Score", formula_value(scores.get("capital_markets"), 2), "P/Tangible Book", formula_value(metrics["bank_price_to_tangible_book"], 2)),
                        make_sheet_row("Capital Markets Valuation Score", formula_value(scores.get("capital_markets_valuation"), 2), "Tangible Equity / Tangible Assets", formula_value(metrics["bank_tangible_equity_to_tangible_assets"], 4)),
                        make_sheet_row("Capital Markets Profitability Score", formula_value(scores.get("capital_markets_profitability"), 2), "Equity / Assets", formula_value(metrics["bank_equity_to_assets"], 4)),
                        make_sheet_row("Capital Markets Capital Score", formula_value(scores.get("capital_markets_capital"), 2), "Tier 1 Capital Ratio", formula_value(metrics["bank_tier1_capital_ratio"], 4)),
                        make_sheet_row("Capital Markets Growth Score", formula_value(scores.get("capital_markets_growth"), 2), "Efficiency Ratio", formula_value(metrics["bank_efficiency_ratio"], 4)),
                    ],
                },
            )

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
            "yfinance and Yahoo public quote-page parsing are the no-key fallbacks for price, profile, estimate-style, and market-derived fields that SEC does not provide.",
            "Growth metrics prefer current quarter versus prior-year quarter comparisons from SEC filing data, then fall back to annual and TTM filing comparisons.",
            "REIT FFO and AFFO are derived from filing-backed components unless the latest filing exposes cleaner company-reported values.",
            "Financial-sector scoring uses bank-specific SEC concepts such as deposits, loans, capital ratios, tangible book, allowance coverage, and efficiency ratio instead of forcing industrial solvency ratios.",
            "Insurance, BDC, utility, midstream, asset-manager, and capital-markets overlays are applied only when the industry classification supports that company type.",
            "The original sheet's custom EP_SPREAD() function was approximated as ROIC minus a 10.84% cost of capital because the underlying Sheets function logic was not provided.",
        ],
        }
