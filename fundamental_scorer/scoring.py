from __future__ import annotations

import math
from statistics import mean
from typing import Any


def has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if not has_value(numerator) or not has_value(denominator) or denominator == 0:
        return None
    return numerator / denominator


def available_average(*values: float | None) -> float | None:
    numbers = [value for value in values if has_value(value)]
    if not numbers:
        return None
    return mean(numbers)


def sigmoid_percent(exponent: float) -> float:
    if exponent >= 0:
        if exponent > 700:
            return 100.0
        exp_term = math.exp(-exponent)
        return 100.0 / (1.0 + exp_term)
    if exponent < -700:
        return 0.0
    exp_term = math.exp(exponent)
    return 100.0 * exp_term / (1.0 + exp_term)


def logistic_score(scale: float, shift: float, value: float | None) -> float | None:
    if not has_value(value):
        return None
    return sigmoid_percent(scale * ((value * 100) - shift))


def valuation_score(metrics: dict[str, float | None]) -> float | None:
    # This mirrors the sheet exactly, including its cell-position mapping:
    # B23=fpe, B24=ey, B25=pb, B26=ps, B27=evebitda, B28=evsales,
    # B29=peg, B30=pe, B31=fcf.
    # In the exported sheet labels, B24 is "PEG Ratio", B29 is "Pe",
    # B30 is "Price/FCF Ratio", and B31 is "Earnings Yield".
    fpe = metrics.get("forward_pe")
    ey = metrics.get("peg_ratio")
    pb = metrics.get("price_to_book")
    ps = metrics.get("price_to_sales")
    evebitda = metrics.get("ev_ebitda")
    evsales = metrics.get("ev_sales")
    peg = metrics.get("pe_ratio")
    pe = metrics.get("price_fcf_ratio")
    fcf = metrics.get("earnings_yield")

    weights = {
        "fpe": 0.12,
        "peg": 0.10,
        "pb": 0.12,
        "ps": 0.10,
        "evebitda": 0.12,
        "evsales": 0.08,
        "pe": 0.10,
        "fcf": 0.08,
        "ey": 0.18,
    }

    def val_penalty(value: float | None) -> float | None:
        if not has_value(value):
            return None
        return max(0.0, sigmoid_percent(-0.09 * (value - 15)))

    scores = {
        "fpe": val_penalty(fpe),
        "peg": val_penalty(peg),
        "pb": val_penalty(pb),
        "ps": val_penalty(ps),
        "evebitda": val_penalty(evebitda),
        "evsales": val_penalty(evsales),
        "pe": val_penalty(pe),
        "fcf": val_penalty(fcf),
        "ey": None
        if not has_value(ey)
        else max(0.0, min(100.0, sigmoid_percent(35 * (ey - 0.06)))),
    }

    fpe_adj = 0.0
    if has_value(fpe) and has_value(pe) and pe not in (None, 0):
        fpe_adj = max(0.0, min(10.0, ((pe - fpe) / pe) * 50))
    if has_value(scores["fpe"]):
        scores["fpe"] = min(100.0, scores["fpe"] + fpe_adj)

    valid_keys = [key for key, score in scores.items() if has_value(score)]
    if not valid_keys:
        return None

    total_weight = sum(weights[key] for key in valid_keys)
    weighted = sum(scores[key] * weights[key] for key in valid_keys if scores[key] is not None)
    return round(clamp(weighted / total_weight, 0, 100), 2)


def profitability_score(metrics: dict[str, float | None]) -> float | None:
    gm = metrics.get("gross_margin")
    opm = metrics.get("operating_margin")
    nm = metrics.get("net_margin")
    roe = metrics.get("roe")
    roa = metrics.get("roa")
    roic = metrics.get("roic")
    fcfm = metrics.get("fcf_margin")
    ebitm = metrics.get("ebit_margin")
    eps = metrics.get("economic_profit_spread")

    roe_adj = None if not has_value(roe) else max(roe, -0.1)

    scores = {
        "gm": logistic_score(0.08, 20, gm),
        "opm": logistic_score(0.10, 10, opm),
        "nm": logistic_score(0.12, 8, nm),
        "roa": logistic_score(0.15, 5, roa),
        "roic": logistic_score(0.15, 6, roic),
        "fcfm": logistic_score(0.08, 10, fcfm),
        "ebitm": logistic_score(0.10, 10, ebitm),
        "eps": logistic_score(0.10, 5, eps),
    }

    s_roe_raw = logistic_score(0.15, 10, roe_adj)
    if has_value(roe_adj) and has_value(roa) and has_value(roic):
        modifier = min(1.0, ((roa * 100) / 10) + ((roic * 100) / 20))
        scores["roe"] = s_roe_raw * modifier if s_roe_raw is not None else None
    else:
        scores["roe"] = s_roe_raw

    weights = {
        "gm": 0.12,
        "opm": 0.12,
        "nm": 0.08,
        "roe": 0.15,
        "roa": 0.10,
        "roic": 0.12,
        "fcfm": 0.10,
        "ebitm": 0.10,
        "eps": 0.15,
    }

    valid_keys = [key for key, score in scores.items() if has_value(score)]
    if not valid_keys:
        return None

    total_weight = sum(weights[key] for key in valid_keys)
    weighted = sum(scores[key] * weights[key] for key in valid_keys if scores[key] is not None)
    return round(clamp(weighted / total_weight, 0, 100), 2)


def growth_score(metrics: dict[str, float | None]) -> float | None:
    rev = metrics.get("revenue_growth")
    earn = metrics.get("earnings_growth")
    fcf = metrics.get("free_cash_flow_growth")
    div = metrics.get("dividend_growth")
    eps = metrics.get("eps_growth")

    s_rev = logistic_score(0.08, 10, rev)
    s_earn = logistic_score(0.08, 10, earn)
    s_fcf = logistic_score(0.08, 8, fcf)

    growth_offset = 1.0
    if has_value(rev) and has_value(earn) and has_value(fcf):
        growth_offset = min(1.0, ((rev * 100) / 20) + ((earn * 100) / 20) + ((fcf * 100) / 25))

    s_div = None if not has_value(div) else sigmoid_percent(0.1 * ((div * 100) - 5)) * growth_offset
    s_eps = None if not has_value(eps) else sigmoid_percent(0.1 * ((eps * 100) - 10)) * growth_offset

    scores = {"rev": s_rev, "earn": s_earn, "fcf": s_fcf, "div": s_div, "eps": s_eps}
    weights = {"rev": 0.25, "earn": 0.25, "fcf": 0.20, "div": 0.10, "eps": 0.20}

    valid_keys = [key for key, score in scores.items() if has_value(score)]
    if not valid_keys:
        return None

    total_weight = sum(weights[key] for key in valid_keys)
    weighted = sum(scores[key] * weights[key] for key in valid_keys if scores[key] is not None)
    return round(clamp(weighted / total_weight, 0, 100), 2)


def financial_strength_score(metrics: dict[str, float | None]) -> float | None:
    dte = metrics.get("debt_to_equity")
    icv = metrics.get("interest_coverage")
    crv = metrics.get("current_ratio")
    deb = metrics.get("debt_ebitda")
    cashr = metrics.get("cash_ratio")
    dfcf = metrics.get("debt_fcf_ratio")
    qr = metrics.get("quick_ratio")
    altz = metrics.get("altman_z")

    scores = {
        "dte": None if not has_value(dte) else clamp(sigmoid_percent(-0.8 * (dte - 1.5)), 0, 100),
        "icv": None if not has_value(icv) else clamp(sigmoid_percent(0.12 * (icv - 5)), 0, 100),
        "crv": None if not has_value(crv) else clamp(sigmoid_percent(crv - 1.5), 0, 100),
        "deb": None if not has_value(deb) else clamp(sigmoid_percent(-0.8 * (deb - 1.5)), 0, 100),
        "cashr": None if not has_value(cashr) else clamp(sigmoid_percent(cashr - 0.8), 0, 100),
        "dfcf": None if not has_value(dfcf) else clamp(sigmoid_percent(-0.8 * (dfcf - 1.5)), 0, 100),
        "qr": None if not has_value(qr) else clamp(sigmoid_percent(qr - 1.2), 0, 100),
        "altz": None if not has_value(altz) else clamp(sigmoid_percent(altz - 2), 0, 100),
    }
    weights = {
        "dte": 0.18,
        "icv": 0.12,
        "crv": 0.10,
        "deb": 0.10,
        "cashr": 0.08,
        "dfcf": 0.06,
        "qr": 0.06,
        "altz": 0.30,
    }
    valid_keys = [key for key, score in scores.items() if has_value(score)]
    if not valid_keys:
        return None

    total_weight = sum(weights[key] for key in valid_keys)
    weighted = sum(scores[key] * weights[key] for key in valid_keys if scores[key] is not None)
    score = weighted / total_weight
    altz_adjust = 1.0 if not has_value(altz) else min(1.0, altz / 3)
    return round(clamp(score * altz_adjust, 0, 100), 2)


def cash_flow_score(metrics: dict[str, float | None]) -> float | None:
    fcf = metrics.get("free_cash_flow")
    ofcm = metrics.get("operating_cash_flow_margin")
    fcf_yield = metrics.get("fcf_yield")
    capex_sales = metrics.get("capex_to_sales")
    cfo_net = metrics.get("cfo_net_income")
    capex_cfo = metrics.get("capex_cfo")
    accrual = metrics.get("accrual_ratio")

    scores = {
        "fcf": None if not has_value(fcf) else clamp(sigmoid_percent(0.0001 * (fcf - 1000)), 0, 100),
        "ofcm": None if not has_value(ofcm) else clamp(sigmoid_percent(8 * (ofcm - 0.1)), 0, 100),
        "fcf_yield": None if not has_value(fcf_yield) else clamp(sigmoid_percent(30 * (fcf_yield - 0.04)), 0, 100),
        "capex_sales": None if not has_value(capex_sales) else clamp(sigmoid_percent(-50 * (capex_sales - 0.05)), 0, 100),
        "cfo_net": None if not has_value(cfo_net) else clamp(sigmoid_percent(5 * (cfo_net - 1)), 0, 100),
        "capex_cfo": None if not has_value(capex_cfo) else clamp(sigmoid_percent(-50 * (capex_cfo - 0.05)), 0, 100),
        "accrual": None if not has_value(accrual) else clamp(sigmoid_percent(-50 * (accrual - 0.05)), 0, 100),
    }
    weights = {
        "fcf": 0.15,
        "ofcm": 0.15,
        "fcf_yield": 0.15,
        "capex_sales": 0.10,
        "cfo_net": 0.15,
        "capex_cfo": 0.15,
        "accrual": 0.15,
    }

    valid_keys = [key for key, score in scores.items() if has_value(score)]
    if not valid_keys:
        return None

    total_weight = sum(weights[key] for key in valid_keys)
    weighted = sum(scores[key] * weights[key] for key in valid_keys if scores[key] is not None)
    return round(clamp(weighted / total_weight, 0, 100), 2)


def composite_score(score_map: dict[str, float | None]) -> float | None:
    v = score_map.get("valuation")
    p = score_map.get("profitability")
    g = score_map.get("growth")
    f = score_map.get("financial_strength")
    c = score_map.get("cash_flow")
    values = [v, p, g, f, c]
    if not any(has_value(value) for value in values):
        return None
    composite = (v or 0) * 0.20 + (p or 0) * 0.25 + (g or 0) * 0.25 + (f or 0) * 0.15 + (c or 0) * 0.15
    return round(clamp(composite, 0, 100), 4)


def _score_higher(value: float | None, *, good: float, bad: float) -> float | None:
    if not has_value(value):
        return None
    if value >= good:
        return 100.0
    if value <= bad:
        return 0.0
    return clamp(((value - bad) / (good - bad)) * 100, 0, 100)


def _score_lower(value: float | None, *, good: float, bad: float) -> float | None:
    if not has_value(value):
        return None
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return clamp(((bad - value) / (bad - good)) * 100, 0, 100)


def _score_band(
    value: float | None,
    *,
    good_low: float,
    good_high: float,
    bad_low: float,
    bad_high: float,
) -> float | None:
    if not has_value(value):
        return None
    if good_low <= value <= good_high:
        return 100.0
    if value < good_low:
        if value <= bad_low:
            return 0.0
        return clamp(((value - bad_low) / (good_low - bad_low)) * 100, 0, 100)
    if value >= bad_high:
        return 0.0
    return clamp(((bad_high - value) / (bad_high - good_high)) * 100, 0, 100)


def _weighted_component_score(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    valid_keys = [key for key, score in scores.items() if has_value(score)]
    if not valid_keys:
        return None
    total_weight = sum(weights[key] for key in valid_keys)
    weighted = sum(scores[key] * weights[key] for key in valid_keys if scores[key] is not None)
    return round(clamp(weighted / total_weight, 0, 100), 2)


def bank_scores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    valuation = _weighted_component_score(
        {
            "pb": _score_lower(metrics.get("price_to_book"), good=1.2, bad=2.5),
            "ptbv": _score_lower(metrics.get("bank_price_to_tangible_book"), good=1.5, bad=3.0),
            "pe": _score_lower(metrics.get("pe_ratio"), good=10.0, bad=22.0),
            "fpe": _score_lower(metrics.get("forward_pe"), good=10.0, bad=20.0),
            "ey": _score_higher(metrics.get("earnings_yield"), good=0.08, bad=0.025),
        },
        {"pb": 0.25, "ptbv": 0.25, "pe": 0.20, "fpe": 0.15, "ey": 0.15},
    )
    profitability = _weighted_component_score(
        {
            "roe": _score_higher(metrics.get("roe"), good=0.12, bad=0.04),
            "roa": _score_higher(metrics.get("roa"), good=0.012, bad=0.002),
            "net_margin": _score_higher(metrics.get("net_margin"), good=0.25, bad=0.06),
            "nim_proxy": _score_higher(metrics.get("bank_net_interest_income_to_assets"), good=0.03, bad=0.005),
            "efficiency": _score_lower(metrics.get("bank_efficiency_ratio"), good=0.55, bad=0.85),
        },
        {"roe": 0.25, "roa": 0.25, "net_margin": 0.15, "nim_proxy": 0.15, "efficiency": 0.20},
    )
    capital = _weighted_component_score(
        {
            "equity_assets": _score_higher(metrics.get("bank_equity_to_assets"), good=0.09, bad=0.035),
            "tangible_equity_assets": _score_higher(metrics.get("bank_tangible_equity_to_tangible_assets"), good=0.07, bad=0.025),
            "capital_rwa": _score_higher(metrics.get("bank_capital_to_rwa"), good=0.12, bad=0.08),
            "tier1": _score_higher(metrics.get("bank_tier1_capital_ratio"), good=0.11, bad=0.075),
            "tier1_leverage": _score_higher(metrics.get("bank_tier1_leverage_ratio"), good=0.06, bad=0.035),
            "loans_deposits": _score_band(metrics.get("bank_loans_to_deposits"), good_low=0.55, good_high=0.90, bad_low=0.20, bad_high=1.15),
            "allowance_loans": _score_higher(metrics.get("bank_allowance_to_loans"), good=0.012, bad=0.0025),
        },
        {
            "equity_assets": 0.20,
            "tangible_equity_assets": 0.15,
            "capital_rwa": 0.18,
            "tier1": 0.16,
            "tier1_leverage": 0.12,
            "loans_deposits": 0.10,
            "allowance_loans": 0.09,
        },
    )
    growth = _weighted_component_score(
        {
            "revenue": _score_higher(metrics.get("revenue_growth"), good=0.06, bad=-0.08),
            "earnings": _score_higher(metrics.get("earnings_growth"), good=0.08, bad=-0.20),
            "eps": _score_higher(metrics.get("eps_growth"), good=0.08, bad=-0.20),
            "deposit": _score_higher(metrics.get("bank_deposit_growth"), good=0.04, bad=-0.10),
            "loan": _score_higher(metrics.get("bank_loan_growth"), good=0.04, bad=-0.10),
            "net_interest": _score_higher(metrics.get("bank_net_interest_income_growth"), good=0.05, bad=-0.12),
        },
        {"revenue": 0.20, "earnings": 0.20, "eps": 0.20, "deposit": 0.15, "loan": 0.15, "net_interest": 0.10},
    )
    credit_quality = _weighted_component_score(
        {
            "efficiency": _score_lower(metrics.get("bank_efficiency_ratio"), good=0.55, bad=0.85),
            "provision_loans": _score_lower(metrics.get("bank_provision_to_loans"), good=0.004, bad=0.025),
            "allowance_loans": _score_higher(metrics.get("bank_allowance_to_loans"), good=0.012, bad=0.0025),
            "deposits_assets": _score_higher(metrics.get("bank_deposits_to_assets"), good=0.50, bad=0.20),
        },
        {"efficiency": 0.30, "provision_loans": 0.25, "allowance_loans": 0.25, "deposits_assets": 0.20},
    )

    total = _weighted_component_score(
        {
            "valuation": valuation,
            "profitability": profitability,
            "capital": capital,
            "growth": growth,
            "credit_quality": credit_quality,
        },
        {"valuation": 0.22, "profitability": 0.24, "capital": 0.26, "growth": 0.14, "credit_quality": 0.14},
    )
    return {
        "bank": total,
        "bank_valuation": valuation,
        "bank_profitability": profitability,
        "bank_capital": capital,
        "bank_growth": growth,
        "bank_credit_quality": credit_quality,
    }


def insurance_scores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    valuation = _weighted_component_score(
        {
            "pb": _score_lower(metrics.get("price_to_book"), good=1.4, bad=2.8),
            "pe": _score_lower(metrics.get("pe_ratio"), good=11.0, bad=24.0),
            "fpe": _score_lower(metrics.get("forward_pe"), good=11.0, bad=22.0),
            "ey": _score_higher(metrics.get("earnings_yield"), good=0.085, bad=0.03),
        },
        {"pb": 0.30, "pe": 0.25, "fpe": 0.20, "ey": 0.25},
    )
    underwriting = _weighted_component_score(
        {
            "combined": _score_lower(metrics.get("insurance_combined_ratio"), good=0.94, bad=1.08),
            "loss": _score_lower(metrics.get("insurance_loss_ratio"), good=0.62, bad=0.82),
            "expense": _score_lower(metrics.get("insurance_expense_ratio"), good=0.30, bad=0.42),
        },
        {"combined": 0.45, "loss": 0.35, "expense": 0.20},
    )
    profitability = _weighted_component_score(
        {
            "roe": _score_higher(metrics.get("roe"), good=0.14, bad=0.05),
            "roa": _score_higher(metrics.get("roa"), good=0.025, bad=0.004),
            "margin": _score_higher(metrics.get("net_margin"), good=0.14, bad=0.02),
            "investment_income": _score_higher(metrics.get("insurance_investment_income_to_revenue"), good=0.12, bad=0.03),
        },
        {"roe": 0.35, "roa": 0.20, "margin": 0.25, "investment_income": 0.20},
    )
    balance = _weighted_component_score(
        {
            "equity_assets": _score_higher(metrics.get("insurance_equity_to_assets"), good=0.22, bad=0.07),
            "debt_equity": _score_lower(metrics.get("debt_to_equity"), good=0.35, bad=1.20),
            "reserves_premiums": _score_band(metrics.get("insurance_reserves_to_premiums"), good_low=0.70, good_high=3.50, bad_low=0.20, bad_high=6.00),
        },
        {"equity_assets": 0.45, "debt_equity": 0.30, "reserves_premiums": 0.25},
    )
    growth = _weighted_component_score(
        {
            "premium": _score_higher(metrics.get("insurance_premium_growth"), good=0.06, bad=-0.08),
            "book": _score_higher(metrics.get("insurance_book_value_growth"), good=0.08, bad=-0.10),
            "earnings": _score_higher(metrics.get("earnings_growth"), good=0.08, bad=-0.20),
        },
        {"premium": 0.40, "book": 0.30, "earnings": 0.30},
    )
    total = _weighted_component_score(
        {
            "valuation": valuation,
            "underwriting": underwriting,
            "profitability": profitability,
            "balance": balance,
            "growth": growth,
        },
        {"valuation": 0.18, "underwriting": 0.25, "profitability": 0.25, "balance": 0.22, "growth": 0.10},
    )
    return {
        "insurance": total,
        "insurance_valuation": valuation,
        "insurance_underwriting": underwriting,
        "insurance_profitability": profitability,
        "insurance_balance": balance,
        "insurance_growth": growth,
    }


def bdc_scores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    valuation = _weighted_component_score(
        {
            "p_nav": _score_band(metrics.get("bdc_price_to_nav"), good_low=0.85, good_high=1.15, bad_low=0.55, bad_high=1.55),
            "nii_yield": _score_higher(metrics.get("bdc_nii_yield"), good=0.10, bad=0.04),
            "ey": _score_higher(metrics.get("earnings_yield"), good=0.09, bad=0.03),
        },
        {"p_nav": 0.45, "nii_yield": 0.35, "ey": 0.20},
    )
    income = _weighted_component_score(
        {
            "dividend_coverage": _score_higher(metrics.get("bdc_dividend_coverage"), good=1.10, bad=0.80),
            "nii_margin": _score_higher(metrics.get("bdc_nii_margin"), good=0.45, bad=0.20),
            "investment_yield": _score_higher(metrics.get("bdc_investment_income_to_assets"), good=0.09, bad=0.035),
        },
        {"dividend_coverage": 0.45, "nii_margin": 0.30, "investment_yield": 0.25},
    )
    capital = _weighted_component_score(
        {
            "asset_coverage": _score_higher(metrics.get("bdc_asset_coverage_ratio"), good=2.00, bad=1.50),
            "leverage": _score_lower(metrics.get("bdc_debt_to_equity"), good=0.90, bad=1.80),
            "equity_assets": _score_higher(metrics.get("bdc_equity_to_assets"), good=0.45, bad=0.25),
        },
        {"asset_coverage": 0.45, "leverage": 0.35, "equity_assets": 0.20},
    )
    growth = _weighted_component_score(
        {
            "nii": _score_higher(metrics.get("bdc_net_investment_income_growth"), good=0.06, bad=-0.12),
            "income": _score_higher(metrics.get("bdc_investment_income_growth"), good=0.06, bad=-0.12),
            "nav": _score_higher(metrics.get("bdc_nav_growth"), good=0.05, bad=-0.12),
        },
        {"nii": 0.40, "income": 0.30, "nav": 0.30},
    )
    total = _weighted_component_score(
        {"valuation": valuation, "income": income, "capital": capital, "growth": growth},
        {"valuation": 0.25, "income": 0.30, "capital": 0.30, "growth": 0.15},
    )
    return {
        "bdc": total,
        "bdc_valuation": valuation,
        "bdc_income": income,
        "bdc_capital": capital,
        "bdc_growth": growth,
    }


def utility_scores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    valuation = _weighted_component_score(
        {
            "pe": _score_lower(metrics.get("pe_ratio"), good=15.0, bad=28.0),
            "fpe": _score_lower(metrics.get("forward_pe"), good=15.0, bad=26.0),
            "ev_ebitda": _score_lower(metrics.get("ev_ebitda"), good=10.0, bad=16.0),
            "ey": _score_higher(metrics.get("earnings_yield"), good=0.06, bad=0.025),
        },
        {"pe": 0.25, "fpe": 0.25, "ev_ebitda": 0.25, "ey": 0.25},
    )
    safety = _weighted_component_score(
        {
            "ffo_debt": _score_higher(metrics.get("utility_ffo_to_debt"), good=0.18, bad=0.08),
            "interest": _score_higher(metrics.get("interest_coverage"), good=4.0, bad=1.8),
            "debt_capital": _score_lower(metrics.get("utility_debt_to_capital"), good=0.55, bad=0.78),
            "payout": _score_lower(metrics.get("dividend_payout_ratio"), good=0.65, bad=0.95),
        },
        {"ffo_debt": 0.35, "interest": 0.25, "debt_capital": 0.25, "payout": 0.15},
    )
    profitability = _weighted_component_score(
        {
            "roe": _score_higher(metrics.get("roe"), good=0.10, bad=0.05),
            "roa": _score_higher(metrics.get("roa"), good=0.035, bad=0.008),
            "margin": _score_higher(metrics.get("operating_margin"), good=0.22, bad=0.08),
        },
        {"roe": 0.40, "roa": 0.25, "margin": 0.35},
    )
    growth = _weighted_component_score(
        {
            "revenue": _score_higher(metrics.get("revenue_growth"), good=0.04, bad=-0.05),
            "eps": _score_higher(metrics.get("eps_growth"), good=0.05, bad=-0.10),
            "dividend": _score_higher(metrics.get("dividend_growth"), good=0.04, bad=-0.05),
            "rate_base_proxy": _score_higher(metrics.get("utility_asset_growth"), good=0.05, bad=-0.03),
        },
        {"revenue": 0.20, "eps": 0.30, "dividend": 0.20, "rate_base_proxy": 0.30},
    )
    cash = _weighted_component_score(
        {
            "ocf_margin": _score_higher(metrics.get("operating_cash_flow_margin"), good=0.22, bad=0.08),
            "capex_ocf": _score_band(metrics.get("utility_capex_to_ocf"), good_low=0.35, good_high=1.20, bad_low=0.05, bad_high=1.90),
        },
        {"ocf_margin": 0.55, "capex_ocf": 0.45},
    )
    total = _weighted_component_score(
        {"valuation": valuation, "safety": safety, "profitability": profitability, "growth": growth, "cash": cash},
        {"valuation": 0.18, "safety": 0.32, "profitability": 0.20, "growth": 0.18, "cash": 0.12},
    )
    return {
        "utility": total,
        "utility_valuation": valuation,
        "utility_safety": safety,
        "utility_profitability": profitability,
        "utility_growth": growth,
        "utility_cash": cash,
    }


def midstream_scores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    valuation = _weighted_component_score(
        {
            "ev_ebitda": _score_lower(metrics.get("ev_ebitda"), good=8.5, bad=14.0),
            "fcf_yield": _score_higher(metrics.get("fcf_yield"), good=0.08, bad=0.025),
            "dcf_yield": _score_higher(metrics.get("midstream_dcf_yield"), good=0.09, bad=0.035),
        },
        {"ev_ebitda": 0.35, "fcf_yield": 0.30, "dcf_yield": 0.35},
    )
    distribution = _weighted_component_score(
        {
            "coverage": _score_higher(metrics.get("midstream_distribution_coverage"), good=1.35, bad=0.90),
            "dividend_growth": _score_higher(metrics.get("dividend_growth"), good=0.04, bad=-0.08),
        },
        {"coverage": 0.75, "dividend_growth": 0.25},
    )
    leverage = _weighted_component_score(
        {
            "debt_ebitda": _score_lower(metrics.get("debt_ebitda"), good=3.8, bad=5.8),
            "debt_fcf": _score_lower(metrics.get("debt_fcf_ratio"), good=5.0, bad=12.0),
            "interest": _score_higher(metrics.get("interest_coverage"), good=4.0, bad=1.8),
        },
        {"debt_ebitda": 0.45, "debt_fcf": 0.30, "interest": 0.25},
    )
    quality = _weighted_component_score(
        {
            "ocf_margin": _score_higher(metrics.get("operating_cash_flow_margin"), good=0.18, bad=0.06),
            "ebit_margin": _score_higher(metrics.get("ebit_margin"), good=0.18, bad=0.06),
            "capex_sales": _score_lower(metrics.get("capex_to_sales"), good=0.08, bad=0.22),
        },
        {"ocf_margin": 0.35, "ebit_margin": 0.35, "capex_sales": 0.30},
    )
    total = _weighted_component_score(
        {"valuation": valuation, "distribution": distribution, "leverage": leverage, "quality": quality},
        {"valuation": 0.25, "distribution": 0.25, "leverage": 0.30, "quality": 0.20},
    )
    return {
        "midstream": total,
        "midstream_valuation": valuation,
        "midstream_distribution": distribution,
        "midstream_leverage": leverage,
        "midstream_quality": quality,
    }


def asset_manager_scores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    valuation = _weighted_component_score(
        {
            "pe": _score_lower(metrics.get("pe_ratio"), good=16.0, bad=32.0),
            "fpe": _score_lower(metrics.get("forward_pe"), good=15.0, bad=30.0),
            "ey": _score_higher(metrics.get("earnings_yield"), good=0.065, bad=0.025),
        },
        {"pe": 0.35, "fpe": 0.30, "ey": 0.35},
    )
    profitability = _weighted_component_score(
        {
            "operating_margin": _score_higher(metrics.get("operating_margin"), good=0.35, bad=0.15),
            "roe": _score_higher(metrics.get("roe"), good=0.16, bad=0.06),
            "roa": _score_higher(metrics.get("roa"), good=0.05, bad=0.01),
            "fee_margin": _score_higher(metrics.get("asset_manager_fee_margin"), good=0.32, bad=0.12),
        },
        {"operating_margin": 0.30, "roe": 0.25, "roa": 0.20, "fee_margin": 0.25},
    )
    stability = _weighted_component_score(
        {
            "debt_equity": _score_lower(metrics.get("debt_to_equity"), good=0.80, bad=2.50),
            "fcf_margin": _score_higher(metrics.get("fcf_margin"), good=0.22, bad=0.05),
            "accrual": _score_lower(metrics.get("accrual_ratio"), good=0.02, bad=0.10),
        },
        {"debt_equity": 0.30, "fcf_margin": 0.45, "accrual": 0.25},
    )
    growth = _weighted_component_score(
        {
            "revenue": _score_higher(metrics.get("revenue_growth"), good=0.07, bad=-0.10),
            "eps": _score_higher(metrics.get("eps_growth"), good=0.08, bad=-0.15),
            "fee": _score_higher(metrics.get("asset_manager_fee_growth"), good=0.06, bad=-0.12),
        },
        {"revenue": 0.35, "eps": 0.35, "fee": 0.30},
    )
    total = _weighted_component_score(
        {"valuation": valuation, "profitability": profitability, "stability": stability, "growth": growth},
        {"valuation": 0.22, "profitability": 0.35, "stability": 0.25, "growth": 0.18},
    )
    return {
        "asset_manager": total,
        "asset_manager_valuation": valuation,
        "asset_manager_profitability": profitability,
        "asset_manager_stability": stability,
        "asset_manager_growth": growth,
    }


def capital_markets_scores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    valuation = _weighted_component_score(
        {
            "pb": _score_lower(metrics.get("price_to_book"), good=1.3, bad=2.8),
            "ptbv": _score_lower(metrics.get("bank_price_to_tangible_book"), good=1.4, bad=3.0),
            "pe": _score_lower(metrics.get("pe_ratio"), good=11.0, bad=24.0),
            "ey": _score_higher(metrics.get("earnings_yield"), good=0.08, bad=0.03),
        },
        {"pb": 0.25, "ptbv": 0.30, "pe": 0.25, "ey": 0.20},
    )
    profitability = _weighted_component_score(
        {
            "roe": _score_higher(metrics.get("roe"), good=0.14, bad=0.05),
            "roa": _score_higher(metrics.get("roa"), good=0.012, bad=0.002),
            "net_margin": _score_higher(metrics.get("net_margin"), good=0.20, bad=0.05),
            "efficiency": _score_lower(metrics.get("bank_efficiency_ratio"), good=0.60, bad=0.85),
        },
        {"roe": 0.35, "roa": 0.20, "net_margin": 0.25, "efficiency": 0.20},
    )
    capital = _weighted_component_score(
        {
            "equity_assets": _score_higher(metrics.get("bank_equity_to_assets"), good=0.08, bad=0.03),
            "tangible_equity_assets": _score_higher(metrics.get("bank_tangible_equity_to_tangible_assets"), good=0.06, bad=0.02),
            "tier1": _score_higher(metrics.get("bank_tier1_capital_ratio"), good=0.12, bad=0.08),
            "leverage": _score_higher(metrics.get("bank_tier1_leverage_ratio"), good=0.055, bad=0.035),
        },
        {"equity_assets": 0.30, "tangible_equity_assets": 0.25, "tier1": 0.25, "leverage": 0.20},
    )
    growth = _weighted_component_score(
        {
            "revenue": _score_higher(metrics.get("revenue_growth"), good=0.06, bad=-0.12),
            "earnings": _score_higher(metrics.get("earnings_growth"), good=0.08, bad=-0.25),
            "eps": _score_higher(metrics.get("eps_growth"), good=0.08, bad=-0.25),
        },
        {"revenue": 0.30, "earnings": 0.35, "eps": 0.35},
    )
    total = _weighted_component_score(
        {"valuation": valuation, "profitability": profitability, "capital": capital, "growth": growth},
        {"valuation": 0.25, "profitability": 0.30, "capital": 0.25, "growth": 0.20},
    )
    return {
        "capital_markets": total,
        "capital_markets_valuation": valuation,
        "capital_markets_profitability": profitability,
        "capital_markets_capital": capital,
        "capital_markets_growth": growth,
    }


def sector_switch(sector: str, mapping: dict[str, float], default: float) -> float:
    return mapping.get(sector, default)


def _build_bank_risk_rating(metrics: dict[str, float | None]) -> str:
    bank_score = metrics.get("bank_score")
    pb = metrics.get("price_to_book")
    ptbv = metrics.get("bank_price_to_tangible_book")
    pe = metrics.get("pe_ratio")
    roe = metrics.get("roe")
    roa = metrics.get("roa")
    equity_assets = metrics.get("bank_equity_to_assets")
    tangible_equity_assets = metrics.get("bank_tangible_equity_to_tangible_assets")
    capital_rwa = metrics.get("bank_capital_to_rwa")
    tier1 = metrics.get("bank_tier1_capital_ratio")
    tier1_leverage = metrics.get("bank_tier1_leverage_ratio")
    loans_deposits = metrics.get("bank_loans_to_deposits")
    allowance_loans = metrics.get("bank_allowance_to_loans")
    efficiency = metrics.get("bank_efficiency_ratio")
    deposit_growth = metrics.get("bank_deposit_growth")
    earnings_growth = metrics.get("earnings_growth")

    bank_inputs = [
        pb,
        ptbv,
        pe,
        roe,
        roa,
        equity_assets,
        tangible_equity_assets,
        capital_rwa,
        tier1,
        tier1_leverage,
        loans_deposits,
        allowance_loans,
        efficiency,
    ]
    if not any(has_value(value) for value in bank_inputs):
        return "no data"

    reasons: list[str] = []
    if has_value(pb) and pb > 2.5:
        reasons.append("High P/B")
    if has_value(ptbv) and ptbv > 3.0:
        reasons.append("High P/Tangible Book")
    if has_value(pe) and pe > 22:
        reasons.append("High P/E")
    if has_value(roe) and roe < 0.07:
        reasons.append("Low ROE")
    if has_value(roa) and roa < 0.005:
        reasons.append("Low ROA")
    if has_value(equity_assets) and equity_assets < 0.05:
        reasons.append("Thin Equity/Assets")
    if has_value(tangible_equity_assets) and tangible_equity_assets < 0.035:
        reasons.append("Thin Tangible Equity")
    if has_value(capital_rwa) and capital_rwa < 0.09:
        reasons.append("Low Capital/RWA")
    if has_value(tier1) and tier1 < 0.085:
        reasons.append("Low Tier 1 Capital")
    if has_value(tier1_leverage) and tier1_leverage < 0.04:
        reasons.append("Low Tier 1 Leverage")
    if has_value(loans_deposits) and loans_deposits > 1.05:
        reasons.append("High Loans/Deposits")
    if has_value(allowance_loans) and allowance_loans < 0.004:
        reasons.append("Low Allowance/Loans")
    if has_value(efficiency) and efficiency > 0.75:
        reasons.append("High Efficiency Ratio")
    if has_value(deposit_growth) and deposit_growth < -0.08:
        reasons.append("Shrinking Deposits")
    if has_value(earnings_growth) and earnings_growth < -0.25:
        reasons.append("Falling Earnings")

    fatal = any(
        [
            has_value(equity_assets) and equity_assets < 0.025,
            has_value(tangible_equity_assets) and tangible_equity_assets < 0.015,
            has_value(capital_rwa) and capital_rwa < 0.065,
            has_value(tier1) and tier1 < 0.06,
            has_value(tier1_leverage) and tier1_leverage < 0.03,
            has_value(loans_deposits) and loans_deposits > 1.25,
            has_value(pb) and pb > 4 and has_value(roe) and roe < 0.04,
        ]
    )
    bad_reasons = ", ".join(dict.fromkeys(reasons))

    if fatal:
        return f"Bad: {bad_reasons or 'Fatal bank capital risk'}"
    if (
        has_value(bank_score)
        and bank_score >= 80
        and has_value(roe)
        and roe >= 0.10
        and has_value(roa)
        and roa >= 0.008
        and (not has_value(efficiency) or efficiency <= 0.65)
        and (not has_value(equity_assets) or equity_assets >= 0.06)
    ):
        return "Excellent (Bank Quality Override)"
    if not bad_reasons:
        return "Excellent"
    if has_value(bank_score) and bank_score >= 65:
        return f"OK: {bad_reasons}"
    return f"Risky: {bad_reasons}"


def _special_rating(score: float | None, reasons: list[str], fatal: bool, override_label: str) -> str:
    bad_reasons = ", ".join(dict.fromkeys(reasons))
    if not has_value(score):
        return "no data"
    if fatal:
        return f"Bad: {bad_reasons or 'Fatal model-specific risk'}"
    if score >= 82 and not bad_reasons:
        return f"Excellent ({override_label} Quality Override)"
    if score >= 78 and len(reasons) <= 1:
        return "Excellent"
    if score >= 65:
        return f"OK: {bad_reasons}" if bad_reasons else "Excellent"
    return f"Risky: {bad_reasons or 'Weak model-specific score'}"


def _build_insurance_risk_rating(metrics: dict[str, float | None]) -> str:
    score = metrics.get("insurance_score")
    combined = metrics.get("insurance_combined_ratio")
    loss = metrics.get("insurance_loss_ratio")
    equity_assets = metrics.get("insurance_equity_to_assets")
    reserves_premiums = metrics.get("insurance_reserves_to_premiums")
    pb = metrics.get("price_to_book")
    roe = metrics.get("roe")
    reasons: list[str] = []
    if has_value(combined) and combined > 1.02:
        reasons.append("High Combined Ratio")
    if has_value(loss) and loss > 0.78:
        reasons.append("High Loss Ratio")
    if has_value(equity_assets) and equity_assets < 0.10:
        reasons.append("Thin Equity/Assets")
    if has_value(reserves_premiums) and (reserves_premiums < 0.40 or reserves_premiums > 5.0):
        reasons.append("Reserve/Premium Outlier")
    if has_value(pb) and pb > 2.8:
        reasons.append("High P/B")
    if has_value(roe) and roe < 0.06:
        reasons.append("Low ROE")
    fatal = any(
        [
            has_value(combined) and combined > 1.15,
            has_value(equity_assets) and equity_assets < 0.05,
            has_value(pb) and pb > 4.0 and has_value(roe) and roe < 0.05,
        ]
    )
    return _special_rating(score, reasons, fatal, "Insurance")


def _build_bdc_risk_rating(metrics: dict[str, float | None]) -> str:
    score = metrics.get("bdc_score")
    p_nav = metrics.get("bdc_price_to_nav")
    coverage = metrics.get("bdc_dividend_coverage")
    asset_coverage = metrics.get("bdc_asset_coverage_ratio")
    leverage = metrics.get("bdc_debt_to_equity")
    reasons: list[str] = []
    if has_value(p_nav) and p_nav > 1.35:
        reasons.append("High Price/NAV")
    if has_value(p_nav) and p_nav < 0.70:
        reasons.append("Distressed NAV Discount")
    if has_value(coverage) and coverage < 1.0:
        reasons.append("Weak Dividend Coverage")
    if has_value(asset_coverage) and asset_coverage < 1.7:
        reasons.append("Thin Asset Coverage")
    if has_value(leverage) and leverage > 1.4:
        reasons.append("High Leverage")
    fatal = any(
        [
            has_value(asset_coverage) and asset_coverage < 1.5,
            has_value(coverage) and coverage < 0.75,
            has_value(leverage) and leverage > 2.0,
        ]
    )
    return _special_rating(score, reasons, fatal, "BDC")


def _build_utility_risk_rating(metrics: dict[str, float | None]) -> str:
    score = metrics.get("utility_score")
    ffo_debt = metrics.get("utility_ffo_to_debt")
    interest = metrics.get("interest_coverage")
    debt_capital = metrics.get("utility_debt_to_capital")
    payout = metrics.get("dividend_payout_ratio")
    reasons: list[str] = []
    if has_value(ffo_debt) and ffo_debt < 0.11:
        reasons.append("Low FFO/Debt")
    if has_value(interest) and interest < 2.2:
        reasons.append("Low Interest Coverage")
    if has_value(debt_capital) and debt_capital > 0.72:
        reasons.append("High Debt/Capital")
    if has_value(payout) and payout > 0.90:
        reasons.append("High Payout")
    fatal = any(
        [
            has_value(ffo_debt) and ffo_debt < 0.07,
            has_value(interest) and interest < 1.4,
            has_value(debt_capital) and debt_capital > 0.82,
        ]
    )
    return _special_rating(score, reasons, fatal, "Utility")


def _build_midstream_risk_rating(metrics: dict[str, float | None]) -> str:
    score = metrics.get("midstream_score")
    coverage = metrics.get("midstream_distribution_coverage")
    debt_ebitda = metrics.get("debt_ebitda")
    dcf_yield = metrics.get("midstream_dcf_yield")
    reasons: list[str] = []
    if has_value(coverage) and coverage < 1.10:
        reasons.append("Low Distribution Coverage")
    if has_value(debt_ebitda) and debt_ebitda > 5.0:
        reasons.append("High Debt/EBITDA")
    if has_value(dcf_yield) and dcf_yield < 0.04:
        reasons.append("Low DCF Yield")
    fatal = any(
        [
            has_value(coverage) and coverage < 0.85,
            has_value(debt_ebitda) and debt_ebitda > 6.5,
        ]
    )
    return _special_rating(score, reasons, fatal, "Midstream")


def _build_asset_manager_risk_rating(metrics: dict[str, float | None]) -> str:
    score = metrics.get("asset_manager_score")
    operating_margin = metrics.get("operating_margin")
    fcf_margin = metrics.get("fcf_margin")
    debt_eq = metrics.get("debt_to_equity")
    pe = metrics.get("pe_ratio")
    reasons: list[str] = []
    if has_value(operating_margin) and operating_margin < 0.18:
        reasons.append("Low Operating Margin")
    if has_value(fcf_margin) and fcf_margin < 0.08:
        reasons.append("Weak FCF Margin")
    if has_value(debt_eq) and debt_eq > 2.0:
        reasons.append("High Debt/Equity")
    if has_value(pe) and pe > 32:
        reasons.append("High P/E")
    fatal = any(
        [
            has_value(operating_margin) and operating_margin < 0.08,
            has_value(debt_eq) and debt_eq > 3.5,
        ]
    )
    return _special_rating(score, reasons, fatal, "Asset Manager")


def _build_capital_markets_risk_rating(metrics: dict[str, float | None]) -> str:
    score = metrics.get("capital_markets_score")
    ptbv = metrics.get("bank_price_to_tangible_book")
    roe = metrics.get("roe")
    equity_assets = metrics.get("bank_equity_to_assets")
    tier1 = metrics.get("bank_tier1_capital_ratio")
    reasons: list[str] = []
    if has_value(ptbv) and ptbv > 2.7:
        reasons.append("High P/Tangible Book")
    if has_value(roe) and roe < 0.07:
        reasons.append("Low ROE")
    if has_value(equity_assets) and equity_assets < 0.04:
        reasons.append("Thin Equity/Assets")
    if has_value(tier1) and tier1 < 0.09:
        reasons.append("Low Tier 1 Capital")
    fatal = any(
        [
            has_value(equity_assets) and equity_assets < 0.025,
            has_value(tier1) and tier1 < 0.065,
            has_value(ptbv) and ptbv > 4 and has_value(roe) and roe < 0.05,
        ]
    )
    return _special_rating(score, reasons, fatal, "Capital Markets")


def build_risk_rating(metrics: dict[str, float | None], sector: str, industry_group: str = "general") -> str:
    if has_value(metrics.get("insurance_score")):
        return _build_insurance_risk_rating(metrics)
    if has_value(metrics.get("bdc_score")):
        return _build_bdc_risk_rating(metrics)
    if has_value(metrics.get("utility_score")):
        return _build_utility_risk_rating(metrics)
    if has_value(metrics.get("midstream_score")):
        return _build_midstream_risk_rating(metrics)
    if has_value(metrics.get("asset_manager_score")):
        return _build_asset_manager_risk_rating(metrics)
    if has_value(metrics.get("capital_markets_score")):
        return _build_capital_markets_risk_rating(metrics)
    if sector == "Financials" and has_value(metrics.get("bank_score")):
        return _build_bank_risk_rating(metrics)

    pb = metrics.get("price_to_book")
    ps = metrics.get("price_to_sales")
    fpe = metrics.get("forward_pe")
    peg = metrics.get("peg_ratio")
    evebitda = metrics.get("ev_ebitda")
    evsales = metrics.get("ev_sales")
    roe = metrics.get("roe")
    roa = metrics.get("roa")
    roic = metrics.get("roic")
    debt_eq = metrics.get("debt_to_equity")
    current = metrics.get("current_ratio")
    altman = metrics.get("altman_z")
    fcf_margin = metrics.get("fcf_margin")
    ey = metrics.get("earnings_yield")
    pffo = metrics.get("p_ffo")
    paffo = metrics.get("p_affo")
    ffo_yield = metrics.get("ffo_yield")
    affo_yield = metrics.get("affo_yield")
    payout_ffo = metrics.get("payout_ratio_ffo")
    noi = metrics.get("noi")
    occupancy = metrics.get("occupancy")
    interest_coverage = metrics.get("interest_coverage")

    capital_intensive_group = industry_group in {"telecom", "autos", "airlines", "aerospace_defense"}

    w_pb = sector_switch(
        sector,
        {
            "Technology": 0.6,
            "Healthcare": 0.9,
            "Financials": 1.3,
            "Utilities": 0.5,
            "Consumer Staples": 0.4,
            "Energy": 0.9,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 0.7,
            "Consumer Discretionary": 0.8,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_ps = sector_switch(
        sector,
        {
            "Technology": 1.1,
            "Healthcare": 0.8,
            "Financials": 0.5,
            "Utilities": 0.6,
            "Consumer Staples": 0.7,
            "Energy": 1.0,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 0.9,
            "Consumer Discretionary": 1.1,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_ev = sector_switch(
        sector,
        {
            "Technology": 1.1,
            "Healthcare": 1.0,
            "Financials": 0.4,
            "Utilities": 0.9,
            "Consumer Staples": 0.7,
            "Energy": 1.2,
            "Industrials": 1.1,
            "Materials": 1.1,
            "Communication Services": 1.0,
            "Consumer Discretionary": 1.0,
            "Real Estate": 0.4,
        },
        1.0,
    )
    w_fcf = sector_switch(
        sector,
        {
            "Technology": 1.3,
            "Healthcare": 1.1,
            "Financials": 0.3,
            "Utilities": 0.9,
            "Consumer Staples": 0.8,
            "Energy": 1.1,
            "Industrials": 1.1,
            "Materials": 1.0,
            "Communication Services": 1.0,
            "Consumer Discretionary": 1.2,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_ey = sector_switch(
        sector,
        {
            "Technology": 1.0,
            "Healthcare": 0.95,
            "Financials": 1.2,
            "Utilities": 1.1,
            "Consumer Staples": 0.85,
            "Energy": 1.0,
            "Industrials": 0.95,
            "Materials": 1.0,
            "Communication Services": 1.0,
            "Consumer Discretionary": 1.0,
            "Real Estate": 1.4,
        },
        1.0,
    )

    pb_thresh = sector_switch(
        sector,
        {
            "Financials": 5.0,
            "Consumer Staples": 18.0,
            "Technology": 12.0,
            "Healthcare": 10.0,
            "Utilities": 7.0,
            "Energy": 10.0,
            "Industrials": 12.0,
            "Materials": 12.0,
            "Communication Services": 11.0,
            "Consumer Discretionary": 13.0,
            "Real Estate": 8.0,
        },
        10.0,
    )
    ps_thresh = sector_switch(
        sector,
        {
            "Financials": 2.5,
            "Consumer Staples": 8.0,
            "Technology": 12.0,
            "Healthcare": 10.0,
            "Utilities": 6.0,
            "Energy": 9.0,
            "Industrials": 10.0,
            "Materials": 10.0,
            "Communication Services": 10.0,
            "Consumer Discretionary": 12.0,
            "Real Estate": 15.0,
        },
        10.0,
    )
    evsales_thresh = sector_switch(
        sector,
        {
            "Financials": 3.0,
            "Consumer Staples": 6.0,
            "Technology": 10.0,
            "Healthcare": 8.0,
            "Utilities": 5.0,
            "Energy": 7.0,
            "Industrials": 8.0,
            "Materials": 8.0,
            "Communication Services": 8.0,
            "Consumer Discretionary": 9.0,
            "Real Estate": 18.0,
        },
        8.0,
    )
    ey_thresh = sector_switch(
        sector,
        {
            "Financials": 0.04,
            "Consumer Staples": 0.03,
            "Technology": 0.045,
            "Healthcare": 0.04,
            "Utilities": 0.04,
            "Energy": 0.045,
            "Industrials": 0.04,
            "Materials": 0.04,
            "Communication Services": 0.04,
            "Consumer Discretionary": 0.045,
            "Real Estate": 0.02,
        },
        0.04,
    )

    def bucket(value: float | None, good: float, ok: float, reverse: bool = False) -> float:
        if not has_value(value):
            return 0.0
        if reverse:
            if value <= good:
                return 1.0
            if value <= ok:
                return 0.7
            return 0.5
        if value >= good:
            return 1.0
        if value >= ok:
            return 0.7
        return 0.5

    val_score_base = (
        bucket(fpe, 30, 50, reverse=True) * w_ev
        + bucket(peg, 2, 3, reverse=True) * w_ev
        + bucket(pb, 4, pb_thresh, reverse=True) * w_pb
        + bucket(ps, 6, ps_thresh, reverse=True) * w_ps
        + bucket(evebitda, 15, 30, reverse=True) * w_ev
        + bucket(evsales, 5, evsales_thresh, reverse=True) * w_ev
    ) / 6

    val_score_re = (
        bucket(pffo, 15, 22, reverse=True) * w_ev
        + bucket(paffo, 15, 22, reverse=True) * w_ev
        + bucket(ffo_yield, 0.065, 0.045) * w_ey
        + bucket(affo_yield, 0.065, 0.045) * w_ey
    ) / 4

    val_score = val_score_re if sector == "Real Estate" else val_score_base

    prof_score_base = (
        bucket(roe, 0.12, 0.08) * w_fcf
        + bucket(roa, 0.05, 0.03) * w_fcf
        + bucket(roic, 0.08, 0.05) * w_fcf
        + bucket(fcf_margin, 0.08, 0.04) * w_fcf
    ) / 4
    prof_score_re = (
        (1.0 if has_value(noi) and noi > 0 else 0.3 if has_value(noi) else 0.0) * 0.4
        + bucket(occupancy, 0.93, 0.88) * 0.6
    )
    prof_score = prof_score_re if sector == "Real Estate" else prof_score_base

    solv_score_base = (
        bucket(
            debt_eq,
            sector_switch(
                sector,
                {
                    "Financials": 4.0,
                    "Utilities": 2.0,
                    "Energy": 2.0,
                    "Industrials": 1.8,
                    "Materials": 1.8,
                    "Communication Services": 2.0,
                    "Consumer Discretionary": 1.5,
                    "Consumer Staples": 1.5,
                    "Healthcare": 1.0,
                    "Technology": 0.8,
                    "Real Estate": 2.5,
                },
                1.5,
            ),
            sector_switch(
                sector,
                {
                    "Financials": 6.0,
                    "Utilities": 3.0,
                    "Energy": 3.0,
                    "Industrials": 2.7,
                    "Materials": 2.7,
                    "Communication Services": 3.0,
                    "Consumer Discretionary": 2.2,
                    "Consumer Staples": 2.2,
                    "Healthcare": 1.5,
                    "Technology": 1.2,
                    "Real Estate": 3.5,
                },
                2.0,
            ),
            reverse=True,
        )
        * w_fcf
        + bucket(current, 1.2, 1.0) * w_fcf
        + bucket(altman, 3.0, 2.2) * w_fcf
    ) / 3
    solv_score_re = (
        bucket(debt_eq, 2.0, 3.0, reverse=True) * 0.5
        + bucket(current, 1.0, 0.8) * 0.3
        + bucket(payout_ffo, 0.85, 0.95, reverse=True) * 0.2
    )
    solv_score = solv_score_re if sector == "Real Estate" else solv_score_base

    ey_score = bucket(ey, ey_thresh, ey_thresh * 0.8) * w_ey
    profit_boost = max(0.0, (prof_score - 0.5) * 0.15)

    if sector == "Real Estate":
        quality_override = (
            prof_score >= 0.75
            and solv_score >= 0.7
            and has_value(ffo_yield)
            and ffo_yield >= 0.06
            and has_value(occupancy)
            and occupancy >= 0.9
        )
    else:
        quality_override = (
            prof_score >= 0.75
            and solv_score >= 0.7
            and has_value(fcf_margin)
            and fcf_margin >= 0.02
        )

    reasons: list[str] = []
    if sector not in {"Real Estate", "Financials"} and has_value(pb) and pb > pb_thresh and w_pb > 0.6:
        reasons.append("High P/B")
    if sector != "Real Estate" and has_value(ps) and ps > ps_thresh and w_ps > 0.75:
        reasons.append("High P/S")
    if sector != "Real Estate" and has_value(evsales) and evsales > evsales_thresh and w_ev > 0.75:
        reasons.append("High EV/Sales")
    if sector not in {"Real Estate", "Financials"} and has_value(ey) and ey < ey_thresh and w_ey > 0.8:
        reasons.append("Low Earnings Yield")
    debt_reason_threshold = sector_switch(
        sector,
        {
            "Financials": 5.0,
            "Utilities": 2.5,
            "Energy": 2.5,
            "Industrials": 2.0,
            "Materials": 2.0,
            "Communication Services": 2.5,
            "Consumer Discretionary": 1.8,
            "Consumer Staples": 1.8,
            "Healthcare": 1.5,
            "Technology": 1.2,
            "Real Estate": 3.2,
        },
        2.0,
    )
    debt_reason_threshold = {
        "telecom": 3.5,
        "autos": 6.0,
        "airlines": 5.0,
        "aerospace_defense": 4.0,
    }.get(industry_group, debt_reason_threshold)
    if has_value(debt_eq) and debt_eq > debt_reason_threshold:
        reasons.append("High Debt/Equity")
    if sector not in {"Real Estate", "Financials"} and has_value(roe) and roe < 0.08 and w_fcf > 0.8:
        reasons.append("Low ROE")
    if sector not in {"Real Estate", "Financials"} and has_value(roa) and roa < 0.03 and w_fcf > 0.8:
        reasons.append("Low ROA")
    if sector not in {"Real Estate", "Financials"} and has_value(roic) and roic < 0.05 and w_fcf > 0.8:
        reasons.append("Low ROIC")
    if sector not in {"Real Estate", "Financials", "Utilities"} and has_value(fcf_margin) and fcf_margin < 0.02 and w_fcf > 0.8:
        reasons.append("Weak FCF Margin")
    if sector == "Real Estate" and has_value(occupancy) and occupancy < 0.88:
        reasons.append("Low Occupancy")
    if sector == "Real Estate" and has_value(ffo_yield) and ffo_yield < 0.045:
        reasons.append("Low FFO Yield")
    if sector == "Real Estate" and has_value(affo_yield) and affo_yield < 0.045:
        reasons.append("Low AFFO Yield")
    if sector == "Real Estate" and has_value(pffo) and pffo > 22:
        reasons.append("High P/FFO")
    if sector == "Real Estate" and has_value(payout_ffo) and payout_ffo > 0.95:
        reasons.append("High FFO Payout")
    if sector == "Financials" and has_value(pb) and pb > 5:
        reasons.append("High P/B")
    if sector == "Financials" and has_value(roe) and roe < 0.06:
        reasons.append("Low ROE")
    if sector == "Utilities" and has_value(debt_eq) and debt_eq > 2.5 and has_value(ey) and ey < 0.035:
        reasons.append("High Debt + Low Yield")
    if (
        capital_intensive_group
        and has_value(altman)
        and altman < 1.1
        and not ((has_value(fcf_margin) and fcf_margin > 0.08) and (has_value(interest_coverage) and interest_coverage > 3.0))
    ):
        reasons.append("Weak Balance Sheet")

    fatal = any(
        [
            has_value(debt_eq)
            and debt_eq
            > {
                "telecom": 6.0,
                "autos": 10.0,
                "airlines": 8.0,
                "aerospace_defense": 6.0,
            }.get(
                industry_group,
                sector_switch(
                    sector,
                    {
                        "Financials": 8.0,
                        "Utilities": 4.0,
                        "Energy": 4.0,
                        "Industrials": 3.5,
                        "Materials": 3.5,
                        "Communication Services": 4.0,
                        "Consumer Discretionary": 3.0,
                        "Consumer Staples": 3.0,
                        "Healthcare": 2.5,
                        "Technology": 2.0,
                        "Real Estate": 5.0,
                    },
                    3.0,
                ),
            ),
            sector != "Real Estate"
            and not (capital_intensive_group and ((has_value(fcf_margin) and fcf_margin > 0.02) or (has_value(interest_coverage) and interest_coverage > 2.0)))
            and has_value(altman)
            and altman < 1.1
            and has_value(current)
            and current < 1.0,
            sector != "Real Estate"
            and not (capital_intensive_group and ((has_value(fcf_margin) and fcf_margin > 0.02) or (has_value(interest_coverage) and interest_coverage > 2.0)))
            and has_value(altman)
            and altman < 1.1
            and has_value(debt_eq)
            and debt_eq > 1.5,
            sector not in {"Real Estate", "Financials"} and has_value(altman) and altman < 1.3 and has_value(fcf_margin) and fcf_margin < 0,
            sector != "Real Estate"
            and all(has_value(value) for value in [pb, ps, evsales, fpe])
            and pb > 50
            and ps > 50
            and evsales > 50
            and fpe > 100,
            sector == "Real Estate" and has_value(occupancy) and occupancy < 0.7,
            sector == "Real Estate" and has_value(debt_eq) and debt_eq > 4 and has_value(ffo_yield) and ffo_yield < 0.04,
            sector == "Financials" and has_value(pb) and pb > 10 and has_value(roe) and roe < 0.04,
        ]
    )

    all_empty = not any(
        has_value(value)
        for value in [fpe, peg, pb, ps, evebitda, evsales, roe, roa, roic, debt_eq, current, altman, fcf_margin, ey]
    )
    total_score = min(1.0, (available_average(val_score, prof_score, solv_score, ey_score) or 0) + profit_boost)
    bad_reasons = ", ".join(dict.fromkeys(reasons))

    if all_empty:
        return "no data"
    if fatal:
        return f"Bad: {bad_reasons or 'Fatal financial risk'}"
    if quality_override and (industry_group == "general" or not bad_reasons):
        return "Excellent (Quality Override)"
    if not bad_reasons:
        return "Excellent"
    if total_score >= 0.85:
        return "Excellent"
    if total_score >= 0.65:
        return f"OK: {bad_reasons}"
    return f"Risky: {bad_reasons}"


def build_recommendation(
    metrics: dict[str, float | None],
    scores: dict[str, float | None],
    profile: dict[str, Any],
    *,
    current_hold: float,
    macro_score: float,
    risk_rating: str,
) -> dict[str, Any]:
    sector = profile.get("sector") or "Unknown"
    pe = metrics.get("forward_pe")
    pb = metrics.get("price_to_book")
    ps = metrics.get("price_to_sales")
    peg = metrics.get("peg_ratio")
    ev_ebitda = metrics.get("ev_ebitda")
    ev_sales = metrics.get("ev_sales")
    gross = metrics.get("gross_margin")
    netm = metrics.get("net_margin")
    roic = metrics.get("roic")
    roe = metrics.get("roe")
    roa = metrics.get("roa")
    revg = metrics.get("revenue_growth")
    epsg = metrics.get("eps_growth")
    debt_eq = metrics.get("debt_to_equity")
    intcov = metrics.get("interest_coverage")
    fcf_margin = metrics.get("fcf_margin")
    accrual = metrics.get("accrual_ratio")
    ey = metrics.get("earnings_yield")
    altman = metrics.get("altman_z")
    current_ratio = metrics.get("current_ratio")
    composite = scores.get("composite")
    pffo = metrics.get("p_ffo")
    paffo = metrics.get("p_affo")
    ffo_yield = metrics.get("ffo_yield")
    payout_ffo = metrics.get("payout_ratio_ffo")
    noi = metrics.get("noi")
    occupancy = metrics.get("occupancy")
    affo_yield = metrics.get("affo_yield")
    bank_score_value = scores.get("bank")
    bank_ptbv = metrics.get("bank_price_to_tangible_book")
    bank_equity_assets = metrics.get("bank_equity_to_assets")
    bank_tangible_equity_assets = metrics.get("bank_tangible_equity_to_tangible_assets")
    bank_capital_rwa = metrics.get("bank_capital_to_rwa")
    bank_tier1 = metrics.get("bank_tier1_capital_ratio")
    bank_tier1_leverage = metrics.get("bank_tier1_leverage_ratio")
    bank_loans_deposits = metrics.get("bank_loans_to_deposits")
    bank_allowance_loans = metrics.get("bank_allowance_to_loans")
    bank_efficiency = metrics.get("bank_efficiency_ratio")
    bank_deposit_growth = metrics.get("bank_deposit_growth")
    bank_loan_growth = metrics.get("bank_loan_growth")
    is_bank_model = sector == "Financials" and has_value(bank_score_value)
    company_type = profile.get("company_type") or "general"
    industry_group = profile.get("industry_group") or "general"
    special_score_keys = {
        "insurance": "insurance",
        "bdc": "bdc",
        "utility": "utility",
        "midstream": "midstream",
        "asset_manager": "asset_manager",
        "capital_markets": "capital_markets",
    }
    special_score_key = special_score_keys.get(company_type)
    special_score_value = scores.get(special_score_key) if special_score_key else None
    is_special_model = has_value(special_score_value)

    def weight(mapping: dict[str, float], default: float) -> float:
        return mapping.get(sector, default)

    re_score = None
    if sector == "Real Estate":
        re_score = (
            (pffo or 0) * (-0.15)
            + (paffo or 0) * (-0.15)
            + (ffo_yield or 0) * 1.5 * 200
            + (affo_yield or 0) * 1.5 * 200
            + (occupancy or 0) * 1.3 * 50
            + (noi or 0) * 0.0008
            + (payout_ffo or 0) * (-15)
            + (debt_eq or 0) * 0.8 * (-0.08)
            + (current_ratio or 0) * 0.08
        )

    w_pe = weight(
        {
            "Technology": 1.0,
            "Healthcare": 0.9,
            "Financials": 0.5,
            "Utilities": 0.7,
            "Consumer Staples": 0.8,
            "Energy": 0.9,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 0.95,
            "Consumer Discretionary": 0.95,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_peg = weight(
        {
            "Technology": 1.0,
            "Healthcare": 0.95,
            "Financials": 0.8,
            "Utilities": 0.7,
            "Consumer Staples": 0.6,
            "Energy": 1.0,
            "Industrials": 0.95,
            "Materials": 0.95,
            "Communication Services": 0.95,
            "Consumer Discretionary": 0.95,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_pb = weight(
        {
            "Technology": 0.8,
            "Healthcare": 0.9,
            "Financials": 1.2,
            "Utilities": 0.6,
            "Consumer Staples": 0.4,
            "Energy": 0.9,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 0.9,
            "Consumer Discretionary": 0.8,
            "Real Estate": 0.4,
        },
        1.0,
    )
    w_ps = weight(
        {
            "Technology": 1.0,
            "Healthcare": 0.9,
            "Financials": 0.8,
            "Utilities": 0.7,
            "Consumer Staples": 0.6,
            "Energy": 1.0,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 0.9,
            "Consumer Discretionary": 1.0,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_ev = weight(
        {
            "Technology": 1.0,
            "Healthcare": 1.0,
            "Financials": 0.5,
            "Utilities": 0.8,
            "Consumer Staples": 0.6,
            "Energy": 1.1,
            "Industrials": 1.1,
            "Materials": 1.0,
            "Communication Services": 1.0,
            "Consumer Discretionary": 1.0,
            "Real Estate": 0.4,
        },
        1.0,
    )
    w_ey = weight(
        {
            "Technology": 1.0,
            "Healthcare": 0.95,
            "Financials": 1.1,
            "Utilities": 1.0,
            "Consumer Staples": 0.85,
            "Energy": 1.0,
            "Industrials": 0.95,
            "Materials": 1.0,
            "Communication Services": 1.0,
            "Consumer Discretionary": 1.0,
            "Real Estate": 1.4,
        },
        1.0,
    )
    w_fcf = weight(
        {
            "Technology": 1.2,
            "Healthcare": 1.0,
            "Financials": 0.5,
            "Utilities": 0.8,
            "Consumer Staples": 0.7,
            "Energy": 1.0,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 1.0,
            "Consumer Discretionary": 1.2,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_roic = weight(
        {
            "Technology": 1.3,
            "Healthcare": 1.1,
            "Financials": 0.9,
            "Utilities": 0.8,
            "Consumer Staples": 0.9,
            "Energy": 1.0,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 1.1,
            "Consumer Discretionary": 1.2,
            "Real Estate": 0.4,
        },
        1.0,
    )
    w_roe = weight(
        {
            "Technology": 1.2,
            "Healthcare": 1.1,
            "Financials": 1.3,
            "Utilities": 0.7,
            "Consumer Staples": 0.8,
            "Energy": 1.0,
            "Industrials": 1.0,
            "Materials": 1.0,
            "Communication Services": 1.0,
            "Consumer Discretionary": 1.1,
            "Real Estate": 0.3,
        },
        1.0,
    )
    w_debt = weight(
        {
            "Financials": 0.3,
            "Utilities": 0.7,
            "Real Estate": 0.8,
            "Materials": 1.2,
            "Industrials": 1.1,
            "Energy": 1.1,
            "Technology": 1.0,
            "Healthcare": 1.0,
            "Consumer Discretionary": 1.0,
            "Consumer Staples": 0.9,
            "Communication Services": 1.0,
        },
        1.0,
    )
    w_altman = weight(
        {
            "Financials": 0.4,
            "Utilities": 0.8,
            "Real Estate": 0.5,
            "Materials": 1.2,
            "Industrials": 1.1,
            "Energy": 1.1,
            "Technology": 1.0,
            "Healthcare": 1.0,
            "Consumer Discretionary": 1.0,
            "Consumer Staples": 0.9,
            "Communication Services": 1.0,
        },
        1.0,
    )
    w_growth = weight(
        {
            "Technology": 1.4,
            "Healthcare": 1.2,
            "Consumer Discretionary": 1.3,
            "Communication Services": 1.2,
            "Industrials": 1.0,
            "Energy": 0.9,
            "Materials": 0.9,
            "Financials": 0.7,
            "Consumer Staples": 0.6,
            "Utilities": 0.5,
            "Real Estate": 0.7,
        },
        1.0,
    )

    industry_weight_adjustments = {
        "telecom": {"w_pe": 0.90, "w_pb": 0.70, "w_ev": 1.10, "w_fcf": 1.20, "w_roic": 0.85, "w_debt": 1.10, "w_growth": 0.70},
        "autos": {"w_pe": 0.90, "w_ps": 0.80, "w_pb": 0.80, "w_fcf": 1.20, "w_roic": 1.10, "w_debt": 0.80, "w_growth": 0.80},
        "airlines": {"w_pe": 0.85, "w_ps": 0.75, "w_pb": 0.70, "w_fcf": 1.30, "w_roic": 1.10, "w_debt": 1.20, "w_growth": 0.75},
        "semiconductors": {"w_pe": 1.05, "w_ps": 1.10, "w_pb": 0.90, "w_fcf": 1.15, "w_roic": 1.35, "w_debt": 1.00, "w_growth": 1.25},
        "saas": {"w_pe": 0.80, "w_ps": 1.35, "w_pb": 0.70, "w_fcf": 1.40, "w_roic": 0.80, "w_debt": 1.05, "w_growth": 1.50},
        "pharma": {"w_pe": 1.00, "w_ps": 0.90, "w_pb": 0.70, "w_fcf": 1.15, "w_roic": 1.10, "w_debt": 0.90, "w_growth": 0.85},
        "retail": {"w_pe": 1.00, "w_ps": 0.75, "w_pb": 0.80, "w_fcf": 1.25, "w_roic": 1.10, "w_debt": 1.00, "w_growth": 1.05},
        "restaurants": {"w_pe": 0.95, "w_ps": 0.90, "w_pb": 0.60, "w_fcf": 1.20, "w_roic": 1.30, "w_debt": 0.90, "w_growth": 1.15},
        "aerospace_defense": {"w_pe": 1.00, "w_ps": 0.90, "w_pb": 0.70, "w_fcf": 1.20, "w_roic": 1.15, "w_debt": 1.10, "w_growth": 0.95},
    }
    adjustments = industry_weight_adjustments.get(industry_group, {})
    w_pe *= adjustments.get("w_pe", 1.0)
    w_peg *= adjustments.get("w_peg", 1.0)
    w_pb *= adjustments.get("w_pb", 1.0)
    w_ps *= adjustments.get("w_ps", 1.0)
    w_ev *= adjustments.get("w_ev", 1.0)
    w_ey *= adjustments.get("w_ey", 1.0)
    w_fcf *= adjustments.get("w_fcf", 1.0)
    w_roic *= adjustments.get("w_roic", 1.0)
    w_roe *= adjustments.get("w_roe", 1.0)
    w_debt *= adjustments.get("w_debt", 1.0)
    w_altman *= adjustments.get("w_altman", 1.0)
    w_growth *= adjustments.get("w_growth", 1.0)

    sector_ps_norm = weight(
        {
            "Technology": 8.0,
            "Healthcare": 4.0,
            "Financials": 2.0,
            "Consumer Discretionary": 2.0,
            "Communication Services": 5.0,
            "Industrials": 2.0,
            "Consumer Staples": 1.5,
            "Energy": 1.5,
            "Materials": 1.5,
            "Utilities": 2.0,
            "Real Estate": 12.0,
        },
        3.0,
    )
    sector_pe_norm = weight(
        {
            "Technology": 25.0,
            "Healthcare": 20.0,
            "Financials": 12.0,
            "Consumer Discretionary": 20.0,
            "Communication Services": 18.0,
            "Industrials": 18.0,
            "Consumer Staples": 18.0,
            "Energy": 15.0,
            "Materials": 15.0,
            "Utilities": 16.0,
            "Real Estate": 35.0,
        },
        18.0,
    )
    sector_pb_norm = weight(
        {
            "Technology": 6.0,
            "Healthcare": 4.0,
            "Financials": 1.2,
            "Consumer Discretionary": 5.0,
            "Communication Services": 4.0,
            "Industrials": 3.0,
            "Consumer Staples": 4.0,
            "Energy": 2.0,
            "Materials": 2.0,
            "Utilities": 1.5,
            "Real Estate": 5.0,
        },
        3.0,
    )
    industry_norms = {
        "telecom": (2.5, 14.0, 2.5),
        "autos": (0.6, 10.0, 1.5),
        "airlines": (0.8, 10.0, 1.5),
        "semiconductors": (8.0, 28.0, 6.0),
        "saas": (10.0, 32.0, 7.0),
        "pharma": (5.0, 18.0, 4.0),
        "retail": (1.0, 20.0, 5.0),
        "restaurants": (4.0, 25.0, 5.0),
        "aerospace_defense": (2.0, 22.0, 4.0),
    }
    if industry_group in industry_norms:
        sector_ps_norm, sector_pe_norm, sector_pb_norm = industry_norms[industry_group]

    if is_bank_model:
        ps_stretch = 1.0
        pe_stretch = (pe or sector_pe_norm) / sector_pe_norm if sector_pe_norm else 1.0
        pb_stretch = (bank_ptbv or pb or sector_pb_norm) / (1.5 if bank_ptbv else sector_pb_norm)
    else:
        ps_stretch = (ps or sector_ps_norm) / sector_ps_norm if sector_ps_norm else 1.0
        pe_stretch = ((pffo or 16.0) / 16.0) if sector == "Real Estate" else ((pe or sector_pe_norm) / sector_pe_norm if sector_pe_norm else 1.0)
        pb_stretch = (pb or sector_pb_norm) / sector_pb_norm if sector_pb_norm else 1.0
    valuation_stretch = (ps_stretch + pe_stretch + pb_stretch) / 3
    valuation_premium = (valuation_stretch - 2) * 10 if valuation_stretch > 2 else 0

    fundamental_quality_base = (
        (pe or 0) * w_pe * (-0.08)
        + (peg or 0) * w_peg * (-0.07)
        + (pb or 0) * w_pb * 0.06
        + (ps or 0) * w_ps * 0.05
        + (ev_ebitda or 0) * w_ev * 0.07
        + (ey or 0) * w_ey * 100 * 0.08
        + (roic or 0) * w_roic * 100 * 0.12
        + (roe or 0) * w_roe * 100 * 0.11
        + (fcf_margin or 0) * w_fcf * 100 * 0.09
        + (revg or 0) * w_growth * 100 * 0.08
        + (epsg or 0) * w_growth * 20 * 0.06
        + (debt_eq or 0) * w_debt * (-0.05)
        + (altman or 0) * w_altman * 0.08
        + (current_ratio or 0) * 0.05
    )
    fundamental_quality_re = (
        (pffo or 0) * (-0.15)
        + (paffo or 0) * (-0.15)
        + (ffo_yield or 0) * 1.5 * 200
        + (affo_yield or 0) * 1.5 * 200
        + (occupancy or 0) * 1.3 * 50
        + (noi or 0) * 0.0008
        + (payout_ffo or 0) * (-15)
        + (debt_eq or 0) * w_debt * (-0.08)
        + (current_ratio or 0) * 0.08
    )
    def recommendation_support_score(weights: dict[str, float]) -> float | None:
        weighted_sum = 0.0
        total_weight = 0.0
        for key, weight_value in weights.items():
            score_value = scores.get(key)
            if has_value(score_value):
                weighted_sum += score_value * weight_value
                total_weight += weight_value
        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    bank_support_score = recommendation_support_score(
        {"valuation": 0.35, "profitability": 0.30, "growth": 0.20, "cash_flow": 0.15}
    )
    special_support_weights = {
        "insurance": {"valuation": 0.25, "profitability": 0.30, "growth": 0.20, "cash_flow": 0.20, "financial_strength": 0.05},
        "bdc": {"valuation": 0.25, "profitability": 0.20, "growth": 0.15, "cash_flow": 0.20, "financial_strength": 0.20},
        "utility": {"valuation": 0.20, "profitability": 0.20, "growth": 0.15, "cash_flow": 0.25, "financial_strength": 0.20},
        "midstream": {"valuation": 0.25, "profitability": 0.20, "growth": 0.10, "cash_flow": 0.25, "financial_strength": 0.20},
        "asset_manager": {"valuation": 0.25, "profitability": 0.35, "growth": 0.20, "cash_flow": 0.15, "financial_strength": 0.05},
        "capital_markets": {"valuation": 0.30, "profitability": 0.30, "growth": 0.20, "cash_flow": 0.10, "financial_strength": 0.10},
    }
    special_support_score = recommendation_support_score(
        special_support_weights.get(company_type, {"valuation": 0.30, "profitability": 0.30, "growth": 0.20, "cash_flow": 0.20})
    )

    def calibrated_model_base(model_score: float | None, support_score: float | None, model_weight: float) -> float | None:
        if not has_value(model_score):
            return support_score if has_value(support_score) else None
        if not has_value(support_score):
            return model_score
        return model_score * model_weight + support_score * (1 - model_weight)

    bank_base_score = calibrated_model_base(bank_score_value, bank_support_score, 0.86)
    special_model_weights = {
        "insurance": 0.78,
        "bdc": 0.84,
        "utility": 0.82,
        "midstream": 0.82,
        "asset_manager": 0.78,
        "capital_markets": 0.82,
    }
    special_base_score = calibrated_model_base(
        special_score_value,
        special_support_score,
        special_model_weights.get(company_type, 0.80),
    )

    fundamental_quality_bank = ((bank_score_value or 50) - 50) * 0.05
    fundamental_quality_special = ((special_score_value or 50) - 50) * 0.05
    if sector == "Real Estate":
        fundamental_quality = fundamental_quality_re
    elif is_special_model:
        fundamental_quality = fundamental_quality_special
    elif is_bank_model:
        fundamental_quality = fundamental_quality_bank
    else:
        fundamental_quality = fundamental_quality_base

    risk_text = (risk_rating or "").upper()
    if "EXCELLENT" in risk_text and not (is_special_model or is_bank_model):
        excellent_bonus = 3
    elif "EXCELLENT" in risk_text:
        excellent_bonus = 1
    else:
        excellent_bonus = 0
    risky_penalty = -10 if "RISKY" in risk_text else 0
    bad_penalty = -999_999_999 if "BAD" in risk_text else 0

    if sector == "Real Estate" and has_value(re_score):
        base_score = re_score
    elif is_special_model:
        base_score = special_base_score
    elif is_bank_model:
        base_score = bank_base_score
    else:
        base_score = composite if has_value(composite) else 50
    adjusted_score = base_score + excellent_bonus + risky_penalty + bad_penalty + (fundamental_quality * 0.15)
    final_score = clamp(adjusted_score, 0, 100)

    if sector == "Real Estate":
        alloc_base = 0 if final_score < 70 else round(12 * math.exp(0.2 * math.log((final_score - 55) / 20)), 2)
    else:
        alloc_base = 0 if final_score < 70 else round(12 * math.exp(0.2 * math.log((final_score - 45) / 20)), 2)

    growth_boost = 0.0
    if sector == "Real Estate":
        if (ffo_yield or 0) > 0.07 and (occupancy or 0) > 0.92 and final_score > 65:
            growth_boost = 1.2
    elif is_special_model:
        if final_score > 68 and (revg or 0) > 0.05 and (epsg or 0) > 0.06:
            growth_boost = 0.8
    elif is_bank_model:
        if (epsg or 0) > 0.12 and (revg or 0) > 0.04 and (bank_deposit_growth or 0) > 0 and final_score > 65:
            growth_boost = 1.0
    else:
        if (epsg or 0) > 0.3 and (revg or 0) > 0.15 and final_score > 65:
            growth_boost = 1.5

    quality_boost = 0.0
    if sector == "Real Estate":
        if (ffo_yield or 0) > 0.075 and (occupancy or 0) > 0.94 and (debt_eq or 9e9) < 2:
            quality_boost = 1.2
    elif is_special_model:
        if (special_score_value or 0) > 82 and "EXCELLENT" in risk_text:
            quality_boost = 0.8
    elif is_bank_model:
        if (
            (bank_score_value or 0) > 80
            and (roe or 0) > 0.10
            and (roa or 0) > 0.008
            and (bank_efficiency or 9e9) < 0.65
            and (bank_equity_assets or 0) > 0.06
        ):
            quality_boost = 1.0
    else:
        if (roic or 0) > 0.15 and (fcf_margin or 0) > 0.15 and (debt_eq or 9e9) < 0.5:
            quality_boost = 1.0

    risk_drag = -2.5 if "RISKY" in risk_text else 0
    bad_override = -999 if "BAD" in risk_text else 0
    alloc_adjusted = alloc_base + growth_boost + quality_boost + risk_drag + bad_override
    alloc = clamp(round(0.3 * current_hold + 0.7 * alloc_adjusted, 1), 0, 30)

    risk_score_base = (
        (15 if (revg or 0) > 0.4 else 0)
        + (10 if (netm or 0) > 0.5 else 0)
        + (10 if (roic or 0) > 0.6 else 0)
        + (10 if (accrual or 0) > 0.05 else 0)
        + (15 if sector == "Technology" and (roe or 0) > 0.8 else 0)
        + (15 if (epsg or 0) > 0.5 else 0)
        + valuation_premium
    )
    risk_score_re = (
        (20 if (pffo or 0) > 25 else 0)
        + (25 if has_value(occupancy) and occupancy < 0.85 else 0)
        + (20 if (debt_eq or 0) > 3 else 0)
        + (15 if (payout_ffo or 0) > 0.95 else 0)
        + (15 if (ffo_yield or 0) < 0.04 else 0)
        + valuation_premium
    )
    risk_score_bank = (
        (15 if (bank_ptbv or 0) > 3 else 0)
        + (15 if (pb or 0) > 2.5 else 0)
        + (20 if has_value(bank_equity_assets) and bank_equity_assets < 0.05 else 0)
        + (20 if has_value(bank_tangible_equity_assets) and bank_tangible_equity_assets < 0.035 else 0)
        + (20 if has_value(bank_capital_rwa) and bank_capital_rwa < 0.09 else 0)
        + (18 if has_value(bank_tier1) and bank_tier1 < 0.085 else 0)
        + (18 if has_value(bank_tier1_leverage) and bank_tier1_leverage < 0.04 else 0)
        + (15 if has_value(bank_loans_deposits) and bank_loans_deposits > 1.05 else 0)
        + (10 if has_value(bank_allowance_loans) and bank_allowance_loans < 0.004 else 0)
        + (12 if has_value(bank_efficiency) and bank_efficiency > 0.75 else 0)
        + (10 if has_value(bank_deposit_growth) and bank_deposit_growth < -0.08 else 0)
        + (8 if has_value(bank_loan_growth) and bank_loan_growth < -0.08 else 0)
        + valuation_premium
    )
    risk_score_special = (
        (20 if "BAD" in risk_text else 0)
        + (12 if "RISKY" in risk_text else 0)
        + (max(0, 70 - (special_score_value or 70)) * 0.8)
        + valuation_premium
    )
    if sector == "Real Estate":
        risk_score = risk_score_re
    elif is_special_model:
        risk_score = risk_score_special
    elif is_bank_model:
        risk_score = risk_score_bank
    else:
        risk_score = risk_score_base

    if risk_score < 20:
        max_alloc_risk = 30
    elif risk_score < 40:
        max_alloc_risk = 20
    elif risk_score < 60:
        max_alloc_risk = 14
    elif risk_score < 80:
        max_alloc_risk = 10
    else:
        max_alloc_risk = 6

    alloc_final = min(alloc, max_alloc_risk)
    if macro_score < 20:
        macro_multiplier = 0.55
    elif macro_score < 40:
        macro_multiplier = 0.75
    elif macro_score < 60:
        macro_multiplier = 1.0
    elif macro_score < 80:
        macro_multiplier = 1.1
    else:
        macro_multiplier = 1.2

    sector_sensitivity = weight(
        {
            "Technology": 0.9,
            "Consumer Discretionary": 1.0,
            "Financials": 1.1,
            "Industrials": 0.85,
            "Energy": 0.95,
            "Materials": 0.85,
            "Real Estate": 1.3,
            "Consumer Staples": 0.7,
            "Healthcare": 0.7,
            "Utilities": 0.6,
        },
        1.0,
    )
    recession_intensity = 1 - max(0.0, 40 - macro_score) / 100
    sector_recession_adjust = 1 + (macro_multiplier - 1) * sector_sensitivity * recession_intensity
    alloc_macro_final = alloc_final * sector_recession_adjust

    if final_score < 50:
        label = "Sell"
    elif final_score < 60:
        label = "Reduce"
    elif final_score < 65:
        label = "Hold"
    elif final_score < 80:
        label = "Buy"
    else:
        label = "Strong Buy"

    return {
        "label": label,
        "final_score": round(final_score, 1),
        "allocation": round(alloc_macro_final, 1),
        "risk_score": round(risk_score, 1),
        "valuation_stretch": round(valuation_stretch, 2),
        "re_score": re_score,
        "bank_score": bank_score_value,
        "special_model": company_type if is_special_model else None,
        "special_score": special_score_value,
        "capped_by_risk": alloc_macro_final < alloc_final,
    }
