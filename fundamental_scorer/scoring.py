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


def sector_switch(sector: str, mapping: dict[str, float], default: float) -> float:
    return mapping.get(sector, default)


def build_risk_rating(metrics: dict[str, float | None], sector: str) -> str:
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
    if has_value(debt_eq) and debt_eq > sector_switch(
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
    ):
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

    fatal = any(
        [
            has_value(debt_eq)
            and debt_eq
            > sector_switch(
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
            sector != "Real Estate" and has_value(altman) and altman < 1.1 and has_value(current) and current < 1.0,
            sector != "Real Estate" and has_value(altman) and altman < 1.1 and has_value(debt_eq) and debt_eq > 1.5,
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
    if quality_override:
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
    fundamental_quality = fundamental_quality_re if sector == "Real Estate" else fundamental_quality_base

    risk_text = (risk_rating or "").upper()
    excellent_bonus = 3 if "EXCELLENT" in risk_text else 0
    risky_penalty = -10 if "RISKY" in risk_text else 0
    bad_penalty = -999_999_999 if "BAD" in risk_text else 0

    base_score = re_score if sector == "Real Estate" and has_value(re_score) else (composite if has_value(composite) else 50)
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
    else:
        if (epsg or 0) > 0.3 and (revg or 0) > 0.15 and final_score > 65:
            growth_boost = 1.5

    quality_boost = 0.0
    if sector == "Real Estate":
        if (ffo_yield or 0) > 0.075 and (occupancy or 0) > 0.94 and (debt_eq or 9e9) < 2:
            quality_boost = 1.2
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
    risk_score = risk_score_re if sector == "Real Estate" else risk_score_base

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
        "capped_by_risk": alloc_macro_final < alloc_final,
    }
