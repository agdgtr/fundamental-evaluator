from __future__ import annotations

import datetime as dt
import html as html_lib
import math
import os
import re
import time
from functools import lru_cache
from typing import Any, Iterable

import requests
import yfinance as yf

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A", "10-QT", "10-QT/A"}
TAXONOMY_ORDER = ("us-gaap", "dei", "ifrs-full", "srt")


def has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def safe_float(value: Any) -> float | None:
    if value in (None, "", "N/A", "-", "Metric not found"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def to_millions(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    return number / 1_000_000.0


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def duration_days(entry: dict[str, Any]) -> int | None:
    start = parse_date(entry.get("start"))
    end = parse_date(entry.get("end"))
    if not start or not end:
        return None
    return (end - start).days


def entry_anchor_date(entry: dict[str, Any], *, instant: bool = False) -> dt.date | None:
    if instant:
        return parse_date(entry.get("instant") or entry.get("end"))
    return parse_date(entry.get("end") or entry.get("instant"))


def dedupe_by_period(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
    for entry in sorted(
        entries,
        key=lambda item: (
            item.get("instant") or "",
            item.get("start") or "",
            item.get("end") or "",
            item.get("filed") or "",
            item.get("fy") or 0,
        ),
        reverse=True,
    ):
        key = (entry.get("instant"), entry.get("start"), entry.get("end"))
        chosen.setdefault(key, entry)
    return list(chosen.values())


class SecEdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        user_agent = user_agent or os.getenv(
            "SEC_USER_AGENT",
            "FundamentalScorer/0.1 (set SEC_USER_AGENT to Name email@example.com)",
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def for_identity(self, name: str | None, email: str | None) -> "SecEdgarClient":
        clean_name = (name or "").strip()
        clean_email = (email or "").strip()
        if not clean_name or not clean_email:
            return self
        return SecEdgarClient(user_agent=f"FundamentalScorer/0.1 ({clean_name} {clean_email})")

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    @lru_cache(maxsize=1)
    def company_tickers(self) -> dict[str, Any]:
        return self._get_json("https://www.sec.gov/files/company_tickers.json")

    def lookup_cik(self, symbol: str) -> int:
        symbol = symbol.upper()
        for item in self.company_tickers().values():
            if item.get("ticker", "").upper() == symbol:
                return int(item["cik_str"])
        raise ValueError(f"Ticker {symbol} was not found in SEC company_tickers.json")

    @lru_cache(maxsize=128)
    def fetch_submissions(self, cik: int) -> dict[str, Any]:
        return self._get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")

    @lru_cache(maxsize=128)
    def fetch_companyfacts(self, cik: int) -> dict[str, Any]:
        return self._get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")

    @lru_cache(maxsize=128)
    def recent_filing_url(
        self,
        cik: int,
        forms: tuple[str, ...] = ("10-Q", "10-K", "20-F", "40-F"),
    ) -> str | None:
        submissions = self.fetch_submissions(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        form_list = recent.get("form", [])
        accession_list = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        for form, accession, primary_doc in zip(form_list, accession_list, primary_docs):
            if form not in forms:
                continue
            accession_compact = accession.replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_compact}/{primary_doc}"
        return None

    @lru_cache(maxsize=256)
    def fetch_text(self, url: str) -> str:
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _parse_inline_xbrl_contexts(text: str) -> dict[str, dict[str, Any]]:
        contexts: dict[str, dict[str, Any]] = {}
        pattern = re.compile(
            r"<(?:xbrli:)?context\b[^>]*\bid=['\"]([^'\"]+)['\"][^>]*>(.*?)</(?:xbrli:)?context>",
            re.I | re.S,
        )
        for match in pattern.finditer(text):
            block = match.group(2)
            start = re.search(r"<(?:xbrli:)?startDate[^>]*>([^<]+)</(?:xbrli:)?startDate>", block, re.I)
            end = re.search(r"<(?:xbrli:)?endDate[^>]*>([^<]+)</(?:xbrli:)?endDate>", block, re.I)
            instant = re.search(r"<(?:xbrli:)?instant[^>]*>([^<]+)</(?:xbrli:)?instant>", block, re.I)
            contexts[match.group(1)] = {
                "start": start.group(1).strip() if start else None,
                "end": end.group(1).strip() if end else None,
                "instant": instant.group(1).strip() if instant else None,
                "has_dimensions": bool(re.search(r"<(?:xbrli:)?segment\b|<(?:xbrldi:)?explicitMember\b", block, re.I)),
            }
        return contexts

    @staticmethod
    def _clean_inline_numeric_value(text: str, scale: int = 0, sign: str | None = None) -> float | None:
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", text)).replace("\xa0", " ").strip()
        if not plain:
            return None
        is_negative = "(" in plain and ")" in plain
        numeric_text = re.sub(r"[^0-9.\-]", "", plain)
        if numeric_text in {"", "-", ".", "-."}:
            return None
        value = safe_float(numeric_text)
        if value is None:
            return None
        if is_negative and value > 0:
            value = -value
        if sign == "-":
            value = -abs(value)
        return value * (10 ** scale)

    @classmethod
    def _parse_inline_xbrl_facts(cls, text: str) -> list[dict[str, Any]]:
        contexts = cls._parse_inline_xbrl_contexts(text)
        facts: list[dict[str, Any]] = []
        fact_pattern = re.compile(
            r"<ix:(?:nonFraction|fraction)\b([^>]*)>(.*?)</ix:(?:nonFraction|fraction)>",
            re.I | re.S,
        )
        attr_pattern = re.compile(r"([A-Za-z_:][\w:.\-]*)=['\"]([^'\"]*)['\"]")
        for match in fact_pattern.finditer(text):
            attrs = {key: value for key, value in attr_pattern.findall(match.group(1))}
            concept = attrs.get("name")
            context_ref = attrs.get("contextRef") or attrs.get("contextref")
            if not concept or not context_ref:
                continue
            scale = int(attrs.get("scale", "0") or 0)
            value = cls._clean_inline_numeric_value(match.group(2), scale=scale, sign=attrs.get("sign"))
            if value is None:
                continue
            context = contexts.get(context_ref, {})
            facts.append(
                {
                    "concept": concept,
                    "context_ref": context_ref,
                    "unit": (attrs.get("unitRef") or attrs.get("unitref") or "").lower(),
                    "value": value,
                    "start": context.get("start"),
                    "end": context.get("end"),
                    "instant": context.get("instant"),
                    "has_dimensions": context.get("has_dimensions", False),
                }
            )
        return facts

    @lru_cache(maxsize=128)
    def recent_filing_facts(self, cik: int) -> list[dict[str, Any]]:
        url = self.recent_filing_url(cik)
        if not url:
            return []
        return self._parse_inline_xbrl_facts(self.fetch_text(url))

    def _recent_filing_plain_text(self, cik: int) -> str:
        url = self.recent_filing_url(cik)
        if not url:
            return ""
        text = self.fetch_text(url)
        text = html_lib.unescape(text)
        return " ".join(re.sub(r"<[^>]+>", " ", text).replace("\xa0", " ").split())

    @staticmethod
    def _parse_filing_number(value: str | None) -> float | None:
        if value is None:
            return None
        return safe_float(value.replace(",", ""))

    @classmethod
    def _extract_percent_after_label(cls, text: str, labels: Iterable[str]) -> float | None:
        blocked_context = re.compile(r"\b(requirement|minimum|stated minimum|definition|primary regulatory)\b", re.I)
        for label in labels:
            pattern = rf"{label}(?:\s*\([^)]*\))*([^0-9%]{{0,90}})([0-9]{{1,3}}(?:\.[0-9]+)?)\s*(?:%|percent)"
            matches = list(re.finditer(pattern, text, re.I))
            for match in matches:
                if blocked_context.search(match.group(1)):
                    continue
                value = cls._parse_filing_number(match.group(2))
                if value is not None:
                    return value / 100
            table_pattern = rf"{label}(?:\s*\([^)]*\))*([^0-9%]{{0,90}})([0-9]{{1,3}}(?:\.[0-9]+)?)"
            table_matches = list(re.finditer(table_pattern, text, re.I))
            for table_match in table_matches:
                if blocked_context.search(table_match.group(1)):
                    continue
                value = cls._parse_filing_number(table_match.group(2))
                if value is not None and 0 <= value <= 100:
                    return value / 100
        return None

    @classmethod
    def _extract_amount_after_label(cls, text: str, labels: Iterable[str]) -> float | None:
        for label in labels:
            pattern = rf"{label}\s*(?:\([^)]*\))?[^0-9$]{{0,40}}(?:\$)?\s*([0-9]{{1,3}}(?:,[0-9]{{3}})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
            for match in re.finditer(pattern, text, re.I):
                value = cls._parse_filing_number(match.group(1))
                if value is not None and value >= 100:
                    return value
        return None

    def extract_recent_bank_capital_metrics(self, cik: int) -> dict[str, float]:
        flat = self._recent_filing_plain_text(cik)
        if not flat:
            return {}
        search_texts: list[str] = []
        for marker in [
            r"Regulatory\s+Capital\s+under\s+Basel",
            r"Risk-based\s+capital\s+metrics",
            r"Capital\s+ratios\s+at\s+period\s+end",
        ]:
            match = re.search(marker, flat, re.I)
            if match:
                search_texts.append(flat[match.start() : match.start() + 6000])
        search_texts.append(flat)

        def extract_percent(labels: Iterable[str]) -> float | None:
            for search_text in search_texts:
                value = self._extract_percent_after_label(search_text, labels)
                if value is not None:
                    return value
            return None

        def extract_amount(labels: Iterable[str]) -> float | None:
            for search_text in search_texts:
                value = self._extract_amount_after_label(search_text, labels)
                if value is not None:
                    return value
            return None

        def extract_row_ratio(text: str, label: str) -> float | None:
            pattern = rf"{label}(?:\s*\([^)]*\))*\s+([0-9]{{1,3}}(?:\.[0-9]+)?)\s*(?:%|\s|$)"
            for match in re.finditer(pattern, text, re.I):
                value = self._parse_filing_number(match.group(1))
                if value is not None and 0 <= value <= 100:
                    return value / 100
            return None

        metrics: dict[str, float] = {}
        cet1_ratio = extract_percent(
            [
                r"Common\s+equity\s+tier\s+1\s+\(CET1\)\s+capital\s+ratio\s+was",
                r"Common\s+equity\s+tier\s+1\s+\(CET1\)\s+capital\s+ratio",
                r"Common\s+equity\s+tier\s+1\s+\(CET1\)",
                r"Common\s+equity\s+tier\s+1\s+capital\s+ratio",
                r"CET1\s+capital\s+ratio",
                r"CET1\s+ratio\s+was",
            ],
        )
        tier1_ratio = extract_percent([r"(?<!equity\s)Tier\s*1\s+capital\s+ratio", r"(?<!equity\s)Tier\s*1\s+capital(?!\s+ratio)"])
        total_capital_ratio = extract_percent([r"Total\s+capital\s+ratio"])
        tier1_leverage_ratio = extract_percent([r"Tier\s*1\s+leverage\s+ratio", r"(?<!Supplementary\s)Leverage\s+ratio"])
        supplementary_leverage_ratio = extract_percent([r"Supplementary\s+leverage\s+ratio"])
        if cet1_ratio is not None:
            metrics["cet1_capital_ratio"] = cet1_ratio
        if tier1_ratio is not None:
            metrics["tier1_capital_ratio"] = tier1_ratio
        if total_capital_ratio is not None:
            metrics["total_capital_ratio"] = total_capital_ratio
        if tier1_leverage_ratio is not None:
            metrics["tier1_leverage_ratio"] = tier1_leverage_ratio
        if supplementary_leverage_ratio is not None:
            metrics["supplementary_leverage_ratio"] = supplementary_leverage_ratio

        std_match = re.search(r"Standardized\s+Approach:?(.*?)(?:Advanced\s+Approach|Leverage-based\s+metrics|Supplementary\s+Leverage|Liquidity\s+Coverage|$)", flat, re.I)
        if std_match:
            std_section = std_match.group(1)
            row_cet1 = extract_row_ratio(std_section, r"Common\s+Equity\s+Tier\s+1\s+\(CET1\)|Common\s+equity\s+tier\s+1\s+capital\s+ratio")
            row_tier1 = extract_row_ratio(std_section, r"(?<!equity\s)Tier\s*1\s+capital(?!\s+ratio)")
            row_total = extract_row_ratio(std_section, r"Total\s+capital(?!\s+ratio)")
            if row_cet1 is not None:
                metrics["cet1_capital_ratio"] = row_cet1
            if row_tier1 is not None:
                metrics["tier1_capital_ratio"] = row_tier1
            if row_total is not None:
                metrics["total_capital_ratio"] = row_total

        required_table_match = re.search(
            r"Required\s+Ratios\s+Standardized\s+Approach\s+Required\s+Ratios\s+Advanced\s+Approaches(.*?)(?:December\s+31,\s*20\d{2}|For\s+all\s+periods\s+presented)",
            flat,
            re.I,
        )
        if required_table_match:
            required_table = required_table_match.group(1)

            def extract_actual_ratio(label: str) -> float | None:
                match = re.search(rf"{label}(?:\s*\([^)]*\))*", required_table, re.I)
                if not match:
                    return None
                values = [
                    self._parse_filing_number(value)
                    for value in re.findall(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%?", required_table[match.end() : match.end() + 100])
                ]
                values = [value for value in values if value is not None]
                if len(values) >= 2:
                    return values[1] / 100
                return None

            required_cet1 = extract_actual_ratio(r"CET1\s+Capital\s+ratio")
            required_tier1 = extract_actual_ratio(r"Tier\s*1\s+Capital\s+ratio")
            required_leverage = extract_actual_ratio(r"(?<!Supplementary\s)Leverage\s+ratio")
            required_supplementary = extract_actual_ratio(r"Supplementary\s+Leverage\s+ratio")
            if required_cet1 is not None:
                metrics["cet1_capital_ratio"] = required_cet1
            if required_tier1 is not None:
                metrics["tier1_capital_ratio"] = required_tier1
            if required_leverage is not None:
                metrics["tier1_leverage_ratio"] = required_leverage
            if required_supplementary is not None:
                metrics["supplementary_leverage_ratio"] = required_supplementary

        risk_weighted_assets = extract_amount(
            [
                r"Risk-weighted\s+assets\s+\(RWAs\)\s+\(in\s+billions\)",
                r"Risk-weighted\s+assets\s+\(in\s+billions\)",
            ],
        )
        if risk_weighted_assets is not None:
            metrics["risk_weighted_assets"] = risk_weighted_assets * 1000
        risk_weighted_assets_millions = extract_amount([r"Total\s+Risk-Weighted\s+Assets", r"Risk-weighted\s+assets"])
        if risk_weighted_assets_millions is not None:
            metrics.setdefault("risk_weighted_assets", risk_weighted_assets_millions)
        tier1_capital = extract_amount([r"(?<!equity\s)Tier\s*1\s+capital(?!\s+ratio)"])
        if tier1_capital is not None:
            metrics["tier1_capital"] = tier1_capital
        cet1_capital = extract_amount([r"Common\s+equity\s+tier\s+1\s+capital(?!\s+ratio)"])
        if cet1_capital is not None:
            metrics["cet1_capital"] = cet1_capital
        return metrics

    def recent_filing_value(
        self,
        cik: int,
        concepts: Iterable[str],
        *,
        kind: str,
        unit_preferences: Iterable[str] | None = None,
    ) -> float | None:
        wanted = {concept.lower() for concept in concepts}
        facts = [fact for fact in self.recent_filing_facts(cik) if fact.get("concept", "").lower() in wanted]
        if unit_preferences:
            wanted_units = {unit.lower() for unit in unit_preferences}
            preferred = [fact for fact in facts if fact.get("unit") in wanted_units]
            if preferred:
                facts = preferred

        def matches_kind(fact: dict[str, Any]) -> bool:
            if kind == "instant":
                return bool(fact.get("instant"))
            duration = duration_days({"start": fact.get("start"), "end": fact.get("end")})
            if duration is None:
                return False
            if kind == "quarterly":
                return 70 <= duration <= 120
            if kind == "annual":
                return 300 <= duration <= 380
            return False

        filtered = [fact for fact in facts if matches_kind(fact)]
        filtered.sort(
            key=lambda fact: (
                1 if not fact.get("has_dimensions") else 0,
                fact.get("instant") or fact.get("end") or "",
                fact.get("start") or "",
            ),
            reverse=True,
        )
        return filtered[0]["value"] if filtered else None

    def extract_recent_occupancy_rate(self, cik: int) -> float | None:
        url = self.recent_filing_url(cik)
        if not url:
            return None
        text = self.fetch_text(url)
        flat = " ".join(re.sub(r"<[^>]+>", " ", text).split())
        patterns = [
            r"occupanc(?:y|ied|ancy rates?)[^.]{0,160}?([0-9]{2,3}(?:\.[0-9]+)?)%",
            r"([0-9]{2,3}(?:\.[0-9]+)?)%[^.]{0,160}?occupanc",
            r"([0-9]{2,3}(?:\.[0-9]+)?)%[^.]{0,160}?leased",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, flat, re.I))
            for match in matches:
                value = safe_float(match.group(1))
                if value is None:
                    continue
                if 0 <= value <= 100:
                    return value / 100.0
        return None

    def extract_named_executive(
        self,
        cik: int,
        *,
        forms: tuple[str, ...] = ("DEF 14A", "10-K", "20-F", "40-F", "10-Q"),
        title_patterns: tuple[str, ...] = ("Chief Executive Officer", "President and Chief Executive Officer"),
    ) -> str | None:
        name_pattern = r"([A-Z][A-Za-z.&'-]+(?:\s+[A-Z][A-Za-z.&'-]+){1,4})"

        def clean_name(raw: str) -> str:
            cleaned = re.sub(r"\s+(?:President|Chief|Executive|Officer|Director|Interim)\b.*$", "", raw).strip(" ,")
            return cleaned

        def looks_like_person_name(raw: str) -> bool:
            tokens = [re.sub(r"[^A-Za-z]", "", part) for part in raw.split()]
            tokens = [token for token in tokens if token]
            if len(tokens) < 2 or len(tokens) > 5:
                return False
            banned = {
                "Inc",
                "Corp",
                "Corporation",
                "Company",
                "Bank",
                "Gas",
                "Electric",
                "Private",
                "Chair",
                "Group",
                "Holdings",
                "Management",
                "Capital",
                "Technologies",
                "Partners",
                "Trust",
                "Financial",
            }
            return not any(token in banned for token in tokens)

        for form in forms:
            url = self.recent_filing_url(cik, forms=(form,))
            if not url:
                continue
            text = self.fetch_text(url)
            flat = " ".join(re.sub(r"<[^>]+>", " ", text).split())
            for title in title_patterns:
                escaped_title = re.escape(title)
                signature_patterns = [
                    rf"/s/\s*{name_pattern}[^.]{{0,180}}{escaped_title}",
                    rf"{name_pattern},\s+President,\s+{escaped_title}",
                    rf"{name_pattern},\s+{escaped_title}(?:\s+and\s+Director)?",
                ]
                for pattern in signature_patterns:
                    matches = list(re.finditer(pattern, flat))
                    if matches:
                        candidate = clean_name(matches[-1].group(1))
                        if looks_like_person_name(candidate):
                            return candidate
                start = 0
                while True:
                    title_index = flat.find(title, start)
                    if title_index == -1:
                        break
                    window = flat[max(0, title_index - 160) : title_index + 160]
                    prefix = window[: max(0, window.find(title))]
                    signature_matches = re.findall(rf"/s/\s*{name_pattern}", prefix)
                    if signature_matches:
                        candidate = clean_name(signature_matches[-1])
                        if looks_like_person_name(candidate):
                            return candidate
                    patterns = [
                        rf"{name_pattern},\s+{escaped_title}",
                        rf"{name_pattern},\s+President,\s+{escaped_title}",
                        rf"{name_pattern}\s*,?\s*President,\s+{escaped_title}",
                        rf"{escaped_title}\s*,?\s+{name_pattern}",
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, window)
                        if match:
                            candidate = clean_name(match.group(1))
                            if looks_like_person_name(candidate):
                                return candidate
                    start = title_index + len(title)
        return None


class YahooFinanceClient:
    @lru_cache(maxsize=128)
    def fetch_snapshot(self, symbol: str) -> dict[str, Any]:
        ticker = yf.Ticker(symbol)
        try:
            info = ticker.get_info() or {}
        except Exception:
            info = {}
        try:
            history = ticker.history(period="max", auto_adjust=False)
        except Exception:
            history = None
        try:
            fast_info = dict(ticker.fast_info or {})
        except Exception:
            fast_info = {}
        return {"info": info, "history": history, "fast_info": fast_info}


class YahooWebClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    @lru_cache(maxsize=128)
    def fetch_quote_page(self, symbol: str) -> str | None:
        try:
            response = self.session.get(f"https://finance.yahoo.com/quote/{symbol}", timeout=20)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    @staticmethod
    def _extract_raw(html: str, key: str) -> float | None:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*\{{\s*"raw"\s*:\s*("?-?[0-9][0-9Ee+\-.,]*"?)',
            rf'"{re.escape(key)}"\s*:\s*("?-?[0-9][0-9Ee+\-.,]*"?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if not match:
                continue
            raw = match.group(1).strip('"').replace(",", "")
            value = safe_float(raw)
            if value is not None:
                return value
        return None

    @staticmethod
    def _extract_string(html: str, key: str) -> str | None:
        match = re.search(rf'"{re.escape(key)}":"([^"]+)"', html)
        if not match:
            return None
        return html_lib.unescape(match.group(1))

    def fetch_metrics(self, symbol: str) -> dict[str, Any]:
        html = self.fetch_quote_page(symbol)
        if not html:
            return {}
        fields = [
            "forwardPE",
            "trailingPE",
            "pegRatio",
            "marketCap",
            "sharesOutstanding",
            "enterpriseValue",
            "enterpriseToRevenue",
            "enterpriseToEbitda",
            "priceToBook",
            "priceToSalesTrailing12Months",
            "bookValue",
            "currentRatio",
            "quickRatio",
            "freeCashflow",
            "operatingCashflow",
            "totalRevenue",
            "totalDebt",
            "totalCash",
            "ebitda",
            "grossProfits",
            "netIncomeToCommon",
            "returnOnEquity",
            "returnOnAssets",
            "debtToEquity",
            "grossMargins",
            "operatingMargins",
            "profitMargins",
            "revenueGrowth",
            "earningsGrowth",
            "earningsQuarterlyGrowth",
            "forwardEps",
            "trailingEps",
            "trailingAnnualDividendRate",
            "dividendRate",
            "currency",
            "financialCurrency",
            "regularMarketPrice",
            "regularMarketPreviousClose",
        ]
        metrics: dict[str, Any] = {}
        for field in fields:
            raw = self._extract_raw(html, field)
            if raw is not None:
                metrics[field] = raw
        for field in ["sector", "industry", "exchange", "longName", "shortName", "currency", "financialCurrency"]:
            text = self._extract_string(html, field)
            if text:
                metrics[field] = text
        return metrics


class APIClientBase:
    def __init__(self, *, base_url: str, api_key: str | None = None, api_key_query: str | None = None, api_key_header: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_query = api_key_query
        self.api_key_header = api_key_header
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": os.getenv(
                    "SEC_USER_AGENT",
                    "FundamentalScorer/0.1 (set SEC_USER_AGENT to Name email@example.com)",
                ),
                "Accept": "application/json",
            }
        )
        if self.api_key and self.api_key_header:
            self.session.headers[self.api_key_header] = self.api_key

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any] | None:
        if not self.available:
            return None
        merged = dict(params or {})
        if self.api_key and self.api_key_query:
            merged[self.api_key_query] = self.api_key
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.session.get(url, params=merged, timeout=20)
                if response.status_code in {429, 503}:
                    retry_after = safe_float(response.headers.get("Retry-After"))
                    if attempt == 0:
                        time.sleep(min(max(retry_after or 0.5, 0.2), 2.0))
                        continue
                    return None
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.25)
                    continue
                return None
        if last_error:
            return None
        return None


class MassiveClient(APIClientBase):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://api.massive.com",
            api_key=os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY"),
            api_key_query="apiKey",
        )

    @lru_cache(maxsize=128)
    def fetch_single_ticker_snapshot(self, symbol: str) -> dict[str, Any] | None:
        payload = self._get_json(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        return payload if isinstance(payload, dict) else None

    @lru_cache(maxsize=128)
    def fetch_ticker_overview(self, symbol: str) -> dict[str, Any] | None:
        payload = self._get_json(f"/v3/reference/tickers/{symbol}")
        if not isinstance(payload, dict):
            return None
        results = payload.get("results")
        return results if isinstance(results, dict) else payload

    @lru_cache(maxsize=128)
    def fetch_ratios(self, symbol: str) -> dict[str, Any] | None:
        payload = self._get_json("/stocks/financials/v1/ratios", params={"ticker": symbol, "limit": 1})
        if not isinstance(payload, dict):
            return None
        results = payload.get("results")
        if isinstance(results, list) and results:
            return results[0]
        if isinstance(results, dict):
            return results
        return None


class FMPClient(APIClientBase):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://financialmodelingprep.com",
            api_key=os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY"),
            api_key_query="apikey",
            api_key_header="apikey",
        )

    @staticmethod
    def _first_item(payload: dict[str, Any] | list[Any] | None) -> dict[str, Any] | None:
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
        if isinstance(payload, dict):
            return payload
        return None

    @lru_cache(maxsize=128)
    def fetch_quote(self, symbol: str) -> dict[str, Any] | None:
        return self._first_item(self._get_json("/stable/quote", params={"symbol": symbol}))

    @lru_cache(maxsize=128)
    def fetch_key_metrics_ttm(self, symbol: str) -> dict[str, Any] | None:
        return self._first_item(self._get_json("/stable/key-metrics-ttm", params={"symbol": symbol}))

    @lru_cache(maxsize=128)
    def fetch_ratios_ttm(self, symbol: str) -> dict[str, Any] | None:
        return self._first_item(self._get_json("/stable/ratios-ttm", params={"symbol": symbol}))

    @lru_cache(maxsize=128)
    def fetch_analyst_estimates(self, symbol: str) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/stable/analyst-estimates",
            params={"symbol": symbol, "period": "annual", "page": 0, "limit": 10},
        )
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []


class AlphaVantageClient(APIClientBase):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://www.alphavantage.co",
            api_key=os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY"),
            api_key_query="apikey",
        )

    def _get_query(self, function: str, *, symbol: str) -> dict[str, Any] | None:
        payload = self._get_json("/query", params={"function": function, "symbol": symbol})
        if not isinstance(payload, dict):
            return None
        if payload.get("Note") or payload.get("Information") or payload.get("Error Message"):
            return None
        return payload

    @lru_cache(maxsize=128)
    def fetch_global_quote(self, symbol: str) -> dict[str, Any] | None:
        payload = self._get_query("GLOBAL_QUOTE", symbol=symbol)
        if not payload:
            return None
        quote = payload.get("Global Quote")
        return quote if isinstance(quote, dict) else None

    @lru_cache(maxsize=128)
    def fetch_company_overview(self, symbol: str) -> dict[str, Any] | None:
        return self._get_query("OVERVIEW", symbol=symbol)

    @lru_cache(maxsize=128)
    def fetch_earnings_estimates(self, symbol: str) -> dict[str, Any] | None:
        return self._get_query("EARNINGS_ESTIMATES", symbol=symbol)


class FactStore:
    def __init__(self, companyfacts: dict[str, Any]) -> None:
        self.facts = companyfacts.get("facts", {})

    def entries(self, concepts: Iterable[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for taxonomy in TAXONOMY_ORDER:
            taxonomy_bucket = self.facts.get(taxonomy, {})
            for concept in concepts:
                concept_bucket = taxonomy_bucket.get(concept)
                if not concept_bucket:
                    continue
                for unit, items in concept_bucket.get("units", {}).items():
                    for item in items:
                        rows.append(
                            {
                                **item,
                                "taxonomy": taxonomy,
                                "concept": concept,
                                "unit": unit,
                            }
                        )
        return rows

    @staticmethod
    def _prefer_units(
        entries: list[dict[str, Any]], unit_preferences: Iterable[str] | None
    ) -> list[dict[str, Any]]:
        if not entries or not unit_preferences:
            return entries
        for unit in unit_preferences:
            filtered = [entry for entry in entries if entry.get("unit") == unit]
            if filtered:
                return filtered
        return entries

    def _filtered(
        self,
        concepts: Iterable[str],
        *,
        kind: str,
        forms: set[str] | None,
        unit_preferences: Iterable[str] | None,
    ) -> list[dict[str, Any]]:
        entries = self.entries(concepts)
        entries = self._prefer_units(entries, unit_preferences)
        if forms:
            entries = [entry for entry in entries if entry.get("form") in forms]
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            duration = duration_days(entry)
            if kind == "annual":
                if duration is None or duration < 300 or duration > 380:
                    continue
            elif kind == "quarterly":
                if duration is None or duration < 70 or duration > 120:
                    continue
            elif kind == "instant":
                if duration is not None and duration > 10:
                    continue
            filtered.append(entry)
        return dedupe_by_period(filtered)

    @staticmethod
    def _apply_max_age(
        entries: list[dict[str, Any]],
        max_age_days: int | None,
        *,
        instant: bool = False,
    ) -> list[dict[str, Any]]:
        if max_age_days is None:
            return entries
        today = dt.date.today()
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            anchor = entry_anchor_date(entry, instant=instant)
            if not anchor:
                continue
            if (today - anchor).days <= max_age_days:
                filtered.append(entry)
        return filtered

    @staticmethod
    def _sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            entries,
            key=lambda item: (item.get("end") or item.get("instant") or "", item.get("filed") or "", item.get("frame") or ""),
            reverse=True,
        )

    def _quarterly_series_with_derived_q4(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        limit: int = 8,
        max_age_days: int | None = None,
    ) -> list[dict[str, Any]]:
        quarterly_entries = self.quarterly_series(
            concepts,
            unit_preferences=unit_preferences,
            limit=max(limit + 6, 12),
            max_age_days=max_age_days,
        )
        annual_entries = self.annual_series(
            concepts,
            unit_preferences=unit_preferences,
            limit=6,
            max_age_days=max_age_days,
        )
        derived_entries: list[dict[str, Any]] = []

        for annual_entry in annual_entries:
            annual_start = parse_date(annual_entry.get("start"))
            annual_end = parse_date(annual_entry.get("end"))
            annual_value = safe_float(annual_entry.get("val"))
            if not annual_start or not annual_end or annual_value is None:
                continue
            if any(parse_date(entry.get("end")) == annual_end for entry in quarterly_entries):
                continue

            quarters_in_period: dict[str, dict[str, Any]] = {}
            for entry in quarterly_entries:
                quarter_start = parse_date(entry.get("start"))
                quarter_end = parse_date(entry.get("end"))
                if not quarter_start or not quarter_end:
                    continue
                if annual_start <= quarter_start and quarter_end < annual_end:
                    quarters_in_period.setdefault(entry.get("end") or "", entry)

            if len(quarters_in_period) != 3:
                continue

            ordered_quarters = sorted(
                quarters_in_period.values(),
                key=lambda item: item.get("end") or "",
            )
            quarter_total = 0.0
            valid = True
            for entry in ordered_quarters:
                value = safe_float(entry.get("val"))
                if value is None:
                    valid = False
                    break
                quarter_total += value
            if not valid:
                continue

            last_quarter_end = parse_date(ordered_quarters[-1].get("end"))
            if not last_quarter_end:
                continue

            derived_entries.append(
                {
                    **annual_entry,
                    "start": (last_quarter_end + dt.timedelta(days=1)).isoformat(),
                    "end": annual_entry.get("end"),
                    "val": annual_value - quarter_total,
                    "form": "DERIVED-Q4",
                    "frame": f"{annual_entry.get('end', '')}-Q4-DERIVED",
                    "derived": True,
                }
            )

        combined = dedupe_by_period(quarterly_entries + derived_entries)
        return self._sort_entries(combined)[:limit]

    def annual_series(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        limit: int = 5,
        max_age_days: int | None = None,
    ) -> list[dict[str, Any]]:
        entries = self._filtered(
            concepts,
            kind="annual",
            forms=ANNUAL_FORMS,
            unit_preferences=unit_preferences,
        )
        entries = self._apply_max_age(entries, max_age_days)
        return self._sort_entries(entries)[:limit]

    def quarterly_series(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        limit: int = 8,
        max_age_days: int | None = None,
    ) -> list[dict[str, Any]]:
        entries = self._filtered(
            concepts,
            kind="quarterly",
            forms=QUARTERLY_FORMS,
            unit_preferences=unit_preferences,
        )
        entries = self._apply_max_age(entries, max_age_days)
        return self._sort_entries(entries)[:limit]

    def instant_series(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        limit: int = 5,
        max_age_days: int | None = None,
    ) -> list[dict[str, Any]]:
        entries = self._filtered(
            concepts,
            kind="instant",
            forms=ANNUAL_FORMS | QUARTERLY_FORMS,
            unit_preferences=unit_preferences,
        )
        entries = self._apply_max_age(entries, max_age_days, instant=True)
        return self._sort_entries(entries)[:limit]

    def latest_annual_value(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        max_age_days: int | None = None,
    ) -> float | None:
        series = self.annual_series(concepts, unit_preferences=unit_preferences, limit=1, max_age_days=max_age_days)
        if not series:
            return None
        return safe_float(series[0].get("val"))

    def latest_instant_value(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        max_age_days: int | None = None,
    ) -> float | None:
        series = self.instant_series(concepts, unit_preferences=unit_preferences, limit=1, max_age_days=max_age_days)
        if not series:
            return None
        return safe_float(series[0].get("val"))

    def annual_values(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        count: int = 2,
        max_age_days: int | None = None,
    ) -> list[float]:
        series = self.annual_series(concepts, unit_preferences=unit_preferences, limit=count, max_age_days=max_age_days)
        values: list[float] = []
        for item in series:
            value = safe_float(item.get("val"))
            if value is not None:
                values.append(value)
        return values

    def trailing_twelve_month_values(
        self,
        concepts: Iterable[str],
        *,
        unit_preferences: Iterable[str] | None = None,
        count: int = 2,
        max_age_days: int | None = None,
    ) -> list[float]:
        series = self._quarterly_series_with_derived_q4(
            concepts,
            unit_preferences=unit_preferences,
            limit=count + 4,
            max_age_days=max_age_days,
        )
        values: list[float] = []
        for item in series:
            value = safe_float(item.get("val"))
            if value is not None:
                values.append(value)

        windows: list[float] = []
        for start_index in range(count):
            window = values[start_index : start_index + 4]
            if len(window) == 4:
                windows.append(sum(window))
        return windows


def first_non_null(*values: Any) -> Any:
    for value in values:
        if has_value(value):
            return value
    return None


def first_item(values: list[Any], index: int = 0) -> Any:
    if len(values) > index:
        return values[index]
    return None




# NEXT TEST PART: removable Polygon fallback support
class UserProvidedMassiveClient(MassiveClient):
    def __init__(self, api_key: str) -> None:
        super().__init__()
        clean_key = (api_key or "").strip()
        self.api_key = clean_key or None
        if self.api_key and self.api_key_header:
            self.session.headers[self.api_key_header] = self.api_key


def build_polygon_massive_client(
    api_key: str | None,
    default_client: MassiveClient | None = None,
) -> MassiveClient | None:
    clean_key = (api_key or "").strip()
    if clean_key:
        return UserProvidedMassiveClient(clean_key)
    if default_client is not None and default_client.available:
        return default_client
    return None


def empty_polygon_fallback_bundle() -> dict[str, Any]:
    return {
        "snapshot": None,
        "overview": None,
        "ratios": None,
        "used": False,
    }


def fetch_polygon_fallback_bundle(
    client: MassiveClient | None,
    symbol: str,
) -> dict[str, Any]:
    if client is None:
        return empty_polygon_fallback_bundle()
    snapshot = client.fetch_single_ticker_snapshot(symbol)
    overview = client.fetch_ticker_overview(symbol)
    ratios = client.fetch_ratios(symbol)
    return {
        "snapshot": snapshot if isinstance(snapshot, dict) else None,
        "overview": overview if isinstance(overview, dict) else None,
        "ratios": ratios if isinstance(ratios, dict) else None,
        "used": any(isinstance(payload, dict) and payload for payload in [snapshot, overview, ratios]),
    }
