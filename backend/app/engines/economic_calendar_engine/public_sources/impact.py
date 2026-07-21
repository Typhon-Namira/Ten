"""Deterministic event-type canonicalization and impact classification — plain lookup tables, no
LLM or fuzzy inference. Every canonical event type and impact tier is auditable by reading this
file; adjusting classification means editing a set, not retraining anything.
"""

from __future__ import annotations

import re

#: Raw title substring (already lowercased, punctuation-stripped) -> canonical event type. Longer,
#: more specific keys are checked first so "core cpi" doesn't get eaten by a bare "cpi" match.
_TITLE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("core personal consumption expenditures", "core_pce"),
    ("core pce", "core_pce"),
    ("personal consumption expenditures", "pce"),
    ("personal income and outlays", "pce"),
    ("core consumer price index", "core_cpi"),
    ("consumer price index", "cpi"),
    ("producer price index", "ppi"),
    ("employment situation", "nonfarm_payrolls"),
    ("nonfarm payroll", "nonfarm_payrolls"),
    ("unemployment rate", "unemployment_rate"),
    ("average hourly earnings", "average_hourly_earnings"),
    ("job openings and labor turnover", "jolts"),
    ("jolts", "jolts"),
    ("import and export price", "import_export_prices"),
    ("employment cost index", "employment_cost_index"),
    ("gross domestic product", "gdp"),
    ("international trade", "international_trade"),
    ("corporate profits", "corporate_profits"),
    ("retail sales", "retail_sales"),
    ("new residential sales", "new_home_sales"),
    ("new home sales", "new_home_sales"),
    ("housing starts", "housing_starts"),
    ("building permits", "building_permits"),
    ("durable goods", "durable_goods"),
    ("factory orders", "factory_orders"),
    ("initial claims", "initial_jobless_claims"),
    ("continued claims", "continuing_jobless_claims"),
    ("continuing claims", "continuing_jobless_claims"),
    ("fomc statement", "fomc_statement"),
    ("federal open market committee", "fomc_rate_decision"),
    ("fomc", "fomc_rate_decision"),
    ("beige book", "beige_book"),
    ("minutes of the federal open market committee", "fomc_minutes"),
    ("governing council", "ecb_rate_decision"),
    ("monetary policy statement", "ecb_rate_decision"),
    ("monetary policy decision", "ecb_rate_decision"),
)

#: Deterministic, auditable impact tiers — exactly the events called out in the brief, expressed
#: as plain sets rather than any inferred/learned classification.
CRITICAL_IMPACT = frozenset({"fomc_rate_decision", "fomc_statement", "ecb_rate_decision"})
HIGH_IMPACT = frozenset(
    {
        "cpi",
        "core_cpi",
        "pce",
        "core_pce",
        "nonfarm_payrolls",
        "unemployment_rate",
        "gdp",
        "ppi",
        "retail_sales",
        "initial_jobless_claims",
        "jolts",
        "fomc_minutes",
        "fed_chair_speech",
    }
)
MEDIUM_IMPACT = frozenset(
    {
        "average_hourly_earnings",
        "import_export_prices",
        "employment_cost_index",
        "international_trade",
        "corporate_profits",
        "new_home_sales",
        "housing_starts",
        "building_permits",
        "durable_goods",
        "factory_orders",
        "continuing_jobless_claims",
        "beige_book",
    }
)

_normalize_pattern = re.compile(r"[^a-z0-9 ]+")


def canonicalize_title(raw_title: str) -> str:
    """Best-effort mapping from a source's raw release title to TEN's canonical event type. Falls
    back to a slugified version of the raw title when no keyword matches — never raises, and never
    silently drops an event just because it's unrecognized."""
    normalized = _normalize_pattern.sub(" ", raw_title.lower()).strip()
    normalized = " ".join(normalized.split())
    for keyword, canonical in _TITLE_KEYWORDS:
        if keyword in normalized:
            return canonical
    slug = normalized.replace(" ", "_")
    return slug or "unknown_event"


def classify_impact(canonical_event_type: str) -> str:
    if canonical_event_type in CRITICAL_IMPACT:
        return "critical"
    if canonical_event_type in HIGH_IMPACT:
        return "high"
    if canonical_event_type in MEDIUM_IMPACT:
        return "medium"
    return "low"


#: Event types worth ingesting at all — everything else from a source's schedule is still parsed
#: (for provenance/debugging) but excluded from the relevance-filtered set the decision engine
#: and dashboard actually consume, per the brief's "does not need every global economic event."
XAUUSD_RELEVANT_EVENT_TYPES = CRITICAL_IMPACT | HIGH_IMPACT | MEDIUM_IMPACT | {"fed_speech"}
