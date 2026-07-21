"""Keyless, public-web economic calendar sources — no API key, no paid provider, no authentication
of any kind. Every adapter here reads an official government/institutional webpage, RSS feed, ICS
calendar, or (for DOL) a deterministic published rule. See `base.py` for the shared architecture
and `../../../../../docs` for the full source policy this subsystem was built to satisfy.
"""

from collections.abc import Callable

from .base import HttpPublicCalendarSource, PublicCalendarSource, RawEconomicEvent, USER_AGENT
from .bea import BeaPublicCalendarSource
from .bls import BlsPublicCalendarSource
from .census import CensusPublicCalendarSource
from .circuit_breaker import CircuitBreakerRegistry, CircuitBreakerState, FailureCategory
from .dol import DolWeeklyClaimsSource, us_federal_holidays
from .ecb import EcbPublicCalendarSource
from .fed import FedPublicCalendarSource
from .impact import CRITICAL_IMPACT, HIGH_IMPACT, MEDIUM_IMPACT, XAUUSD_RELEVANT_EVENT_TYPES, canonicalize_title, classify_impact
from .robots import RobotsPolicy, evaluate_robots_policy
from .ssrf import ALLOWED_DOMAINS, UnsafePublicUrlError, assert_safe_public_url

#: name (matches `ProviderConfig.name` in `configs/economic_calendar.yaml`) -> constructor. One
#: entry per official source — `build_providers()` in `..providers` looks sources up here when a
#: config entry's mode is `public_web_source`.
PUBLIC_SOURCE_CLASSES: dict[str, Callable[..., HttpPublicCalendarSource]] = {
    "bls": BlsPublicCalendarSource,
    "bea": BeaPublicCalendarSource,
    "federal_reserve": FedPublicCalendarSource,
    "census_bureau": CensusPublicCalendarSource,
    "dol_weekly_claims": DolWeeklyClaimsSource,
    "ecb": EcbPublicCalendarSource,
}

__all__ = [
    "ALLOWED_DOMAINS",
    "CRITICAL_IMPACT",
    "HIGH_IMPACT",
    "MEDIUM_IMPACT",
    "PUBLIC_SOURCE_CLASSES",
    "USER_AGENT",
    "XAUUSD_RELEVANT_EVENT_TYPES",
    "BeaPublicCalendarSource",
    "BlsPublicCalendarSource",
    "CensusPublicCalendarSource",
    "CircuitBreakerRegistry",
    "CircuitBreakerState",
    "DolWeeklyClaimsSource",
    "EcbPublicCalendarSource",
    "FailureCategory",
    "FedPublicCalendarSource",
    "HttpPublicCalendarSource",
    "PublicCalendarSource",
    "RawEconomicEvent",
    "RobotsPolicy",
    "UnsafePublicUrlError",
    "assert_safe_public_url",
    "canonicalize_title",
    "classify_impact",
    "evaluate_robots_policy",
    "us_federal_holidays",
]
