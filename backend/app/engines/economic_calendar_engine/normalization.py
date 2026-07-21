from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import EconomicCalendarConfig
from .models import (
    EconomicEvent,
    EconomicEventStatus,
    EventCategory,
    EventImportance,
    ProviderEventObservation,
    PublicationState,
    TimingPrecision,
    ValueType,
    stable_id,
)
from .public_sources.impact import canonicalize_title


@dataclass(frozen=True)
class ParsedValue:
    value: float | None
    text: str | None
    value_type: ValueType
    unit: str | None
    scale: float
    precision: int | None


def canonical_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def parse_value(raw: str | float | int | None, unit_hint: str | None = None) -> ParsedValue:
    if raw is None:
        return ParsedValue(None, None, ValueType.UNKNOWN, unit_hint, 1, None)
    if isinstance(raw, (float, int)):
        return ParsedValue(float(raw), str(raw), ValueType.NUMBER, unit_hint, 1, _precision(str(raw)))
    text = raw.strip()
    if not text or text.lower() in {"n/a", "na", "-", "tentative", "tbd", "--"}:
        return ParsedValue(None, text or None, ValueType.NOT_APPLICABLE, unit_hint, 1, None)
    negative = text.startswith("(") and text.endswith(")")
    cleaned = text.strip("() ").replace(" ", "")
    if cleaned.startswith("<") or cleaned.startswith(">"):
        cleaned = cleaned[1:]
    currency = cleaned[0] if cleaned and cleaned[0] in "$€£¥" else None
    cleaned = cleaned.lstrip("$€£¥")
    suffix = cleaned[-1:].upper()
    scale = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0, "T": 1_000_000_000_000.0}.get(suffix, 1.0)
    if scale != 1:
        cleaned = cleaned[:-1]
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "") if cleaned.rfind(".") > cleaned.rfind(",") else cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(",") == 1 and len(cleaned.rsplit(",", 1)[1]) <= 2:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        value = float(Decimal(cleaned)) * scale
    except (InvalidOperation, ValueError):
        return ParsedValue(None, text, ValueType.TEXT, unit_hint, 1, None)
    if negative:
        value = -value
    value_type = ValueType.PERCENT if percent else ValueType.CURRENCY if currency else ValueType.NUMBER
    return ParsedValue(value, text, value_type, "%" if percent else unit_hint or currency, scale, _precision(cleaned))


def _precision(value: str) -> int:
    return len(value.rsplit(".", 1)[1]) if "." in value else 0


def parse_schedule(raw: str | None, timezone: str | None) -> tuple[datetime | None, TimingPrecision, tuple[str, ...]]:
    if not raw or raw.strip().lower() in {"tentative", "tbd"}:
        return None, TimingPrecision.TENTATIVE, ()
    value = raw.strip()
    warnings: list[str] = []
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, TimingPrecision.UNKNOWN, ("unparseable scheduled time",)
    precision = TimingPrecision.DATE_ONLY if "T" not in value and " " not in value else TimingPrecision.MINUTE
    if parsed.tzinfo is None:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone or "UTC"))
        except ZoneInfoNotFoundError:
            parsed = parsed.replace(tzinfo=UTC)
            warnings.append("unknown provider timezone; UTC assumed")
    return parsed.astimezone(UTC), precision, tuple(warnings)


def _enum(
    enum: type[EventImportance] | type[EventCategory] | type[EconomicEventStatus], raw: str | None, aliases: dict[str, str]
) -> EventImportance | EventCategory | EconomicEventStatus:
    key = canonical_name(raw or "unknown").replace(" ", "_")
    key = aliases.get(key, key)
    try:
        return enum(key)
    except ValueError:
        return enum("unknown")


def normalize_observation(item: ProviderEventObservation, config: EconomicCalendarConfig) -> EconomicEvent:
    name = canonical_name(item.raw_name)
    country = (item.raw_country or "").upper()[:2] or None
    currency = (item.raw_currency or config.country_currency.get(country or "", "")).upper() or None
    scheduled, precision, warnings = parse_schedule(item.raw_scheduled_time, item.raw_timezone)
    actual = parse_value(item.raw_actual, item.raw_unit)
    forecast = parse_value(item.raw_forecast, item.raw_unit)
    previous = parse_value(item.raw_previous, item.raw_unit)
    revised_previous = parse_value(item.raw_revised_previous, item.raw_unit)
    importance = _enum(EventImportance, item.raw_importance, {"red": "high", "orange": "medium", "yellow": "low"})
    category = _enum(EventCategory, item.raw_category, {"rates": "interest_rate", "jobs": "employment"})
    status = _enum(EconomicEventStatus, item.raw_status, {"published": "released", "canceled": "cancelled"})
    if status == EconomicEventStatus.UNKNOWN:
        status = (
            EconomicEventStatus.RELEASED
            if actual.value is not None
            else EconomicEventStatus.TENTATIVE
            if precision == TimingPrecision.TENTATIVE
            else EconomicEventStatus.SCHEDULED
        )
    publication = PublicationState.FIRST_RELEASE if actual.value is not None else PublicationState.NOT_PUBLISHED
    # Dedup key: canonical event type (not free-text title — "Consumer Price Index" from one
    # source and "CPI" from another must collapse to the same key) + country + scheduled UTC date
    # (the "release period"). Two sources reporting the same event on the same day always
    # reconcile to one canonical TEN event; `reconcile()` then picks a winner by provider priority
    # and flags `conflict_state` if the sources actually disagree on time/value.
    canonical_event_type = canonicalize_title(item.raw_name)
    identity_time = scheduled.date().isoformat() if scheduled else "tentative"
    event_id = stable_id("economic-event", canonical_event_type, country, identity_time)
    unit = actual.unit or forecast.unit or previous.unit or item.raw_unit
    return EconomicEvent(
        event_id=event_id,
        configuration_version=config.version,
        schema_version=config.versions.schema_version,
        normalization_version=config.versions.normalization_version,
        canonical_name=name,
        canonical_event_type=canonical_event_type,
        display_name=item.raw_name.strip(),
        category=category,
        importance=importance,
        country_code=country,
        currency_codes=(currency,) if currency else (),
        scheduled_at=scheduled,
        scheduled_timezone=item.raw_timezone or "UTC",
        scheduled_at_utc=scheduled,
        timing_precision=precision,
        status=status,
        publication_state=publication,
        actual_value=actual.value,
        forecast_value=forecast.value,
        previous_value=previous.value,
        revised_previous_value=revised_previous.value,
        actual_text=actual.text,
        forecast_text=forecast.text,
        previous_text=previous.text,
        value_type=actual.value_type if actual.text else forecast.value_type,
        unit=unit,
        scale=max(actual.scale, forecast.scale, previous.scale),
        precision=actual.precision if actual.value is not None else forecast.precision,
        first_published_at=item.provider_published_at if actual.value is not None else None,
        last_updated_at=item.provider_updated_at or item.available_at,
        available_at=item.available_at,
        ingested_at=item.ingested_at,
        is_all_day=precision == TimingPrecision.DATE_ONLY,
        is_tentative=precision == TimingPrecision.TENTATIVE,
        is_cancelled=status == EconomicEventStatus.CANCELLED,
        is_postponed=status == EconomicEventStatus.POSTPONED,
        is_rescheduled=status == EconomicEventStatus.RESCHEDULED,
        provider_records=(item.observation_id,),
        source_quality=0.8 if warnings or item.parse_warnings else 1,
        normalization_confidence=0.7 if not country or not currency else 1,
        data_completeness=sum(value is not None for value in (scheduled, country, currency, item.raw_importance, item.raw_category)) / 5,
        conflict_state="none",
        affected_currencies=(currency,) if currency else (),
        metadata={"provider": item.provider_name, "provider_event_id": item.provider_event_id, "parse_warnings": (*item.parse_warnings, *warnings)},
    )
