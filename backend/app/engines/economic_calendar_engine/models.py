from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


NAMESPACE = UUID("2bf5f2a8-3881-4e76-91da-492e5a466476")


def stable_id(kind: str, *parts: object) -> UUID:
    return uuid5(NAMESPACE, "|".join((kind, *(str(part) for part in parts))))


def payload_hash(payload: object) -> str:
    import json

    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class CalendarModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def aware_datetimes(cls, value: object) -> object:
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class ProviderMode(StrEnum):
    LIVE_PROVIDER = "live_provider"
    FILE_IMPORT = "file_import"
    STATIC_FIXTURE = "static_fixture"
    IN_MEMORY_TEST_PROVIDER = "in_memory_test_provider"
    DISABLED = "disabled"


class EconomicEventStatus(StrEnum):
    SCHEDULED = "scheduled"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    RELEASED = "released"
    REVISED = "revised"
    CORRECTED = "corrected"
    DELAYED = "delayed"
    POSTPONED = "postponed"
    RESCHEDULED = "rescheduled"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class PublicationState(StrEnum):
    NOT_PUBLISHED = "not_published"
    PARTIALLY_PUBLISHED = "partially_published"
    FIRST_RELEASE = "first_release"
    REVISED_RELEASE = "revised_release"
    CORRECTED_RELEASE = "corrected_release"
    FINAL_RELEASE = "final_release"
    UNAVAILABLE = "unavailable"


class EventImportance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EventCategory(StrEnum):
    CENTRAL_BANK = "central_bank"
    INTEREST_RATE = "interest_rate"
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    GROWTH = "growth"
    GDP = "gdp"
    PMI = "pmi"
    RETAIL = "retail"
    CONSUMER = "consumer"
    HOUSING = "housing"
    MANUFACTURING = "manufacturing"
    INDUSTRIAL = "industrial"
    TRADE = "trade"
    FISCAL = "fiscal"
    DEBT = "debt"
    MONEY = "money"
    SENTIMENT = "sentiment"
    CONFIDENCE = "confidence"
    COMMODITY = "commodity"
    ENERGY = "energy"
    AUCTION = "auction"
    SPEECH = "speech"
    MINUTES = "minutes"
    PRESS_CONFERENCE = "press_conference"
    HOLIDAY = "holiday"
    CLOSURE = "closure"
    OTHER = "other"
    UNKNOWN = "unknown"


class ValueType(StrEnum):
    NUMBER = "number"
    PERCENT = "percent"
    RATE = "rate"
    INDEX = "index"
    COUNT = "count"
    CURRENCY = "currency"
    DURATION = "duration"
    BOOLEAN = "boolean"
    TEXT = "text"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class TimingPrecision(StrEnum):
    EXACT = "exact"
    MINUTE = "minute"
    HOUR = "hour"
    DATE_ONLY = "date_only"
    TENTATIVE = "tentative"
    UNKNOWN = "unknown"


class RelevanceLevel(StrEnum):
    DIRECT = "direct"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"
    UNKNOWN = "unknown"


class RiskWindowPhase(StrEnum):
    OUTSIDE = "outside"
    PRE_EVENT = "pre_event"
    IMMINENT = "imminent"
    AT_EVENT = "at_event"
    POST_EVENT = "post_event"
    COOLDOWN = "cooldown"
    OVERLAPPING = "overlapping"
    UNKNOWN = "unknown"


class ConflictState(StrEnum):
    NONE = "none"
    MINOR = "minor"
    MATERIAL = "material"
    UNRESOLVED = "unresolved"


class RevisionType(StrEnum):
    INITIAL_DISCOVERY = "initial_discovery"
    SCHEDULE_CHANGE = "schedule_change"
    STATUS_CHANGE = "status_change"
    FIRST_RELEASE = "first_release"
    VALUE_REVISION = "value_revision"
    CORRECTION = "correction"
    CANCELLATION = "cancellation"
    POSTPONEMENT = "postponement"
    RESCHEDULE = "reschedule"
    METADATA_UPDATE = "metadata_update"
    PROVIDER_CONFLICT = "provider_conflict"
    MERGE = "merge"


class FreshnessState(StrEnum):
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ConnectionState(StrEnum):
    """A provider's actual, live connection outcome — the single fact every other status field
    (reachable/authenticated/endpoint_valid/entitlement_valid/data_available/message) is derived
    from, so they can never disagree. Each value answers a DIFFERENT layer of "what happened":
    UNREACHABLE/TIMEOUT mean we never got an HTTP response at all; everything else means the
    server responded, so `reachable` is true for all of them — a 404 proves the server was
    reached, it does not mean the provider is "unreachable"."""

    CONNECTED = "connected"
    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    """HTTP 401 — credentials missing or rejected outright."""
    FORBIDDEN = "forbidden"
    """HTTP 403 — credentials were accepted, but this endpoint/resource isn't entitled under the
    current plan (e.g. a retired legacy endpoint, or a paid-tier-only route)."""
    INVALID_ENDPOINT = "invalid_endpoint"
    """HTTP 404 — the server was reached and authenticated the request path itself doesn't exist.
    Never conflate this with UNREACHABLE: a 404 is proof of a successful connection."""
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    """HTTP 5xx — the provider's own infrastructure failed; not a fact about our request."""
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class CalendarContextState(StrEnum):
    """The single categorical answer to "why does the economic context look the way it does" —
    computed once (analyzer.instrument_context) and consumed identically by signal_decision_engine,
    the explainability layer, market_intelligence, and diagnostics, so they can never contradict
    each other. Only the PROVIDER_* / NO_CALENDAR_DATA states represent genuine unavailability;
    NO_RELEVANT_EVENTS / OUTSIDE_RISK_WINDOW / INSIDE_RISK_WINDOW are routine, healthy states that
    must never degrade or block anything downstream."""

    PROVIDER_UNREACHABLE = "provider_unreachable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_AUTH_FAILED = "provider_auth_failed"
    PROVIDER_ENTITLEMENT_INVALID = "provider_entitlement_invalid"
    PROVIDER_INVALID_ENDPOINT = "provider_invalid_endpoint"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_ERROR = "provider_error"
    NO_CALENDAR_DATA = "no_calendar_data"
    NO_RELEVANT_EVENTS = "no_relevant_events"
    OUTSIDE_RISK_WINDOW = "outside_risk_window"
    INSIDE_RISK_WINDOW = "inside_risk_window"


GENUINELY_UNAVAILABLE_STATES = frozenset(
    {
        CalendarContextState.PROVIDER_UNREACHABLE,
        CalendarContextState.PROVIDER_TIMEOUT,
        CalendarContextState.PROVIDER_AUTH_FAILED,
        CalendarContextState.PROVIDER_ENTITLEMENT_INVALID,
        CalendarContextState.PROVIDER_INVALID_ENDPOINT,
        CalendarContextState.PROVIDER_RATE_LIMITED,
        CalendarContextState.PROVIDER_ERROR,
        CalendarContextState.NO_CALENDAR_DATA,
    }
)


class ProviderCapabilities(CalendarModel):
    historical_events: bool = False
    future_events: bool = True
    actual_values: bool = True
    forecast_values: bool = True
    previous_values: bool = True
    revisions: bool = False
    statuses: bool = True
    rescheduling: bool = False
    cancellations: bool = False
    incremental_updates: bool = False
    publication_time: bool = False
    importance: bool = True
    country: bool = True
    currency: bool = True
    unit: bool = True
    precision: bool = False
    source_url: bool = False
    # Explicit per-dataset entitlement flags — a successful call to one FMP endpoint (e.g. a
    # lightweight quote used only to verify connectivity) must never be read as proof that every
    # other dataset is included in the current subscription plan; each is verified independently.
    live_quote: bool = False
    candles_1min: bool = False
    candles_5min: bool = False
    candles_15min: bool = False
    candles_1hour: bool = False
    candles_daily: bool = False
    economic_calendar: bool = False


class ProviderEventObservation(CalendarModel):
    observation_id: UUID
    provider_name: str
    provider_version: str = "unknown"
    provider_event_id: str
    provider_event_url: str | None = None
    request_id: str | None = None
    request_started_at: datetime | None = None
    response_received_at: datetime
    raw_name: str
    raw_category: str | None = None
    raw_country: str | None = None
    raw_currency: str | None = None
    raw_importance: str | None = None
    raw_status: str | None = None
    raw_scheduled_time: str | None = None
    raw_timezone: str | None = None
    raw_actual: str | float | None = None
    raw_forecast: str | float | None = None
    raw_previous: str | float | None = None
    raw_revised_previous: str | float | None = None
    raw_unit: str | None = None
    provider_published_at: datetime | None = None
    provider_updated_at: datetime | None = None
    available_at: datetime
    ingested_at: datetime
    payload_hash: str
    raw_payload_reference: str | None = None
    parse_warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class EconomicEvent(CalendarModel):
    event_id: UUID
    engine_name: str = "economic_calendar"
    engine_version: str = "1.0.0"
    schema_version: str = "1.0"
    configuration_version: str = "1.0.0"
    normalization_version: str = "1.0.0"
    canonical_name: str
    display_name: str
    description: str | None = None
    category: EventCategory = EventCategory.UNKNOWN
    subcategory: str | None = None
    importance: EventImportance = EventImportance.UNKNOWN
    country_code: str | None = None
    country_name: str | None = None
    currency_codes: tuple[str, ...] = ()
    region_codes: tuple[str, ...] = ()
    scheduled_at: datetime | None = None
    scheduled_timezone: str = "UTC"
    scheduled_at_utc: datetime | None = None
    timing_precision: TimingPrecision = TimingPrecision.UNKNOWN
    status: EconomicEventStatus = EconomicEventStatus.UNKNOWN
    publication_state: PublicationState = PublicationState.NOT_PUBLISHED
    actual_value: float | None = None
    forecast_value: float | None = None
    previous_value: float | None = None
    revised_previous_value: float | None = None
    actual_text: str | None = None
    forecast_text: str | None = None
    previous_text: str | None = None
    value_type: ValueType = ValueType.UNKNOWN
    unit: str | None = None
    scale: float = 1.0
    precision: int | None = Field(default=None, ge=0, le=12)
    first_published_at: datetime | None = None
    last_updated_at: datetime
    available_at: datetime
    ingested_at: datetime
    is_all_day: bool = False
    is_tentative: bool = False
    is_cancelled: bool = False
    is_postponed: bool = False
    is_rescheduled: bool = False
    is_revised: bool = False
    is_corrected: bool = False
    original_scheduled_at: datetime | None = None
    rescheduled_from: datetime | None = None
    rescheduled_to: datetime | None = None
    provider_records: tuple[UUID, ...] = ()
    revision_count: int = Field(default=0, ge=0)
    latest_revision_id: UUID | None = None
    source_quality: float = Field(default=1.0, ge=0, le=1)
    normalization_confidence: float = Field(default=1.0, ge=0, le=1)
    data_completeness: float = Field(default=1.0, ge=0, le=1)
    conflict_state: ConflictState = ConflictState.NONE
    affected_assets: tuple[str, ...] = ()
    affected_currencies: tuple[str, ...] = ()
    affected_instruments: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    probabilistic_context: bool = True
    trading_instruction: bool = False

    @model_validator(mode="after")
    def validate_semantics(self) -> EconomicEvent:
        if not self.probabilistic_context or self.trading_instruction:
            raise ValueError("calendar outputs are context only")
        if self.scheduled_at and self.scheduled_at_utc and self.scheduled_at.astimezone(UTC) != self.scheduled_at_utc.astimezone(UTC):
            raise ValueError("scheduled timestamps disagree")
        if self.status == EconomicEventStatus.CANCELLED and not self.is_cancelled:
            raise ValueError("cancelled status requires is_cancelled")
        return self

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def currency(self) -> str:
        return self.currency_codes[0] if self.currency_codes else ""

    @property
    def actual(self) -> float | None:
        return self.actual_value

    @property
    def forecast(self) -> float | None:
        return self.forecast_value

    @property
    def previous(self) -> float | None:
        return self.previous_value


class EconomicEventRevision(CalendarModel):
    revision_id: UUID
    event_id: UUID
    revision_number: int = Field(ge=1)
    revision_type: RevisionType
    observed_at: datetime
    available_at: datetime
    provider_published_at: datetime | None = None
    ingested_at: datetime
    previous_status: EconomicEventStatus | None = None
    new_status: EconomicEventStatus | None = None
    previous_scheduled_at: datetime | None = None
    new_scheduled_at: datetime | None = None
    previous_actual_value: float | None = None
    new_actual_value: float | None = None
    previous_forecast_value: float | None = None
    new_forecast_value: float | None = None
    previous_previous_value: float | None = None
    new_previous_value: float | None = None
    previous_revised_previous_value: float | None = None
    new_revised_previous_value: float | None = None
    changed_fields: tuple[str, ...]
    provider_observation_ids: tuple[UUID, ...]
    reason: str
    payload_hash: str


class ProviderStatus(CalendarModel):
    provider_name: str
    provider_version: str = "unknown"
    base_url: str | None = None
    mode: ProviderMode
    enabled: bool
    api_key_configured: bool = False
    # Each of these answers ONE independent layer of "what happened," derived solely from
    # `connection_state` (see `ConnectionState`'s docstring) — they can never contradict each
    # other or `connection_state` because there is exactly one source of truth for all of them.
    # A 404 sets reachable=True, authenticated=True, endpoint_valid=False: the server was reached
    # and the credentials were fine, only the route itself is wrong — never "unreachable".
    reachable: bool = False
    authenticated: bool = False
    endpoint_valid: bool = True
    entitlement_valid: bool = True
    data_available: bool = False
    stale: bool = False
    rate_limited: bool = False
    connection_state: ConnectionState = ConnectionState.UNKNOWN
    failure_reason: str | None = None
    http_status: int | None = None
    last_request: datetime | None = None
    last_success: datetime | None = None
    last_failure: datetime | None = None
    last_cursor: str | None = None
    response_time_ms: float | None = None
    retry_count: int = 0
    backoff_until: datetime | None = None
    rate_limit_remaining: int | None = None
    rate_limit_limit: int | None = None
    daily_quota_used: int | None = None
    daily_quota_limit: int | None = None
    monthly_quota_used: int | None = None
    monthly_quota_limit: int | None = None
    # Sanitized (never contains the API key/token) — the last raw error text/body TEN actually
    # received, so a human can see exactly what the provider said instead of just a category.
    raw_error: str | None = None
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    message: str = ""


class DegradationState(CalendarModel):
    is_degraded: bool = False
    # "healthy" or one of the `CalendarContextState` genuine-failure values — computed once from
    # the highest-priority provider's actual `connection_state`, never re-derived downstream.
    category: str = "healthy"
    reasons: tuple[str, ...] = ()
    unavailable_providers: tuple[str, ...] = ()
    partial_results: bool = False


class EconomicCalendarSnapshot(CalendarModel):
    snapshot_id: UUID
    engine_name: str = "economic_calendar"
    engine_version: str = "1.0.0"
    schema_version: str = "1.0"
    configuration_version: str = "1.0.0"
    normalization_version: str = "1.0.0"
    analysis_timestamp: datetime
    historical_boundary: datetime
    created_at: datetime
    window_start: datetime
    window_end: datetime
    events: tuple[EconomicEvent, ...]
    upcoming_events: tuple[EconomicEvent, ...]
    active_events: tuple[EconomicEvent, ...]
    recent_events: tuple[EconomicEvent, ...]
    event_count: int = 0
    high_importance_count: int = 0
    critical_importance_count: int = 0
    released_count: int = 0
    revised_count: int = 0
    conflicting_count: int = 0
    unavailable_count: int = 0
    provider_status: tuple[ProviderStatus, ...] = ()
    degradation: DegradationState = Field(default_factory=DegradationState)
    quality: float = Field(default=0, ge=0, le=1)
    freshness: FreshnessState = FreshnessState.UNKNOWN
    probabilistic_context: bool = True
    trading_instruction: bool = False


class InstrumentEventContext(CalendarModel):
    context_id: UUID
    symbol: str
    instrument_type: str = "generic"
    base_currency: str | None = None
    quote_currency: str | None = None
    analysis_timestamp: datetime
    historical_boundary: datetime
    previous_relevant_event: EconomicEvent | None = None
    next_relevant_event: EconomicEvent | None = None
    active_relevant_events: tuple[EconomicEvent, ...] = ()
    minutes_since_previous_event: float | None = None
    minutes_until_next_event: float | None = None
    risk_window_phase: RiskWindowPhase = RiskWindowPhase.OUTSIDE
    risk_score: float = Field(default=0, ge=0, le=1)
    cluster_score: float = Field(default=0, ge=0, le=1)
    importance_score: float = Field(default=0, ge=0, le=1)
    relevance_score: float = Field(default=0, ge=0, le=1)
    freshness_score: float = Field(default=0, ge=0, le=1)
    quality_score: float = Field(default=0, ge=0, le=1)
    pre_event_window_minutes: int = Field(default=0, ge=0)
    post_event_window_minutes: int = Field(default=0, ge=0)
    cooldown_window_minutes: int = Field(default=0, ge=0)
    direct_currency_matches: tuple[str, ...] = ()
    indirect_currency_matches: tuple[str, ...] = ()
    country_matches: tuple[str, ...] = ()
    asset_class_matches: tuple[str, ...] = ()
    conflicting_events: tuple[UUID, ...] = ()
    unavailable_context: tuple[str, ...] = ()
    # The canonical, single-source-of-truth categorical state — see `CalendarContextState`.
    # signal_decision_engine, the explainability layer, market_intelligence, and diagnostics all
    # read THIS field rather than each re-deriving their own notion of "is this unavailable."
    context_state: CalendarContextState = CalendarContextState.OUTSIDE_RISK_WINDOW
    primary_explanation: str = ""
    limitations: tuple[str, ...] = ()
    probabilistic_context: bool = True
    trading_instruction: bool = False


class EventCluster(CalendarModel):
    cluster_id: UUID
    window_start: datetime
    window_end: datetime
    event_ids: tuple[UUID, ...]
    countries: tuple[str, ...] = ()
    currencies: tuple[str, ...] = ()
    categories: tuple[EventCategory, ...] = ()
    event_count: int = 0
    high_importance_count: int = 0
    critical_importance_count: int = 0
    cluster_density: float = Field(default=0, ge=0, le=1)
    cluster_importance: float = Field(default=0, ge=0, le=1)
    overlap_score: float = Field(default=0, ge=0, le=1)
    conflict_score: float = Field(default=0, ge=0, le=1)
    affected_instruments: tuple[str, ...] = ()
    explanation: str = ""


class EconomicCalendarExplanation(CalendarModel):
    headline: str
    summary: str
    relevant_events: tuple[UUID, ...] = ()
    excluded_events: tuple[UUID, ...] = ()
    unavailable_events: tuple[UUID, ...] = ()
    conflicting_events: tuple[UUID, ...] = ()
    provider_conflicts: tuple[str, ...] = ()
    revision_history: tuple[str, ...] = ()
    mapping_explanation: tuple[str, ...] = ()
    risk_components: dict[str, float] = Field(default_factory=dict)
    quality_components: dict[str, float] = Field(default_factory=dict)
    freshness_components: dict[str, float] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()


class EconomicCalendarCheckpoint(CalendarModel):
    checkpoint_id: UUID
    engine_name: str = "economic_calendar"
    engine_version: str = "1.0.0"
    schema_version: str = "1.0"
    configuration_version: str = "1.0.0"
    normalization_version: str = "1.0.0"
    last_successful_sync_at: datetime | None = None
    last_provider_cursor: dict[str, str] = Field(default_factory=dict)
    last_provider_update_token: dict[str, str] = Field(default_factory=dict)
    last_processed_observation: UUID | None = None
    state_payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    created_at: datetime


class NewsRiskResult(CalendarModel):
    risk_level: EventImportance = EventImportance.LOW
    no_trade: bool = False
    active_events: tuple[EconomicEvent, ...] = ()
    minutes_to_nearest: float | None = None
    observations: tuple[str, ...] = ()
    probabilistic_context: bool = True
    trading_instruction: bool = False
