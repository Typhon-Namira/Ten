"""Shared base for every keyless public-source calendar adapter.

Architectural boundary: `fetch_schedule()` is the one method each concrete source actually
implements (one parser per source, per the brief) — it returns plain `RawEconomicEvent` records
with no knowledge of TEN's broader pipeline. `HttpPublicCalendarSource.fetch_events()` is a thin,
shared adaptation layer that wraps those into `ProviderFetchResult` so every existing pipeline
mechanism (`reconcile`, `build_snapshot`, `instrument_context`, the decision engine, explainability)
keeps working unmodified — a public source is just another `EconomicCalendarProvider`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from time import perf_counter
from typing import Protocol, runtime_checkable

import httpx

from ..models import ConnectionState, ProviderCapabilities, ProviderEventObservation, ProviderMode, ProviderStatus, SourceType, stable_id
from ..providers import EconomicCalendarProvider, ProviderFetchRequest, ProviderFetchResult, observation_from_mapping
from .circuit_breaker import CircuitBreakerState, FailureCategory
from .robots import RobotsPolicy, evaluate_robots_policy
from .ssrf import assert_safe_public_url

logger = logging.getLogger(__name__)

#: TEN identifies itself honestly on every request — no impersonation of a browser, per the
#: source-policy requirement to never bypass automated-access detection.
USER_AGENT = "TEN-Economic-Calendar/1.0 (+https://github.com/; institutional market analysis; contact via repository issues)"


@dataclass(frozen=True)
class RawEconomicEvent:
    """What one source-specific parser produces — deliberately the same field set
    `observation_from_mapping()` already expects, so there is exactly one normalization path for
    every provider in this engine, keyed or keyless."""

    raw_name: str
    raw_scheduled_time: str  # ISO 8601, always timezone-aware
    provider_event_id: str
    raw_country: str | None = None
    raw_currency: str | None = None
    raw_importance: str | None = None
    raw_status: str | None = None
    raw_timezone: str = "UTC"
    raw_actual: str | float | None = None
    raw_forecast: str | float | None = None
    raw_previous: str | float | None = None
    raw_unit: str | None = None
    source_url: str | None = None


@runtime_checkable
class PublicCalendarSource(Protocol):
    async def fetch_schedule(self, start_date: date, end_date: date) -> list[RawEconomicEvent]: ...


class HttpPublicCalendarSource(EconomicCalendarProvider):
    """Base for every official-domain, keyless calendar source. Owns: SSRF-checked HTTP fetch,
    robots.txt evaluation (checked once at construction, never re-fetched mid-poll-cycle), a
    per-source circuit breaker, and every `ProviderStatus` diagnostic field a public source can
    honestly report — no API key, no quota, because there is neither."""

    parser_version = "1.0.0"

    def __init__(self, name: str, *, source_url: str, timeout_seconds: float = 15, source_type: SourceType = SourceType.PUBLIC_WEBPAGE, max_retries: int = 2) -> None:
        assert_safe_public_url(source_url)
        self.name = name
        self.version = self.parser_version
        self.timezone = "UTC"
        self.mode = ProviderMode.PUBLIC_WEB_SOURCE
        self.source_url = source_url
        self.source_type = source_type
        self.max_retries = max_retries
        self.capabilities = ProviderCapabilities(historical_events=False, economic_calendar=True)
        self._client = httpx.AsyncClient(timeout=timeout_seconds, headers={"User-Agent": USER_AGENT})
        self._circuit = CircuitBreakerState()
        self._robots_policy: RobotsPolicy = RobotsPolicy.UNKNOWN
        self._robots_checked = False
        self.last_latency_ms = 0.0
        self._last_request: datetime | None = None
        self._last_success: datetime | None = None
        self._last_failure: datetime | None = None
        self._connection_state = ConnectionState.UNKNOWN
        self._failure_reason: str | None = None
        self._http_status: int | None = None
        self._raw_error: str | None = None
        self._events_parsed = 0
        self._last_schedule_date: datetime | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _ensure_robots_checked(self) -> None:
        if self._robots_checked:
            return
        self._robots_checked = True
        self._robots_policy = await evaluate_robots_policy(self._client, self.source_url, user_agent=USER_AGENT)
        if self._robots_policy == RobotsPolicy.DISALLOWED:
            logger.error("public calendar source disallowed by robots.txt: source=%s url=%s — this source will not be fetched", self.name, self.source_url)

    async def _get_text(self, url: str) -> str | None:
        """Fetches `url` (which must already be on the SSRF allowlist) and returns the response
        body, or `None` on any failure — every failure is logged and recorded on the circuit
        breaker with the right category, and this never raises to the caller."""
        assert_safe_public_url(url)
        await self._ensure_robots_checked()
        if self._robots_policy == RobotsPolicy.DISALLOWED:
            self._connection_state, self._failure_reason = ConnectionState.FORBIDDEN, "disallowed by robots.txt"
            return None
        now = datetime.now(UTC)
        self._last_request = now
        if not self._circuit.should_attempt(now):
            self._connection_state = ConnectionState.RATE_LIMITED
            self._failure_reason = f"circuit breaker open until {self._circuit.open_until.isoformat() if self._circuit.open_until else 'unknown'}"
            return None
        started = perf_counter()
        attempt = 0
        last_exception: Exception | None = None
        while attempt <= self.max_retries:
            try:
                response = await self._client.get(url, follow_redirects=True)
            except httpx.TimeoutException as exc:
                last_exception, category = exc, FailureCategory.TIMEOUT
                self._connection_state, self._failure_reason = ConnectionState.TIMEOUT, "request timed out"
            except httpx.HTTPError as exc:
                last_exception, category = exc, FailureCategory.NETWORK_ERROR
                self._connection_state, self._failure_reason = ConnectionState.UNREACHABLE, f"{type(exc).__name__}: {exc}"[:300]
            else:
                self.last_latency_ms = (perf_counter() - started) * 1000
                self._http_status = response.status_code
                if response.status_code == 404:
                    self._connection_state, self._failure_reason = ConnectionState.INVALID_ENDPOINT, "HTTP 404: page not found — the source URL may have moved"
                    self._raw_error, self._last_failure = response.text[:500], now
                    logger.error("public calendar source page not found: source=%s url=%s status=404", self.name, url)
                    self._circuit.record_failure(now, FailureCategory.HTTP_ERROR)
                    return None
                if response.status_code == 403:
                    self._connection_state, self._failure_reason = ConnectionState.FORBIDDEN, "HTTP 403: automated access appears to be blocked"
                    self._raw_error, self._last_failure = response.text[:500], now
                    logger.error("public calendar source blocked automated access: source=%s url=%s status=403", self.name, url)
                    self._circuit.record_failure(now, FailureCategory.BLOCKED)
                    return None
                if response.status_code == 429:
                    self._connection_state, self._failure_reason = ConnectionState.RATE_LIMITED, "HTTP 429: rate limited"
                    self._raw_error, self._last_failure = response.text[:300], now
                    logger.warning("public calendar source rate limited: source=%s url=%s", self.name, url)
                    self._circuit.record_failure(now, FailureCategory.HTTP_ERROR)
                    return None
                if response.status_code >= 500:
                    self._connection_state, self._failure_reason = ConnectionState.SERVER_ERROR, f"HTTP {response.status_code}"
                    if attempt < self.max_retries:
                        attempt += 1
                        await self._sleep_backoff(attempt)
                        continue
                    self._raw_error, self._last_failure = response.text[:300], now
                    self._circuit.record_failure(now, FailureCategory.HTTP_ERROR)
                    return None
                if response.status_code >= 400:
                    self._connection_state, self._failure_reason = ConnectionState.INVALID_ENDPOINT, f"HTTP {response.status_code}"
                    self._raw_error, self._last_failure = response.text[:300], now
                    self._circuit.record_failure(now, FailureCategory.HTTP_ERROR)
                    return None
                self._connection_state, self._failure_reason, self._raw_error = ConnectionState.CONNECTED, None, None
                self._last_success = now
                self._circuit.record_success(now)
                logger.info("public calendar source fetch succeeded: source=%s url=%s status=%s latency_ms=%.1f", self.name, url, response.status_code, self.last_latency_ms)
                return response.text
            attempt += 1
            if attempt <= self.max_retries:
                await self._sleep_backoff(attempt)
        self.last_latency_ms = (perf_counter() - started) * 1000
        self._raw_error = f"{type(last_exception).__name__}: {last_exception}"[:300] if last_exception else self._failure_reason
        self._last_failure = now
        self._circuit.record_failure(now, category if last_exception else FailureCategory.NETWORK_ERROR)
        logger.warning("public calendar source unreachable: source=%s url=%s connection_state=%s message=%s", self.name, url, self._connection_state.value, self._raw_error)
        return None

    async def _sleep_backoff(self, attempt: int) -> None:
        await asyncio.sleep(min(0.5 * attempt, 5.0))

    async def fetch_schedule(self, start_date: date, end_date: date) -> list[RawEconomicEvent]:  # pragma: no cover - overridden
        raise NotImplementedError

    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        """Adapts `fetch_schedule()` (the source-owned method) into the shared
        `EconomicCalendarProvider` contract every existing pipeline stage already consumes."""
        now = datetime.now(UTC)
        try:
            raw_events = await self.fetch_schedule(request.start.date(), request.end.date())
        except Exception as exc:  # a parser bug in one source must never take down the whole sync cycle
            self._connection_state, self._failure_reason = ConnectionState.UNREACHABLE, f"parser raised {type(exc).__name__}"
            self._circuit.record_failure(now, FailureCategory.PARSER_MISMATCH)
            logger.error("public calendar source parser failed: source=%s error=%s", self.name, exc, exc_info=True)
            return ProviderFetchResult(observations=(), warnings=(f"parser_failed:{type(exc).__name__}",))
        if self._connection_state not in {ConnectionState.CONNECTED} and not raw_events:
            return ProviderFetchResult(observations=(), warnings=(f"public_source_fetch_failed:{self._connection_state.value}",))
        values: list[ProviderEventObservation] = []
        failures = 0
        latest_scheduled: datetime | None = None
        for index, raw in enumerate(raw_events):
            try:
                mapping = {
                    "provider_event_id": raw.provider_event_id,
                    "raw_name": raw.raw_name,
                    "raw_country": raw.raw_country,
                    "raw_currency": raw.raw_currency,
                    "raw_importance": raw.raw_importance,
                    "raw_status": raw.raw_status,
                    "raw_scheduled_time": raw.raw_scheduled_time,
                    "raw_timezone": raw.raw_timezone,
                    "raw_actual": raw.raw_actual,
                    "raw_forecast": raw.raw_forecast,
                    "raw_previous": raw.raw_previous,
                    "raw_unit": raw.raw_unit,
                    "provider_event_url": raw.source_url,
                }
                item = observation_from_mapping(self.name, self.version, mapping, now, index)
                if request.start <= item.available_at <= request.end and len(values) < request.limit:
                    values.append(item)
                scheduled = _parse_iso(raw.raw_scheduled_time)
                if scheduled and (latest_scheduled is None or scheduled > latest_scheduled):
                    latest_scheduled = scheduled
            except (TypeError, ValueError, KeyError):
                failures += 1
        self._events_parsed = len(values)
        if latest_scheduled:
            self._last_schedule_date = latest_scheduled
        if not raw_events and self._connection_state == ConnectionState.CONNECTED:
            # A structurally successful fetch that parsed zero events is a DIFFERENT problem than
            # a fetch failure — most likely the page layout changed under the parser.
            self._circuit.record_failure(now, FailureCategory.EMPTY_SCHEDULE)
            logger.warning("public calendar source returned zero events from an otherwise successful fetch: source=%s — likely a layout change", self.name)
            return ProviderFetchResult(observations=(), warnings=("empty_valid_schedule",))
        return ProviderFetchResult(observations=tuple(values), cursor=str(len(values)), success_count=len(values), failure_count=failures)

    async def health(self) -> ProviderStatus:
        reachable = self._connection_state not in {ConnectionState.UNREACHABLE, ConnectionState.TIMEOUT, ConnectionState.UNKNOWN, ConnectionState.DISABLED}
        return ProviderStatus(
            provider_name=self.name,
            provider_version=self.version,
            base_url=self.source_url,
            mode=self.mode,
            enabled=True,
            api_key_configured=False,
            authenticated=True,  # meaningless for a public source — never gated by credentials
            reachable=reachable,
            endpoint_valid=self._connection_state != ConnectionState.INVALID_ENDPOINT,
            entitlement_valid=self._connection_state != ConnectionState.FORBIDDEN,
            data_available=self._connection_state == ConnectionState.CONNECTED and self._events_parsed > 0,
            rate_limited=self._connection_state == ConnectionState.RATE_LIMITED,
            connection_state=self._connection_state,
            failure_reason=self._failure_reason,
            http_status=self._http_status,
            last_request=self._last_request,
            last_success=self._last_success,
            last_failure=self._last_failure,
            response_time_ms=self.last_latency_ms or None,
            retry_count=0,
            raw_error=self._raw_error,
            capabilities=self.capabilities,
            message=self._failure_reason or "",
            source_type=self.source_type,
            robots_policy_status=self._robots_policy.value,
            parser_version=self.parser_version,
            events_parsed=self._events_parsed,
            last_schedule_date=self._last_schedule_date,
            cache_age_seconds=(datetime.now(UTC) - self._last_success).total_seconds() if self._last_success else None,
            circuit_breaker_open=not self._circuit.should_attempt(datetime.now(UTC)),
            circuit_breaker_open_until=self._circuit.open_until,
            last_failure_category=self._circuit.last_failure_category.value if self._circuit.last_failure_category else None,
        )


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def stable_event_id(source: str, *parts: object) -> str:
    return str(stable_id(f"public-source-{source}", *parts))


__all__ = ["HttpPublicCalendarSource", "PublicCalendarSource", "RawEconomicEvent", "USER_AGENT", "stable_event_id"]
