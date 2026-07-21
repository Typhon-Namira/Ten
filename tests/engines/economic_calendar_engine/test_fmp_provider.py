"""Financial Modeling Prep provider + provider-priority/failover coverage.

Uses `httpx.MockTransport` (the established pattern in this codebase, see
tests/engines/market_data_engine/test_market_data_adapters.py) so these tests never make a real
network call and never require a real API key.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from backend.app.engines.economic_calendar_engine.analyzer import build_snapshot, instrument_context
from backend.app.engines.economic_calendar_engine.config import EconomicCalendarConfig, ProviderConfig
from backend.app.engines.economic_calendar_engine.models import CalendarContextState, ConnectionState, FreshnessState, ProviderMode
from backend.app.engines.economic_calendar_engine.providers import DisabledProvider, FinancialModelingPrepProvider, FinnhubEconomicCalendarProvider, ProviderFetchRequest, build_providers

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)
WINDOW_START = NOW - timedelta(days=1)
WINDOW_END = NOW + timedelta(days=1)


def _request() -> ProviderFetchRequest:
    return ProviderFetchRequest(start=WINDOW_START, end=WINDOW_END)


def _mounted(provider: FinancialModelingPrepProvider | FinnhubEconomicCalendarProvider, handler) -> None:
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.retry_backoff_seconds = 0  # keep retry-path tests fast/deterministic


@pytest.mark.asyncio
async def test_fmp_provider_maps_events_and_reports_healthy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["apikey"] == "secret"
        return httpx.Response(
            200,
            json=[
                {"event": "Nonfarm Payrolls", "date": "2026-07-20 08:30:00", "country": "US", "currency": "USD", "impact": "High", "actual": 216000, "estimate": 185000, "previous": 199000, "unit": ""},
                {"event": "ECB Rate Decision", "date": "2026-07-20 11:45:00", "country": "EMU", "currency": "EUR", "impact": "High", "actual": None, "estimate": 4.0, "previous": 4.0, "unit": "%"},
            ],
            headers={"x-ratelimit-remaining": "97", "x-ratelimit-limit": "100"},
        )

    provider = FinancialModelingPrepProvider(api_key="secret")
    _mounted(provider, handler)
    result = await provider.fetch_events(_request())
    assert result.success_count == 2
    assert {item.raw_currency for item in result.observations} == {"USD", "EUR"}

    status = await provider.health()
    assert status.provider_name == "financial_modeling_prep"
    assert status.connection_state == ConnectionState.CONNECTED
    assert status.reachable is True
    assert status.authenticated is True
    assert status.http_status == 200
    assert status.rate_limit_remaining == 97
    assert status.rate_limit_limit == 100
    assert status.failure_reason is None
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_reports_authentication_failure_on_401() -> None:
    provider = FinancialModelingPrepProvider(api_key="bad-key")
    _mounted(provider, lambda request: httpx.Response(401, text="Unauthorized"))
    result = await provider.fetch_events(_request())
    assert result.observations == ()
    assert result.warnings

    status = await provider.health()
    assert status.connection_state == ConnectionState.UNAUTHORIZED
    assert status.authenticated is False
    assert status.reachable is False
    assert status.http_status == 401
    assert "authentication failed" in (status.failure_reason or "")
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_detects_error_body_returned_with_http_200() -> None:
    """FMP returns HTTP 200 with an error body for some invalid-key/invalid-plan cases — this is
    a real authentication failure, not "zero events today", and must be reported as such."""
    provider = FinancialModelingPrepProvider(api_key="bad-key")
    _mounted(provider, lambda request: httpx.Response(200, json={"Error Message": "Invalid API KEY."}))
    result = await provider.fetch_events(_request())
    assert result.observations == ()
    assert result.warnings == ("fmp_error_response",)

    status = await provider.health()
    assert status.connection_state == ConnectionState.UNAUTHORIZED
    assert "Invalid API KEY" in (status.failure_reason or "")
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_reports_rate_limited_on_429_without_retrying() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, headers={"retry-after": "30"})

    provider = FinancialModelingPrepProvider(api_key="secret")
    _mounted(provider, handler)
    result = await provider.fetch_events(_request())
    assert calls["count"] == 1  # rate limiting backs off; it must not be retried immediately
    assert result.observations == ()

    status = await provider.health()
    assert status.connection_state == ConnectionState.RATE_LIMITED
    assert status.rate_limited is True
    assert status.backoff_until is not None
    assert status.backoff_until > NOW - timedelta(days=365)  # sanity: a real future-ish timestamp
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_retries_server_errors_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=[])

    provider = FinancialModelingPrepProvider(api_key="secret", max_retries=2)
    _mounted(provider, handler)
    result = await provider.fetch_events(_request())
    assert calls["count"] == 3
    assert result.success_count == 0
    status = await provider.health()
    assert status.connection_state == ConnectionState.CONNECTED
    assert status.retry_count == 2
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_reports_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("deadline exceeded", request=request)

    provider = FinancialModelingPrepProvider(api_key="secret", max_retries=0)
    _mounted(provider, handler)
    result = await provider.fetch_events(_request())
    assert result.observations == ()
    status = await provider.health()
    assert status.connection_state == ConnectionState.TIMEOUT
    assert status.reachable is False
    await provider.close()


def test_build_providers_prefers_fmp_over_finnhub_in_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEN_FMP_API_KEY", "fmp-secret")
    monkeypatch.setenv("TEN_FINNHUB_API_KEY", "finnhub-secret")
    configs = (
        ProviderConfig(name="fmp", mode=ProviderMode.LIVE_PROVIDER, enabled=True, priority=10, api_key_env="TEN_FMP_API_KEY"),
        ProviderConfig(name="finnhub", mode=ProviderMode.LIVE_PROVIDER, enabled=True, priority=20, api_key_env="TEN_FINNHUB_API_KEY"),
        ProviderConfig(name="disabled", mode=ProviderMode.DISABLED, enabled=False, priority=100),
    )
    providers = build_providers(configs)
    assert [type(item).__name__ for item in providers] == ["FinancialModelingPrepProvider", "FinnhubEconomicCalendarProvider", "DisabledProvider"]
    assert providers[0].mode == ProviderMode.LIVE_PROVIDER


def test_build_providers_degrades_finnhub_to_disabled_when_its_key_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finnhub is only an optional fallback — a missing fallback key must never block FMP (the
    primary) from starting, and must never raise."""
    monkeypatch.setenv("TEN_FMP_API_KEY", "fmp-secret")
    monkeypatch.delenv("TEN_FINNHUB_API_KEY", raising=False)
    configs = (
        ProviderConfig(name="fmp", mode=ProviderMode.LIVE_PROVIDER, enabled=True, priority=10, api_key_env="TEN_FMP_API_KEY"),
        ProviderConfig(name="finnhub", mode=ProviderMode.LIVE_PROVIDER, enabled=True, priority=20, api_key_env="TEN_FINNHUB_API_KEY"),
    )
    providers = build_providers(configs)
    assert isinstance(providers[0], FinancialModelingPrepProvider)
    assert isinstance(providers[1], DisabledProvider)


def _status(name: str, *, enabled: bool, reachable: bool, connection_state: ConnectionState, failure_reason: str | None = None):
    from backend.app.engines.economic_calendar_engine.models import ProviderStatus

    return ProviderStatus(
        provider_name=name, mode=ProviderMode.LIVE_PROVIDER, enabled=enabled, reachable=reachable, connection_state=connection_state, failure_reason=failure_reason, last_success=NOW if reachable else None
    )


def test_snapshot_not_degraded_when_fallback_provider_covers_a_failed_primary() -> None:
    """A working fallback (Finnhub) covering for a failed primary (FMP) must never read as
    degraded — genuine unavailability only means NO configured provider produced data."""
    statuses = (
        _status("financial_modeling_prep", enabled=True, reachable=False, connection_state=ConnectionState.UNREACHABLE, failure_reason="connection refused"),
        _status("finnhub", enabled=True, reachable=True, connection_state=ConnectionState.CONNECTED),
    )
    snapshot = build_snapshot((), NOW, WINDOW_START, WINDOW_END, statuses, EconomicCalendarConfig())
    assert snapshot.degradation.is_degraded is False
    assert snapshot.degradation.category == "healthy"

    context = instrument_context("XAUUSD", snapshot, EconomicCalendarConfig())
    assert context.context_state in {CalendarContextState.NO_RELEVANT_EVENTS, CalendarContextState.OUTSIDE_RISK_WINDOW}
    assert context.unavailable_context == ()


@pytest.mark.parametrize(
    ("connection_state", "expected_category"),
    [
        (ConnectionState.UNREACHABLE, CalendarContextState.PROVIDER_UNREACHABLE),
        (ConnectionState.TIMEOUT, CalendarContextState.PROVIDER_TIMEOUT),
        (ConnectionState.UNAUTHORIZED, CalendarContextState.PROVIDER_AUTH_FAILED),
        (ConnectionState.RATE_LIMITED, CalendarContextState.PROVIDER_RATE_LIMITED),
    ],
)
def test_snapshot_degradation_category_matches_primary_providers_actual_failure(connection_state: ConnectionState, expected_category: CalendarContextState) -> None:
    """Every configured provider unreachable — the category must name the SPECIFIC failure the
    primary (highest-priority) provider actually hit, not a generic "unavailable" label."""
    statuses = (_status("financial_modeling_prep", enabled=True, reachable=False, connection_state=connection_state, failure_reason="boom"),)
    snapshot = build_snapshot((), NOW, WINDOW_START, WINDOW_END, statuses, EconomicCalendarConfig())
    assert snapshot.degradation.is_degraded is True
    assert snapshot.degradation.category == expected_category.value

    context = instrument_context("XAUUSD", snapshot, EconomicCalendarConfig())
    assert context.context_state == expected_category
    assert context.unavailable_context == ("boom",)


def test_stale_calendar_data_is_reported_as_unavailable_even_if_last_reachable() -> None:
    """A provider that WAS reachable a long time ago but hasn't synced since is not trustworthy
    "live" data — freshness must independently gate availability, not just the last connection
    outcome."""
    stale_status = _status("financial_modeling_prep", enabled=True, reachable=True, connection_state=ConnectionState.CONNECTED)
    stale_status = stale_status.model_copy(update={"last_success": NOW - timedelta(hours=6)})
    snapshot = build_snapshot((), NOW, WINDOW_START, WINDOW_END, (stale_status,), EconomicCalendarConfig())
    assert snapshot.freshness == FreshnessState.STALE
    context = instrument_context("XAUUSD", snapshot, EconomicCalendarConfig())
    assert context.context_state == CalendarContextState.NO_CALENDAR_DATA
    assert context.unavailable_context != ()
