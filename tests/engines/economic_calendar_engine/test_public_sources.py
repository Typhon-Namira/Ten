"""Keyless public-source economic calendar architecture — SSRF, circuit breaker, ICS/RSS/HTML
parsing, deterministic impact classification, dedup, risk windows, freshness, and the "no API key
anywhere in the active pipeline" guarantee. Every test here uses fixtures or a mocked
`httpx.MockTransport` — none make a real network call.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from backend.app.engines.economic_calendar_engine.analyzer import build_snapshot, instrument_context, materialize_risk_windows, reconcile
from backend.app.engines.economic_calendar_engine.config import EconomicCalendarConfig, ProviderConfig
from backend.app.engines.economic_calendar_engine.models import (
    CalendarContextState,
    ConnectionState,
    EventImportance,
    FreshnessState,
    ProviderMode,
    ProviderStatus,
    SourceType,
)
from backend.app.engines.economic_calendar_engine.normalization import normalize_observation
from backend.app.engines.economic_calendar_engine.providers import ProviderFetchRequest, build_providers
from backend.app.engines.economic_calendar_engine.public_sources import (
    ALLOWED_DOMAINS,
    PUBLIC_SOURCE_CLASSES,
    UnsafePublicUrlError,
    assert_safe_public_url,
    canonicalize_title,
    classify_impact,
)
from backend.app.engines.economic_calendar_engine.public_sources.base import RawEconomicEvent
from backend.app.engines.economic_calendar_engine.public_sources.bea import BeaPublicCalendarSource
from backend.app.engines.economic_calendar_engine.public_sources.bls import BlsPublicCalendarSource
from backend.app.engines.economic_calendar_engine.public_sources.circuit_breaker import CircuitBreakerState, FailureCategory
from backend.app.engines.economic_calendar_engine.public_sources.dol import DolWeeklyClaimsSource
from backend.app.engines.economic_calendar_engine.public_sources.html_dates import find_date_entries
from backend.app.engines.economic_calendar_engine.public_sources.ics import parse_ics_events

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


def _mounted(provider, handler) -> None:
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# 1. No API key is required.
# ---------------------------------------------------------------------------


def test_public_sources_require_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TEN_FMP_API_KEY", "TEN_FINNHUB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    configs = tuple(ProviderConfig(name=name, mode=ProviderMode.PUBLIC_WEB_SOURCE, enabled=True, priority=index) for index, name in enumerate(PUBLIC_SOURCE_CLASSES))
    providers = build_providers(configs)
    assert len(providers) == len(PUBLIC_SOURCE_CLASSES)
    for provider in providers:
        assert type(provider).__name__ != "DisabledProvider"


# ---------------------------------------------------------------------------
# 2. No FMP or Finnhub request is made when they are disabled.
# ---------------------------------------------------------------------------


def test_fmp_and_finnhub_are_never_constructed_when_disabled() -> None:
    configs = (
        ProviderConfig(name="fmp", mode=ProviderMode.LIVE_PROVIDER, enabled=False, priority=900, api_key_env="TEN_FMP_API_KEY"),
        ProviderConfig(name="finnhub", mode=ProviderMode.LIVE_PROVIDER, enabled=False, priority=910, api_key_env="TEN_FINNHUB_API_KEY"),
        ProviderConfig(name="bls", mode=ProviderMode.PUBLIC_WEB_SOURCE, enabled=True, priority=10),
    )
    providers = build_providers(configs)
    names = [type(item).__name__ for item in providers]
    # Disabled entries degrade to DisabledProvider, which never makes an HTTP request of any kind.
    assert names == ["DisabledProvider", "DisabledProvider", "BlsPublicCalendarSource"]


# ---------------------------------------------------------------------------
# 3. Public source fetch succeeds (BLS ICS).
# ---------------------------------------------------------------------------

_BLS_ICS_FIXTURE = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:cpi-2026-08-12@bls.gov
SUMMARY:Consumer Price Index
DTSTART:20260812T123000Z
DESCRIPTION:July 2026 CPI release
END:VEVENT
BEGIN:VEVENT
UID:nfp-2026-08-07@bls.gov
SUMMARY:Employment Situation
DTSTART;VALUE=DATE:20260807
END:VEVENT
END:VCALENDAR
"""


@pytest.mark.asyncio
async def test_bls_ics_source_fetch_succeeds() -> None:
    source = BlsPublicCalendarSource()
    _mounted(source, lambda request: httpx.Response(200, text=_BLS_ICS_FIXTURE))
    events = await source.fetch_schedule(date(2026, 7, 1), date(2026, 9, 1))
    assert {item.raw_name for item in events} == {"Consumer Price Index", "Employment Situation"}
    status = await source.health()
    assert status.connection_state == ConnectionState.CONNECTED
    assert status.source_type == SourceType.ICS_CALENDAR
    await source.close()


# ---------------------------------------------------------------------------
# 4. Static HTML parser extracts events.
# ---------------------------------------------------------------------------

_BEA_HTML_FIXTURE = """
<html><body>
<table>
<tr><td>August 27, 2026</td><td>Gross Domestic Product, 2nd Quarter 2026</td></tr>
<tr><td>August 29, 2026</td><td>Personal Income and Outlays, July 2026</td></tr>
<tr><td>September 3, 2026</td><td>Regional Personal Income (not XAUUSD-relevant)</td></tr>
</table>
</body></html>
"""


@pytest.mark.asyncio
async def test_bea_html_parser_extracts_relevant_events() -> None:
    source = BeaPublicCalendarSource()
    _mounted(source, lambda request: httpx.Response(200, text=_BEA_HTML_FIXTURE))
    events = await source.fetch_schedule(date(2026, 8, 1), date(2026, 9, 30))
    names = {item.raw_name for item in events}
    assert any("Gross Domestic Product" in name for name in names)
    assert any("Personal Income" in name for name in names)
    # "Regional Personal Income" is not in XAUUSD_RELEVANT_EVENT_TYPES — excluded.
    assert not any("Regional" in name for name in names)
    await source.close()


def test_find_date_entries_extracts_dates_from_various_table_layouts() -> None:
    html = "<ul><li>Jan 15, 2026 — Retail Sales</li><li>15 January 2026 duplicate should still be found once per block</li></ul>"
    entries = find_date_entries(html, start_date=date(2026, 1, 1), end_date=date(2026, 12, 31))
    assert any(entry.event_date == date(2026, 1, 15) for entry in entries)


# ---------------------------------------------------------------------------
# 5. JavaScript-rendered source uses browser fallback only when needed.
# ---------------------------------------------------------------------------


def test_no_browser_automation_dependency_is_required() -> None:
    """None of the six official sources required JavaScript rendering to expose their release
    schedule (ICS/RSS feeds and server-rendered government HTML) — so no Playwright/Selenium
    dependency was introduced. This is a deliberate architecture fact, asserted here so it doesn't
    silently regress: if a future source genuinely needs a browser, `PublicCalendarSource`'s
    protocol (`fetch_schedule`) does not preclude a browser-backed implementation, but none exists
    today and none of the current adapters import a browser automation library."""
    import backend.app.engines.economic_calendar_engine.public_sources as public_sources_pkg

    source_module = public_sources_pkg.__file__
    assert source_module is not None
    package_dir = os.path.dirname(source_module)
    for filename in os.listdir(package_dir):
        if filename.endswith(".py"):
            with open(os.path.join(package_dir, filename), encoding="utf-8") as handle:
                contents = handle.read()
            assert "playwright" not in contents.lower()
            assert "selenium" not in contents.lower()


# ---------------------------------------------------------------------------
# 6. One parser failure does not fail the engine.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_source_parser_failure_does_not_raise() -> None:
    source = BlsPublicCalendarSource()

    async def _boom(start_date: date, end_date: date) -> list[RawEconomicEvent]:
        raise ValueError("simulated parser bug")

    source.fetch_schedule = _boom  # type: ignore[method-assign]
    result = await source.fetch_events(ProviderFetchRequest(start=NOW - timedelta(days=1), end=NOW + timedelta(days=1)))
    assert result.observations == ()
    assert result.warnings
    status = await source.health()
    assert status.last_failure_category == "parser_mismatch"
    await source.close()


# ---------------------------------------------------------------------------
# 7 & 8. Times normalize to UTC; DST transitions are handled.
# ---------------------------------------------------------------------------


def test_ics_parser_normalizes_utc_and_tzid_datetimes() -> None:
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
SUMMARY:UTC Event
DTSTART:20260812T123000Z
END:VEVENT
BEGIN:VEVENT
SUMMARY:Eastern Event Winter
DTSTART;TZID=America/New_York:20260115T083000
END:VEVENT
BEGIN:VEVENT
SUMMARY:Eastern Event Summer DST
DTSTART;TZID=America/New_York:20260715T083000
END:VEVENT
END:VCALENDAR
"""
    events = parse_ics_events(ics)
    by_name = {item.summary: item for item in events}
    assert by_name["UTC Event"].dtstart == datetime(2026, 8, 12, 12, 30, tzinfo=UTC)
    # EST is UTC-5 in January (no DST) -> 08:30 local = 13:30 UTC.
    assert by_name["Eastern Event Winter"].dtstart == datetime(2026, 1, 15, 13, 30, tzinfo=UTC)
    # EDT is UTC-4 in July (DST active) -> 08:30 local = 12:30 UTC. Confirms DST is handled
    # correctly rather than applying a fixed UTC offset year-round.
    assert by_name["Eastern Event Summer DST"].dtstart == datetime(2026, 7, 15, 12, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_dol_deterministic_schedule_uses_timezone_aware_eastern_time() -> None:
    source = DolWeeklyClaimsSource()
    results = await source.fetch_schedule(date(2026, 7, 1), date(2026, 7, 31))
    assert results
    for item in results:
        parsed = datetime.fromisoformat(item.raw_scheduled_time)
        assert parsed.tzinfo is not None
        assert parsed.astimezone(UTC).tzinfo is UTC


# ---------------------------------------------------------------------------
# 9 & 10. Duplicate events from two sources merge; issuing agency wins conflicts.
# ---------------------------------------------------------------------------


def _config_with_priority(*names: str) -> EconomicCalendarConfig:
    return EconomicCalendarConfig(providers=tuple(ProviderConfig(name=name, mode=ProviderMode.PUBLIC_WEB_SOURCE, enabled=True) for name in names), provider_priority=names)


def test_duplicate_events_from_two_sources_merge_and_issuing_agency_wins() -> None:
    config = _config_with_priority("bls", "some_aggregator")
    from backend.app.engines.economic_calendar_engine.providers import observation_from_mapping

    scheduled = "2026-08-12T12:30:00Z"
    bls_row = {"provider_event_id": "bls-1", "raw_name": "Consumer Price Index", "raw_country": "US", "raw_currency": "USD", "raw_scheduled_time": scheduled, "raw_timezone": "UTC"}
    aggregator_row = {"provider_event_id": "agg-1", "raw_name": "US Consumer Price Index (CPI) release", "raw_country": "US", "raw_currency": "USD", "raw_scheduled_time": scheduled, "raw_timezone": "UTC"}
    bls_obs = observation_from_mapping("bls", "1", bls_row, NOW)
    agg_obs = observation_from_mapping("some_aggregator", "1", aggregator_row, NOW)
    bls_event = normalize_observation(bls_obs, config)
    agg_event = normalize_observation(agg_obs, config)
    # Same canonical event type + country + scheduled date -> identical dedup key, regardless of
    # the free-text title difference ("Consumer Price Index" vs "CPI (US)").
    assert bls_event.event_id == agg_event.event_id

    reconciled = reconcile((bls_event, agg_event), config.provider_priority)
    assert len(reconciled) == 1
    # "bls" is listed first in provider_priority -> the issuing agency's own record wins.
    assert reconciled[0].metadata["provider"] == "bls"


# ---------------------------------------------------------------------------
# 11. High-impact XAUUSD events are classified correctly; 12. low-relevance events excluded.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected_type", "expected_impact"),
    [
        ("FOMC Rate Decision", "fomc_rate_decision", "critical"),
        ("Consumer Price Index", "cpi", "high"),
        ("Employment Situation", "nonfarm_payrolls", "high"),
        ("Personal Consumption Expenditures", "pce", "high"),
        ("New Residential Sales", "new_home_sales", "medium"),
        ("Some Obscure Regional Survey", "some_obscure_regional_survey", "low"),
    ],
)
def test_impact_classification_is_deterministic(title: str, expected_type: str, expected_impact: str) -> None:
    canonical = canonicalize_title(title)
    assert canonical == expected_type
    assert classify_impact(canonical) == expected_impact


# ---------------------------------------------------------------------------
# 13. Risk windows are generated correctly.
# ---------------------------------------------------------------------------


def test_risk_windows_are_generated_with_correct_bounds() -> None:
    config = EconomicCalendarConfig()
    from backend.app.engines.economic_calendar_engine.providers import observation_from_mapping

    obs = observation_from_mapping("bls", "1", {"provider_event_id": "1", "raw_name": "Consumer Price Index", "raw_country": "US", "raw_currency": "USD", "raw_importance": "high", "raw_scheduled_time": "2026-08-12T12:30:00Z", "raw_timezone": "UTC"}, NOW)
    event = normalize_observation(obs, config)
    windows = materialize_risk_windows((event,), config, schedule_freshness=FreshnessState.FRESH)
    assert len(windows) == 1
    window = windows[0]
    high_bounds = config.windows["high"]
    assert window.window_start == event.scheduled_at_utc - timedelta(minutes=high_bounds.pre_minutes)
    assert window.window_end == event.scheduled_at_utc + timedelta(minutes=high_bounds.post_minutes + high_bounds.cooldown_minutes)
    assert window.impact == EventImportance.HIGH
    assert window.schedule_freshness == FreshnessState.FRESH


# ---------------------------------------------------------------------------
# 14, 15, 16. Cached schedules survive temporary failures; stale schedules eventually expire;
# missing actual/forecast never invalidates an upcoming schedule.
# ---------------------------------------------------------------------------


def _status(name: str, *, enabled: bool, reachable: bool, connection_state: ConnectionState, last_success: datetime | None = None) -> ProviderStatus:
    return ProviderStatus(provider_name=name, mode=ProviderMode.PUBLIC_WEB_SOURCE, enabled=enabled, reachable=reachable, connection_state=connection_state, last_success=last_success, source_type=SourceType.ICS_CALENDAR)


def test_cached_schedule_survives_a_temporary_source_failure() -> None:
    config = EconomicCalendarConfig()
    from backend.app.engines.economic_calendar_engine.providers import observation_from_mapping

    obs = observation_from_mapping("bls", "1", {"provider_event_id": "1", "raw_name": "Consumer Price Index", "raw_country": "US", "raw_currency": "USD", "raw_scheduled_time": "2026-08-12T12:30:00Z", "raw_timezone": "UTC"}, NOW)
    event = normalize_observation(obs, config)
    failing_status = _status("bls", enabled=True, reachable=False, connection_state=ConnectionState.UNREACHABLE)
    snapshot = build_snapshot((event,), NOW, NOW - timedelta(days=1), NOW + timedelta(days=30), (failing_status,), config)
    assert snapshot.degradation.is_degraded is True
    assert snapshot.event_count == 1  # the previously-fetched event is still present
    assert snapshot.available_from_cached_schedule is True


def test_stale_schedule_eventually_becomes_unavailable() -> None:
    config = EconomicCalendarConfig()
    old_status = _status("bls", enabled=True, reachable=True, connection_state=ConnectionState.CONNECTED, last_success=NOW - timedelta(days=5))
    snapshot = build_snapshot((), NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), (old_status,), config)
    assert snapshot.freshness == FreshnessState.CRITICAL
    context = instrument_context("XAUUSD", snapshot, config)
    assert context.context_state == CalendarContextState.NO_CALENDAR_DATA
    assert context.unavailable_context != ()


def test_missing_actual_and_forecast_do_not_invalidate_an_upcoming_schedule() -> None:
    config = EconomicCalendarConfig()
    from backend.app.engines.economic_calendar_engine.providers import observation_from_mapping

    obs = observation_from_mapping(
        "bls", "1", {"provider_event_id": "1", "raw_name": "Consumer Price Index", "raw_country": "US", "raw_currency": "USD", "raw_scheduled_time": "2026-08-12T12:30:00Z", "raw_timezone": "UTC"}, NOW
    )
    event = normalize_observation(obs, config)
    assert event.schedule_available is True
    assert event.result_available is False
    assert event.forecast_available is False
    fresh_status = _status("bls", enabled=True, reachable=True, connection_state=ConnectionState.CONNECTED, last_success=NOW)
    snapshot = build_snapshot((event,), NOW, NOW - timedelta(days=1), NOW + timedelta(days=30), (fresh_status,), config)
    context = instrument_context("XAUUSD", snapshot, config)
    assert context.context_state not in {CalendarContextState.NO_CALENDAR_DATA, CalendarContextState.PROVIDER_UNREACHABLE}
    assert context.unavailable_context == ()


# ---------------------------------------------------------------------------
# 17. HTTP 404 activates source-specific backoff.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_404_activates_the_circuit_breaker() -> None:
    source = BlsPublicCalendarSource()
    _mounted(source, lambda request: httpx.Response(404, text="Not Found"))
    await source.fetch_schedule(date(2026, 7, 1), date(2026, 9, 1))
    assert source._circuit.should_attempt(NOW + timedelta(seconds=1)) is False
    assert source._circuit.last_failure_category == FailureCategory.HTTP_ERROR
    status = await source.health()
    assert status.connection_state == ConnectionState.INVALID_ENDPOINT
    assert status.reachable is True  # a 404 proves the server was reached
    assert status.circuit_breaker_open is True
    await source.close()


def test_circuit_breaker_backs_off_longer_for_structural_failures() -> None:
    breaker = CircuitBreakerState()
    now = NOW
    breaker.record_failure(now, FailureCategory.NETWORK_ERROR)
    transient_backoff = breaker.open_until
    breaker.record_success(now)
    breaker.record_failure(now, FailureCategory.PARSER_MISMATCH)
    structural_backoff = breaker.open_until
    assert transient_backoff is not None and structural_backoff is not None
    assert (structural_backoff - now) > (transient_backoff - now)


# ---------------------------------------------------------------------------
# 18. Parser layout mismatch is surfaced clearly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_schedule_from_a_successful_fetch_is_flagged_as_a_layout_change() -> None:
    """A 200 response that parses to zero events is a DIFFERENT, more suspicious signal than a
    fetch failure — most likely the page's layout changed under the parser."""
    source = BeaPublicCalendarSource()
    _mounted(source, lambda request: httpx.Response(200, text="<html><body>No recognizable content here</body></html>"))
    result = await source.fetch_events(ProviderFetchRequest(start=NOW - timedelta(days=1), end=NOW + timedelta(days=1)))
    assert result.warnings == ("empty_valid_schedule",)
    await source.close()


# ---------------------------------------------------------------------------
# 19. No secret appears in logs — covered directly in test_fmp_provider.py /
# test_fmp_market_data_provider.py (API-key redaction) and tests/core/test_logging_setup.py
# (httpx's own request-line logging suppression). Public sources have no secret to leak at all.
# ---------------------------------------------------------------------------


def test_public_sources_have_no_credential_to_leak() -> None:
    for source_class in PUBLIC_SOURCE_CLASSES.values():
        assert not hasattr(source_class, "api_key")


# ---------------------------------------------------------------------------
# 20. SSRF protections reject private and arbitrary URLs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/schedule.ics",
        "http://127.0.0.1/schedule.ics",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.5/internal",
        "file:///etc/passwd",
        "https://evil.example.com/bls.ics",  # not on the allowlist, even though HTTPS
        "ftp://www.bls.gov/schedule.ics",  # disallowed scheme
    ],
)
def test_ssrf_protection_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(UnsafePublicUrlError):
        assert_safe_public_url(url)


def test_ssrf_protection_allows_every_configured_official_domain() -> None:
    for domain in ALLOWED_DOMAINS:
        assert_safe_public_url(f"https://{domain}/schedule")


def test_public_source_construction_rejects_a_non_allowlisted_url() -> None:
    with pytest.raises(UnsafePublicUrlError):
        BlsPublicCalendarSource(source_url="https://not-bls.example.com/schedule.ics")


# ---------------------------------------------------------------------------
# 22. Economic context becomes available from a fresh public schedule.
# ---------------------------------------------------------------------------


def test_economic_context_available_from_a_fresh_public_schedule() -> None:
    config = EconomicCalendarConfig()
    from backend.app.engines.economic_calendar_engine.providers import observation_from_mapping

    obs = observation_from_mapping("bls", "1", {"provider_event_id": "1", "raw_name": "Consumer Price Index", "raw_country": "US", "raw_currency": "USD", "raw_scheduled_time": "2026-08-12T12:30:00Z", "raw_timezone": "UTC"}, NOW)
    event = normalize_observation(obs, config)
    fresh_status = _status("bls", enabled=True, reachable=True, connection_state=ConnectionState.CONNECTED, last_success=NOW)
    snapshot = build_snapshot((event,), NOW, NOW - timedelta(days=1), NOW + timedelta(days=30), (fresh_status,), config)
    assert snapshot.degradation.is_degraded is False
    context = instrument_context("XAUUSD", snapshot, config)
    assert context.unavailable_context == ()


# ---------------------------------------------------------------------------
# 26. No recurring task creates duplicate fetches.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_starting_the_service_twice_does_not_create_two_scheduler_tasks() -> None:
    from backend.app.engines.economic_calendar_engine.service import EconomicCalendarService
    from backend.app.events import InMemoryEventBus
    from backend.app.features import InMemoryFeatureStore

    config = _config_with_priority("bls")
    service = EconomicCalendarService(InMemoryEventBus(), InMemoryFeatureStore(), config, providers=build_providers(config.providers))
    await service.start()
    first_task = service._scheduler
    await service.start()
    assert service._scheduler is first_task  # the guard (`if self._scheduler is None`) held
    await service.stop()
