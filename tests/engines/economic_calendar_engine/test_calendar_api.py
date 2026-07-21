from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI

from backend.app.api.routes.economic_calendar import router
from backend.app.engines.economic_calendar_engine import (
    ConnectionState,
    EconomicCalendarConfig,
    EconomicCalendarService,
    FixedClock,
    InMemoryProvider,
    ProviderConfig,
    ProviderMode,
    ProviderStatus,
)
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore

NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)


async def application() -> tuple[FastAPI, EconomicCalendarService]:
    config = EconomicCalendarConfig(
        providers=(ProviderConfig(name="fixture", mode=ProviderMode.IN_MEMORY_TEST_PROVIDER, enabled=True),), provider_priority=("fixture",)
    )
    rows = (
        {
            "id": "cpi",
            "name": "CPI",
            "country": "US",
            "currency": "USD",
            "category": "inflation",
            "importance": "high",
            "scheduled_at": (NOW + timedelta(minutes=5)).isoformat(),
            "available_at": (NOW - timedelta(hours=1)).isoformat(),
            "response_received_at": (NOW - timedelta(hours=1)).isoformat(),
        },
    )
    service = EconomicCalendarService(InMemoryEventBus(), InMemoryFeatureStore(), config, providers=(InMemoryProvider("fixture", rows),), clock=FixedClock(NOW))
    await service.restore()
    await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1))
    app = FastAPI()
    app.include_router(router)
    app.state.economic_calendar_service = service
    return app, service


@pytest.mark.asyncio
async def test_all_read_only_routes_and_safe_validation() -> None:
    app, service = await application()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path in (
            "health",
            "config",
            "metrics",
            "providers",
            "diagnostics",
            "events",
            "upcoming",
            "recent",
            "active",
            "snapshot",
            "history",
            "context",
            "context/XAUUSD",
            "clusters",
            "conflicts",
            "explanations",
        ):
            response = await client.get(f"/economic-calendar/{path}")
            assert response.status_code == 200, (path, response.text)
        event_id = str((await service.events())[0].event_id)
        for suffix in ("", "/revisions", "/observations"):
            assert (await client.get(f"/economic-calendar/events/{event_id}{suffix}")).status_code == 200
        assert (await client.get("/economic-calendar/events/not-a-uuid")).status_code == 422


@pytest.mark.asyncio
async def test_diagnostics_reports_five_independent_stages() -> None:
    """A synced, reachable provider with a relevant USD event scheduled must report every
    stage healthy — proving 'unavailable' isn't reported just because *some* symbol elsewhere
    has no relevant events right now, and proving each stage is genuinely independent."""
    app, _ = await application()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/economic-calendar/diagnostics", params={"symbol": "XAUUSD"})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"provider_health", "downloaded_events", "mapped_events", "relevant_events", "trading_context"}
        assert body["provider_health"]["status"] == "healthy"
        assert body["downloaded_events"]["status"] == "ok" and body["downloaded_events"]["count"] >= 1
        assert body["mapped_events"]["status"] == "ok"
        assert body["relevant_events"]["status"] == "available"
        assert body["trading_context"]["status"] == "ready"

        # An irrelevant symbol has no matching currency: `relevant_events` reports it while every
        # other stage — including `trading_context` — stays healthy, since "no news for this
        # symbol right now" is the routine, expected state and must never be conflated with the
        # provider/calendar sync actually being unavailable (a real failure). Regression coverage
        # for a bug where `trading_context` used to go "unavailable" for every symbol with no
        # imminent news, which permanently HARD_BLOCKed decisions even with a fully healthy,
        # live-syncing provider.
        unrelated = await client.get("/economic-calendar/diagnostics", params={"symbol": "GBPCHF"})
        unrelated_body = unrelated.json()
        assert unrelated_body["provider_health"]["status"] == "healthy"
        assert unrelated_body["relevant_events"]["status"] == "none_relevant"
        assert unrelated_body["trading_context"]["status"] == "ready"
        assert (await client.get(f"/economic-calendar/events/{'0' * 8}-0000-0000-0000-000000000000")).status_code == 404
        assert (await client.get("/economic-calendar/context/../../bad")).status_code in {404, 422}
        assert (await client.get("/economic-calendar/events", params={"country": "USA"})).status_code == 422
        assert (await client.get("/economic-calendar/events", params={"currency": "US"})).status_code == 422
        assert (
            await client.get("/economic-calendar/events", params={"start": NOW.isoformat(), "end": (NOW - timedelta(days=1)).isoformat()})
        ).status_code == 422
        assert (
            await client.get(
                "/economic-calendar/events", params={"start": (NOW - timedelta(days=200)).isoformat(), "end": (NOW + timedelta(days=200)).isoformat()}
            )
        ).status_code == 422
        assert (await client.post("/economic-calendar/events")).status_code == 405
        payload = (await client.get("/economic-calendar/config")).json()
        assert "file_path" not in payload["configuration"]["providers"][0]
        assert payload["configuration"]["providers"][0].get("token") is None


class _AlwaysFailingProvider:
    """A minimal `EconomicCalendarProvider` double that simulates one public source being
    completely unreachable — used to prove the dashboard stays queryable (HTTP 200) even while a
    source is down, per regression item 21."""

    def __init__(self, name: str = "always_failing") -> None:
        self.name, self.version, self.timezone, self.mode = name, "1", "UTC", ProviderMode.PUBLIC_WEB_SOURCE
        self.capabilities = None  # type: ignore[assignment]

    async def fetch_events(self, request):  # type: ignore[no-untyped-def]
        raise ConnectionError("simulated source outage")

    async def health(self) -> ProviderStatus:
        return ProviderStatus(provider_name=self.name, mode=self.mode, enabled=True, reachable=False, connection_state=ConnectionState.UNREACHABLE, failure_reason="simulated source outage")


@pytest.mark.asyncio
async def test_dashboard_endpoints_stay_available_when_one_source_is_down() -> None:
    config = EconomicCalendarConfig(
        providers=(ProviderConfig(name="fixture", mode=ProviderMode.IN_MEMORY_TEST_PROVIDER, enabled=True), ProviderConfig(name="always_failing", mode=ProviderMode.PUBLIC_WEB_SOURCE, enabled=True)),
        provider_priority=("fixture", "always_failing"),
    )
    rows = (
        {
            "id": "cpi",
            "name": "CPI",
            "country": "US",
            "currency": "USD",
            "category": "inflation",
            "importance": "high",
            "scheduled_at": (NOW + timedelta(minutes=5)).isoformat(),
            "available_at": (NOW - timedelta(hours=1)).isoformat(),
            "response_received_at": (NOW - timedelta(hours=1)).isoformat(),
        },
    )
    service = EconomicCalendarService(
        InMemoryEventBus(), InMemoryFeatureStore(), config, providers=(InMemoryProvider("fixture", rows), _AlwaysFailingProvider()), clock=FixedClock(NOW)
    )
    await service.restore()
    await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1))
    app = FastAPI()
    app.include_router(router)
    app.state.economic_calendar_service = service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ("health", "providers", "diagnostics", "events", "snapshot", "context"):
            response = await client.get(f"/economic-calendar/{path}")
            assert response.status_code == 200, (path, response.text)
        providers_payload = (await client.get("/economic-calendar/providers")).json()
        by_name = {item["provider_name"]: item for item in providers_payload}
        assert by_name["always_failing"]["reachable"] is False
        assert by_name["fixture"]["reachable"] is True
        # The healthy fixture source's data still surfaces even though the other source is down.
        events_payload = (await client.get("/economic-calendar/events")).json()
        assert len(events_payload) >= 1

        layered = await service.layered_health()
        assert layered["engine_readiness"] == "ready"
        assert layered["is_degraded"] is True
        assert layered["degraded_reason"] == "one_optional_source_failed"
