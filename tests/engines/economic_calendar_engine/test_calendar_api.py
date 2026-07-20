from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI

from backend.app.api.routes.economic_calendar import router
from backend.app.engines.economic_calendar_engine import (
    EconomicCalendarConfig,
    EconomicCalendarService,
    FixedClock,
    InMemoryProvider,
    ProviderConfig,
    ProviderMode,
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
    stage healthy — proving 'degraded' isn't reported just because *some* symbol elsewhere
    might have no relevant events, and proving each stage is genuinely independent."""
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

        # An irrelevant symbol has no matching currency: relevant/trading_context degrade
        # independently while provider_health/downloaded_events/mapped_events stay healthy —
        # proving the five stages don't collapse into one shared boolean.
        unrelated = await client.get("/economic-calendar/diagnostics", params={"symbol": "GBPCHF"})
        unrelated_body = unrelated.json()
        assert unrelated_body["provider_health"]["status"] == "healthy"
        assert unrelated_body["relevant_events"]["status"] == "none_relevant"
        assert unrelated_body["trading_context"]["status"] == "unavailable"
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
