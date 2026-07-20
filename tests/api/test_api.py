from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.integration import CanonicalEventEnvelope
from backend.app.main import create_app


def test_unhandled_exception_on_get_degrades_to_200_not_500() -> None:
    """No observability GET endpoint may ever return HTTP 500 — an unguarded, unexpected
    exception (here a plain RuntimeError raised somewhere a repository call was never wrapped
    in a try/except) must degrade to a graceful `status: "error"` body at 200 instead of an
    opaque server error, per the app-wide exception handler in `create_app()`."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        service = client.app.state.market_data_service
        service.repository.candle_at = AsyncMock(side_effect=RuntimeError("simulated repository failure"))
        response = client.get("/market/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["exception_class"] == "RuntimeError"
    assert body["path"] == "/market/status"


def test_unhandled_exception_on_post_still_fails_closed() -> None:
    """The never-500 safety net only widens GET (read-only, observability) endpoints. A
    mutation endpoint hitting the same kind of unguarded exception must keep failing closed,
    since this app's exception-handling carve-out is scoped to observability, not writes."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        service = client.app.state.ai_scoring_service
        service.calculate = AsyncMock(side_effect=RuntimeError("simulated engine failure"))
        response = client.post("/ai-scoring/score", json={"instrument": "XAUUSD", "timeframe": "M15"})
    assert response.status_code == 500


def test_health_and_status_endpoints() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        engines = client.get("/engines/status")
        market = client.get("/market/status")
        diagnostics = client.get("/api/v1/system/diagnostics")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert engines.status_code == 200 and len(engines.json()) == 11
    assert market.status_code == 200 and market.json()["symbol"] == "XAUUSD"
    assert market.json()["market_status"] in {"OPEN", "CLOSED_WEEKEND", "CLOSED_DAILY_BREAK", "HOLIDAY_OR_PROVIDER_CLOSED", "UNKNOWN"}
    assert diagnostics.status_code == 200
    payload = diagnostics.json()
    assert payload["replay"]["status"] == "disabled"
    assert payload["workers"]["market_data_worker"]["enabled"] is False
    assert "api_key" not in str(payload).lower()
    assert "database_url" not in str(payload).lower()


def test_dashboard_endpoints_default_to_the_configured_primary_timeframe_not_a_hardcoded_m15() -> None:
    """Regression test: `/market-intelligence`, `/pipeline/stages/latest`, `/performance`, and
    `/market/status` used to default their instrument/timeframe query params to hardcoded
    "XAUUSD"/"M15" literals — so a deployment configured for a different primary timeframe (e.g.
    M1) left every dashboard-facing endpoint silently querying a candle series the pipeline never
    actually produces data for, while the pipeline's own events/logs were for the real (M1)
    series. `/api/v1/system/selection` is the one authoritative source every endpoint (and the
    frontend) must agree with instead of each hardcoding its own default."""
    settings = Settings(market_data_symbols=("XAUUSD",), market_data_timeframes=("M1",))
    with TestClient(create_app(settings_override=settings)) as client:
        selection = client.get("/api/v1/system/selection")
        assert selection.status_code == 200
        assert selection.json() == {"instrument": "XAUUSD", "timeframe": "M1", "configured_instruments": ["XAUUSD"], "configured_timeframes": ["M1"]}

        market_intelligence = client.get("/api/v1/system/market-intelligence")
        assert market_intelligence.status_code == 200
        assert market_intelligence.json()["timeframe"] == "M1"

        stages = client.get("/api/v1/pipeline/stages/latest")
        assert stages.status_code == 200
        assert stages.json()["timeframe"] == "M1"

        performance = client.get("/api/v1/system/performance")
        assert performance.status_code == 200
        assert performance.json()["timeframe"] == "M1"

        market_status = client.get("/market/status")
        assert market_status.status_code == 200
        assert market_status.json()["symbol"] == "XAUUSD"

        diagnostics = client.get("/api/v1/system/diagnostics")
        assert diagnostics.status_code == 200
        assert diagnostics.json()["market"]["timeframe"] == "M1"

        # Explicit query params still override the configured default.
        overridden = client.get("/api/v1/system/market-intelligence", params={"timeframe": "M15"})
        assert overridden.json()["timeframe"] == "M15"


def test_market_intelligence_differentiates_economic_status_into_five_stages() -> None:
    """Regression test: the dashboard's economic status widget and the signal decision engine's
    `economic_context_unavailable` blocker used to trace back to the same underlying fact (no
    relevant event mapping right now) but render as two unrelated-looking states with no way to
    tell whether the provider itself is actually down. `economic_status.stages` must always be
    present with all five independently-reported stages, not collapsed into the single `degraded`
    boolean."""
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/system/market-intelligence")
    assert response.status_code == 200
    stages = response.json()["economic_status"]["stages"]
    assert stages is not None
    assert set(stages) == {"provider_health", "downloaded_events", "mapped_events", "relevant_events", "trading_context"}


def test_performance_reports_in_flight_latency_instead_of_null_while_a_cycle_is_running() -> None:
    """Regression test: `pipeline_latency_ms` only ever reflected a COMPLETED cycle's duration, so
    it read as unavailable ("--" in the UI) exactly when the pipeline was busiest — a cycle that
    has started but not yet finished has no completed duration, but "elapsed so far" is exactly
    the number an operator wants at that moment."""
    with TestClient(create_app()) as client:
        app = client.app
        settings = app.state.settings
        instrument, timeframe = settings.market_data_symbols[0].upper(), settings.market_data_timeframes[0]
        app.state.pipeline_stage_tracker.begin(instrument, timeframe, datetime(2026, 1, 1, tzinfo=UTC))
        response = client.get("/api/v1/system/performance")
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_latency_ms"] is None
    assert body["pipeline_in_flight_ms"] is not None
    assert body["pipeline_in_flight_ms"] >= 0


@pytest.mark.asyncio
async def test_performance_reports_queue_backlog_age_when_nothing_has_started_processing_yet() -> None:
    """Regression test: `queue_length` (outbox backlog) could be non-zero while
    `pipeline_latency_ms` stayed null forever if the worker hadn't picked anything up yet —
    `queue_oldest_pending_age_seconds` must be populated whenever backlog exists, independent of
    whether the stage tracker has any cycle recorded at all."""
    app = create_app()
    async with app.router.lifespan_context(app):
        settings = app.state.settings
        instrument, timeframe = settings.market_data_symbols[0].upper(), settings.market_data_timeframes[0]
        candle = Candle(timestamp=datetime(2026, 1, 1, tzinfo=UTC), symbol=instrument, timeframe=Timeframe(timeframe), open=1, high=2, low=1, close=1.5, volume=10, provider="test")
        envelope = CanonicalEventEnvelope.final_candle(candle, uuid4(), datetime.now(UTC))
        await app.state.integration_repository.enqueue(envelope)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/system/performance")
    assert response.status_code == 200
    body = response.json()
    assert body["queue_length"] >= 1
    assert body["queue_oldest_pending_age_seconds"] is not None
    assert body["queue_oldest_pending_age_seconds"] >= 0


def test_signal_endpoints_start_empty() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/signals").json() == []
        assert client.get("/signals/latest").status_code == 404
