from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

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
    assert market.status_code == 200 and market.json()["symbol"] == "XAU/USD"
    assert market.json()["market_status"] in {"OPEN", "CLOSED_WEEKEND", "CLOSED_DAILY_BREAK", "HOLIDAY_OR_PROVIDER_CLOSED", "UNKNOWN"}
    assert diagnostics.status_code == 200
    payload = diagnostics.json()
    assert payload["replay"]["status"] == "disabled"
    assert payload["workers"]["market_data_worker"]["enabled"] is False
    assert "api_key" not in str(payload).lower()
    assert "database_url" not in str(payload).lower()


def test_signal_endpoints_start_empty() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/signals").json() == []
        assert client.get("/signals/latest").status_code == 404
