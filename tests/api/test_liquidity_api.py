import asyncio
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_liquidity_service
from backend.app.engines.liquidity_engine import InMemoryLiquidityRepository, LiquidityContext, LiquidityService
from backend.app.engines.liquidity_engine.analyzer import BaselineLiquidityAnalyzer
from backend.app.engines.market_data_engine import Timeframe
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.main import create_app


class Market:
    def __init__(self, candles: list[Any]) -> None:
        self.candles = candles
        self.sessions = BaselineLiquidityAnalyzer().sessions

    async def history(self, *_: object, **__: object) -> list[Any]:
        return self.candles

    async def replay(self, _symbol: str, _timeframe: Timeframe, timestamp: object, **_: object) -> list[Any]:
        return self.candles


class SMC:
    async def liquidity_context(self, *_: object) -> None:
        return None


def test_all_liquidity_read_endpoints_and_bounds(candles: list[Any]) -> None:
    service = LiquidityService(
        cast(Any, Market(candles)), cast(Any, SMC()), InMemoryEventBus(), InMemoryFeatureStore(), repository=InMemoryLiquidityRepository()
    )
    snapshot = asyncio.run(service.analyze_context(LiquidityContext(tuple(candles))))
    app = create_app()
    app.dependency_overrides[get_liquidity_service] = lambda: service
    timestamp = snapshot.analysis_timestamp.isoformat()
    with TestClient(app) as client:
        paths = (
            "state",
            "snapshot",
            "levels",
            "equal-levels",
            "pools",
            "events",
            "sweeps",
            "grabs",
            "raids",
            "stop-hunts",
            "false-breaks",
            "sessions",
            "reference-levels",
            "confluences",
            "targets",
            "map",
            "mtf",
            "health",
            "metrics",
            "config",
        )
        assert all(client.get(f"/liquidity/{path}").status_code == 200 for path in paths)
        assert client.get("/liquidity/replay", params={"timestamp": timestamp}).status_code == 200
        assert client.get("/liquidity/pools", params={"offset": 1, "limit": 1}).status_code == 200
        pools = client.get("/liquidity/pools").json()
        assert pools
        selected = pools[0]
        filtered = client.get(
            "/liquidity/pools",
            params={
                "side": selected["side"],
                "scope": selected["scope"],
                "type": selected["pool_type"],
                "lifecycle_state": selected["lifecycle_state"],
                "min_confidence": selected["confidence_score"],
                "min_strength": selected["strength_score"],
                "active_only": "true",
            },
        )
        assert filtered.status_code == 200
        assert all(item["side"] == selected["side"] for item in filtered.json())
        assert client.get("/liquidity/levels", params={"source": "not-a-source"}).json() == []
        assert client.get("/liquidity/pools", params={"limit": 5001}).status_code == 422
        assert client.get("/liquidity/pools", params={"min_strength": 101}).status_code == 422
        assert client.post("/liquidity/pools").status_code == 405
        health = client.get("/liquidity/health").json()
        assert health["status"] == "degraded" and health["repository_mode"] == "memory"
        assert "database_url" not in client.get("/liquidity/config").text.lower()


def test_liquidity_source_degradation_is_503() -> None:
    class EmptyMarket(Market):
        async def history(self, *_: object, **__: object) -> list[Any]:
            raise RuntimeError("unavailable")

    service = LiquidityService(cast(Any, EmptyMarket([])), cast(Any, SMC()), InMemoryEventBus(), InMemoryFeatureStore())
    app = create_app()
    app.dependency_overrides[get_liquidity_service] = lambda: service
    with TestClient(app) as client:
        response = client.get("/liquidity/snapshot")
    assert response.status_code == 503 and response.json()["detail"] == "Liquidity source data is unavailable"
