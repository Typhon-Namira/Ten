from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.app.api.dependencies import get_market_regime_service
from backend.app.api.routes.market_regime import router
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_regime_engine import BaselineMarketRegimeAnalyzer, MarketRegimeConfig, MarketRegimeContext


BASE = datetime(2026, 7, 1, tzinfo=UTC)


def snapshot() -> object:
    values = tuple(Candle(symbol="XAU/USD", timeframe=Timeframe.M15, timestamp=BASE + timedelta(minutes=15 * index), ingestion_timestamp=BASE + timedelta(minutes=15 * index), open=3300 + index, high=3301.5 + index, low=3299.5 + index, close=3301 + index, volume=100) for index in range(30))
    return BaselineMarketRegimeAnalyzer().analyze_snapshot(MarketRegimeContext(values))


class Repository:
    def __init__(self, value: object) -> None:
        self.value = value

    async def get_snapshot(self, identifier: object) -> object | None:
        return self.value if str(identifier) == str(self.value.snapshot_id) else None

    async def list_transitions(self, *_: object) -> tuple[object, ...]:
        return ()

    async def list_evidence(self, *_: object) -> tuple[object, ...]:
        return self.value.evidence


class Service:
    def __init__(self, value: object) -> None:
        self.value = value
        self.repository = Repository(value)
        self.config = MarketRegimeConfig()
        self.analyzer = SimpleNamespace(version="1.0.0")
        self.metrics = SimpleNamespace(snapshot=lambda: {"analysis_count": 1})

    def health(self) -> dict[str, object]:
        return {"status": "healthy", "probabilistic_inference": True, "trading_instruction": False}

    async def state(self, *_: object) -> object:
        return self.value

    async def history(self, *_: object) -> tuple[object, ...]:
        return (self.value,)

    async def multi_timeframe(self, *_: object) -> object:
        return self.value.multi_timeframe


@pytest.mark.asyncio
async def test_all_market_regime_endpoints_are_read_only_typed_and_bounded() -> None:
    value = snapshot()
    service = Service(value)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_market_regime_service] = lambda: service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        paths = ("/health", "/config", "/metrics", "/state", "/history", "/trend", "/volatility", "/auction", "/compression", "/expansion", "/transitions", "/persistence", "/sessions", "/mtf", "/evidence", "/explanations")
        for path in paths:
            response = await client.get("/market-regime" + path)
            assert response.status_code == 200, (path, response.text)
        found = await client.get(f"/market-regime/snapshots/{value.snapshot_id}")
        assert found.status_code == 200
        assert found.json()["probabilistic_inference"] is True
        assert found.json()["trading_instruction"] is False
        assert (await client.get("/market-regime/state?timeframe=BAD")).status_code == 422
        assert (await client.get("/market-regime/state?timestamp=bad")).status_code == 422
        assert (await client.get("/market-regime/history?limit=1001")).status_code == 422
        assert (await client.get("/market-regime/evidence?limit=1001")).status_code == 422
        assert (await client.get("/market-regime/snapshots/not-a-uuid")).status_code == 404
        assert (await client.post("/market-regime/state")).status_code == 405
