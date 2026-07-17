import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_smc_service
from backend.app.engines.smc_engine import SMCService
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.main import create_app


def test_all_smc_read_endpoints_and_validation(candles: list[Any]) -> None:
    market = SimpleNamespace(history=AsyncMock(return_value=candles), replay=AsyncMock(return_value=candles))
    service = SMCService(cast(Any, market), InMemoryEventBus(), InMemoryFeatureStore())
    snapshot = asyncio.run(service.analyze_candles(candles))
    app = create_app()
    app.dependency_overrides[get_smc_service] = lambda: service
    timestamp = snapshot.analysis_timestamp.isoformat()
    with TestClient(app) as client:
        paths = (
            "/smc/state",
            "/smc/swings",
            "/smc/structure",
            "/smc/events",
            "/smc/snapshot",
            "/smc/health",
            "/smc/metrics",
            "/smc/config",
        )
        assert all(client.get(path).status_code == 200 for path in paths)
        assert client.get("/smc/replay", params={"timestamp": timestamp}).status_code == 200
        assert client.get("/smc/swings?minimum_confidence=101").status_code == 422
        assert client.get("/smc/events?limit=5001").status_code == 422
