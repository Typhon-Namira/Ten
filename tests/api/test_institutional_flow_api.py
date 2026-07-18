from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_institutional_flow_service
from backend.app.api.routes.institutional_flow import router
from backend.app.engines.institutional_flow_engine import BaselineInstitutionalFlowAnalyzer, InstitutionalFlowConfig, InstitutionalFlowContext
from backend.app.engines.market_data_engine import Candle, Timeframe


def make_snapshot() -> object:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    candles = tuple(
        Candle(
            symbol="XAU/USD",
            timeframe=Timeframe.M15,
            timestamp=start + timedelta(minutes=15 * index),
            open=3300 + index,
            high=3301.5 + index,
            low=3299.5 + index,
            close=3301 + index,
            volume=100 + index,
        )
        for index in range(10)
    )
    return BaselineInstitutionalFlowAnalyzer().analyze_snapshot(InstitutionalFlowContext(candles))


class Metrics:
    def snapshot(self) -> dict[str, int]:
        return {"analyses_completed": 1}


class Service:
    def __init__(self) -> None:
        self.item = make_snapshot()
        self.config = InstitutionalFlowConfig()
        self.analyzer = BaselineInstitutionalFlowAnalyzer()
        self.metrics = Metrics()

    def health(self) -> dict[str, str]:
        return {"status": "healthy"}

    async def state(self, *_: object) -> object:
        return self.item

    async def replay(self, *_: object) -> object:
        return self.item

    async def analyze(self, *_: object, **__: object) -> object:
        return self.item

    async def multi_timeframe(self, *_: object) -> dict[str, object]:
        return {"aligned": True, "direction_by_timeframe": {}}


def test_institutional_flow_read_only_api_surface() -> None:
    app = FastAPI()
    app.include_router(router)
    service = Service()
    app.dependency_overrides[get_institutional_flow_service] = lambda: service
    with TestClient(app) as client:
        for path in (
            "/institutional-flow/health",
            "/institutional-flow/metrics",
            "/institutional-flow/config",
            "/institutional-flow/snapshot",
            "/institutional-flow/state",
            "/institutional-flow/participation",
            "/institutional-flow/initiative",
            "/institutional-flow/responsive",
            "/institutional-flow/absorption",
            "/institutional-flow/exhaustion",
            "/institutional-flow/inventory",
            "/institutional-flow/campaign",
            "/institutional-flow/pressure",
            "/institutional-flow/persistence",
            "/institutional-flow/cross-session",
            "/institutional-flow/confluences",
            "/institutional-flow/explanation",
            "/institutional-flow/evidence?limit=1",
            "/institutional-flow/mtf",
            "/institutional-flow/replay?timestamp=2026-07-01T00:00:00Z",
        ):
            response = client.get(path)
            assert response.status_code == 200, (path, response.text)
        assert client.put("/institutional-flow/state").status_code == 405
        assert client.get("/institutional-flow/state?timeframe=INVALID").status_code == 422
        assert client.get("/institutional-flow/evidence?limit=1001").status_code == 422


def test_api_source_unavailable_is_degraded_not_stack_trace() -> None:
    class EmptyService(Service):
        async def state(self, *_: object) -> None:
            return None

        async def analyze(self, *_: object, **__: object) -> object:
            raise RuntimeError("secret internal stack")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_institutional_flow_service] = EmptyService
    with TestClient(app) as client:
        response = client.get("/institutional-flow/state")
    assert response.status_code == 503
    assert response.json() == {"detail": "Institutional Flow source data is unavailable"}
