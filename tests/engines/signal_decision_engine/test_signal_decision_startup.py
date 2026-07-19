import httpx
import pytest

from backend.app.main import create_app


@pytest.mark.asyncio
async def test_application_lifecycle_openapi_and_all_health_routes() -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        assert app.state.signal_decision_service.initialized is True
        paths = app.openapi()["paths"]
        assert "/signal-decisions/health" in paths
        assert "/signal-decisions/evaluate" in paths
        assert paths["/signal-decisions/evaluate"]["post"]["tags"] == ["signal-decisions"]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for path in (
                "/health",
                "/market/health",
                "/smc/health",
                "/liquidity/health",
                "/volume-profile/health",
                "/institutional-flow/health",
                "/market-regime/health",
                "/economic-calendar/health",
                "/ai-scoring/health",
                "/signal-decisions/health",
            ):
                assert (await client.get(path)).status_code == 200
    assert app.state.signal_decision_service.closed is True
