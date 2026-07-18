import httpx
import pytest

from backend.app.main import create_app


@pytest.mark.asyncio
async def test_application_startup_shutdown_schema_and_engine_health_routes() -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        assert app.state.ai_scoring_service.initialized is True
        schema_paths = app.openapi()["paths"]
        assert "/ai-scoring/health" in schema_paths
        assert "/ai-scoring/score" in schema_paths
        assert schema_paths["/ai-scoring/score"]["post"]["tags"] == ["ai-scoring"]
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
            ):
                response = await client.get(path)
                assert response.status_code == 200, (path, response.text)
    assert app.state.ai_scoring_service.closed is True
