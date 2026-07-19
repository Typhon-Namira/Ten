from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from backend.app.api.routes.replays import router
from backend.app.engines.replay_engine import ReplayConfig, ReplayMode
from backend.app.main import create_app
from tests.engines.replay_engine.test_replay_engine import build_service, historical_event, replay_request


@pytest.mark.asyncio
async def test_replay_api_complete_control_and_bounded_reads() -> None:
    service, _ = await build_service((historical_event(5, 1), historical_event(10, 2)))
    app = FastAPI()
    app.state.replay_service = service
    app.include_router(router)
    request = replay_request(mode=ReplayMode.STEP).model_dump(mode="json")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/replays/health")).status_code == 200
        assert (await client.get("/replays/config")).status_code == 200
        assert (await client.get("/replays/metrics")).status_code == 200
        created_response = await client.post("/replays", json=request)
        assert created_response.status_code == 201
        replay_id = created_response.json()["replay_id"]
        assert (await client.get("/replays?limit=10")).status_code == 200
        assert (await client.get(f"/replays/{replay_id}")).status_code == 200
        stepped = await client.post(f"/replays/{replay_id}/step", json={"units": 1})
        assert stepped.status_code == 200
        assert (await client.get(f"/replays/{replay_id}/checkpoints")).status_code == 200
        assert (await client.get(f"/replays/{replay_id}/transitions")).status_code == 200
        assert (await client.get(f"/replays/{replay_id}/summary")).status_code == 200
        assert (await client.get(f"/replays/{replay_id}/outputs")).status_code == 200
        assert (await client.get(f"/replays/{replay_id}/ai-scores")).status_code == 200
        assert (await client.get(f"/replays/{replay_id}/signal-decisions")).status_code == 200
        assert (await client.get(f"/replays/{replay_id}/trace")).status_code == 404
        assert (await client.post(f"/replays/{replay_id}/resume")).status_code == 202
        assert (await client.post(f"/replays/{replay_id}/pause")).status_code == 202
        assert (await client.post(f"/replays/{replay_id}/cancel")).status_code == 202
        assert (await client.get("/replays/not-a-uuid")).status_code == 422
        assert (await client.get(f"/replays/{uuid4()}")).status_code == 404
        assert (await client.get("/replays?limit=1000")).status_code == 422


@pytest.mark.asyncio
async def test_compare_api_and_invalid_transition_statuses() -> None:
    service, _ = await build_service((historical_event(5, 1),))
    left = await service.create(replay_request())
    right = await service.create(replay_request())
    app = FastAPI()
    app.state.replay_service = service
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/replays/compare", json={"left_replay_id": str(left.replay_id), "right_replay_id": str(right.replay_id)})
        assert response.status_code == 200 and response.json()["comparable"] is True
        assert (await client.post(f"/replays/{left.replay_id}/resume")).status_code == 409
        assert (await client.post(f"/replays/{left.replay_id}/step", json={"units": 1})).status_code == 409


def test_strict_configuration_validation() -> None:
    with pytest.raises(ValidationError, match="heartbeat"):
        ReplayConfig(worker={"lease_seconds": 10, "heartbeat_seconds": 10})
    with pytest.raises(ValidationError, match="live analytical state"):
        ReplayConfig(isolation={"allow_live_event_publication": True})
    with pytest.raises(ValidationError, match="concurrency"):
        ReplayConfig(limits={"max_concurrent_sessions": 1}, worker={"max_concurrency": 2})
    with pytest.raises(ValidationError, match="cannot be empty"):
        ReplayConfig(approved_sources=frozenset())


@pytest.mark.asyncio
async def test_application_lifecycle_openapi_and_health_regression() -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        assert app.state.replay_service.initialized is True
        paths = app.openapi()["paths"]
        assert "/replays/health" in paths and "/replays/{replay_id}/step" in paths
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
                "/replays/health",
            ):
                assert (await client.get(path)).status_code == 200
    assert app.state.replay_service.closed is True
    await app.state.replay_service.stop()
