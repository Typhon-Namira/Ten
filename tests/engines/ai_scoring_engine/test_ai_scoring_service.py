from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI

from backend.app.api.routes.ai_scoring import router
from backend.app.engines.ai_scoring_engine import (
    AIScoringConfig,
    AIScoringService,
    FixedClock,
    InMemoryAIScoringRepository,
    ScoreMode,
    ScoreRequest,
)
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from tests.engines.ai_scoring_engine.test_ai_scoring import NOW, aligned_input


async def service() -> tuple[AIScoringService, InMemoryAIScoringRepository, InMemoryEventBus, InMemoryFeatureStore]:
    repository = InMemoryAIScoringRepository()
    events = InMemoryEventBus()
    features = InMemoryFeatureStore()
    value = AIScoringService(repository, events, features, AIScoringConfig(), repository_mode="memory", clock=FixedClock(NOW))
    await value.start()
    return value, repository, events, features


@pytest.mark.asyncio
async def test_repository_idempotency_filters_pagination_and_retention() -> None:
    value, repository, _, _ = await service()
    first = await value.calculate_input(aligned_input(), publish_events=False)
    duplicate = await value.calculate_input(aligned_input(), publish_events=False)
    assert duplicate.snapshot_id == first.snapshot_id
    assert await repository.get_snapshot(first.snapshot_id) == first
    assert await repository.get_latest_snapshot("XAUUSD", "M15") == first
    assert await repository.find_by_fingerprint(first.metadata.input_fingerprint, ScoreMode.LIVE) == first
    assert await repository.list_snapshots("XAUUSD", "M15", status=first.status, policy_version="1.0.0", mode=ScoreMode.LIVE, limit=1) == (first,)
    assert await repository.list_snapshots("OTHER", "M15") == ()
    assert await repository.prune(NOW + timedelta(days=1), ScoreMode.LIVE, 1) == 1
    assert await repository.get_snapshot(first.snapshot_id) is None


@pytest.mark.asyncio
async def test_feature_event_publication_replay_suppression_and_health() -> None:
    value, _, events, features = await service()
    snapshot = await value.calculate_input(aligned_input(), publish_events=True)
    assert len(events.history()) == 1
    feature = await features.snapshot(next(iter(events.history())).correlation_id)
    assert feature.features["ai_score"]["trading_instruction"] is False
    await value.calculate_input(aligned_input(mode=ScoreMode.REPLAY), persist=False, publish_events=True)
    assert len(events.history()) == 1
    health = value.health()
    assert health["status"] == "degraded"
    assert health["persistence"] == {"status": "degraded", "mode": "memory"}
    assert value.metrics.latest_snapshot_id is None
    await value.stop()
    assert value.health()["status"] == "unavailable"
    await value.stop()
    assert snapshot.metadata.trading_instruction is False


@pytest.mark.asyncio
async def test_cleanup_and_api_contracts() -> None:
    value, _, _, _ = await service()
    stored = await value.calculate_input(aligned_input(), publish_events=False)
    app = FastAPI()
    app.include_router(router)
    app.state.ai_scoring_service = value
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for path in ("health", "config", "metrics"):
            assert (await client.get(f"/ai-scoring/{path}")).status_code == 200
        assert (await client.get("/ai-scoring/latest")).status_code == 200
        assert (await client.get("/ai-scoring/history")).status_code == 200
        assert (await client.get(f"/ai-scoring/snapshots/{stored.snapshot_id}")).status_code == 200
        explanation = await client.get(f"/ai-scoring/snapshots/{stored.snapshot_id}/explanation")
        assert explanation.status_code == 200
        assert explanation.json()["financial_safety_code"] == "analytical_intelligence_only"
        assert (await client.get(f"/ai-scoring/snapshots/{'0' * 8}-0000-0000-0000-000000000000")).status_code == 404
        assert (await client.get("/ai-scoring/history", params={"start": NOW.isoformat(), "end": (NOW - timedelta(days=1)).isoformat()})).status_code == 422
        assert (await client.get("/ai-scoring/history", params={"start": (NOW - timedelta(days=500)).isoformat(), "end": NOW.isoformat()})).status_code == 422
        assert (await client.post("/ai-scoring/replay", json={"instrument": "XAUUSD", "timeframe": "M15"})).status_code == 422
        replay = await client.post("/ai-scoring/replay", json={"instrument": "XAUUSD", "timeframe": "M15", "as_of": NOW.isoformat(), "persist": False})
        assert replay.status_code == 200
        assert replay.json()["status"] == "insufficient_evidence"
        score = await client.post("/ai-scoring/score", json={"instrument": "XAUUSD", "timeframe": "M15", "persist": False})
        assert score.status_code == 200
        assert score.json()["status"] == "insufficient_evidence"
        assert (await client.post("/ai-scoring/score", json={"instrument": "bad path", "timeframe": "M15"})).status_code == 422
    assert await value.cleanup() == 0


@pytest.mark.asyncio
async def test_future_live_request_is_rejected() -> None:
    value, _, _, _ = await service()
    with pytest.raises(ValueError, match="future scoring boundary"):
        await value.calculate(ScoreRequest(as_of=NOW + timedelta(minutes=1)))
