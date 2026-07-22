from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI

from backend.app.api.routes.signal_decisions import router
from backend.app.engines.ai_scoring_engine import FixedClock
from backend.app.engines.signal_decision_engine import DecisionRequest
from tests.engines.signal_decision_engine.test_signal_decision_engine import NOW
from tests.engines.signal_decision_engine.test_signal_decision_service import build_service


@pytest.mark.asyncio
async def test_api_contract_health_history_replay_and_safety() -> None:
    service, _, score = await build_service()
    stored = await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id))
    app = FastAPI()
    app.include_router(router)
    app.state.signal_decision_service = service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for path in ("health", "config", "metrics"):
            assert (await client.get(f"/signal-decisions/{path}")).status_code == 200
        latest = await client.get("/signal-decisions/latest")
        assert latest.status_code == 200
        assert latest.json()["state"] == "eligible"
        service.clock = FixedClock(stored.valid_until + timedelta(seconds=1))
        expired_latest = await client.get("/signal-decisions/latest")
        assert expired_latest.status_code == 200
        assert expired_latest.json()["decision_id"] == str(stored.decision_id)
        assert "order_side" not in latest.text
        assert "position_size" not in latest.text
        assert (await client.get("/signal-decisions/history")).status_code == 200
        assert (await client.get(f"/signal-decisions/{stored.decision_id}")).status_code == 200
        assert (await client.get(f"/signal-decisions/{stored.decision_id}/rules")).status_code == 200
        explanation = await client.get(f"/signal-decisions/{stored.decision_id}/explanation")
        assert explanation.json()["financial_safety_notice"] == "analytical_decision_only"
        missing = "00000000-0000-0000-0000-000000000000"
        assert (await client.get(f"/signal-decisions/{missing}")).status_code == 404
        assert (await client.get("/signal-decisions/latest", params={"instrument": "OTHER"})).status_code == 404
        assert (await client.get("/signal-decisions/history", params={"start": NOW.isoformat(), "end": (NOW - timedelta(days=1)).isoformat()})).status_code == 422
        assert (await client.get("/signal-decisions/history", params={"start": (NOW - timedelta(days=500)).isoformat(), "end": NOW.isoformat()})).status_code == 422
        assert (await client.post("/signal-decisions/replay", json={"instrument": "XAUUSD", "timeframe": "M15", "ai_score_snapshot_id": str(score.snapshot_id)})).status_code == 422
        replay = await client.post("/signal-decisions/replay", json={"instrument": "XAUUSD", "timeframe": "M15", "ai_score_snapshot_id": str(score.snapshot_id), "as_of": NOW.isoformat(), "persist": False})
        assert replay.status_code == 200
        assert replay.json()["mode"] == "replay"
        missing_score = await client.post("/signal-decisions/evaluate", json={"instrument": "XAUUSD", "timeframe": "M15", "ai_score_snapshot_id": missing})
        assert missing_score.status_code == 404
        invalid_policy = await client.post("/signal-decisions/evaluate", json={"instrument": "XAUUSD", "timeframe": "M15", "ai_score_snapshot_id": str(score.snapshot_id), "decision_policy_name": "missing_policy"})
        assert invalid_policy.status_code == 422
        assert (await client.post("/signal-decisions/evaluate", json={"instrument": "bad path", "timeframe": "M15", "ai_score_snapshot_id": str(score.snapshot_id)})).status_code == 422
