"""Explainability API tests. The fake provider never makes a real network call, so these tests can assert on exactly what
grounded payload the model was handed and what happens when it fails or returns garbage, without
touching a real API key."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.ai.provider_client import AIProviderClient, AIProviderCompletion
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.exceptions import ExternalServiceError
from backend.app.explainability import service as explainability_service_module
from backend.app.explainability import ExplainabilityService
from backend.app.main import create_app

PROMPTS_DIR = Path(explainability_service_module.__file__).resolve().parent / "prompts"

VALID_EXPLANATION = {
    "summary": "Confidence is low because only two of five directional engines currently agree.",
    "primary_reasons": ["Institutional Flow has no snapshot yet"],
    "opposing_factors": ["SMC shows a bullish order block"],
    "engine_breakdown": [{"engine": "smc", "influence": "neutral", "note": "No snapshot yet"}],
    "required_for_change": ["A liquidity snapshot must be published"],
    "caveats": ["Institutional Flow has no snapshot yet"],
}


class FakeProviderClient(AIProviderClient):
    provider = "cerebras"
    base_url = "https://api.cerebras.ai/v1"
    configured = True

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def available_models(self) -> tuple[str, ...]:
        return ("test-model",)

    async def complete_json(self, *, system_prompt: str, payload: dict[str, Any], model: str, **_: Any) -> AIProviderCompletion:
        self.calls.append({"system_prompt": system_prompt, "payload": payload, "model": model})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return AIProviderCompletion(
            content=self.response,
            provider="cerebras",
            model=model,
            status_code=200,
            latency_ms=1,
            provider_request_id=None,
            token_usage=None,
            rate_limit_limit=None,
            rate_limit_remaining=None,
            rate_limit_reset=None,
            retry_after=None,
        )


class FakeDecision:
    """Stands in for a persisted `SignalDecision` — only the fields context.py/explain.py
    actually read, with a real `model_dump` so `_summarize()` treats it like a Pydantic model."""

    def __init__(self, *, state: str, instrument: str = "XAUUSD", timeframe: str = "M15") -> None:
        self.decision_id = uuid4()
        self.instrument = instrument
        self.timeframe = timeframe
        self.as_of = datetime(2026, 1, 1, tzinfo=UTC)
        self.state = _StateValue(state)
        self.ai_score_snapshot_id = None

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"decision_id": str(self.decision_id), "state": self.state.value, "instrument": self.instrument, "timeframe": self.timeframe}


class _StateValue:
    def __init__(self, value: str) -> None:
        self.value = value


def _install(client: TestClient, fake: FakeProviderClient) -> None:
    client.app.state.explainability_service = ExplainabilityService(fake, PromptLoader(PROMPTS_DIR), model="test-model")


def test_explain_current_grounds_context_and_cites_evidence() -> None:
    fake = FakeProviderClient(response=VALID_EXPLANATION)
    with TestClient(create_app()) as client:
        _install(client, fake)
        response = client.get("/api/v1/explain/current")
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["explanation"] == VALID_EXPLANATION
    assert body["explainability_score"]["engines_total"] == 8
    assert isinstance(body["evidence"], list)
    assert len(body["engines"]) == 8
    # The grounded payload sent to the model must contain the same engine facts the response cites.
    sent_payload = fake.calls[0]["payload"]
    assert sent_payload["instrument"] == body["instrument"]
    assert len(sent_payload["engines"]) == 8


def test_explain_current_degrades_gracefully_when_provider_fails() -> None:
    """A failing provider call must never 500 and never fabricate an explanation — it degrades
    to `explanation: null` with the failure reported in `error`."""
    fake = FakeProviderClient(error=ExternalServiceError("Provider returned an invalid scoring response"))
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        _install(client, fake)
        response = client.get("/api/v1/explain/current")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] is None
    assert body["error"]
    assert body["explainability_score"]["engines_total"] == 8


def test_explain_current_degrades_when_model_returns_an_invalid_shape() -> None:
    """The model returning JSON that doesn't match `Explanation` (e.g. missing `summary`) must
    also degrade instead of 500ing or being rendered as-is."""
    fake = FakeProviderClient(response={"primary_reasons": ["not a valid explanation"]})
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        _install(client, fake)
        response = client.get("/api/v1/explain/current")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] is None
    assert body["error"]


def test_explain_decision_not_found_returns_404() -> None:
    fake = FakeProviderClient(response=VALID_EXPLANATION)
    with TestClient(create_app()) as client:
        _install(client, fake)
        response = client.get(f"/api/v1/explain/decision/{uuid4()}")
    assert response.status_code == 404


def test_explain_rejection_rejects_an_eligible_decision_with_422() -> None:
    fake = FakeProviderClient(response=VALID_EXPLANATION)
    decision = FakeDecision(state="eligible")
    with TestClient(create_app()) as client:
        _install(client, fake)
        client.app.state.signal_decision_service.get_decision = AsyncMock(return_value=decision)
        response = client.get(f"/api/v1/explain/rejection/{decision.decision_id}")
    assert response.status_code == 422


def test_explain_rejection_explains_a_blocked_decision_with_a_rejection_specific_question() -> None:
    fake = FakeProviderClient(response=VALID_EXPLANATION)
    decision = FakeDecision(state="blocked")
    with TestClient(create_app()) as client:
        _install(client, fake)
        client.app.state.signal_decision_service.get_decision = AsyncMock(return_value=decision)
        response = client.get(f"/api/v1/explain/rejection/{decision.decision_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] == VALID_EXPLANATION
    sent_payload = fake.calls[0]["payload"]
    assert "rejected" in sent_payload["question"].lower()
    assert sent_payload["decision"]["state"] == "blocked"


def test_explain_chat_folds_conversation_history_into_the_grounded_payload() -> None:
    fake = FakeProviderClient(response=VALID_EXPLANATION)
    with TestClient(create_app()) as client:
        _install(client, fake)
        response = client.post(
            "/api/v1/explain/chat",
            json={"message": "Why is confidence low?", "history": [{"role": "user", "content": "Explain the current market."}, {"role": "assistant", "content": "Confidence is currently unavailable."}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] == VALID_EXPLANATION
    sent_payload = fake.calls[0]["payload"]
    assert sent_payload["question"] == "Why is confidence low?"
    assert sent_payload["conversation_history"] == [
        {"role": "user", "content": "Explain the current market."},
        {"role": "assistant", "content": "Confidence is currently unavailable."},
    ]
