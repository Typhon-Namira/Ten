from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.app.core.config import Settings
from backend.app.signal_notifications.service import (
    SignalEmailOutboxRepository,
    SignalEmailWorker,
    primary_publication_ineligibility_reason,
    primary_scenario_email_outbox_values,
    render_signal_email,
)
from tests.conftest import FakeSessionFactory
from backend.app.engines.signal_decision_engine import ConservativeSignalDecisionPolicy
from backend.app.scenario_forecasting.simulation_engine import (
    MarketSimulationConfig,
    MarketSimulationEngine,
)
from tests.engines.signal_decision_engine.test_signal_decision_engine import (
    ai_score,
    decision_input,
)
from tests.scenario_forecasting.test_scenario_forecasting import scenario_inputs
from tests.signal_synthesis.test_multi_timeframe_signal import aligned_analysis


def payload(*, blocked: bool = False) -> dict[str, object]:
    return {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "combined_confidence": 64.0,
        "combined_strength": "MODERATE",
        "entry": 4026.89,
        "stop_loss": 4018.0,
        "take_profit": 4045.0,
        "risk_reward": 2.04,
        "current_market_price": 4027.0,
        "expected_horizon_seconds": 900,
        "market_time": "2026-07-30T10:00:00+00:00",
        "created_at": "2026-07-30T10:00:05+00:00",
        "expires_at": "2026-07-30T10:15:00+00:00",
        "execution_status": "BLOCKED" if blocked else "READY",
        "guardrail_status": "REJECTED" if blocked else "APPROVED",
        "publication_status": "INELIGIBLE" if blocked else "ELIGIBLE",
        "blockers": ["cooldown_active"] if blocked else [],
        "analytical_thesis": "Same-cycle M5/M15 structure is bullish.",
        "geometry_owner_timeframe": "M15",
        "structural_source_ids": ["order-block-1", "liquidity-1"],
        "timeframe_summaries": [
            {
                "timeframe": "M5",
                "direction": "BUY",
                "confidence": 58.0,
                "strength": "MODERATE",
                "execution_status": "READY",
            },
            {
                "timeframe": "M15",
                "direction": "BUY",
                "confidence": 69.0,
                "strength": "MODERATE",
                "execution_status": "READY",
            },
        ],
        "cycle_id": "cycle",
        "analysis_id": "analysis",
        "synthesis_id": "synthesis",
        "signal_id": "signal",
        "decision_id": "decision",
    }


def test_legacy_signal_email_is_rejected_without_primary_authority() -> None:
    with pytest.raises(
        ValueError, match="requires an authoritative Primary Scenario"
    ):
        render_signal_email(payload())


def test_blocked_legacy_signal_email_is_also_rejected() -> None:
    with pytest.raises(
        ValueError, match="requires an authoritative Primary Scenario"
    ):
        render_signal_email(payload(blocked=True))


def test_primary_scenario_email_uses_authoritative_contract() -> None:
    value = payload() | {
        "primary_scenario_id": "22222222-2222-2222-2222-222222222222",
        "primary_scenario_score": 74.0,
        "scenario_type": "bullish_pullback_continuation",
        "market_cutoff": "2026-07-30T12:00:00+00:00",
        "reference_price": 4098.78,
        "expected_path": (
            "1. Hold structure: 4098.70-4098.90",
            "2. Expand: 4102.00-4102.20",
        ),
        "entry_type": "PULLBACK",
        "entry_zone": {"low": 4097.8, "high": 4098.1},
        "invalidation": "M15 closes below demand",
        "supporting_evidence": ("smc:1", "liquidity:2"),
        "alternative_summary": {
            "direction": "BEARISH",
            "scenario_type": "upside_liquidity_sweep_reversal",
            "score": 63.0,
        },
    }

    subject, body = render_signal_email(value)

    assert subject == "TEN Primary Scenario · XAUUSD BUY · 74% · M15"
    assert "Expected path:" in body
    assert "Entry type: PULLBACK" in body
    assert "Alternative Scenario: BEARISH" in body
    assert "Analytical Intelligence Only" in body
    assert "No Broker Execution" in body


def test_email_configuration_validation_is_safe_and_secret_free() -> None:
    settings = Settings(signal_email_enabled=True, smtp_username="user")

    assert settings.signal_email_configuration_errors == (
        "TEN_SMTP_HOST",
        "TEN_EMAIL_FROM",
        "TEN_SMTP_PASSWORD",
    )
    assert "user" not in repr(settings.signal_email_configuration_errors)


@pytest.mark.asyncio
async def test_claimed_primary_scenario_email_is_dispatched_once_and_marked_sent() -> None:
    repository = SimpleNamespace(mark_sent=AsyncMock(), mark_failed=AsyncMock())
    sender = SimpleNamespace(send=AsyncMock(return_value="<provider-message-id>"))
    worker = SignalEmailWorker(
        repository,
        sender,
        enabled=True,
        poll_seconds=10,
        max_attempts=5,
    )
    event = SimpleNamespace(
        id=uuid4(),
        signal_id=uuid4(),
        primary_scenario_id=uuid4(),
        recipient="operator@example.com",
        status="PROCESSING",
        attempt_count=1,
        payload={
            "primary_scenario_id": str(uuid4()),
            "market_cutoff": datetime.now(UTC).isoformat(),
        },
    )

    await worker._deliver(event)

    sender.send.assert_awaited_once_with(event.payload, event.recipient)
    repository.mark_sent.assert_awaited_once()
    repository.mark_failed.assert_not_awaited()


async def authoritative_selection_and_decision():
    state, quant, synthesis = await scenario_inputs()
    _, selection = MarketSimulationEngine(
        MarketSimulationConfig(
            primary_scenario_threshold=0,
            email_scenario_threshold=0,
        )
    ).simulate(state, quant, synthesis)
    score = ai_score(as_of=state.market_data_boundary, calculated_at=state.market_data_boundary)
    decision = ConservativeSignalDecisionPolicy().evaluate(
        decision_input(score=score, as_of=state.market_data_boundary).model_copy(
            update={
                "current_ai_analysis": aligned_analysis(state, quant),
                "market_snapshot_id": state.state_id,
                "quantitative_forecast_id": quant.result_id,
                "current_primary_scenario": selection,
            }
        )
    )
    return selection, decision


@pytest.mark.asyncio
async def test_authoritative_primary_owns_email_payload_and_deduplication() -> None:
    selection, decision = await authoritative_selection_and_decision()
    now = selection.selected_at

    first = primary_scenario_email_outbox_values(
        selection, decision, "operator@example.com", now
    )
    repeated = primary_scenario_email_outbox_values(
        selection, decision, "operator@example.com", now + timedelta(seconds=1)
    )

    assert first is not None and repeated is not None
    assert first["id"] == repeated["id"]
    assert first["signal_id"] == selection.selection_id
    assert first["primary_scenario_id"] == selection.primary_candidate_id
    assert first["payload"]["entry"] == selection.primary.geometry.entry
    assert first["payload"]["publication_status"] == "ELIGIBLE"


@pytest.mark.asyncio
async def test_publication_ineligible_primary_has_exact_guardrail_reason() -> None:
    selection, decision = await authoritative_selection_and_decision()
    blocked = decision.model_copy(update={"publication_eligible": False})

    assert primary_publication_ineligibility_reason(
        selection, blocked, selection.selected_at
    ) == "guardrails_rejected"
    assert (
        primary_scenario_email_outbox_values(
            selection,
            blocked,
            "operator@example.com",
            selection.selected_at,
        )
        is None
    )


@pytest.mark.asyncio
async def test_guardrail_decision_for_another_selection_cannot_publish() -> None:
    selection, decision = await authoritative_selection_and_decision()
    mismatched = selection.model_copy(update={"selection_id": uuid4()})

    assert primary_publication_ineligibility_reason(
        mismatched, decision, selection.selected_at
    ) == "guardrail_decision_selection_mismatch"


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _InsertResult:
    def __init__(self, value: object | None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> object | None:
        return self.value


class _Scalars:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def all(self) -> list[object]:
        return self.values


@pytest.mark.asyncio
async def test_publication_evaluation_atomically_creates_one_outbox() -> None:
    selection, decision = await authoritative_selection_and_decision()
    outbox_id = uuid4()
    database = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[None, _InsertResult(outbox_id), None]
        ),
        begin=lambda: _Transaction(),
    )
    repository = SignalEmailOutboxRepository(FakeSessionFactory(database))

    created = await repository.evaluate_primary_scenario(
        selection,
        decision,
        "operator@example.com",
        email_enabled=True,
        now=selection.selected_at,
    )

    assert created is True
    assert database.execute.await_count == 3


@pytest.mark.asyncio
async def test_reconciliation_repairs_missing_outbox_without_analysis_or_simulation() -> None:
    selection, decision = await authoritative_selection_and_decision()
    values = primary_scenario_email_outbox_values(
        selection, decision, "operator@example.com", selection.selected_at
    )
    assert values is not None
    publication = SimpleNamespace(
        selection_id=selection.selection_id,
        primary_scenario_id=selection.primary_candidate_id,
        decision_id=decision.decision_id,
        payload={
            "market_cutoff": selection.market_cutoff.isoformat(),
            "outbox": {
                "id": str(values["id"]),
                "signal_id": str(values["signal_id"]),
                "deduplication_key": values["deduplication_key"],
                "recipient": values["recipient"],
                "payload": values["payload"],
            },
        },
    )
    database = SimpleNamespace(
        scalars=AsyncMock(return_value=_Scalars([publication])),
        execute=AsyncMock(return_value=_InsertResult(values["id"])),
        begin=lambda: _Transaction(),
    )
    repository = SignalEmailOutboxRepository(FakeSessionFactory(database))

    assert await repository.reconcile_primary_scenario_publications() == 1
    database.scalars.assert_awaited_once()
    database.execute.assert_awaited_once()
