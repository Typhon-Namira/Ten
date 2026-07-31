from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.app.core.config import Settings
from backend.app.signal_notifications.service import SignalEmailWorker, render_signal_email


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
