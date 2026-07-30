from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.signal_notifications.service import render_signal_email


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


def test_executable_email_contains_complete_same_cycle_contract() -> None:
    subject, body = render_signal_email(payload())

    assert subject.startswith("[TEN AI] XAUUSD BUY")
    for expected in (
        "TEN AI ANALYTICAL PLATFORM",
        "M5: BUY",
        "M15: BUY",
        "Entry Price: 4026.89",
        "Stop Loss: 4018.0",
        "Take Profit: 4045.0",
        "Risk/Reward: 2.04",
        "Cycle ID: cycle",
        "Decision ID: decision",
    ):
        assert expected in body


def test_blocked_email_prominently_preserves_guardrail_reason() -> None:
    subject, body = render_signal_email(payload(blocked=True))

    assert subject.startswith("[TEN AI][BLOCKED]")
    assert "cooldown_active" in subject
    assert "Publication status: INELIGIBLE" in body
    assert "Guardrail status: REJECTED" in body


def test_email_configuration_validation_is_safe_and_secret_free() -> None:
    settings = Settings(signal_email_enabled=True, smtp_username="user")

    assert settings.signal_email_configuration_errors == (
        "TEN_SMTP_HOST",
        "TEN_EMAIL_FROM",
        "TEN_SMTP_PASSWORD",
    )
    assert "user" not in repr(settings.signal_email_configuration_errors)
