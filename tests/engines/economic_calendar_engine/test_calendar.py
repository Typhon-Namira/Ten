from datetime import UTC, datetime, timedelta

from backend.app.engines.economic_calendar_engine import BaselineEconomicCalendarEngine, EconomicEvent, EventImportance


def test_high_impact_event_activates_no_trade_window() -> None:
    now = datetime(2026, 1, 9, 13, 20, tzinfo=UTC)
    event = EconomicEvent(event_id="nfp-1", name="Nonfarm Payrolls", scheduled_at=now + timedelta(minutes=10), importance=EventImportance.HIGH)
    result = BaselineEconomicCalendarEngine().analyze((now, [event]))
    assert result.no_trade is True
    assert result.risk_level == EventImportance.HIGH

