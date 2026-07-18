from abc import ABC
from datetime import datetime

from backend.app.engines.common import AnalysisEngine

from .analyzer import _phase
from .config import EconomicCalendarConfig
from .models import EconomicEvent, EventImportance, NewsRiskResult, RiskWindowPhase


class EconomicCalendarEngine(AnalysisEngine[tuple[datetime, list[EconomicEvent]], NewsRiskResult], ABC):
    """Provider-neutral compatibility contract for pipeline calendar context."""


class BaselineEconomicCalendarEngine(EconomicCalendarEngine):
    name = "economic_calendar"
    version = "1.0.0"

    def __init__(self, config: EconomicCalendarConfig | None = None) -> None:
        self.config = config or EconomicCalendarConfig()

    def analyze(self, data: tuple[datetime, list[EconomicEvent]]) -> NewsRiskResult:
        now, events = data
        relevant = tuple(
            item for item in events if _phase(item, now, self.config) not in {RiskWindowPhase.OUTSIDE, RiskWindowPhase.COOLDOWN, RiskWindowPhase.UNKNOWN}
        )
        scheduled = [item for item in events if item.scheduled_at_utc]
        nearest = min((abs((item.scheduled_at_utc - now).total_seconds() / 60) for item in scheduled), default=None)  # type: ignore[operator]
        rank = {EventImportance.UNKNOWN: 0, EventImportance.LOW: 1, EventImportance.MEDIUM: 2, EventImportance.HIGH: 3, EventImportance.CRITICAL: 4}
        level = max((item.importance for item in relevant), key=rank.__getitem__, default=EventImportance.LOW)
        return NewsRiskResult(
            risk_level=level,
            no_trade=False,
            active_events=relevant,
            minutes_to_nearest=nearest,
            observations=("Context only; the engine does not create trading restrictions or instructions.",),
        )
