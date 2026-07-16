from abc import ABC
from datetime import datetime

from backend.app.engines.common import AnalysisEngine

from .config import EconomicCalendarConfig
from .models import EconomicEvent, EventImportance, NewsRiskResult


class EconomicCalendarEngine(AnalysisEngine[tuple[datetime, list[EconomicEvent]], NewsRiskResult], ABC):
    """Contract for provider-neutral macro-event risk filtering."""


class BaselineEconomicCalendarEngine(EconomicCalendarEngine):
    name = "economic_calendar"
    version = "1.0.0"

    def __init__(self, config: EconomicCalendarConfig | None = None) -> None:
        self.config = config or EconomicCalendarConfig()

    def analyze(self, data: tuple[datetime, list[EconomicEvent]]) -> NewsRiskResult:
        now, events = data
        relevant: list[EconomicEvent] = []
        nearest: float | None = None
        level = EventImportance.LOW
        for event in events:
            minutes = (event.scheduled_at - now).total_seconds() / 60
            distance = abs(minutes)
            nearest = distance if nearest is None else min(nearest, distance)
            pre, post = self._window(event.importance)
            if -post <= minutes <= pre:
                relevant.append(event)
                if event.importance == EventImportance.HIGH or level == EventImportance.LOW:
                    level = event.importance
        return NewsRiskResult(risk_level=level, no_trade=any(item.importance == EventImportance.HIGH for item in relevant), active_events=relevant, minutes_to_nearest=nearest)

    def _window(self, importance: EventImportance) -> tuple[int, int]:
        if importance == EventImportance.HIGH:
            return self.config.high_impact_pre_minutes, self.config.high_impact_post_minutes
        if importance == EventImportance.MEDIUM:
            return self.config.medium_impact_pre_minutes, self.config.medium_impact_post_minutes
        return 5, 5

