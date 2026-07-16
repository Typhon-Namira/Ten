"""DST-aware global market-session classification."""

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from .models import MarketSession


class MarketSessionEngine:
    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays = holidays or set()
        self.london = ZoneInfo("Europe/London")
        self.new_york = ZoneInfo("America/New_York")

    def session_at(self, timestamp: datetime) -> MarketSession:
        instant = timestamp.astimezone(UTC)
        if instant.date() in self.holidays:
            return MarketSession.HOLIDAY
        if instant.weekday() == 5 or (instant.weekday() == 6 and instant.time() < time(22)):
            return MarketSession.WEEKEND
        london_hour = instant.astimezone(self.london).hour
        new_york_hour = instant.astimezone(self.new_york).hour
        london_open = 8 <= london_hour < 17
        new_york_open = 8 <= new_york_hour < 17
        if london_open and new_york_open:
            return MarketSession.LONDON_NEW_YORK_OVERLAP
        if london_open:
            return MarketSession.LONDON
        if new_york_open:
            return MarketSession.NEW_YORK
        if 0 <= instant.hour < 9:
            return MarketSession.ASIA
        return MarketSession.CLOSED

    def is_open(self, timestamp: datetime) -> bool:
        return self.session_at(timestamp) not in {MarketSession.CLOSED, MarketSession.WEEKEND, MarketSession.HOLIDAY}
