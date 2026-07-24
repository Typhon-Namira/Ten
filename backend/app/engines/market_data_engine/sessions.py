"""DST-aware global market-session classification."""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .models import MarketScheduleStatus, MarketSession, MarketStatusCode


class MarketSessionEngine:
    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays = holidays or set()
        self.london = ZoneInfo("Europe/London")
        self.new_york = ZoneInfo("America/New_York")

    def session_at(self, timestamp: datetime) -> MarketSession:
        instant = timestamp.astimezone(UTC)
        local = instant.astimezone(self.new_york)
        if local.date() in self.holidays or instant.date() in self.holidays:
            return MarketSession.HOLIDAY
        if local.weekday() == 5 or (
            local.weekday() == 6 and local.timetz().replace(tzinfo=None) < time(18)
        ):
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

    def status_at(self, timestamp: datetime) -> MarketScheduleStatus:
        if timestamp.tzinfo is None:
            return MarketScheduleStatus(
                market_status=MarketStatusCode.UNKNOWN,
                market_open=False,
                active_session=None,
                closure_reason="timezone_unknown",
                server_time_utc=timestamp.replace(tzinfo=UTC),
            )
        instant = timestamp.astimezone(UTC)
        local = instant.astimezone(self.new_york)
        if local.date() in self.holidays or instant.date() in self.holidays:
            return MarketScheduleStatus(
                market_status=MarketStatusCode.HOLIDAY_OR_PROVIDER_CLOSED,
                market_open=False,
                active_session=None,
                closure_reason="holiday_or_provider_closed",
                server_time_utc=instant,
            )
        weekday = local.weekday()
        local_time = local.timetz().replace(tzinfo=None)
        weekend = weekday == 5 or (weekday == 6 and local_time < time(18)) or (weekday == 4 and local_time >= time(17))
        if weekend:
            days_until_sunday = (6 - weekday) % 7
            sunday = local.date() + timedelta(days=days_until_sunday)
            next_open = datetime.combine(sunday, time(18), self.new_york).astimezone(UTC)
            return MarketScheduleStatus(
                market_status=MarketStatusCode.CLOSED_WEEKEND,
                market_open=False,
                active_session=None,
                closure_reason="weekend",
                next_expected_open_at=next_open,
                server_time_utc=instant,
            )
        if weekday in {0, 1, 2, 3} and time(17) <= local_time < time(18):
            next_open = datetime.combine(local.date(), time(18), self.new_york).astimezone(UTC)
            return MarketScheduleStatus(
                market_status=MarketStatusCode.MAINTENANCE,
                market_open=False,
                active_session=None,
                closure_reason="daily_rollover_maintenance",
                next_expected_open_at=next_open,
                server_time_utc=instant,
            )
        session = self.session_at(instant)
        if session in {MarketSession.CLOSED, MarketSession.WEEKEND, MarketSession.HOLIDAY}:
            session = MarketSession.ASIA
        return MarketScheduleStatus(
            market_status=MarketStatusCode.OPEN,
            market_open=True,
            active_session=session,
            server_time_utc=instant,
        )

    def is_open(self, timestamp: datetime) -> bool:
        return self.status_at(timestamp).market_open
