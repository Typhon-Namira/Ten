"""U.S. Department of Labor — Unemployment Insurance Weekly Claims (Initial + Continuing Claims).

Authoritative and, unusually among these adapters, does not require scraping a page at all: DOL
publishes the UI Weekly Claims release every Thursday at 8:30am ET (dol.gov/newsroom/releases/eta),
with the only irregularity being a shift when Thursday falls on a federal holiday. Rather than
guess which direction DOL shifts an affected release (their published exception list is the only
authoritative source for that, and this adapter does not fabricate a specific alternate date it
cannot verify), this generates the regular Thursday schedule and flags holiday-colliding weeks as
tentative with reduced confidence — an honest "this date needs the exception list checked" signal
rather than a fabricated shifted date.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ..models import ConnectionState, SourceType
from .base import HttpPublicCalendarSource, RawEconomicEvent, stable_event_id

ET = ZoneInfo("America/New_York")


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def us_federal_holidays(year: int) -> set[date]:
    """Fixed-date and floating U.S. federal holidays observed by executive-branch agencies
    (including DOL) — the set that can plausibly shift a Thursday release, not a general-purpose
    holiday calendar. Deterministic, auditable, no external dependency."""
    holidays = {
        date(year, 1, 1),  # New Year's Day
        date(year, 6, 19),  # Juneteenth
        date(year, 7, 4),  # Independence Day
        date(year, 11, 11),  # Veterans Day
        date(year, 12, 25),  # Christmas Day
        _nth_weekday_of_month(year, 1, 0, 3),  # MLK Day: 3rd Monday of January
        _nth_weekday_of_month(year, 2, 0, 3),  # Washington's Birthday: 3rd Monday of February
        _last_weekday_of_month(year, 5, 0),  # Memorial Day: last Monday of May
        _nth_weekday_of_month(year, 9, 0, 1),  # Labor Day: 1st Monday of September
        _nth_weekday_of_month(year, 10, 0, 2),  # Columbus Day: 2nd Monday of October
        _nth_weekday_of_month(year, 11, 3, 4),  # Thanksgiving: 4th Thursday of November
    }
    return holidays


class DolWeeklyClaimsSource(HttpPublicCalendarSource):
    """Deterministic-rule source — `source_url` points at DOL's release page for provenance/
    citation purposes only; the schedule itself is computed, not scraped."""

    def __init__(self, *, source_url: str = "https://www.dol.gov/newsroom/releases/eta", timeout_seconds: float = 15) -> None:
        super().__init__("dol_weekly_claims", source_url=source_url, timeout_seconds=timeout_seconds, source_type=SourceType.DETERMINISTIC_RULE)
        # A deterministic rule has nothing to "reach" over HTTP — mark it permanently connected
        # rather than leaving connection_state at UNKNOWN, which would misreport it as unavailable.
        self._connection_state = ConnectionState.CONNECTED

    async def fetch_schedule(self, start_date: date, end_date: date) -> list[RawEconomicEvent]:
        results: list[RawEconomicEvent] = []
        cursor = start_date
        # Walk forward to the first Thursday on/after start_date.
        cursor += timedelta(days=(3 - cursor.weekday()) % 7)
        holidays_by_year: dict[int, set[date]] = {}
        while cursor <= end_date:
            year_holidays = holidays_by_year.setdefault(cursor.year, us_federal_holidays(cursor.year))
            is_holiday_week = cursor in year_holidays
            scheduled = datetime.combine(cursor, time(8, 30), tzinfo=ET)
            for canonical_name, importance in (("Initial Jobless Claims", "high"), ("Continuing Jobless Claims", "medium")):
                results.append(
                    RawEconomicEvent(
                        raw_name=f"{canonical_name} (tentative — federal holiday week, verify exact date)" if is_holiday_week else canonical_name,
                        raw_scheduled_time=scheduled.isoformat(),
                        provider_event_id=stable_event_id("dol", canonical_name, cursor.isoformat()),
                        raw_country="US",
                        raw_currency="USD",
                        raw_importance=importance,
                        raw_status="tentative" if is_holiday_week else "scheduled",
                        raw_timezone="America/New_York",
                        source_url=self.source_url,
                    )
                )
            cursor += timedelta(days=7)
        return results
