"""U.S. Census Bureau — Economic Indicator release schedule (Retail Sales, New Home Sales, Housing
Starts, Building Permits, Durable Goods, Factory Orders). Census is the issuing agency for all of
these.

NOTE: census.gov's calendar-listview page's exact DOM was not independently verified against a
live fetch from this environment — see `html_dates.py`'s module docstring for the resilience
strategy and verification caveat that applies here.
"""

from __future__ import annotations

from datetime import date

from ..models import SourceType
from .base import HttpPublicCalendarSource, RawEconomicEvent, stable_event_id
from .html_dates import find_date_entries
from .impact import canonicalize_title, XAUUSD_RELEVANT_EVENT_TYPES

SCHEDULE_URL = "https://www.census.gov/economic-indicators/calendar-listview.html"


class CensusPublicCalendarSource(HttpPublicCalendarSource):
    def __init__(self, *, source_url: str = SCHEDULE_URL, timeout_seconds: float = 15) -> None:
        super().__init__("census_bureau", source_url=source_url, timeout_seconds=timeout_seconds, source_type=SourceType.PUBLIC_WEBPAGE)

    async def fetch_schedule(self, start_date: date, end_date: date) -> list[RawEconomicEvent]:
        html = await self._get_text(self.source_url)
        if html is None:
            return []
        results: list[RawEconomicEvent] = []
        for entry in find_date_entries(html, start_date=start_date, end_date=end_date):
            canonical = canonicalize_title(entry.nearby_text)
            if canonical not in XAUUSD_RELEVANT_EVENT_TYPES:
                continue  # Census's calendar also lists many indicators irrelevant to XAUUSD
            results.append(
                RawEconomicEvent(
                    raw_name=entry.nearby_text[:200],
                    raw_scheduled_time=f"{entry.event_date.isoformat()}T13:00:00+00:00",  # Census releases typically 8:30am/10am ET
                    provider_event_id=stable_event_id("census_bureau", canonical, entry.event_date.isoformat()),
                    raw_country="US",
                    raw_currency="USD",
                    raw_importance="medium",
                    raw_status="scheduled",
                    raw_timezone="UTC",
                    source_url=self.source_url,
                )
            )
        return results
