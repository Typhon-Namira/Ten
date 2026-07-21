"""European Central Bank — Governing Council meeting calendar (schedule) + press-release RSS
(results). ECB is the issuing body for its own rate decisions and monetary-policy statements;
ecb.europa.eu is the canonical, authoritative source.

NOTE: the Governing Council calendar page's exact DOM was not independently verified against a
live fetch from this environment — see `html_dates.py`'s module docstring for the resilience
strategy and verification caveat that applies here.
"""

from __future__ import annotations

from datetime import date

from ..models import SourceType
from .base import HttpPublicCalendarSource, RawEconomicEvent, stable_event_id
from .html_dates import find_date_entries
from .rss import parse_press_rss

CALENDAR_URL = "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
PRESS_RSS_URL = "https://www.ecb.europa.eu/rss/press.xml"


class EcbPublicCalendarSource(HttpPublicCalendarSource):
    def __init__(self, *, calendar_url: str = CALENDAR_URL, press_rss_url: str = PRESS_RSS_URL, timeout_seconds: float = 15) -> None:
        super().__init__("ecb", source_url=calendar_url, timeout_seconds=timeout_seconds, source_type=SourceType.PUBLIC_WEBPAGE)
        self.press_rss_url = press_rss_url

    async def fetch_schedule(self, start_date: date, end_date: date) -> list[RawEconomicEvent]:
        results: list[RawEconomicEvent] = []
        html = await self._get_text(self.source_url)
        if html is not None:
            for entry in find_date_entries(html, start_date=start_date, end_date=end_date):
                title = entry.nearby_text if "governing council" in entry.nearby_text.lower() or "monetary policy" in entry.nearby_text.lower() else "ECB Governing Council Meeting"
                results.append(
                    RawEconomicEvent(
                        raw_name=title[:200],
                        raw_scheduled_time=f"{entry.event_date.isoformat()}T13:15:00+00:00",  # ECB decisions publish ~13:15 UTC (approximate)
                        provider_event_id=stable_event_id("ecb", "governing_council", entry.event_date.isoformat()),
                        raw_country="EU",
                        raw_currency="EUR",
                        raw_importance="critical",
                        raw_status="scheduled",
                        raw_timezone="UTC",
                        source_url=self.source_url,
                    )
                )
        rss = await self._get_text(self.press_rss_url)
        if rss is not None:
            results.extend(parse_press_rss(rss, start_date, end_date, source_name="ecb", source_url=self.press_rss_url, country="EU", currency="EUR"))
        return results
