"""Federal Reserve — FOMC meeting calendar (schedule) + monetary-policy press releases (results).

Authoritative: the Federal Reserve Board is the issuing body for FOMC rate decisions, statements,
minutes, and the Beige Book — federalreserve.gov is the canonical source, not a secondary
aggregator. Two feeds are combined:
  - the FOMC calendar page for forward-looking meeting dates (schedule)
  - the monetary-policy press-release RSS feed for confirmed statement/minutes releases (result)

NOTE: `fomccalendars.htm`'s exact DOM was not independently verified against a live fetch from
this environment — see `html_dates.py`'s module docstring for the resilience strategy and the
verification caveat that applies here.
"""

from __future__ import annotations

from datetime import date, datetime

from ..models import SourceType
from .base import HttpPublicCalendarSource, RawEconomicEvent, stable_event_id
from .html_dates import find_date_entries
from .impact import canonicalize_title
from .rss import parse_press_rss

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
PRESS_RSS_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"


class FedPublicCalendarSource(HttpPublicCalendarSource):
    def __init__(self, *, calendar_url: str = CALENDAR_URL, press_rss_url: str = PRESS_RSS_URL, timeout_seconds: float = 15) -> None:
        super().__init__("federal_reserve", source_url=calendar_url, timeout_seconds=timeout_seconds, source_type=SourceType.PUBLIC_WEBPAGE)
        self.press_rss_url = press_rss_url

    async def fetch_schedule(self, start_date: date, end_date: date) -> list[RawEconomicEvent]:
        results: list[RawEconomicEvent] = []
        html = await self._get_text(self.source_url)
        if html is not None:
            for entry in find_date_entries(html, start_date=start_date, end_date=end_date):
                title = entry.nearby_text if "fomc" in entry.nearby_text.lower() or "meeting" in entry.nearby_text.lower() else "FOMC Meeting"
                results.append(_fomc_event(title, entry.event_date, self.source_url))
        # The RSS feed announces AT release time — it naturally only ever contains items already
        # published, so it contributes recently-released statements/minutes within the window
        # rather than future schedule entries.
        rss = await self._get_text(self.press_rss_url)
        if rss is not None:
            results.extend(parse_press_rss(rss, start_date, end_date, source_name="federal_reserve", source_url=self.press_rss_url, country="US", currency="USD"))
        return results


def _fomc_event(title: str, event_date: date, source_url: str) -> RawEconomicEvent:
    canonical = canonicalize_title(title)
    scheduled = datetime(event_date.year, event_date.month, event_date.day, 18, 0)  # FOMC statement ~2pm ET / 18:00 UTC (approx, DST-naive placeholder)
    return RawEconomicEvent(
        raw_name=title[:200],
        raw_scheduled_time=scheduled.isoformat() + "+00:00",
        provider_event_id=stable_event_id("federal_reserve", canonical, event_date.isoformat()),
        raw_country="US",
        raw_currency="USD",
        raw_importance="critical",
        raw_status="scheduled",
        raw_timezone="UTC",
        source_url=source_url,
    )
