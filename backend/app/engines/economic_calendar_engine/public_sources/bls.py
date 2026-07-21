"""U.S. Bureau of Labor Statistics — official release-schedule iCalendar feed.

Authoritative because BLS is the issuing agency for every event type this adapter produces (CPI,
Core CPI, PPI, the Employment Situation/Nonfarm Payrolls, Unemployment Rate, Average Hourly
Earnings, JOLTS, Import/Export Prices, Employment Cost Index) — there is no more authoritative
source for "when will BLS publish X" than BLS's own calendar. The ICS feed is documented on BLS's
own help page (bls.gov/help/hlpical.htm) as the intended machine-readable subscription mechanism,
so parsing it is exactly the access pattern BLS designed for — not a scrape of incidental content.
"""

from __future__ import annotations

from datetime import date

from ..models import SourceType
from .base import HttpPublicCalendarSource, RawEconomicEvent, stable_event_id
from .ics import parse_ics_events
from .impact import canonicalize_title

SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/bls.ics"


class BlsPublicCalendarSource(HttpPublicCalendarSource):
    def __init__(self, *, source_url: str = SCHEDULE_URL, timeout_seconds: float = 15) -> None:
        super().__init__("bls", source_url=source_url, timeout_seconds=timeout_seconds, source_type=SourceType.ICS_CALENDAR)

    async def fetch_schedule(self, start_date: date, end_date: date) -> list[RawEconomicEvent]:
        raw = await self._get_text(self.source_url)
        if raw is None:
            return []
        events = parse_ics_events(raw)
        results: list[RawEconomicEvent] = []
        for event in events:
            event_date = event.dtstart.date()
            if not (start_date <= event_date <= end_date):
                continue
            canonical = canonicalize_title(event.summary)
            results.append(
                RawEconomicEvent(
                    raw_name=event.summary,
                    raw_scheduled_time=event.dtstart.isoformat(),
                    provider_event_id=event.uid or stable_event_id("bls", event.summary, event_date.isoformat()),
                    raw_country="US",
                    raw_currency="USD",
                    raw_importance="high" if canonical in {"cpi", "core_cpi", "nonfarm_payrolls", "ppi", "unemployment_rate"} else "medium",
                    raw_status="scheduled",
                    raw_timezone="UTC",
                    source_url=self.source_url,
                )
            )
        return results
