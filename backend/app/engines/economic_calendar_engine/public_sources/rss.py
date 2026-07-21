"""Generic RSS/XML result-feed parsing shared by every adapter that has an official press-release
RSS feed (Federal Reserve, ECB) — these feeds announce AT release time, so they only ever surface
already-published items within the requested window, never a forward schedule."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime

from .base import RawEconomicEvent, stable_event_id
from .impact import canonicalize_title


def parse_press_rss(raw_xml: str, start_date: date, end_date: date, *, source_name: str, source_url: str, country: str, currency: str, importance: str = "critical") -> list[RawEconomicEvent]:
    try:
        # `raw_xml` only ever comes from an SSRF-allowlisted official government/institutional domain.
        root = ET.fromstring(raw_xml)
    except ET.ParseError:
        return []
    results: list[RawEconomicEvent] = []
    for item in root.iter("item"):
        title_el, pub_date_el = item.find("title"), item.find("pubDate")
        if title_el is None or title_el.text is None or pub_date_el is None or pub_date_el.text is None:
            continue
        try:
            published = _parse_rfc822(pub_date_el.text)
        except (ValueError, TypeError):
            continue
        if not (start_date <= published.date() <= end_date):
            continue
        canonical = canonicalize_title(title_el.text)
        results.append(
            RawEconomicEvent(
                raw_name=title_el.text[:200],
                raw_scheduled_time=published.isoformat(),
                provider_event_id=stable_event_id(source_name, canonical, published.isoformat()),
                raw_country=country,
                raw_currency=currency,
                raw_importance=importance,
                raw_status="released",
                raw_timezone="UTC",
                source_url=source_url,
            )
        )
    return results


def _parse_rfc822(value: str) -> datetime:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
