"""Defensive date/title extraction for the HTML-based public sources (BEA, Census, Federal
Reserve, ECB). These four pages' exact DOM structure was not independently verified against a
live fetch from this environment (outbound access to *.gov/*.europa.eu was not available) — this
module deliberately favors resilience to markup changes over precision: it extracts
(date, nearby text) pairs from the page's visible text rather than depending on specific CSS
selectors, so a class-name or table-structure change is less likely to silently return nothing.

Every adapter using this must still be verified against the live page once deployed — a parser
mismatch here degrades to `empty_valid_schedule` (surfaced clearly in diagnostics, circuit-broken
with a long backoff) rather than crashing or fabricating dates, per the source-adapter contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from bs4 import BeautifulSoup

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_NAMES = "|".join(sorted(_MONTHS, key=len, reverse=True))

#: "January 15, 2026" / "Jan 15 2026" / "15 January 2026" — the handful of date spellings actually
#: seen on U.S. and EU government release-schedule pages.
_DATE_PATTERNS = (
    re.compile(rf"\b({_MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE),
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES})\.?,?\s+(\d{{4}})\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class ExtractedDateEntry:
    event_date: date
    nearby_text: str


def extract_visible_text_blocks(html: str) -> list[str]:
    """Strips scripts/styles and returns one text block per reasonably-sized element (table row,
    list item, or paragraph) — the unit `find_date_entries` scans for a date + a title together."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks: list[str] = []
    for element in soup.find_all(["tr", "li", "p", "div"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if text and 5 <= len(text) <= 400:
            blocks.append(text)
    return blocks


def find_date_entries(html: str, *, start_date: date, end_date: date) -> list[ExtractedDateEntry]:
    """Scans each visible text block for a date within [start_date, end_date] and pairs it with
    the block's own text as the nearby title — resilient to nesting/class changes because it
    never depends on a specific element being the "title cell" vs. "date cell"."""
    results: list[ExtractedDateEntry] = []
    seen: set[tuple[date, str]] = set()
    for block in extract_visible_text_blocks(html):
        parsed = _first_date_in(block)
        if parsed is None or not (start_date <= parsed <= end_date):
            continue
        key = (parsed, block)
        if key in seen:
            continue
        seen.add(key)
        results.append(ExtractedDateEntry(event_date=parsed, nearby_text=block))
    return results


def _first_date_in(text: str) -> date | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if groups[0].lower() in _MONTHS:
            month_name, day, year = groups
        else:
            day, month_name, year = groups
        month = _MONTHS.get(month_name.lower())
        if not month:
            continue
        try:
            return date(int(year), month, int(day))
        except ValueError:
            continue
    return None
