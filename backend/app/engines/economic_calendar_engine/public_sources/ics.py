"""Minimal RFC 5545 (iCalendar) VEVENT parser — deliberately narrow: only the fields TEN's
calendar adapters actually need (SUMMARY, DTSTART, UID, DESCRIPTION, LAST-MODIFIED). No recurrence
expansion, no VALARM/VTIMEZONE component support beyond a bare TZID passthrough. Government
release-schedule ICS feeds (e.g. BLS) publish one VEVENT per release with an absolute DTSTART, so
this scope is sufficient without pulling in a full third-party iCalendar dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class IcsEvent:
    uid: str | None
    summary: str
    description: str | None
    dtstart: datetime
    dtstart_is_date_only: bool
    last_modified: datetime | None


def _unfold_lines(raw: str) -> list[str]:
    """RFC 5545 §3.1: a line beginning with a single space or tab is a continuation of the
    previous line, not a new property."""
    lines: list[str] = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        if line.startswith((" ", "\t")) and lines:
            lines[-1] += line[1:]
        elif line:
            lines.append(line)
    return lines


def _unescape(value: str) -> str:
    return value.replace("\\n", "\n").replace("\\N", "\n").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")


def _parse_property(line: str) -> tuple[str, dict[str, str], str]:
    name_and_params, _, value = line.partition(":")
    parts = name_and_params.split(";")
    name = parts[0].upper()
    params: dict[str, str] = {}
    for part in parts[1:]:
        key, _, val = part.partition("=")
        if key:
            params[key.upper()] = val
    return name, params, value


def _parse_datetime(value: str, params: dict[str, str]) -> tuple[datetime, bool]:
    value = value.strip()
    if params.get("VALUE") == "DATE" or (len(value) == 8 and value.isdigit()):
        parsed_date = date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC), True
    is_utc = value.endswith("Z")
    core = value.rstrip("Z")
    naive = datetime.strptime(core, "%Y%m%dT%H%M%S")
    if is_utc:
        return naive.replace(tzinfo=UTC), False
    tzid = params.get("TZID")
    if tzid:
        try:
            return naive.replace(tzinfo=ZoneInfo(tzid)).astimezone(UTC), False
        except ZoneInfoNotFoundError:
            pass
    # No timezone information at all (a "floating" time, RFC 5545 §3.3.5) — treating it as UTC is
    # a documented assumption, not a silent guess: callers see `dtstart_is_date_only=False` and a
    # source-level warning is the adapter's responsibility to raise if this matters.
    return naive.replace(tzinfo=UTC), False


def parse_ics_events(raw: str) -> list[IcsEvent]:
    """Parses every VEVENT block in `raw`. A malformed individual event (missing DTSTART/SUMMARY,
    unparseable date) is skipped rather than aborting the whole feed — one bad entry must not
    blank out an otherwise-valid release calendar."""
    events: list[IcsEvent] = []
    in_event = False
    current: dict[str, tuple[dict[str, str], str]] = {}
    for line in _unfold_lines(raw):
        name, params, value = _parse_property(line)
        if name == "BEGIN" and value.upper() == "VEVENT":
            in_event, current = True, {}
            continue
        if name == "END" and value.upper() == "VEVENT":
            in_event = False
            event = _build_event(current)
            if event is not None:
                events.append(event)
            continue
        if in_event:
            current[name] = (params, value)
    return events


def _build_event(fields: dict[str, tuple[dict[str, str], str]]) -> IcsEvent | None:
    if "SUMMARY" not in fields or "DTSTART" not in fields:
        return None
    summary = _unescape(fields["SUMMARY"][1])
    try:
        dtstart, is_date_only = _parse_datetime(fields["DTSTART"][1], fields["DTSTART"][0])
    except (ValueError, IndexError):
        return None
    last_modified = None
    if "LAST-MODIFIED" in fields:
        try:
            last_modified, _ = _parse_datetime(fields["LAST-MODIFIED"][1], fields["LAST-MODIFIED"][0])
        except (ValueError, IndexError):
            last_modified = None
    description = _unescape(fields["DESCRIPTION"][1]) if "DESCRIPTION" in fields else None
    uid = _unescape(fields["UID"][1]) if "UID" in fields else None
    return IcsEvent(uid=uid, summary=summary, description=description, dtstart=dtstart, dtstart_is_date_only=is_date_only, last_modified=last_modified)
