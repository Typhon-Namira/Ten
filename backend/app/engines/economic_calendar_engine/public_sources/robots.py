"""Minimal robots.txt evaluation — checked once per source at first use, not on every poll cycle.
Uses the standard library's `urllib.robotparser`, which already implements the standard matching
rules; this module only wires it up to TEN's async HTTP client and fails safe (treats an
unreachable/missing robots.txt as "allowed", since the absence of a robots.txt is not a
disallowance — but a robots.txt that explicitly disallows the path is always honored).
"""

from __future__ import annotations

import logging
from enum import StrEnum
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)


class RobotsPolicy(StrEnum):
    ALLOWED = "allowed"
    DISALLOWED = "disallowed"
    UNKNOWN = "unknown"


async def evaluate_robots_policy(client: httpx.AsyncClient, url: str, *, user_agent: str) -> RobotsPolicy:
    parts = urlsplit(url)
    robots_url = urljoin(f"{parts.scheme}://{parts.netloc}", "/robots.txt")
    try:
        response = await client.get(robots_url, timeout=10)
    except httpx.HTTPError:
        logger.info("robots.txt unreachable, defaulting to allowed: url=%s", robots_url)
        return RobotsPolicy.UNKNOWN
    if response.status_code >= 400:
        # No robots.txt (404) or a server error is not a disallowance — most .gov sites don't
        # publish one at all, and its absence has never meant "no automated access."
        return RobotsPolicy.UNKNOWN
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    try:
        allowed = parser.can_fetch(user_agent, url)
    except Exception:  # a malformed robots.txt must never crash source evaluation
        return RobotsPolicy.UNKNOWN
    return RobotsPolicy.ALLOWED if allowed else RobotsPolicy.DISALLOWED
