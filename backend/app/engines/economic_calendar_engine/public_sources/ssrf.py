"""SSRF-safe URL validation for the public-source calendar adapters.

Every outbound request in this subsystem — the configured source URL and any manual-override URL
an administrator sets — must pass through `assert_safe_public_url()` first. The generic safety
checks (scheme, private/reserved IP, localhost) live in `backend.app.core.net.ssrf`, shared with
every other public-source adapter in TEN; this module only adds the domain allowlist that is
specific to the economic-calendar sources.
"""

from __future__ import annotations

from backend.app.core.net.ssrf import ALLOWED_SCHEMES, UnsafePublicUrlError, resolved_addresses_are_safe
from backend.app.core.net.ssrf import assert_safe_public_url as _assert_safe_public_url

__all__ = ["ALLOWED_DOMAINS", "ALLOWED_SCHEMES", "UnsafePublicUrlError", "assert_safe_public_url", "resolved_addresses_are_safe"]

#: Official domains this subsystem is permitted to fetch from. Adding a new source means adding
#: its domain here first — an administrator cannot point an adapter at an arbitrary URL (see
#: `public_sources.config.SourceOverride`), which is the actual SSRF boundary: not just "no
#: private IPs" but "no domain outside this explicit, reviewed list."
ALLOWED_DOMAINS = frozenset(
    {
        "www.bls.gov",
        "bls.gov",
        "www.bea.gov",
        "bea.gov",
        "apps.bea.gov",
        "www.federalreserve.gov",
        "federalreserve.gov",
        "www.census.gov",
        "census.gov",
        "www.dol.gov",
        "dol.gov",
        "www.ecb.europa.eu",
        "ecb.europa.eu",
    }
)


def assert_safe_public_url(url: str, *, allowed_domains: frozenset[str] = ALLOWED_DOMAINS) -> None:
    _assert_safe_public_url(url, allowed_domains=allowed_domains)
