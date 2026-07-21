"""SSRF-safe URL validation for the public-source calendar adapters.

Every outbound request in this subsystem — the configured source URL and any manual-override URL
an administrator sets — must pass through `assert_safe_public_url()` first. Nothing here bypasses
CAPTCHA, Cloudflare challenges, or authentication; it only decides whether a URL is *structurally*
safe to request at all (right scheme, not a private/internal address, on an approved domain).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})

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


class UnsafePublicUrlError(ValueError):
    """Raised when a URL fails the SSRF safety check — never caught silently; callers must treat
    this as a configuration error, not a transient fetch failure."""


def _is_private_or_reserved(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # not a literal IP — resolved separately by the caller
    return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast or addr.is_unspecified


def assert_safe_public_url(url: str, *, allowed_domains: frozenset[str] = ALLOWED_DOMAINS) -> None:
    """Rejects non-HTTP(S) schemes, literal private/loopback/link-local IPs, `localhost`, and any
    host outside `allowed_domains`. Does not perform DNS resolution itself (that happens inside
    httpx at request time); this is a fast, dependency-free pre-check against the URL as written."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafePublicUrlError(f"scheme not allowed: {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise UnsafePublicUrlError("URL has no host")
    if host in {"localhost", "0.0.0.0"} or host.endswith(".localhost"):
        raise UnsafePublicUrlError(f"host not allowed: {host!r}")
    if _is_private_or_reserved(host):
        raise UnsafePublicUrlError(f"host resolves to a private/reserved address: {host!r}")
    if host not in allowed_domains:
        raise UnsafePublicUrlError(f"host not on the official-domain allowlist: {host!r}")


def resolved_addresses_are_safe(host: str) -> bool:
    """Defense in depth: resolve the hostname and reject it if every A/AAAA record is itself
    private/loopback/link-local — catches DNS rebinding onto an internal address. Best-effort;
    resolution failures are treated as unsafe (fail closed) rather than silently passing through."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not infos:
        return False
    return not any(_is_private_or_reserved(str(info[4][0])) for info in infos)
