"""Generic SSRF-safe URL validation, shared by every public-source adapter across TEN.

Nothing here bypasses CAPTCHA, Cloudflare challenges, or authentication; it only decides whether a
URL is *structurally* safe to request at all (right scheme, not a private/internal address, on an
approved domain). The domain allowlist itself is deliberately NOT defined here — it is engine- and
source-specific policy, supplied by the caller via `allowed_domains`.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafePublicUrlError(ValueError):
    """Raised when a URL fails the SSRF safety check — never caught silently; callers must treat
    this as a configuration error, not a transient fetch failure."""


def is_private_or_reserved(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # not a literal IP — resolved separately by the caller
    return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast or addr.is_unspecified


def assert_safe_public_url(url: str, *, allowed_domains: frozenset[str]) -> None:
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
    if is_private_or_reserved(host):
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
    return not any(is_private_or_reserved(str(info[4][0])) for info in infos)
