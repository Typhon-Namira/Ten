"""SSRF-safe URL validation for the Market Data Engine's keyless public-source adapters.

The generic safety checks (scheme, private/reserved IP, localhost) live in
`backend.app.core.net.ssrf`, shared with the economic calendar engine's public sources; this
module only adds the domain allowlist specific to market-data sources. Every `base_url` a
`ProviderConfig` can set — including an operator-supplied override — must pass through
`assert_safe_public_url()` before a provider is constructed.
"""

from __future__ import annotations

from backend.app.core.net.ssrf import ALLOWED_SCHEMES, UnsafePublicUrlError, resolved_addresses_are_safe
from backend.app.core.net.ssrf import assert_safe_public_url as _assert_safe_public_url

__all__ = ["ALLOWED_DOMAINS", "ALLOWED_SCHEMES", "UnsafePublicUrlError", "assert_safe_public_url", "resolved_addresses_are_safe"]

#: Every domain any `HttpMarketDataProvider` subclass is permitted to fetch from — keyed and
#: keyless alike. `HttpMarketDataProvider.__init__` checks every configured `base_url` against
#: this list unconditionally (defense in depth on "any configurable base_url", not just the new
#: keyless sources), so it also has to include the existing paid providers' known-good hosts.
ALLOWED_DOMAINS = frozenset(
    {
        # Keyless public sources (this migration).
        "prices.lbma.org.uk",
        "api.kraken.com",
        "www.okx.com",
        "okx.com",
        # Disabled-by-default legacy adapters — Yahoo/Stooq/Binance's `base_url` is still
        # SSRF-checked at construction even though they never run by default. See `adapters.py`.
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
        "stooq.com",
        "www.stooq.com",
        "api.binance.com",
        # Existing paid providers — unchanged behavior, just now also SSRF-checked.
        "api.twelvedata.com",
        "www.alphavantage.co",
        "financialmodelingprep.com",
        "api-fxpractice.oanda.com",
        "api-fxtrade.oanda.com",
    }
)


def assert_safe_public_url(url: str, *, allowed_domains: frozenset[str] = ALLOWED_DOMAINS) -> None:
    _assert_safe_public_url(url, allowed_domains=allowed_domains)
