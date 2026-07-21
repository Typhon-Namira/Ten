"""HTTP provider adapters. Provider-specific formats terminate in this module."""

import asyncio
import logging
import os
from abc import abstractmethod
from datetime import UTC, datetime
from time import monotonic, perf_counter
from typing import Any

import httpx

from backend.app.core.net.robots import RobotsPolicy, evaluate_robots_policy

from .exceptions import ProviderRateLimitedError, ProviderResponseError
from .models import Candle, Timeframe
from .providers import (
    MarketDataProvider,
    ProviderCapabilities,
    ProviderName,
    ProviderQuota,
    ProviderRequest,
)
from .ssrf import assert_safe_public_url
from .symbols import provider_symbol

logger = logging.getLogger(__name__)

#: Identifies TEN to every public source it fetches from without a key — matches the convention
#: already established in `economic_calendar_engine/public_sources/base.py`.
USER_AGENT = "TEN-MarketData/1.0 (+https://github.com/; institutional market analysis; contact via repository issues)"


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _http_error_detail(error: ProviderResponseError) -> tuple[int | None, str]:
    """Best-effort status code + message extraction from a `ProviderResponseError` — `_get()`
    chains the original `httpx` exception via `__cause__`, so this recovers the real HTTP status
    and response body instead of only the generic "request failed" wrapper message."""
    if isinstance(error, ProviderRateLimitedError):
        return 429, str(error)
    cause = error.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        return cause.response.status_code, cause.response.text[:300]
    return None, str(cause) if cause else str(error)


def _classify_http_status(status_code: int | None) -> str:
    """A response was received in every branch here — a 404 proves the server was reached and is
    a completely different failure than a dropped connection, so it is never labeled
    "unreachable". `None` (no status recovered at all) means the request never got a response."""
    if status_code is None:
        return "unreachable"
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "invalid_endpoint"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "server_error"
    if status_code >= 400:
        return "invalid_endpoint"
    return "reachable"


class HttpMarketDataProvider(MarketDataProvider):
    #: Overridden `False` by keyless public-source adapters (LBMA, Kraken, OKX, ...) — everything
    #: else in this shared base (retry/backoff, rate gating, latency tracking, quota tracking) is
    #: identical whether or not a provider needs a key, so keyless adapters reuse this base class
    #: directly rather than duplicating it.
    requires_api_key: bool = True
    #: Overridden `True` by every public-source adapter added in the keyless migration (LBMA,
    #: Kraken, OKX, and the disabled-by-default Yahoo/Stooq/Binance legacy adapters). Left `False`
    #: for the pre-existing keyed adapters so their tests' `base_url="https://example.test"`
    #: fixtures keep working unchanged — SSRF-checking a developer-configured, already-trusted paid
    #: API host adds little, while enforcing it retroactively would break existing test contracts
    #: for no safety benefit.
    enforce_domain_allowlist: bool = False
    #: Overridden `True`/`False` per public-source adapter — `False` means "never checked, always
    #: treated as allowed" (used by sources like LBMA that don't need the extra request), `True`
    #: means robots.txt must be evaluated before the first real fetch (used by every adapter whose
    #: host is known or suspected to restrict automated access).
    check_robots: bool = False

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str,
        timeout_seconds: float = 15,
        account_id: str | None = None,
        requests_per_minute: int | None = None,
        max_rate_limit_retries: int = 2,
    ) -> None:
        if self.requires_api_key and not api_key:
            raise ValueError(f"{self.provider_name.value} API key is required")
        if self.enforce_domain_allowlist:
            assert_safe_public_url(base_url)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.account_id = account_id
        self._client = httpx.AsyncClient(timeout=timeout_seconds, headers={"User-Agent": USER_AGENT})
        self._robots_policy: RobotsPolicy = RobotsPolicy.UNKNOWN
        self._robots_checked = False
        # `last_latency_ms` is the true bottleneck — the raw duration of the single HTTP call that
        # actually finished last. `last_total_latency_ms` is the full wall-clock time `_get()`
        # spent including every 429 retry's backoff sleep and rate-gate queueing wait. Reporting
        # only a wrapping timer around the whole retrying call (the old behavior, still visible as
        # `ProviderManager._execute()`'s own outer timer) makes a provider that took 400ms to
        # actually respond look like it took 7+ seconds once two backoff sleeps are folded in —
        # indistinguishable from a genuinely slow API without this split.
        self.last_latency_ms = 0.0
        self.last_total_latency_ms = 0.0
        self.last_retry_count = 0
        self.last_rate_gate_wait_ms = 0.0
        self._quota = ProviderQuota()
        self._min_interval_seconds = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._max_rate_limit_retries = max_rate_limit_retries
        self._rate_gate = asyncio.Lock()
        self._next_allowed_at: float = 0.0

    async def _await_rate_gate(self) -> None:
        """Proactively pace requests to `requests_per_minute` (config-driven, opt-in).

        Root cause this addresses: bootstrap loops through every configured timeframe
        sequentially with zero delay between requests. Against a provider with a tight
        free-tier quota (e.g. a handful of requests/minute), that alone is enough to
        trip a 429 before the loop even finishes. This lock also means every caller of
        this provider instance — the background worker AND any concurrent dashboard-
        triggered `refresh=True` fetch — shares one pacing queue instead of racing.
        """
        if self._min_interval_seconds <= 0:
            return
        async with self._rate_gate:
            now = monotonic()
            wait = self._next_allowed_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = monotonic()
            self._next_allowed_at = now + self._min_interval_seconds

    async def _get(self, path: str, *, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        total_started = perf_counter()
        rate_limited: ProviderRateLimitedError | None = None
        for attempt in range(self._max_rate_limit_retries + 1):
            gate_started = perf_counter()
            await self._await_rate_gate()
            self.last_rate_gate_wait_ms = (perf_counter() - gate_started) * 1000
            started = perf_counter()
            try:
                response = await self._client.get(f"{self.base_url}{path}", params=params, headers=headers)
                if response.status_code == 429:
                    self._update_quota(response.headers)
                    retry_after = _parse_retry_after(response.headers)
                    rate_limited = ProviderRateLimitedError(f"{self.provider_name.value} rate limited (429)", retry_after_seconds=retry_after)
                    if attempt < self._max_rate_limit_retries:
                        await asyncio.sleep(retry_after if retry_after is not None else min(2**attempt, 30))
                        continue
                    raise rate_limited
                response.raise_for_status()
                self._update_quota(response.headers)
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise ProviderResponseError(f"{self.provider_name.value} request failed") from exc
            finally:
                self.last_latency_ms = (perf_counter() - started) * 1000
                self.last_retry_count = attempt
                self.last_total_latency_ms = (perf_counter() - total_started) * 1000
        raise rate_limited or ProviderResponseError(f"{self.provider_name.value} request failed")  # pragma: no cover - loop always returns/raises above

    def _update_quota(self, headers: httpx.Headers) -> None:
        def integer(name: str) -> int | None:
            value = headers.get(name)
            return int(value) if value and value.isdigit() else None

        self._quota = ProviderQuota(
            limit=integer("x-ratelimit-limit"),
            remaining=integer("x-ratelimit-remaining"),
        )

    async def quota(self) -> ProviderQuota:
        return self._quota

    async def close(self) -> None:
        await self._client.aclose()

    async def _ensure_robots_allowed(self, url: str) -> None:
        """Checked once per provider instance, not on every poll — mirrors
        `HttpPublicCalendarSource._ensure_robots_checked()` in the economic calendar engine.
        Raises `ProviderResponseError` on an explicit `Disallow`; never bypasses it."""
        if not self.check_robots or self._robots_checked:
            return
        self._robots_policy = await evaluate_robots_policy(self._client, url, user_agent=USER_AGENT)
        self._robots_checked = True
        if self._robots_policy is RobotsPolicy.DISALLOWED:
            logger.warning("market_data.provider.robots_disallowed: provider=%s url=%s", self.provider_name.value, url)
            raise ProviderResponseError(f"{self.provider_name.value} robots.txt disallows automated access to {url}")

    async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        candles = await self.fetch_history(ProviderRequest(symbol=symbol, timeframe=timeframe, limit=1))
        if not candles:
            raise ProviderResponseError(f"{self.provider_name.value} returned no candles")
        return candles[-1]

    @abstractmethod
    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        raise NotImplementedError


def _parse_timestamp(value: str | int | float) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _period_has_closed(timestamp: datetime, timeframe: Timeframe, *, now: datetime | None = None) -> bool:
    """No-lookahead guard shared by every public-source adapter: a candle is final only once its
    own period has fully elapsed. None of the new keyless sources provide OANDA's explicit
    `complete` flag reliably (OKX does via `confirm`; Kraken and LBMA provide nothing), so this
    computed check is the one mechanism every adapter can rely on — it is deliberately applied even
    where a source-native completion flag also exists, as a second, source-independent backstop."""
    boundary = now or datetime.now(UTC)
    return timestamp + timeframe.duration <= boundary


class TwelveDataProvider(HttpMarketDataProvider):
    provider_name = ProviderName.TWELVE_DATA
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        spread=False,
        supported_symbols=("XAUUSD",),
        supported_timeframes=tuple(Timeframe),
        maximum_history_candles=5000,
    )
    _intervals = {
        Timeframe.M1: "1min",
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "1h",
        Timeframe.H4: "4h",
        Timeframe.D1: "1day",
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        data = await self._get(
            "/time_series",
            params={
                "symbol": provider_symbol(self.provider_name.value, request.symbol),
                "interval": self._intervals[request.timeframe],
                "outputsize": min(request.limit, 5000),
                "start_date": request.start.isoformat() if request.start else None,
                "end_date": request.end.isoformat() if request.end else None,
                "apikey": self.api_key,
                "timezone": "UTC",
                "order": "ASC",
            },
        )
        if data.get("status") == "error" or not isinstance(data.get("values"), list):
            raise ProviderResponseError(f"TwelveData invalid response: {data.get('message', 'missing values')}")
        return [
            Candle(
                timestamp=_parse_timestamp(row["datetime"]),
                symbol=request.symbol,
                timeframe=request.timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0),
                provider=self.provider_name.value,
            )
            for row in data["values"]
        ]


class AlphaVantageProvider(HttpMarketDataProvider):
    provider_name = ProviderName.ALPHA_VANTAGE
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        supported_symbols=("XAUUSD",),
        supported_timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.D1),
    )
    _intervals = {
        Timeframe.M1: "1min",
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "60min",
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        if request.timeframe == Timeframe.D1:
            function = "FX_DAILY"
            interval = None
            series_key = "Time Series FX (Daily)"
        else:
            function = "FX_INTRADAY"
            interval = self._intervals[request.timeframe]
            series_key = f"Time Series FX ({interval})"
        params = {
            "function": function,
            "from_symbol": "XAU",
            "to_symbol": "USD",
            "interval": interval,
            "outputsize": "full" if request.limit > 100 else "compact",
            "apikey": self.api_key,
        }
        data = await self._get("", params=params)
        rows = data.get(series_key)
        if not isinstance(rows, dict):
            raise ProviderResponseError(f"AlphaVantage invalid response: {data.get('Note') or data.get('Error Message') or 'missing series'}")
        candles = [
            Candle(
                timestamp=_parse_timestamp(timestamp),
                symbol=request.symbol,
                timeframe=request.timeframe,
                open=float(row["1. open"]),
                high=float(row["2. high"]),
                low=float(row["3. low"]),
                close=float(row["4. close"]),
                volume=0,
                provider=self.provider_name.value,
            )
            for timestamp, row in rows.items()
        ]
        return sorted(candles, key=lambda candle: candle.timestamp)[-request.limit :]


class FinancialModelingPrepProvider(HttpMarketDataProvider):
    provider_name = ProviderName.FMP
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        spread=False,  # no bid/ask on this feed — never fabricate a spread value from it
        live_quote=True,
        supported_symbols=("XAUUSD",),
        # Only intervals with a confirmed, documented `/stable` endpoint. 15-minute and 30-minute
        # `historical-chart` routes were NOT independently verified against current FMP
        # documentation — do not add them back without confirming the exact endpoint first.
        supported_timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.H1, Timeframe.D1),
    )
    #: `/stable` is a base URL only — FMP 404s if it is ever requested directly.
    QUOTE_ENDPOINT = "/quote-short"
    DAILY_ENDPOINT = "/historical-price-eod/full"
    _intervals = {
        Timeframe.M1: "1min",
        Timeframe.M5: "5min",
        Timeframe.H1: "1hour",
    }

    async def check_connectivity(self) -> None:
        """Active health probe against a real, lightweight, always-available endpoint — never the
        bare `/stable` base URL, which is not itself a resource and 404s."""
        await self._get(self.QUOTE_ENDPOINT, params={"symbol": provider_symbol(self.provider_name.value, "XAUUSD"), "apikey": self.api_key})

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        # FMP's stable API moved the symbol out of the path and into a query parameter (the
        # legacy /api/v3 endpoint took it as a path segment) — interval stays in the path, e.g.
        # /stable/historical-chart/1hour?symbol=GCUSD. Daily data is a separate endpoint entirely
        # (not a `historical-chart` interval) with a different response shape.
        provider_symbol_value = provider_symbol(self.provider_name.value, request.symbol)
        is_daily = request.timeframe == Timeframe.D1
        path = self.DAILY_ENDPOINT if is_daily else f"/historical-chart/{self._intervals[request.timeframe]}"
        params = {
            "symbol": provider_symbol_value,
            "from": request.start.date().isoformat() if request.start else None,
            "to": request.end.date().isoformat() if request.end else None,
            "apikey": self.api_key,
        }
        try:
            data = await self._get(path, params=params)
        except ProviderResponseError as exc:
            status_code, message = _http_error_detail(exc)
            category = _classify_http_status(status_code)
            level = logger.error if category in {"unauthorized", "forbidden", "invalid_endpoint"} else logger.warning
            # Never silently fail — every failure is logged with method, sanitized URL (apikey
            # never included), TEN's canonical symbol, the provider-specific symbol actually sent,
            # the requested timeframe, status, classification, latency, retry count, and the
            # provider's own message, so this is diagnosable from logs alone.
            level(
                "Financial Modeling Prep request failed: method=GET url=%s%s canonical_symbol=%s provider_symbol=%s timeframe=%s status=%s classification=%s "
                "latency_ms=%.1f retry_count=%s message=%s",
                self.base_url, path, request.symbol, provider_symbol_value, request.timeframe.value, status_code, category,
                self.last_latency_ms, self.last_retry_count, message,
            )
            raise
        logger.info(
            "Financial Modeling Prep request succeeded: method=GET url=%s%s canonical_symbol=%s provider_symbol=%s timeframe=%s latency_ms=%.1f retry_count=%s",
            self.base_url, path, request.symbol, provider_symbol_value, request.timeframe.value, self.last_latency_ms, self.last_retry_count,
        )
        rows = data.get("historical") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ProviderResponseError("Financial Modeling Prep response must be a list")
        candles: list[Candle] = []
        seen_timestamps: set[datetime] = set()
        skipped = 0
        for row in rows:
            try:
                timestamp = _parse_timestamp(row["date"])
                open_, high, low, close = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if timestamp in seen_timestamps:
                continue  # duplicate row within this response — reject before it ever reaches persistence
            seen_timestamps.add(timestamp)
            candles.append(
                Candle(
                    timestamp=timestamp,
                    # `request.symbol` is TEN's own canonical symbol ("XAUUSD") — `provider_symbol_value`
                    # ("GCUSD") is only ever used for the outgoing request, never stored.
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=float(row.get("volume") or 0),
                    # This feed has no real bid/ask; `spread` defaults to 0.0 on the `Candle` model
                    # itself (not Optional), so `capabilities.spread=False` above is what tells
                    # every downstream consumer to treat that 0.0 as "not measured", not "flat".
                    provider=self.provider_name.value,
                )
            )
        if skipped:
            logger.warning("Financial Modeling Prep returned %s malformed row(s) that were skipped: endpoint=%s canonical_symbol=%s", skipped, path, request.symbol)
        return sorted(candles, key=lambda candle: candle.timestamp)[-request.limit :]


class OandaProvider(HttpMarketDataProvider):
    provider_name = ProviderName.OANDA
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        spread=True,
        supported_symbols=("XAUUSD",),
        supported_timeframes=tuple(Timeframe),
        maximum_history_candles=5000,
    )
    _granularity = {
        Timeframe.M1: "M1",
        Timeframe.M5: "M5",
        Timeframe.M15: "M15",
        Timeframe.M30: "M30",
        Timeframe.H1: "H1",
        Timeframe.H4: "H4",
        Timeframe.D1: "D",
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        data = await self._get(
            f"/v3/instruments/{provider_symbol(self.provider_name.value, request.symbol)}/candles",
            params={
                "granularity": self._granularity[request.timeframe],
                "count": min(request.limit, 5000),
                "from": request.start.isoformat() if request.start else None,
                "to": request.end.isoformat() if request.end else None,
                "price": "MBA",
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        rows = data.get("candles")
        if not isinstance(rows, list):
            raise ProviderResponseError("OANDA response missing candles")
        candles: list[Candle] = []
        for row in rows:
            mid = row.get("mid")
            if not mid or not row.get("complete", True):
                continue
            bid = row.get("bid", mid)
            ask = row.get("ask", mid)
            spread = max(0.0, float(ask["c"]) - float(bid["c"]))
            candles.append(
                Candle(
                    timestamp=_parse_timestamp(row["time"]),
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=float(row.get("volume") or 0),
                    spread=spread,
                    provider=self.provider_name.value,
                )
            )
        return candles


class LbmaGoldPriceProvider(HttpMarketDataProvider):
    """The London Bullion Market Association's own daily AM/PM gold price fix — the actual
    industry benchmark the global gold market prices against. Published at
    https://prices.lbma.org.uk with no robots.txt at all (checked empirically: an empty response,
    not an explicit Disallow) and no authentication of any kind. This is the one source in this
    engine that is NOT a proxy instrument — it is the official reference price itself.

    LBMA publishes one AM fixing and, on most trading days, a second PM fixing — never intraday
    ticks — so this adapter only supports `D1`. Each daily candle uses the AM fix as `open` and the
    PM fix (when published that day) as `close`, with `high`/`low` taken as the max/min of the two
    real observed fixes — both are genuine published values, never interpolated or fabricated. On
    a day with only an AM fix (LBMA occasionally omits the PM fix, e.g. around some holidays), all
    four OHLC fields equal that single fix rather than guessing a second value.
    """

    provider_name = ProviderName.LBMA_GOLD_PRICE
    requires_api_key = False
    enforce_domain_allowlist = True
    check_robots = True
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=False,  # a fixing is set once or twice per session, never polled live
        volume=False,  # LBMA publishes no trade volume alongside the fix
        supported_symbols=("XAUUSD",),
        supported_timeframes=(Timeframe.D1,),
    )
    AM_ENDPOINT = "/json/gold_am.json"
    PM_ENDPOINT = "/json/gold_pm.json"

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        if request.timeframe != Timeframe.D1:
            raise ProviderResponseError("LBMA Gold Price only publishes daily fixings")
        await self._ensure_robots_allowed(f"{self.base_url}{self.AM_ENDPOINT}")
        am_data = await self._get(self.AM_ENDPOINT, params={})
        if not isinstance(am_data, list):
            raise ProviderResponseError("LBMA gold_am.json response must be a list")
        pm_by_date: dict[str, float] = {}
        try:
            pm_data = await self._get(self.PM_ENDPOINT, params={})
            if isinstance(pm_data, list):
                for row in pm_data:
                    try:
                        value = row["v"][0]
                    except (KeyError, IndexError, TypeError):
                        continue
                    if value is not None:
                        pm_by_date[row["d"]] = float(value)
        except ProviderResponseError:
            # The PM fix is best-effort supplementary detail — an AM-only candle (O=H=L=C) is
            # still honest, real data, so a failed PM fetch must not fail the whole request.
            logger.warning("LBMA PM fix fetch failed; continuing with AM-only candles: provider=%s", self.provider_name.value)
        today = datetime.now(UTC).date()
        candles: list[Candle] = []
        for row in am_data:
            try:
                day = row["d"]
                am_value = row["v"][0]
            except (KeyError, IndexError, TypeError):
                continue
            if am_value is None:
                continue  # LBMA marks a non-trading day (e.g. bank holiday) with a null value
            timestamp = datetime.fromisoformat(day).replace(tzinfo=UTC)
            if timestamp.date() >= today:
                continue  # no-lookahead: only a fully-elapsed trading day's fix is ever final
            if request.start and timestamp < request.start:
                continue
            if request.end and timestamp > request.end:
                continue
            am = float(am_value)
            pm = pm_by_date.get(day)
            candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    timeframe=Timeframe.D1,
                    open=am,
                    high=max(am, pm) if pm is not None else am,
                    low=min(am, pm) if pm is not None else am,
                    close=pm if pm is not None else am,
                    volume=0,
                    provider=self.provider_name.value,
                )
            )
        return sorted(candles, key=lambda candle: candle.timestamp)[-request.limit :]


# Both PAXG (Paxos Gold, used by KrakenGoldProxyProvider) and XAUT (Tether Gold, used by
# OkxGoldProxyProvider) are ERC-20 tokens redeemable for allocated LBMA-grade physical gold bars,
# each 1 token = 1 fine troy ounce. Both trade on open, liquid exchange order books and closely
# track the LBMA spot benchmark, but neither IS spot XAU/USD: exchange supply/demand can introduce
# a small premium or discount versus the benchmark, particularly in thin liquidity. Per the
# explicit migration brief, these are used only as supplementary, clearly-labeled proxy
# instruments for intraday coverage and cross-source validation — never as the sole source for a
# request. Deliberately two different token issuers on two different exchanges, so a single
# issuer's peg anomaly or a single exchange's order-book glitch cannot look like two
# independently-confirming sources agreeing.


class KrakenGoldProxyProvider(HttpMarketDataProvider):
    """Kraken's public, keyless OHLC endpoint for PAXG/USD (Paxos Gold) — a supplementary,
    clearly-labeled gold-token proxy instrument, never the sole source (see the module-level note
    above this class). `api.kraken.com` publishes no robots.txt at all (a 404, not an explicit
    Disallow) — treated as allowed per TEN's robots policy (absence is not a disallowance)."""

    provider_name = ProviderName.KRAKEN
    requires_api_key = False
    enforce_domain_allowlist = True
    check_robots = True
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        volume=True,
        supported_symbols=("XAUUSD",),
        supported_timeframes=tuple(Timeframe),
        maximum_history_candles=720,  # Kraken's public OHLC endpoint returns ~720 recent points
    )
    OHLC_ENDPOINT = "/0/public/OHLC"
    _intervals = {
        Timeframe.M1: 1,
        Timeframe.M5: 5,
        Timeframe.M15: 15,
        Timeframe.M30: 30,
        Timeframe.H1: 60,
        Timeframe.H4: 240,
        Timeframe.D1: 1440,
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        pair = provider_symbol(self.provider_name.value, request.symbol)
        await self._ensure_robots_allowed(f"{self.base_url}{self.OHLC_ENDPOINT}")
        data = await self._get(self.OHLC_ENDPOINT, params={"pair": pair, "interval": self._intervals[request.timeframe]})
        if not isinstance(data, dict):
            raise ProviderResponseError("Kraken OHLC response must be an object")
        errors = data.get("error")
        if errors:
            raise ProviderResponseError(f"Kraken error: {errors}")
        result = data.get("result")
        rows = result.get(pair) if isinstance(result, dict) else None
        if not isinstance(rows, list) and isinstance(result, dict):
            # Kraken's response key for a pair does not always echo the exact requested spelling —
            # `result` otherwise only ever contains a `last` pagination cursor alongside it, so a
            # single remaining series is unambiguously the one requested.
            candidates = {key: value for key, value in result.items() if key != "last" and isinstance(value, list)}
            if len(candidates) == 1:
                rows = next(iter(candidates.values()))
        if not isinstance(rows, list):
            raise ProviderResponseError("Kraken OHLC response missing the requested pair's series")
        now = datetime.now(UTC)
        candles: list[Candle] = []
        for row in rows:
            try:
                timestamp = datetime.fromtimestamp(float(row[0]), UTC)
                open_, high, low, close = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                volume = float(row[6])
            except (IndexError, TypeError, ValueError):
                continue
            if not _period_has_closed(timestamp, request.timeframe, now=now):
                continue  # Kraken's own docs: the last row is always the still-forming interval
            if request.start and timestamp < request.start:
                continue
            if request.end and timestamp > request.end:
                continue
            candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    provider=self.provider_name.value,
                )
            )
        return sorted(candles, key=lambda candle: candle.timestamp)[-request.limit :]


class OkxGoldProxyProvider(HttpMarketDataProvider):
    """OKX's public, keyless candles endpoint for XAUT/USDT (Tether Gold) — a supplementary,
    clearly-labeled gold-token proxy instrument, never the sole source (see the module-level note
    above `KrakenGoldProxyProvider`). `www.okx.com/robots.txt` has no blanket disallow and
    explicitly `Allow`s `/api/*?` query-parameterized paths — the clearest allow signal found
    during this migration."""

    provider_name = ProviderName.OKX
    requires_api_key = False
    enforce_domain_allowlist = True
    check_robots = True
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        volume=True,
        supported_symbols=("XAUUSD",),
        supported_timeframes=tuple(Timeframe),
        maximum_history_candles=300,  # OKX's public candles endpoint caps `limit` at 300 per call
    )
    CANDLES_ENDPOINT = "/api/v5/market/candles"
    _bars = {
        Timeframe.M1: "1m",
        Timeframe.M5: "5m",
        Timeframe.M15: "15m",
        Timeframe.M30: "30m",
        Timeframe.H1: "1H",
        Timeframe.H4: "4H",
        Timeframe.D1: "1D",
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        inst_id = provider_symbol(self.provider_name.value, request.symbol)
        await self._ensure_robots_allowed(f"{self.base_url}{self.CANDLES_ENDPOINT}")
        data = await self._get(
            self.CANDLES_ENDPOINT, params={"instId": inst_id, "bar": self._bars[request.timeframe], "limit": min(request.limit, 300)}
        )
        if not isinstance(data, dict) or data.get("code") != "0":
            raise ProviderResponseError(f"OKX error: {data.get('msg') if isinstance(data, dict) else 'invalid response'}")
        rows = data.get("data")
        if not isinstance(rows, list):
            raise ProviderResponseError("OKX candles response missing data")
        now = datetime.now(UTC)
        candles: list[Candle] = []
        for row in rows:
            try:
                timestamp = datetime.fromtimestamp(int(row[0]) / 1000, UTC)
                open_, high, low, close, volume = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
                confirmed = row[8] == "1"
            except (IndexError, TypeError, ValueError):
                continue
            # `confirm` is OKX's own explicit completion flag — trusted first; the computed
            # period-elapsed check runs as a second, source-independent backstop regardless.
            if not confirmed or not _period_has_closed(timestamp, request.timeframe, now=now):
                continue
            if request.start and timestamp < request.start:
                continue
            if request.end and timestamp > request.end:
                continue
            candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    provider=self.provider_name.value,
                )
            )
        # OKX returns newest-first.
        return sorted(candles, key=lambda candle: candle.timestamp)[-request.limit :]


class YahooFinanceProvider(HttpMarketDataProvider):
    """Yahoo Finance's public `/v8/finance/chart` endpoint — technically keyless and returns clean
    OHLCV JSON for `GC=F` (COMEX gold futures; Yahoo no longer serves `XAUUSD=X`/`XAU=X` spot from
    this endpoint as of this migration — both return "No data found, symbol may be delisted").

    **Permanently disabled by default and not part of the active provider set.** Both
    `query1.finance.yahoo.com` and `query2.finance.yahoo.com` publish a robots.txt with
    `User-agent: * / Disallow: /` — an explicit, blanket disallow of all automated access, checked
    empirically during this migration. `check_robots=True` below means this adapter will refuse to
    fetch and raise even if an operator manually flips `enabled: true` in config, rather than
    silently bypassing that policy. Kept fully implemented (not a stub) in case Yahoo's robots.txt
    policy changes in the future, or so an operator can consciously fork/override it having
    understood the ToS implication — never enable this without re-checking robots.txt first."""

    provider_name = ProviderName.YAHOO_FINANCE
    requires_api_key = False
    enforce_domain_allowlist = True
    check_robots = True
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        volume=True,
        supported_symbols=("XAUUSD",),
        supported_timeframes=tuple(Timeframe),
    )
    CHART_ENDPOINT = "/v8/finance/chart"
    _intervals = {
        Timeframe.M1: "1m",
        Timeframe.M5: "5m",
        Timeframe.M15: "15m",
        Timeframe.M30: "30m",
        Timeframe.H1: "60m",
        Timeframe.H4: "60m",  # Yahoo has no native 4h interval; H4 is aggregated from H1 below
        Timeframe.D1: "1d",
    }
    _ranges = {
        Timeframe.M1: "1d",
        Timeframe.M5: "5d",
        Timeframe.M15: "5d",
        Timeframe.M30: "1mo",
        Timeframe.H1: "1mo",
        Timeframe.H4: "3mo",
        Timeframe.D1: "2y",
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        symbol = provider_symbol(self.provider_name.value, request.symbol)
        url = f"{self.base_url}{self.CHART_ENDPOINT}/{symbol}"
        await self._ensure_robots_allowed(url)  # always raises today — see class docstring
        fetch_interval = "60m" if request.timeframe == Timeframe.H4 else self._intervals[request.timeframe]
        data = await self._get(f"{self.CHART_ENDPOINT}/{symbol}", params={"interval": fetch_interval, "range": self._ranges[request.timeframe]})
        result = data.get("chart", {}).get("result") if isinstance(data, dict) else None
        if not result or not isinstance(result, list):
            error = data.get("chart", {}).get("error") if isinstance(data, dict) else None
            raise ProviderResponseError(f"Yahoo Finance chart error: {error}")
        payload = result[0]
        timestamps = payload.get("timestamp") or []
        quote = (payload.get("indicators", {}).get("quote") or [{}])[0]
        now = datetime.now(UTC)
        base_candles: list[Candle] = []
        for index, epoch in enumerate(timestamps):
            try:
                open_, high, low, close = quote["open"][index], quote["high"][index], quote["low"][index], quote["close"][index]
                if None in (open_, high, low, close):
                    continue  # Yahoo pads illiquid intraday minutes with nulls; never fabricate a fill
                timestamp = datetime.fromtimestamp(epoch, UTC)
            except (KeyError, IndexError, TypeError):
                continue
            source_interval = Timeframe.H1 if request.timeframe == Timeframe.H4 else request.timeframe
            if not _period_has_closed(timestamp, source_interval, now=now):
                continue
            base_candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    timeframe=source_interval,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(quote.get("volume", [0] * len(timestamps))[index] or 0),
                    provider=self.provider_name.value,
                )
            )
        base_candles.sort(key=lambda candle: candle.timestamp)
        if request.timeframe == Timeframe.H4:
            base_candles = _aggregate_candles(base_candles, Timeframe.H4, self.provider_name.value)
        if request.start:
            base_candles = [item for item in base_candles if item.timestamp >= request.start]
        if request.end:
            base_candles = [item for item in base_candles if item.timestamp <= request.end]
        return base_candles[-request.limit :]


class StooqProvider(HttpMarketDataProvider):
    """Stooq's keyless CSV download endpoint (`/q/d/l/?s=xauusd&i=d`) for spot gold history.

    **Permanently disabled by default and not part of the active provider set, for two independent
    reasons found during this migration:** (1) `stooq.com/robots.txt` disallows all automated
    access for every user-agent except Googlebot/Bingbot by name; (2) the endpoint itself now
    serves a JavaScript proof-of-work "verify your browser" challenge instead of CSV data — solving
    it would be exactly the kind of automated-access-restriction bypass this project must never do.
    Because of (2), the CSV column format below could NOT be empirically re-verified against a live
    response during this migration; it is written from Stooq's long-stable, widely-documented
    `Date,Open,High,Low,Close,Volume` format used by numerous other open-source integrations, but
    should be treated as unverified until checked against a real response. `check_robots=True`
    means this adapter refuses to fetch even if manually enabled."""

    provider_name = ProviderName.STOOQ
    requires_api_key = False
    enforce_domain_allowlist = True
    check_robots = True
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=False,
        volume=True,
        supported_symbols=("XAUUSD",),
        supported_timeframes=(Timeframe.D1,),
    )
    DOWNLOAD_ENDPOINT = "/q/d/l/"

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        if request.timeframe != Timeframe.D1:
            raise ProviderResponseError("this Stooq adapter only supports daily (i=d) history")
        symbol = provider_symbol(self.provider_name.value, request.symbol)
        url = f"{self.base_url}{self.DOWNLOAD_ENDPOINT}?s={symbol}&i=d"
        await self._ensure_robots_allowed(url)  # always raises today — see class docstring
        raw = await self._get_text(self.DOWNLOAD_ENDPOINT, params={"s": symbol, "i": "d"})
        lines = raw.strip().splitlines()
        if len(lines) < 2:
            raise ProviderResponseError("Stooq CSV response was empty")
        header = [column.strip().lower() for column in lines[0].split(",")]
        today = datetime.now(UTC).date()
        candles: list[Candle] = []
        for line in lines[1:]:
            fields = dict(zip(header, line.split(","), strict=False))
            try:
                timestamp = datetime.fromisoformat(fields["date"]).replace(tzinfo=UTC)
                open_, high, low, close = float(fields["open"]), float(fields["high"]), float(fields["low"]), float(fields["close"])
            except (KeyError, ValueError):
                continue
            if timestamp.date() >= today:
                continue
            candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    timeframe=Timeframe.D1,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=float(fields.get("volume") or 0),
                    provider=self.provider_name.value,
                )
            )
        return sorted(candles, key=lambda candle: candle.timestamp)[-request.limit :]

    async def _get_text(self, path: str, *, params: dict[str, Any]) -> str:
        response = await self._client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        return response.text


class BinanceProvider(HttpMarketDataProvider):
    """Binance's public `/api/v3/klines` endpoint for a gold-token pair (e.g. `PAXGUSDT`) — kept
    as a second potential crypto-proxy source.

    **Permanently disabled by default and not part of the active provider set.**
    `api.binance.com/robots.txt` disallows all automated access for every user-agent, checked
    empirically during this migration, even though the endpoint itself works and needs no key.
    Kraken and OKX (both robots-clean) already cover the crypto-proxy cross-validation role this
    engine needs, so Binance is not required for correctness — kept implemented for completeness
    and in case its robots.txt policy changes. `check_robots=True` means this adapter refuses to
    fetch even if manually enabled."""

    provider_name = ProviderName.BINANCE
    requires_api_key = False
    enforce_domain_allowlist = True
    check_robots = True
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        volume=True,
        supported_symbols=("XAUUSD",),
        supported_timeframes=tuple(Timeframe),
    )
    KLINES_ENDPOINT = "/api/v3/klines"
    _intervals = {
        Timeframe.M1: "1m",
        Timeframe.M5: "5m",
        Timeframe.M15: "15m",
        Timeframe.M30: "30m",
        Timeframe.H1: "1h",
        Timeframe.H4: "4h",
        Timeframe.D1: "1d",
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        symbol = provider_symbol(self.provider_name.value, request.symbol)
        url = f"{self.base_url}{self.KLINES_ENDPOINT}"
        await self._ensure_robots_allowed(url)  # always raises today — see class docstring
        data = await self._get(
            self.KLINES_ENDPOINT, params={"symbol": symbol, "interval": self._intervals[request.timeframe], "limit": min(request.limit, 1000)}
        )
        if not isinstance(data, list):
            raise ProviderResponseError("Binance klines response must be a list")
        now = datetime.now(UTC)
        candles: list[Candle] = []
        for row in data:
            try:
                timestamp = datetime.fromtimestamp(int(row[0]) / 1000, UTC)
                open_, high, low, close, volume = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
                close_time = datetime.fromtimestamp(int(row[6]) / 1000, UTC)
            except (IndexError, TypeError, ValueError):
                continue
            if close_time > now or not _period_has_closed(timestamp, request.timeframe, now=now):
                continue  # Binance's last kline is always the still-forming interval
            candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    provider=self.provider_name.value,
                )
            )
        return sorted(candles, key=lambda candle: candle.timestamp)[-request.limit :]


def _aggregate_candles(source: list[Candle], target: Timeframe, provider: str) -> list[Candle]:
    """Deterministic OHLCV aggregation of consecutive smaller candles into a larger, aligned bar —
    e.g. four consecutive H1 candles into one H4 candle. This is arithmetic on already-real,
    already-final candles (open of the first, close of the last, max high, min low, summed volume),
    not interpolation or fabrication: every aggregated bar is backed entirely by genuine observed
    data, and a bar is only emitted once all of its constituent smaller candles are present and
    themselves already final. Used because Yahoo Finance has no native 4-hour interval; Kraken and
    OKX are not aggregated this way since both provide native 4h bars directly."""
    if not source:
        return []
    step = target.duration
    groups: dict[datetime, list[Candle]] = {}
    for candle in source:
        epoch_minutes = int(candle.timestamp.timestamp() // step.total_seconds())
        bucket_start = datetime.fromtimestamp(epoch_minutes * step.total_seconds(), UTC)
        groups.setdefault(bucket_start, []).append(candle)
    aggregated: list[Candle] = []
    expected_members = int(step.total_seconds() // source[0].timeframe.duration.total_seconds())
    for bucket_start, members in sorted(groups.items()):
        if len(members) < expected_members:
            continue  # an incomplete group (e.g. a still-forming or gapped bucket) is never emitted
        ordered = sorted(members, key=lambda item: item.timestamp)
        aggregated.append(
            Candle(
                timestamp=bucket_start,
                symbol=ordered[0].symbol,
                timeframe=target,
                open=ordered[0].open,
                high=max(item.high for item in ordered),
                low=min(item.low for item in ordered),
                close=ordered[-1].close,
                volume=sum(item.volume for item in ordered),
                provider=provider,
            )
        )
    return aggregated


def provider_api_key(environment_name: str) -> str:
    return os.getenv(environment_name, "")
