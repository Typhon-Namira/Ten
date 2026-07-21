"""HTTP provider adapters. Provider-specific formats terminate in this module."""

import asyncio
import logging
import os
from abc import abstractmethod
from datetime import UTC, datetime
from time import monotonic, perf_counter
from typing import Any

import httpx

from .exceptions import ProviderRateLimitedError, ProviderResponseError
from .models import Candle, Timeframe
from .providers import (
    MarketDataProvider,
    ProviderCapabilities,
    ProviderName,
    ProviderQuota,
    ProviderRequest,
)
from .symbols import provider_symbol

logger = logging.getLogger(__name__)


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


class HttpMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 15,
        account_id: str | None = None,
        requests_per_minute: int | None = None,
        max_rate_limit_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError(f"{self.provider_name.value} API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.account_id = account_id
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
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
        supported_symbols=("XAUUSD",),
        supported_timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.D1),
    )
    _intervals = {
        Timeframe.M1: "1min",
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "1hour",
        Timeframe.D1: "1day",
    }

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        # FMP's stable API moved the symbol out of the path and into a query parameter (the
        # legacy /api/v3 endpoint took it as a path segment) — interval stays in the path, e.g.
        # /stable/historical-chart/1hour?symbol=XAUUSD.
        interval = self._intervals[request.timeframe]
        path = f"/historical-chart/{interval}"
        try:
            data = await self._get(
                path,
                params={
                    "symbol": provider_symbol(self.provider_name.value, request.symbol),
                    "from": request.start.date().isoformat() if request.start else None,
                    "to": request.end.date().isoformat() if request.end else None,
                    "apikey": self.api_key,
                },
            )
        except ProviderResponseError as exc:
            status_code, message = _http_error_detail(exc)
            if status_code in (401, 403):
                logger.error("Financial Modeling Prep authentication failed: endpoint=%s status=%s message=%s", path, status_code, message)
            else:
                logger.warning("Financial Modeling Prep request failed: endpoint=%s status=%s message=%s", path, status_code, message)
            raise
        if not isinstance(data, list):
            raise ProviderResponseError("Financial Modeling Prep response must be a list")
        candles = [
            Candle(
                timestamp=_parse_timestamp(row["date"]),
                symbol=request.symbol,
                timeframe=request.timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0),
                provider=self.provider_name.value,
            )
            for row in data
        ]
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


def provider_api_key(environment_name: str) -> str:
    return os.getenv(environment_name, "")
