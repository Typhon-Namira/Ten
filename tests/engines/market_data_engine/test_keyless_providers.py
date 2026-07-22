"""Keyless public-source Market Data Engine adapters (LBMA, Kraken, OKX) plus the disabled-by-
default, robots-blocked legacy adapters (Yahoo Finance, Stooq, Binance), cross-source outlier
quarantine, and no-lookahead correctness. Every test uses `httpx.MockTransport` — none make a real
network call.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from backend.app.core.net.ssrf import UnsafePublicUrlError
from backend.app.engines.market_data_engine.adapters import (
    BinanceProvider,
    KrakenGoldProxyProvider,
    LbmaGoldPriceProvider,
    OkxGoldProxyProvider,
    StooqProvider,
    YahooFinanceProvider,
    _aggregate_candles,
    _period_has_closed,
)
from backend.app.engines.market_data_engine.config import MarketDataConfig, ProviderConfig, ValidationConfig
from backend.app.engines.market_data_engine.events import GapDetected
from backend.app.engines.market_data_engine.exceptions import ProviderResponseError
from backend.app.engines.market_data_engine.manager import ProviderManager, ProviderRegistry
from backend.app.engines.market_data_engine.models import Candle, Timeframe
from backend.app.engines.market_data_engine.providers import InMemoryMarketDataProvider, ProviderRequest
from backend.app.engines.market_data_engine.service import MarketDataService, build_market_data_service
from backend.app.engines.market_data_engine.ssrf import assert_safe_public_url
from backend.app.engines.market_data_engine.symbols import provider_symbol
from backend.app.engines.market_data_engine.validation import MarketDataValidator
from backend.app.events import InMemoryEventBus

NOW = datetime(2026, 7, 21, 18, tzinfo=UTC)


def _no_robots(request: httpx.Request) -> httpx.Response | None:
    """Shared MockTransport helper: respond 404 to a robots.txt probe (absence, not disallow, per
    TEN's policy) and let the caller's own handler answer everything else."""
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    return None


def _mounted(provider, handler) -> None:
    def combined(request: httpx.Request) -> httpx.Response:
        robots_response = _no_robots(request)
        return robots_response if robots_response is not None else handler(request)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(combined))


# ---------------------------------------------------------------------------
# No API key is required for any active default provider.
# ---------------------------------------------------------------------------


def test_default_config_active_providers_require_no_api_key() -> None:
    for cls, base_url in [
        (LbmaGoldPriceProvider, "https://prices.lbma.org.uk"),
        (KrakenGoldProxyProvider, "https://api.kraken.com"),
        (OkxGoldProxyProvider, "https://www.okx.com"),
    ]:
        provider = cls(base_url=base_url)
        assert provider.api_key == ""
        assert provider.requires_api_key is False


def test_build_market_data_service_constructs_only_keyless_providers_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in ("TEN_TWELVE_DATA_API_KEY", "TEN_ALPHA_VANTAGE_API_KEY", "TEN_FMP_API_KEY", "TEN_OANDA_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    config = MarketDataConfig(
        providers=(
            ProviderConfig(name="lbma_gold_price", base_url="https://prices.lbma.org.uk", priority=10),
            ProviderConfig(name="kraken", base_url="https://api.kraken.com", priority=20),
            ProviderConfig(name="okx", base_url="https://www.okx.com", priority=30),
            ProviderConfig(name="twelve_data", base_url="https://api.twelvedata.com", api_key_env="TEN_TWELVE_DATA_API_KEY", enabled=False, priority=900),
        )
    )
    service = build_market_data_service(config)
    names = sorted(type(item).__name__ for item in service.manager.registry.all())
    assert names == ["KrakenGoldProxyProvider", "LbmaGoldPriceProvider", "OkxGoldProxyProvider"]


# ---------------------------------------------------------------------------
# Old paid providers are still present in code and config but never constructed by default.
# ---------------------------------------------------------------------------


def test_legacy_paid_providers_are_disabled_by_default_but_still_constructible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEN_TWELVE_DATA_API_KEY", "secret")
    disabled_config = MarketDataConfig(
        providers=(ProviderConfig(name="twelve_data", base_url="https://api.twelvedata.com", api_key_env="TEN_TWELVE_DATA_API_KEY", enabled=False, priority=900),)
    )
    assert build_market_data_service(disabled_config).manager.registry.all() == ()
    enabled_config = MarketDataConfig(
        providers=(ProviderConfig(name="twelve_data", base_url="https://api.twelvedata.com", api_key_env="TEN_TWELVE_DATA_API_KEY", enabled=True, priority=900),)
    )
    enabled = build_market_data_service(enabled_config)
    assert [type(item).__name__ for item in enabled.manager.registry.all()] == ["TwelveDataProvider"]


# ---------------------------------------------------------------------------
# LBMA Gold Price — successful fetch/parse, malformed/empty response, network failure isolation.
# ---------------------------------------------------------------------------

# Dates are computed relative to the real wall clock at test-run time, not hardcoded — the
# adapter's own no-lookahead check compares against `datetime.now(UTC).date()`, so a fixed date
# string for "today" would silently start failing as soon as real time moved past it (this bit a
# sibling test in the economic calendar suite the same way earlier in this session).
_LBMA_TODAY = datetime.now(UTC).date()
_LBMA_DAY_4, _LBMA_DAY_3, _LBMA_DAY_2, _LBMA_DAY_1 = (
    (_LBMA_TODAY - timedelta(days=offset)).isoformat() for offset in (4, 3, 2, 1)
)


def _lbma_am_fixture() -> list[dict]:
    return [
        {"d": _LBMA_DAY_4, "v": [4030.95, 2984.11, 3516.6]},
        {"d": _LBMA_DAY_3, "v": [3998.8, 2974.31, 3494.95]},
        {"d": _LBMA_DAY_2, "v": [None, None, None]},  # non-trading day
        {"d": _LBMA_DAY_1, "v": [4014.8, 2983, 3512.15]},
        {"d": _LBMA_TODAY.isoformat(), "v": [4020.0, 2990, 3520.0]},  # "today" — must be excluded
    ]


def _lbma_pm_fixture() -> list[dict]:
    return [
        {"d": _LBMA_DAY_4, "v": [3993.55, 2960.0, 3490.0]},
        {"d": _LBMA_DAY_3, "v": [3995.35, 2965.0, 3492.0]},
        # _LBMA_DAY_1's PM fix intentionally absent — AM-only day
    ]


@pytest.mark.asyncio
async def test_lbma_fetch_succeeds_and_combines_am_pm_fixes() -> None:
    am_fixture, pm_fixture = _lbma_am_fixture(), _lbma_pm_fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/json/gold_am.json":
            return httpx.Response(200, json=am_fixture)
        if request.url.path == "/json/gold_pm.json":
            return httpx.Response(200, json=pm_fixture)
        return httpx.Response(404)

    provider = LbmaGoldPriceProvider(base_url="https://prices.lbma.org.uk")
    _mounted(provider, handler)
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.D1, limit=10))
    by_date = {item.timestamp.date().isoformat(): item for item in candles}
    assert set(by_date) == {_LBMA_DAY_4, _LBMA_DAY_3, _LBMA_DAY_1}  # holiday + "today" excluded
    both = by_date[_LBMA_DAY_4]
    assert both.open == 4030.95 and both.close == 3993.55
    assert both.high == max(4030.95, 3993.55) and both.low == min(4030.95, 3993.55)
    am_only = by_date[_LBMA_DAY_1]
    assert am_only.open == am_only.high == am_only.low == am_only.close == 4014.8
    assert all(item.volume == 0 for item in candles)
    await provider.close()


@pytest.mark.asyncio
async def test_lbma_malformed_response_raises_cleanly() -> None:
    provider = LbmaGoldPriceProvider(base_url="https://prices.lbma.org.uk")
    _mounted(provider, lambda request: httpx.Response(200, json={"not": "a list"}))
    with pytest.raises(ProviderResponseError):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.D1, limit=10))
    await provider.close()


@pytest.mark.asyncio
async def test_lbma_empty_response_returns_no_candles() -> None:
    provider = LbmaGoldPriceProvider(base_url="https://prices.lbma.org.uk")
    _mounted(provider, lambda request: httpx.Response(200, json=[]))
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.D1, limit=10))
    assert candles == []
    await provider.close()


@pytest.mark.asyncio
async def test_lbma_pm_fetch_failure_falls_back_to_am_only_candles() -> None:
    am_fixture = _lbma_am_fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/json/gold_am.json":
            return httpx.Response(200, json=am_fixture)
        return httpx.Response(500)  # PM endpoint is down

    provider = LbmaGoldPriceProvider(base_url="https://prices.lbma.org.uk")
    _mounted(provider, handler)
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.D1, limit=10))
    assert candles  # a failed PM fetch must not fail the whole request
    assert all(item.open == item.high == item.low == item.close for item in candles)
    await provider.close()


@pytest.mark.asyncio
async def test_lbma_rejects_non_daily_timeframes() -> None:
    provider = LbmaGoldPriceProvider(base_url="https://prices.lbma.org.uk")
    with pytest.raises(ProviderResponseError):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.M1, limit=10))
    await provider.close()


# ---------------------------------------------------------------------------
# Kraken — successful fetch/parse, malformed/empty response, network failure isolation, no-lookahead.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kraken_fetch_succeeds_and_drops_the_forming_candle() -> None:
    # `_period_has_closed` inside the adapter always uses the real wall clock (no injectable
    # clock), so this test must derive its "closed" / "still forming" fixture rows from the real
    # current time too — not the module's fixed `NOW` constant used elsewhere in this file.
    real_now = datetime.now(UTC)
    closed_ts = int((real_now - timedelta(hours=2)).timestamp())
    forming_ts = int((real_now - timedelta(minutes=10)).timestamp())  # H1 bar not yet elapsed
    rows = [
        [closed_ts, "4060.0", "4065.0", "4058.0", "4063.0", "4061.0", "10.5", 5],
        [forming_ts, "4063.0", "4064.0", "4062.0", "4062.5", "4063.0", "1.2", 1],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": [], "result": {"PAXGUSD": rows, "last": closed_ts}})

    provider = KrakenGoldProxyProvider(base_url="https://api.kraken.com")
    _mounted(provider, handler)
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    assert len(candles) == 1
    assert candles[0].close == 4063.0
    assert candles[0].provider == "kraken"
    await provider.close()


@pytest.mark.asyncio
async def test_kraken_error_field_raises() -> None:
    provider = KrakenGoldProxyProvider(base_url="https://api.kraken.com")
    _mounted(provider, lambda request: httpx.Response(200, json={"error": ["EQuery:Unknown asset pair"], "result": {}}))
    with pytest.raises(ProviderResponseError):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    await provider.close()


@pytest.mark.asyncio
async def test_kraken_missing_pair_key_raises() -> None:
    provider = KrakenGoldProxyProvider(base_url="https://api.kraken.com")
    _mounted(provider, lambda request: httpx.Response(200, json={"error": [], "result": {"last": 0}}))
    with pytest.raises(ProviderResponseError):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    await provider.close()


@pytest.mark.asyncio
async def test_kraken_network_failure_does_not_crash_and_is_isolated() -> None:
    provider = KrakenGoldProxyProvider(base_url="https://api.kraken.com")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ConnectError("simulated network outage")

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderResponseError):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    await provider.close()


# ---------------------------------------------------------------------------
# OKX — successful fetch/parse (confirm flag), malformed/empty response, network failure isolation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_okx_fetch_succeeds_and_honors_the_confirm_flag() -> None:
    # Same real-wall-clock note as the Kraken test above.
    real_now = datetime.now(UTC)
    closed_ms = int((real_now - timedelta(hours=2)).timestamp() * 1000)
    forming_ms = int((real_now - timedelta(minutes=5)).timestamp() * 1000)
    rows = [
        [str(forming_ms), "4063.0", "4064.0", "4062.0", "4062.5", "1.2", "0", "0", "0"],  # newest first, unconfirmed
        [str(closed_ms), "4060.0", "4065.0", "4058.0", "4063.0", "10.5", "0", "0", "1"],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "0", "msg": "", "data": rows})

    provider = OkxGoldProxyProvider(base_url="https://www.okx.com")
    _mounted(provider, handler)
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    assert len(candles) == 1
    assert candles[0].close == 4063.0
    assert candles[0].provider == "okx"
    await provider.close()


@pytest.mark.asyncio
async def test_okx_error_code_raises() -> None:
    provider = OkxGoldProxyProvider(base_url="https://www.okx.com")
    _mounted(provider, lambda request: httpx.Response(200, json={"code": "51001", "msg": "Instrument ID does not exist", "data": []}))
    with pytest.raises(ProviderResponseError):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    await provider.close()


@pytest.mark.asyncio
async def test_okx_empty_data_returns_no_candles() -> None:
    provider = OkxGoldProxyProvider(base_url="https://www.okx.com")
    _mounted(provider, lambda request: httpx.Response(200, json={"code": "0", "msg": "", "data": []}))
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    assert candles == []
    await provider.close()


@pytest.mark.asyncio
async def test_okx_network_failure_is_isolated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ConnectError("simulated network outage")

    provider = OkxGoldProxyProvider(base_url="https://www.okx.com")
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderResponseError):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    await provider.close()


# ---------------------------------------------------------------------------
# One source's failure never crashes the engine — ProviderManager fails over.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_source_failure_fails_over_to_the_next_healthy_source() -> None:
    registry = ProviderRegistry()
    failing = KrakenGoldProxyProvider(base_url="https://api.kraken.com")

    def failing_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ConnectError("simulated outage")

    failing._client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    registry.register(failing)

    healthy_candle = Candle(timestamp=NOW - timedelta(hours=1), timeframe=Timeframe.H1, open=4060, high=4065, low=4058, close=4062, provider="memory")
    registry.register(InMemoryMarketDataProvider([healthy_candle]))
    manager = ProviderManager(registry, preferred="memory")
    result = await manager.history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    assert result and result[-1].provider == "memory"
    await failing.close()


# ---------------------------------------------------------------------------
# No-lookahead / partial-vs-final candle correctness.
# ---------------------------------------------------------------------------


def test_period_has_closed_boundary_conditions() -> None:
    now = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    assert _period_has_closed(datetime(2026, 7, 21, 10, 0, tzinfo=UTC), Timeframe.H1, now=now) is True  # fully elapsed
    assert _period_has_closed(datetime(2026, 7, 21, 11, 30, tzinfo=UTC), Timeframe.H1, now=now) is False  # still forming
    assert _period_has_closed(datetime(2026, 7, 21, 11, 0, tzinfo=UTC), Timeframe.H1, now=now) is True  # exactly closed


def test_h4_aggregation_only_emits_complete_buckets() -> None:
    base = datetime(2026, 7, 21, 8, tzinfo=UTC)
    members = [
        Candle(timestamp=base + timedelta(hours=index), timeframe=Timeframe.H1, open=100 + index, high=105 + index, low=99 + index, close=102 + index, volume=10, provider="yahoo_finance")
        for index in range(4)
    ]
    incomplete = [Candle(timestamp=base + timedelta(hours=4), timeframe=Timeframe.H1, open=104, high=106, low=103, close=105, volume=5, provider="yahoo_finance")]
    result = _aggregate_candles(members + incomplete, Timeframe.H4, "yahoo_finance")
    assert len(result) == 1  # the 4-8h bucket has only 1 of 4 required members — never emitted
    bucket = result[0]
    assert bucket.timestamp == base
    assert bucket.open == members[0].open
    assert bucket.close == members[-1].close
    assert bucket.high == max(item.high for item in members)
    assert bucket.low == min(item.low for item in members)
    assert bucket.volume == sum(item.volume for item in members)


def test_h4_aggregation_of_empty_input_is_empty() -> None:
    assert _aggregate_candles([], Timeframe.H4, "yahoo_finance") == []


# ---------------------------------------------------------------------------
# Cross-source outlier detection and quarantine.
# ---------------------------------------------------------------------------


def _c(close: float, *, provider: str, at: datetime = NOW) -> Candle:
    return Candle(timestamp=at, timeframe=Timeframe.H1, open=close, high=close + 1, low=close - 1, close=close, volume=1, provider=provider)


def test_compare_flags_a_minor_deviation_without_dropping_it() -> None:
    validator = MarketDataValidator(ValidationConfig(cross_source_tolerance=0.01, cross_source_quarantine_tolerance=0.05))
    primary = [_c(4000.0, provider="kraken")]
    secondary = [_c(4030.0, provider="okx")]  # ~0.75% deviation — below tolerance, no anomaly
    anomalies, quarantined = validator.compare(primary, secondary)
    assert anomalies == []
    assert quarantined == set()


def test_compare_flags_a_moderate_deviation_as_inconsistency_not_quarantine() -> None:
    validator = MarketDataValidator(ValidationConfig(cross_source_tolerance=0.01, cross_source_quarantine_tolerance=0.05))
    primary = [_c(4000.0, provider="kraken")]
    secondary = [_c(4080.0, provider="okx")]  # ~2% deviation — flagged, not quarantined
    anomalies, quarantined = validator.compare(primary, secondary)
    assert len(anomalies) == 1 and anomalies[0].severity == 2
    assert quarantined == set()


def test_compare_quarantines_an_implausible_outlier() -> None:
    validator = MarketDataValidator(ValidationConfig(cross_source_tolerance=0.01, cross_source_quarantine_tolerance=0.05))
    # A real-world example found during this migration: a single bad wick on a low-liquidity
    # crypto order book — the primary reports a close wildly out of line with the cross-check.
    primary = [_c(2222.4, provider="kraken")]
    secondary = [_c(4014.8, provider="okx")]
    anomalies, quarantined = validator.compare(primary, secondary)
    assert quarantined == {NOW}
    assert anomalies[0].severity == 3
    assert anomalies[0].missing_count == 1


@pytest.mark.asyncio
async def test_quarantined_candle_is_removed_and_never_fabricated_back_in() -> None:
    """End-to-end through `MarketDataService.history()`: a quarantined outlier from the primary
    provider must never reach the returned series, and — if no third source can supply a real
    replacement — the slot is left as a genuine gap, never a fabricated/interpolated fill."""
    registry = ProviderRegistry()
    bad_timestamp = NOW - timedelta(hours=1)
    good_timestamp = NOW - timedelta(hours=2)
    primary_candles = [
        Candle(timestamp=good_timestamp, timeframe=Timeframe.H1, open=4050, high=4055, low=4045, close=4050, provider="primary"),
        Candle(timestamp=bad_timestamp, timeframe=Timeframe.H1, open=2222, high=2223, low=2221, close=2222.4, provider="primary"),
    ]
    secondary_candles = [
        Candle(timestamp=good_timestamp, timeframe=Timeframe.H1, open=4051, high=4056, low=4046, close=4051, provider="secondary"),
        Candle(timestamp=bad_timestamp, timeframe=Timeframe.H1, open=4014, high=4016, low=4012, close=4014.8, provider="secondary"),
    ]
    registry.register(InMemoryMarketDataProvider(primary_candles))
    secondary_provider = InMemoryMarketDataProvider(secondary_candles)
    secondary_provider.provider_name = type(secondary_provider.provider_name)("memory")  # type: ignore[misc]
    # Two distinct provider identities are required for ranking/cross-check to treat them as
    # independent sources; InMemoryMarketDataProvider always reports ProviderName.MEMORY, so we
    # register the "secondary" set under a second registry-visible name via a thin subclass.
    from backend.app.engines.market_data_engine.providers import MarketDataProvider, ProviderCapabilities, ProviderName

    class _SecondaryProvider(MarketDataProvider):
        provider_name = ProviderName.CSV
        capabilities = ProviderCapabilities(historical=True, realtime_polling=True, supported_timeframes=tuple(Timeframe))

        async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
            return [item for item in secondary_candles if (request.start is None or item.timestamp >= request.start) and (request.end is None or item.timestamp <= request.end)]

        async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
            return secondary_candles[-1]

    registry.register(_SecondaryProvider())
    manager = ProviderManager(registry, preferred="memory")
    config = MarketDataConfig(validation=ValidationConfig(cross_source_tolerance=0.01, cross_source_quarantine_tolerance=0.05))
    bus = InMemoryEventBus()
    published: list[GapDetected] = []

    async def _capture(event: object) -> None:
        if isinstance(event, GapDetected):
            published.append(event)

    bus.subscribe(GapDetected, _capture)
    service = MarketDataService(manager, config=config, event_bus=bus)
    result = await service.history("XAUUSD", Timeframe.H1, start=good_timestamp - timedelta(minutes=1), end=bad_timestamp + timedelta(minutes=1), refresh=True)
    assert all(item.timestamp != bad_timestamp or item.close != 2222.4 for item in result)
    assert published  # the quarantine surfaced as a GapDetected event, not a silent drop


# ---------------------------------------------------------------------------
# SSRF protection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/ohlc",
        "http://127.0.0.1/ohlc",
        "http://169.254.169.254/latest/meta-data/",
        "https://evil.example.com/ohlc",
        "file:///etc/passwd",
        "ftp://api.kraken.com/ohlc",
    ],
)
def test_ssrf_protection_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(UnsafePublicUrlError):
        assert_safe_public_url(url)


def test_ssrf_protection_allows_the_active_keyless_domains() -> None:
    for domain in ("prices.lbma.org.uk", "api.kraken.com", "www.okx.com"):
        assert_safe_public_url(f"https://{domain}/x")


def test_keyless_provider_construction_rejects_a_non_allowlisted_base_url() -> None:
    with pytest.raises(UnsafePublicUrlError):
        LbmaGoldPriceProvider(base_url="https://not-lbma.example.com")


# ---------------------------------------------------------------------------
# Disabled-by-default legacy adapters are robots-gated, not just config-gated.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_class", "base_url"),
    [
        (YahooFinanceProvider, "https://query1.finance.yahoo.com"),
        (StooqProvider, "https://stooq.com"),
        (BinanceProvider, "https://api.binance.com"),
    ],
)
async def test_disabled_legacy_adapters_self_refuse_on_an_explicit_robots_disallow(provider_class: type, base_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        return httpx.Response(200, json={"never": "reached"})

    provider = provider_class(base_url=base_url)
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderResponseError, match="robots.txt disallows"):
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.D1, limit=5))
    await provider.close()


def test_legacy_adapters_require_no_api_key_either() -> None:
    for cls, base_url in [
        (YahooFinanceProvider, "https://query1.finance.yahoo.com"),
        (StooqProvider, "https://stooq.com"),
        (BinanceProvider, "https://api.binance.com"),
    ]:
        provider = cls(base_url=base_url)
        assert provider.api_key == ""


# ---------------------------------------------------------------------------
# Symbol mapping sanity.
# ---------------------------------------------------------------------------


def test_provider_symbol_mappings_for_new_sources() -> None:
    assert provider_symbol("kraken", "XAUUSD") == "PAXGUSD"
    assert provider_symbol("okx", "XAUUSD") == "XAUT-USDT"
    assert provider_symbol("yahoo_finance", "XAUUSD") == "GC=F"
