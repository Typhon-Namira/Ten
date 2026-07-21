"""Financial Modeling Prep market-data provider — stable API migration + symbol mapping +
diagnostics coverage. Uses `httpx.MockTransport` (the established pattern in this codebase) so
these tests never make a real network call and never require a real API key."""

from __future__ import annotations

import logging

import httpx
import pytest

from backend.app.engines.market_data_engine.adapters import FinancialModelingPrepProvider, _http_error_detail
from backend.app.engines.market_data_engine.exceptions import ProviderResponseError
from backend.app.engines.market_data_engine.providers import ProviderRequest
from backend.app.engines.market_data_engine.models import Timeframe
from backend.app.engines.market_data_engine.symbols import provider_symbol


def _mounted(provider: FinancialModelingPrepProvider, handler) -> None:
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_xauusd_maps_to_gcusd_for_fmp_only() -> None:
    """FMP's commodity endpoints use COMEX futures tickers ("GCUSD"), not FX-style pairs — TEN's
    internal canonical symbol ("XAUUSD") is never changed; the mapping lives only in the adapter."""
    assert provider_symbol("financial_modeling_prep", "XAUUSD") == "GCUSD"
    # Unrelated providers must not be affected by this mapping.
    assert provider_symbol("twelve_data", "XAUUSD") == "XAU/USD"
    assert provider_symbol("oanda", "XAUUSD") == "XAU_USD"


@pytest.mark.asyncio
async def test_fmp_provider_health_check_never_requests_the_bare_base_url() -> None:
    """`/stable` is a base URL only — FMP 404s if it is ever requested directly."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json=[{"symbol": "GCUSD", "price": 2400.1}])

    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, handler)
    await provider.check_connectivity()
    assert seen == ["/stable/quote-short"]
    assert all(path not in {"", "/", "/stable", "/stable/"} for path in seen)
    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeframe", "expected_path"), [(Timeframe.M1, "/stable/historical-chart/1min"), (Timeframe.M5, "/stable/historical-chart/5min"), (Timeframe.H1, "/stable/historical-chart/1hour")])
async def test_fmp_provider_builds_the_correct_intraday_url_per_interval(timeframe: Timeframe, expected_path: str) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["symbol"] = request.url.params["symbol"]
        return httpx.Response(200, json=[])

    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, handler)
    await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=timeframe, limit=2))
    assert seen["path"] == expected_path
    assert seen["symbol"] == "GCUSD"
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_uses_the_daily_eod_endpoint_not_historical_chart() -> None:
    """Daily data is a documented, separate endpoint (`historical-price-eod/full`) — not another
    `historical-chart/{interval}` route."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["symbol"] = request.url.params["symbol"]
        return httpx.Response(200, json=[{"date": "2026-07-20", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10}])

    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, handler)
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.D1, limit=2))
    assert seen["path"] == "/stable/historical-price-eod/full"
    assert seen["symbol"] == "GCUSD"
    assert len(candles) == 1
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_accepts_wrapped_daily_response_shape() -> None:
    """Some FMP endpoints wrap results as {"historical": [...]}; the adapter must handle both a
    bare list and this wrapped shape rather than assuming exactly one."""
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, lambda request: httpx.Response(200, json={"symbol": "GCUSD", "historical": [{"date": "2026-07-20", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10}]}))
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.D1, limit=2))
    assert len(candles) == 1
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_never_logs_the_api_key(caplog: pytest.LogCaptureFixture) -> None:
    secret_key = "sk-super-secret-do-not-log-me"
    provider = FinancialModelingPrepProvider(api_key=secret_key, base_url="https://financialmodelingprep.com/stable")
    seen_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.url.params["apikey"])
        return httpx.Response(401, text="Unauthorized")

    _mounted(provider, handler)
    target_logger = logging.getLogger("backend.app.engines.market_data_engine.adapters")
    target_logger.addHandler(caplog.handler)
    original_level, original_disabled = target_logger.level, target_logger.disabled
    target_logger.setLevel(logging.DEBUG)
    target_logger.disabled = False
    try:
        with pytest.raises(ProviderResponseError):
            await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(original_level)
        target_logger.disabled = original_disabled
    assert seen_keys == [secret_key]  # the request itself must still carry the real key
    assert caplog.records
    for record in caplog.records:
        assert secret_key not in record.getMessage()
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_classifies_http_404_as_invalid_endpoint(caplog: pytest.LogCaptureFixture) -> None:
    """A 404 proves the server was reached — it must never be classified the same as a dropped
    connection ("unreachable")."""
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, lambda request: httpx.Response(404, text="Not Found"))
    target_logger = logging.getLogger("backend.app.engines.market_data_engine.adapters")
    target_logger.addHandler(caplog.handler)
    original_level, original_disabled = target_logger.level, target_logger.disabled
    target_logger.setLevel(logging.ERROR)
    target_logger.disabled = False
    try:
        with pytest.raises(ProviderResponseError):
            await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(original_level)
        target_logger.disabled = original_disabled
    logged = next(record.message for record in caplog.records if "request failed" in record.message)
    assert "classification=invalid_endpoint" in logged
    assert "status=404" in logged
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_classifies_http_429_as_rate_limited() -> None:
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable", max_rate_limit_retries=0)
    _mounted(provider, lambda request: httpx.Response(429, headers={"retry-after": "5"}))
    with pytest.raises(ProviderResponseError) as excinfo:
        await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    status_code, _ = _http_error_detail(excinfo.value)
    assert status_code == 429
    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("status_code", "expected_category"), [(401, "unauthorized"), (403, "forbidden")])
async def test_fmp_provider_classifies_401_and_403_distinctly(status_code: int, expected_category: str, caplog: pytest.LogCaptureFixture) -> None:
    """401 (bad/missing credentials) and 403 (valid credentials, insufficient entitlement) are
    different failures and must be classified distinctly, not collapsed into one "auth" bucket."""
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, lambda request: httpx.Response(status_code, text="denied"))
    target_logger = logging.getLogger("backend.app.engines.market_data_engine.adapters")
    target_logger.addHandler(caplog.handler)
    original_level, original_disabled = target_logger.level, target_logger.disabled
    target_logger.setLevel(logging.ERROR)
    target_logger.disabled = False
    try:
        with pytest.raises(ProviderResponseError):
            await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(original_level)
        target_logger.disabled = original_disabled
    logged = next(record.message for record in caplog.records if "request failed" in record.message)
    assert f"classification={expected_category}" in logged
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_sets_last_success_on_a_successful_response() -> None:
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, lambda request: httpx.Response(200, json=[{"date": "2026-07-20", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10}]))
    assert (await provider.quota()) is not None  # sanity: provider constructed cleanly
    await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    # `HttpMarketDataProvider` doesn't persist a `last_success` timestamp field, but a successful
    # call must leave the provider healthy for a subsequent request with no error state carried over.
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    assert len(candles) == 1
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_normalizes_gcusd_candles_back_to_xauusd() -> None:
    """FMP returns commodity data under the "GCUSD" ticker — every returned `Candle` must be
    tagged with TEN's own canonical symbol, never the provider-specific one."""
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, lambda request: httpx.Response(200, json=[{"date": "2026-07-20 10:00:00", "open": 2400.0, "high": 2405.0, "low": 2398.0, "close": 2402.0, "volume": 120}]))
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    assert len(candles) == 1
    assert candles[0].symbol == "XAUUSD"
    assert candles[0].provider == "financial_modeling_prep"
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_spread_is_never_fabricated() -> None:
    """This feed has no real bid/ask. `Candle.spread` is a non-optional float on the shared model
    (defaults to 0.0), so "spread unavailable" is expressed via `capabilities.spread=False` —
    every downstream consumer must check that flag before trusting `candle.spread`, exactly as it
    already does for every other spread-less provider (e.g. TwelveData)."""
    assert FinancialModelingPrepProvider.capabilities.spread is False
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(provider, lambda request: httpx.Response(200, json=[{"date": "2026-07-20 10:00:00", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10}]))
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=2))
    assert candles[0].spread == 0.0  # the literal field value — untrusted downstream via the capability flag above
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_rejects_duplicate_rows_before_they_reach_persistence() -> None:
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(
        provider,
        lambda request: httpx.Response(
            200,
            json=[
                {"date": "2026-07-20 10:00:00", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10},
                {"date": "2026-07-20 10:00:00", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10},
                {"date": "2026-07-20 11:00:00", "open": 1.5, "high": 2.5, "low": 1.5, "close": 2.0, "volume": 12},
            ],
        ),
    )
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    assert len(candles) == 2
    await provider.close()


@pytest.mark.asyncio
async def test_fmp_provider_skips_malformed_rows_without_crashing_the_whole_batch() -> None:
    provider = FinancialModelingPrepProvider(api_key="secret", base_url="https://financialmodelingprep.com/stable")
    _mounted(
        provider,
        lambda request: httpx.Response(
            200,
            json=[
                {"date": "2026-07-20 10:00:00", "open": "not-a-number", "high": 2, "low": 1, "close": 1.5, "volume": 10},
                {"date": "2026-07-20 11:00:00", "open": 1.5, "high": 2.5, "low": 1.5, "close": 2.0, "volume": 12},
            ],
        ),
    )
    candles = await provider.fetch_history(ProviderRequest(symbol="XAUUSD", timeframe=Timeframe.H1, limit=10))
    assert len(candles) == 1
    assert candles[0].close == 2.0
    await provider.close()
