from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.cache import CacheStatistics
from backend.app.engines.market_data_engine.freshness import (
    evaluate_market_data_freshness,
)
from backend.app.engines.market_data_engine.sessions import MarketSessionEngine


def candle(timeframe: Timeframe, timestamp: datetime) -> Candle:
    return Candle(
        timestamp=timestamp,
        symbol="XAUUSD",
        timeframe=timeframe,
        open=2400,
        high=2401,
        low=2399,
        close=2400,
        provider="kraken",
    )


def service_with(candles: dict[Timeframe, Candle]) -> SimpleNamespace:
    repository = SimpleNamespace(
        candle_at=AsyncMock(
            side_effect=lambda _symbol, timeframe, _now: candles.get(timeframe)
        )
    )
    statistics = {
        "kraken": SimpleNamespace(
            last_success_at=datetime(2026, 7, 28, 13, 59, 5, tzinfo=UTC)
        )
    }
    return SimpleNamespace(
        repository=repository,
        sessions=MarketSessionEngine(),
        manager=SimpleNamespace(current_provider="kraken", statistics=statistics),
        cache=SimpleNamespace(
            statistics=CacheStatistics(
                last_read_at=datetime(2026, 7, 28, 13, 59, 6, tzinfo=UTC),
                last_write_at=datetime(2026, 7, 28, 13, 59, 5, tzinfo=UTC),
            )
        ),
    )


@pytest.mark.asyncio
async def test_required_m1_m5_m15_closed_candles_are_fresh() -> None:
    now = datetime(2026, 7, 28, 14, tzinfo=UTC)
    service = service_with(
        {
            Timeframe.M1: candle(Timeframe.M1, datetime(2026, 7, 28, 13, 59, tzinfo=UTC)),
            Timeframe.M5: candle(Timeframe.M5, datetime(2026, 7, 28, 13, 55, tzinfo=UTC)),
            Timeframe.M15: candle(Timeframe.M15, datetime(2026, 7, 28, 13, 45, tzinfo=UTC)),
        }
    )

    result = await evaluate_market_data_freshness(
        service,
        symbol="XAUUSD",
        timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.M15),
        worker_utc_now=now,
        freshness_limit_seconds=1800,
        ums_market_timestamp=datetime(2026, 7, 28, 12, 20, tzinfo=UTC),
    )

    assert result["status"] == "FRESH"
    assert result["age_seconds"] == 0
    assert result["ums_market_timestamp"] < result["latest_candle_timestamp"]
    assert result["stale_reason"] is None


@pytest.mark.asyncio
async def test_old_required_timeframe_reports_exact_stale_rule() -> None:
    now = datetime(2026, 7, 28, 14, tzinfo=UTC)
    service = service_with(
        {
            Timeframe.M1: candle(Timeframe.M1, datetime(2026, 7, 28, 13, 59, tzinfo=UTC)),
            Timeframe.M5: candle(Timeframe.M5, datetime(2026, 7, 28, 13, 55, tzinfo=UTC)),
            Timeframe.M15: candle(Timeframe.M15, datetime(2026, 7, 28, 12, tzinfo=UTC)),
        }
    )

    result = await evaluate_market_data_freshness(
        service,
        symbol="XAUUSD",
        timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.M15),
        worker_utc_now=now,
        freshness_limit_seconds=1800,
        ums_market_timestamp=None,
    )

    assert result["status"] == "STALE"
    assert result["age_seconds"] == 6300
    assert result["stale_reason"] == (
        "completed_candle_age_exceeds_limit:M15:6300.000>1800"
    )


@pytest.mark.asyncio
async def test_closed_market_is_not_misreported_as_stale_provider_data() -> None:
    now = datetime(2026, 8, 1, 14, tzinfo=UTC)
    service = service_with({})

    result = await evaluate_market_data_freshness(
        service,
        symbol="XAUUSD",
        timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.M15),
        worker_utc_now=now,
        freshness_limit_seconds=1800,
        ums_market_timestamp=None,
    )

    assert result["status"] == "MARKET_CLOSED"
    assert result["market_status"] == "CLOSED_WEEKEND"
    assert result["stale_reason"] == "weekend"
    assert result["rule"] == "closed_session_does_not_require_advancing_candles"


@pytest.mark.asyncio
async def test_missing_open_market_timeframe_is_stale_without_fabrication() -> None:
    now = datetime(2026, 7, 28, 14, tzinfo=UTC)
    service = service_with(
        {
            Timeframe.M1: candle(Timeframe.M1, datetime(2026, 7, 28, 13, 59, tzinfo=UTC)),
            Timeframe.M5: candle(Timeframe.M5, datetime(2026, 7, 28, 13, 55, tzinfo=UTC)),
        }
    )

    result = await evaluate_market_data_freshness(
        service,
        symbol="XAUUSD",
        timeframes=(Timeframe.M1, Timeframe.M5, Timeframe.M15),
        worker_utc_now=now,
        freshness_limit_seconds=1800,
        ums_market_timestamp=None,
    )

    assert result["status"] == "STALE"
    assert result["missing_timeframes"] == ("M15",)
    assert result["stale_reason"] == "missing_required_timeframes:M15"
