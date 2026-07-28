"""Session-aware diagnostics for the live market-data boundary."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
import logging
from typing import Any

from .models import Timeframe, canonical_symbol
from .service import MarketDataService

logger = logging.getLogger(__name__)


async def evaluate_market_data_freshness(
    service: MarketDataService,
    *,
    symbol: str,
    timeframes: Iterable[Timeframe],
    worker_utc_now: datetime,
    freshness_limit_seconds: int,
    ums_market_timestamp: datetime | None,
) -> dict[str, Any]:
    """Measure freshness from durable closed candles, independently of AI-cycle cadence.

    Candle timestamps are open boundaries, so freshness is measured from each candle's
    close boundary. A closed XAUUSD session is an explicit schedule state, not a stale
    provider failure, and no synthetic candle is created to make a closed market look fresh.
    """

    if worker_utc_now.tzinfo is None:
        raise ValueError("worker_utc_now must be timezone-aware")
    now = worker_utc_now.astimezone(UTC)
    instrument = canonical_symbol(symbol)
    required = tuple(dict.fromkeys(timeframes))
    schedule = service.sessions.status_at(now)
    candles = {
        timeframe: await service.repository.candle_at(instrument, timeframe, now)
        for timeframe in required
    }
    close_timestamps = {
        timeframe.value: (
            candle.timestamp + candle.timeframe.duration if candle is not None else None
        )
        for timeframe, candle in candles.items()
    }
    ages = {
        timeframe: max(0.0, (now - close_timestamp).total_seconds())
        for timeframe, close_timestamp in close_timestamps.items()
        if close_timestamp is not None
    }
    missing = tuple(
        timeframe.value for timeframe, candle in candles.items() if candle is None
    )
    provider = service.manager.current_provider
    provider_stats = service.manager.statistics.get(provider) if provider else None
    oldest_timeframe = max(ages, key=lambda timeframe: ages[timeframe]) if ages else None
    age_seconds = max(ages.values()) if ages else None

    if not schedule.market_open:
        status = "MARKET_CLOSED"
        stale_reason = schedule.closure_reason or schedule.market_status.value
        rule = "closed_session_does_not_require_advancing_candles"
    elif missing:
        status = "STALE"
        stale_reason = f"missing_required_timeframes:{','.join(missing)}"
        rule = "every_required_timeframe_must_have_a_completed_candle"
    elif age_seconds is not None and age_seconds > freshness_limit_seconds:
        status = "STALE"
        stale_reason = (
            f"completed_candle_age_exceeds_limit:{oldest_timeframe}"
            f":{age_seconds:.3f}>{freshness_limit_seconds}"
        )
        rule = "oldest_required_completed_candle_age_lte_freshness_limit"
    else:
        status = "FRESH"
        stale_reason = None
        rule = "oldest_required_completed_candle_age_lte_freshness_limit"

    market_boundary = max(
        (value for value in close_timestamps.values() if value is not None),
        default=None,
    )
    market_session = (
        schedule.active_session.value.upper()
        if schedule.active_session is not None
        else "CLOSED"
    )
    diagnostics: dict[str, Any] = {
        "event": "market_data.freshness",
        "symbol": instrument,
        "provider": provider,
        "provider_timestamp": (
            provider_stats.last_success_at if provider_stats is not None else None
        ),
        # TEN's production market-data path is closed-candle polling. It has no tick
        # stream, so null is explicit rather than inferred as a missing provider event.
        "latest_tick_timestamp": None,
        "latest_candle_timestamp": market_boundary,
        "latest_candle_timestamps": close_timestamps,
        "cache_write_timestamp": service.cache.statistics.last_write_at,
        "cache_read_timestamp": service.cache.statistics.last_read_at,
        "ums_market_timestamp": ums_market_timestamp,
        "worker_utc_now": now,
        "age_seconds": age_seconds,
        "freshness_limit_seconds": freshness_limit_seconds,
        "market_session": market_session,
        "market_status": schedule.market_status.value,
        "market_status_source": schedule.status_source,
        "status": status,
        "stale_reason": stale_reason,
        "rule": rule,
        "missing_timeframes": missing,
        "tick_stream_status": "not_configured_candle_polling",
    }
    logger.info("market_data.freshness", extra=diagnostics)
    return diagnostics
