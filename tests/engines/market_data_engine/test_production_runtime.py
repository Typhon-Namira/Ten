import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from backend.app.engines.market_data_engine import MarketDataWorker, MarketStatusCode, Timeframe
from backend.app.engines.market_data_engine.sessions import MarketSessionEngine


@pytest.mark.parametrize(
    ("instant", "expected"),
    (
        (datetime(2026, 7, 17, 20, 59, tzinfo=UTC), MarketStatusCode.OPEN),
        (datetime(2026, 7, 17, 21, 0, tzinfo=UTC), MarketStatusCode.CLOSED_WEEKEND),
        (datetime(2026, 7, 18, 12, 0, tzinfo=UTC), MarketStatusCode.CLOSED_WEEKEND),
        (datetime(2026, 7, 19, 21, 59, tzinfo=UTC), MarketStatusCode.CLOSED_WEEKEND),
        (datetime(2026, 7, 19, 22, 0, tzinfo=UTC), MarketStatusCode.OPEN),
        (datetime(2026, 7, 20, 21, 30, tzinfo=UTC), MarketStatusCode.CLOSED_DAILY_BREAK),
    ),
)
def test_xau_schedule_boundaries(instant: datetime, expected: MarketStatusCode) -> None:
    status = MarketSessionEngine().status_at(instant)
    assert status.market_status == expected
    assert status.market_open is (expected == MarketStatusCode.OPEN)
    assert (status.active_session is not None) is status.market_open
    assert status.server_time_utc.tzinfo is UTC


def test_xau_schedule_next_open_tracks_new_york_dst() -> None:
    engine = MarketSessionEngine()
    summer = engine.status_at(datetime(2026, 7, 18, 12, tzinfo=UTC))
    winter = engine.status_at(datetime(2026, 12, 5, 12, tzinfo=UTC))
    assert summer.next_expected_open_at == datetime(2026, 7, 19, 22, tzinfo=UTC)
    assert winter.next_expected_open_at == datetime(2026, 12, 6, 23, tzinfo=UTC)


class StubMarketService:
    def __init__(self) -> None:
        self.sessions = SimpleNamespace(status_at=lambda _: SimpleNamespace(market_open=False))
        self.history_calls = 0

    async def history(self, *_: object, **__: object) -> list[object]:
        self.history_calls += 1
        return []

    async def latest(self, *_: object, **__: object) -> None:
        raise AssertionError("closed market must not poll latest")


@pytest.mark.asyncio
async def test_enabled_worker_bootstraps_while_market_is_closed_and_stops() -> None:
    service = StubMarketService()
    worker = MarketDataWorker(
        service,  # type: ignore[arg-type]
        enabled=True,
        symbols=("XAUUSD",),
        timeframes=(Timeframe.M15,),
        bootstrap_enabled=True,
        bootstrap_candles=500,
        poll_seconds=60,
    )
    worker.start()
    for _ in range(100):
        if service.history_calls:
            break
        await asyncio.sleep(0)
    assert service.history_calls == 1
    assert worker.status()["running"] is True
    await worker.stop()
    assert worker.status()["running"] is False


@pytest.mark.asyncio
async def test_disabled_worker_never_starts() -> None:
    worker = MarketDataWorker(
        StubMarketService(),  # type: ignore[arg-type]
        enabled=False,
        symbols=("XAUUSD",),
        timeframes=(Timeframe.M15,),
        bootstrap_enabled=True,
        bootstrap_candles=500,
        poll_seconds=60,
    )
    worker.start()
    assert worker.status()["processing_state"] == "disabled"
    assert worker.status()["running"] is False
