from datetime import UTC, datetime, timedelta

import pytest

from backend.app.engines.market_data_engine import Candle, Timeframe


class FakeSessionFactory:
    """Wraps a hand-built fake/mock session so it satisfies the `async_sessionmaker` calling
    convention (callable; returns an async context manager) that `SqlAlchemyXRepository` classes
    now require (see `backend/app/storage/scoped_session.py`). Tests that build a fake session by
    hand — rather than a real `AsyncSession` — wrap it in `FakeSessionFactory(session)` instead of
    passing the raw session directly; the same fake session instance is returned on every call,
    matching these tests' existing assertions against call history on that one mock.
    """

    def __init__(self, session: object) -> None:
        self._session = session

    def __call__(self) -> "FakeSessionFactory":
        return self

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


@pytest.fixture
def candles() -> list[Candle]:
    """Repeatable XAU/USD-like candles with volume and a final bullish break."""

    start = datetime(2026, 1, 5, 8, tzinfo=UTC)
    rows = [
        (2640.0, 2644.0, 2638.0, 2642.0, 100),
        (2642.0, 2646.0, 2640.0, 2644.0, 120),
        (2644.0, 2646.2, 2641.0, 2642.0, 110),
        (2642.0, 2649.0, 2641.5, 2648.0, 180),
        (2648.0, 2654.0, 2647.0, 2653.0, 240),
    ]
    return [Candle(timeframe=Timeframe.M15, timestamp=start + timedelta(minutes=15 * index), open=open_price, high=high, low=low, close=close, volume=volume) for index, (open_price, high, low, close, volume) in enumerate(rows)]
