import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from backend.app.engines.market_data_engine import Candle, MarketDataWorker, MarketStatusCode, Timeframe
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


class BootstrapRecoveryService(StubMarketService):
    async def history(self, symbol: str, timeframe: Timeframe, **_: object) -> list[Candle]:
        self.history_calls += 1
        return [
            Candle(
                timestamp=datetime(2026, 7, 23, 10, tzinfo=UTC),
                ingestion_timestamp=datetime(2026, 7, 23, 10, 16, tzinfo=UTC),
                symbol=symbol,
                timeframe=timeframe,
                open=3300,
                high=3302,
                low=3298,
                close=3301,
                volume=10,
                provider="test",
            )
        ]


@pytest.mark.asyncio
async def test_bootstrap_recovers_after_one_failed_event_and_logs_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = BootstrapRecoveryService()
    analyzed: list[Timeframe] = []

    async def historical_analysis(value: Candle) -> None:
        if not analyzed:
            analyzed.append(value.timeframe)
            error = RuntimeError("first historical event failed")
            error.__dict__["integration_event_id"] = "a" * 64
            error.__dict__["integration_stage"] = "mark_processed"
            raise error
        analyzed.append(value.timeframe)

    worker = MarketDataWorker(
        service,  # type: ignore[arg-type]
        enabled=True,
        symbols=("XAUUSD",),
        timeframes=(Timeframe.M1, Timeframe.M5),
        bootstrap_enabled=True,
        bootstrap_candles=500,
        poll_seconds=60,
        historical_analysis=historical_analysis,
    )

    target_logger = logging.getLogger("backend.app.engines.market_data_engine.worker")
    target_logger.addHandler(caplog.handler)
    original_level, original_disabled = target_logger.level, target_logger.disabled
    target_logger.setLevel(logging.WARNING)
    target_logger.disabled = False
    try:
        await worker._bootstrap()
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(original_level)
        target_logger.disabled = original_disabled

    assert service.history_calls == 2
    assert analyzed == [Timeframe.M1, Timeframe.M5]
    assert worker.processing_state == "idle"
    failure = next(record for record in caplog.records if record.message == "market_data.bootstrap.failed")
    assert failure.event_id == "a" * 64
    assert failure.failure_stage == "mark_processed"


class _FlakySessionsMarketService(StubMarketService):
    """Raises from `sessions.status_at()` (a call outside `_poll()`'s own try/except) on the
    first N invocations, then behaves normally — reproduces the exact failure surface found during
    the "market data healthy but SMC-onward chain silent for hours" investigation: an exception
    thrown from inside `MarketDataWorker.run()`'s loop body, but outside any per-poll boundary."""

    def __init__(self, failures_before_recovery: int) -> None:
        super().__init__()
        self._remaining_failures = failures_before_recovery
        self.status_at_calls = 0

        def status_at(_: object) -> SimpleNamespace:
            self.status_at_calls += 1
            if self._remaining_failures > 0:
                self._remaining_failures -= 1
                raise RuntimeError("simulated failure outside the per-poll boundary")
            return SimpleNamespace(market_open=False)

        self.sessions = SimpleNamespace(status_at=status_at)


@pytest.mark.asyncio
async def test_worker_survives_an_exception_outside_the_per_poll_boundary_and_keeps_running() -> None:
    """Regression test: before this fix, an exception raised anywhere in `run()`'s loop body other
    than inside `_poll()` (e.g. `sessions.status_at()`) propagated straight out of the coroutine,
    silently ending the `asyncio.Task` forever — `status()["enabled"]` stayed `True` but nothing
    ever polled again. The loop must now catch it, record it, and keep going."""
    service = _FlakySessionsMarketService(failures_before_recovery=2)
    worker = MarketDataWorker(
        service,  # type: ignore[arg-type]
        enabled=True,
        symbols=("XAUUSD",),
        timeframes=(Timeframe.M15,),
        bootstrap_enabled=False,
        bootstrap_candles=500,
        poll_seconds=0.01,
    )
    worker.start()
    for _ in range(500):
        if service.status_at_calls >= 4:
            break
        await asyncio.sleep(0.01)  # real time must pass for `poll_seconds`' wait_for to elapse
    assert service.status_at_calls >= 4  # the loop kept iterating past the injected failures
    assert worker.status()["running"] is True  # the task is still alive, not dead
    assert worker.status()["crashed"] is False
    assert worker.status()["last_error"] == "RuntimeError"
    assert worker.status()["consecutive_failures"] >= 1
    await worker.stop()


@pytest.mark.asyncio
async def test_worker_task_death_is_surfaced_via_status_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """If `run()` ever dies anyway (a `BaseException` the loop deliberately does not swallow, or a
    future bug that reintroduces an unguarded path), `status()["crashed"]` must become visible and
    the death must be logged at a level that would appear in the live log stream — not vanish as an
    unretrieved `asyncio.Task` exception."""
    worker = MarketDataWorker(
        StubMarketService(),  # type: ignore[arg-type]
        enabled=True,
        symbols=("XAUUSD",),
        timeframes=(Timeframe.M15,),
        bootstrap_enabled=False,
        bootstrap_candles=500,
        poll_seconds=60,
    )

    async def _dies_immediately() -> None:
        raise RuntimeError("simulated unrecoverable worker crash")

    worker.run = _dies_immediately  # type: ignore[method-assign]
    # `create_app()` (exercised by other tests in the full suite) reconfigures logging via
    # `configure_logging()`'s `logging.basicConfig(force=True)`, which strips pytest's caplog
    # handler off the root logger for the rest of the process — so relying on root propagation
    # here is order-dependent. Attach caplog's handler directly to the specific logger under test,
    # matching the pattern already established in test_fmp_provider.py.
    target_logger = logging.getLogger("backend.app.engines.market_data_engine.worker")
    target_logger.addHandler(caplog.handler)
    original_level, original_disabled = target_logger.level, target_logger.disabled
    target_logger.setLevel(logging.CRITICAL)
    target_logger.disabled = False
    try:
        worker.start()
        for _ in range(500):
            # Wait for the done-callback specifically, not just `task.done()` — the callback runs
            # via `call_soon` slightly after the task itself finishes, and it's what sets
            # `last_fatal_error`.
            if worker.status()["last_fatal_error"] is not None:
                break
            await asyncio.sleep(0.01)
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(original_level)
        target_logger.disabled = original_disabled
    status = worker.status()
    assert status["crashed"] is True
    assert status["running"] is False
    assert status["last_fatal_error"] and "RuntimeError" in status["last_fatal_error"]
    own_records = [record for record in caplog.records if record.name == target_logger.name]
    assert any(record.levelname == "CRITICAL" and "task_died" in record.message for record in own_records)
