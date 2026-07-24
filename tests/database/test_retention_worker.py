"""Unit coverage for RetentionWorker's scheduling/crash-guard behavior, no database required.

`RetentionRepository.prune()` itself is only meaningfully exercised against real PostgreSQL (see
test_retention_postgresql.py) — this file covers everything above that boundary: the worker loop,
its status() reporting, and that one failing engine cleanup() cannot block the others or the
repository prune from running.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import logging

import pytest

from backend.app.core.database.retention import RetentionWorker


@contextmanager
def _captured(caplog: pytest.LogCaptureFixture, logger_name: str) -> Iterator[None]:
    """Attach caplog's handler directly to `logger_name`, bypassing root-logger propagation.

    `configure_logging()` (backend/app/core/logging/setup.py) calls `logging.basicConfig(...,
    force=True)`, which replaces the root logger's handlers — if that has already run earlier in
    the same pytest session (e.g. via another test's app-startup fixture), a bare
    `caplog.at_level(...)` can silently miss records here. See the identical rationale/pattern in
    tests/engines/market_data_engine/test_keyless_providers.py.
    """
    target = logging.getLogger(logger_name)
    original_level, original_disabled, original_propagate = target.level, target.disabled, target.propagate
    target.addHandler(caplog.handler)
    target.setLevel(logging.DEBUG)
    target.disabled = False
    target.propagate = False
    try:
        yield
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(original_level)
        target.disabled = original_disabled
        target.propagate = original_propagate


class _FakeRepository:
    def __init__(self) -> None:
        self.calls = 0
        self.result: dict[str, int] = {"smc_objects": 3, "integration_events": 1}

    async def prune(self, **kwargs: object) -> dict[str, int]:
        self.calls += 1
        return dict(self.result)


class _FakeCleanable:
    def __init__(self, count: int = 0, *, raises: bool = False) -> None:
        self.count = count
        self.raises = raises
        self.calls = 0

    async def cleanup(self) -> int:
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return self.count


def _worker(repository: _FakeRepository, cleanable_services: tuple[object, ...] = ()) -> RetentionWorker:
    return RetentionWorker(
        repository,  # type: ignore[arg-type]
        enabled=True,
        interval_seconds=60,
        batch_size=500,
        analytical_object_retention_days=14,
        analytical_snapshot_retention_days=14,
        integration_audit_retention_days=14,
        operational_signal_retention_days=180,
        market_data_history_retention_days=7,
        cleanable_services=cleanable_services,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_run_once_returns_repository_deletes_and_engine_cleanup_counts() -> None:
    repository = _FakeRepository()
    ai_scoring = _FakeCleanable(count=5)
    worker = _worker(repository, (ai_scoring,))

    result = await worker.run_once()

    assert repository.calls == 1
    assert ai_scoring.calls == 1
    assert result["smc_objects"] == 3
    assert result["integration_events"] == 1
    assert result["_FakeCleanable"] == 5


@pytest.mark.asyncio
async def test_one_failing_cleanup_does_not_block_others_or_the_repository_prune(caplog: pytest.LogCaptureFixture) -> None:
    repository = _FakeRepository()
    failing = _FakeCleanable(raises=True)
    healthy = _FakeCleanable(count=7)
    worker = _worker(repository, (failing, healthy))

    with _captured(caplog, "backend.app.core.database.retention"):
        result = await worker.run_once()

    assert failing.calls == 1
    assert healthy.calls == 1
    assert repository.calls == 1
    assert result["smc_objects"] == 3
    assert "retention.engine_cleanup.failed" in caplog.text


@pytest.mark.asyncio
async def test_start_stop_lifecycle_and_status_fields() -> None:
    repository = _FakeRepository()
    worker = _worker(repository)
    assert worker.status()["running"] is False

    worker.start()
    try:
        assert worker.status()["running"] is True
        assert worker.status()["enabled"] is True
        assert worker.status()["crashed"] is False
    finally:
        await worker.stop()

    assert worker.status()["running"] is False


@pytest.mark.asyncio
async def test_disabled_worker_never_starts() -> None:
    repository = _FakeRepository()
    worker = RetentionWorker(
        repository,  # type: ignore[arg-type]
        enabled=False,
        interval_seconds=60,
        batch_size=500,
        analytical_object_retention_days=14,
        analytical_snapshot_retention_days=14,
        integration_audit_retention_days=14,
        operational_signal_retention_days=180,
        market_data_history_retention_days=7,
    )
    worker.start()
    assert worker.status()["running"] is False
    assert repository.calls == 0
    await worker.stop()


@pytest.mark.asyncio
async def test_worker_survives_a_per_cycle_repository_exception_and_keeps_running() -> None:
    class _RaisingRepository:
        async def prune(self, **kwargs: object) -> dict[str, int]:
            raise RuntimeError("db unavailable")

    worker = _worker(_RaisingRepository())  # type: ignore[arg-type]
    worker.interval_seconds = 0.01
    worker.start()
    try:
        for _ in range(200):
            if worker.last_error is not None:
                break
            import asyncio

            await asyncio.sleep(0.01)
        assert worker.last_error == "RuntimeError"
        assert worker.status()["running"] is True
        assert worker.status()["crashed"] is False
        assert worker.consecutive_failures >= 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_task_death_is_surfaced_via_status_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    class _BrokenClock:
        def __call__(self) -> datetime:
            raise RuntimeError("clock exploded")

    repository = _FakeRepository()
    worker = _worker(repository)
    worker.clock = _BrokenClock()  # breaks before the per-iteration try/except can catch it

    with _captured(caplog, "backend.app.core.database.retention"):
        worker.start()
        for _ in range(200):
            if worker.status()["last_fatal_error"] is not None:
                break
            import asyncio

            await asyncio.sleep(0.01)

    assert worker.status()["crashed"] is True
    assert worker.status()["last_fatal_error"] is not None
    assert "retention.worker.task_died" in caplog.text
    await worker.stop()
