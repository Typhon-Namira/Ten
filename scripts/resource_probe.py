"""Deterministic local resource probe for market-data polling and in-process retention.

This intentionally uses only standard-library instrumentation and TEN's in-memory adapters so it
can run in CI without a live provider or PostgreSQL. Production-only counters are reported by the
runtime health endpoint and must be collected separately after deployment.
"""

from __future__ import annotations

import asyncio
import ctypes
from datetime import UTC, datetime
import gc
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc

sys.path.insert(0, os.environ.get("TEN_PROBE_SOURCE_ROOT", str(Path(__file__).resolve().parents[1])))

from backend.app.engines.market_data_engine.cache import MarketDataCache
from backend.app.engines.market_data_engine.manager import ProviderManager, ProviderRegistry
from backend.app.engines.market_data_engine.models import Candle, Timeframe
from backend.app.engines.market_data_engine.providers import InMemoryMarketDataProvider, ProviderName
from backend.app.engines.market_data_engine.repository import InMemoryMarketDataRepository
from backend.app.engines.market_data_engine.service import MarketDataService
from backend.app.engines.market_data_engine.worker import MarketDataWorker
from backend.app.events import InMemoryEventBus


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _rss_bytes() -> int | None:
    if os.name != "nt":
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize) if ok else None


def _candle() -> Candle:
    now = datetime(2026, 7, 22, 9, 0, tzinfo=UTC)
    return Candle(
        symbol="XAUUSD",
        timeframe=Timeframe.M1,
        timestamp=now,
        open=4100,
        high=4102,
        low=4099,
        close=4101,
        volume=10,
        provider="memory",
        ingestion_timestamp=now + Timeframe.M1.duration,
    )


async def _event_loop_lag(samples: int = 100) -> dict[str, float]:
    observed: list[float] = []
    interval = 0.001
    for _ in range(samples):
        started = time.perf_counter()
        await asyncio.sleep(interval)
        observed.append(max(0.0, (time.perf_counter() - started - interval) * 1000))
    ordered = sorted(observed)
    return {
        "mean_ms": statistics.fmean(observed),
        "p95_ms": ordered[max(0, round(0.95 * len(ordered)) - 1)],
        "max_ms": max(observed),
    }


async def probe(iterations: int = 600) -> dict[str, object]:
    registry = ProviderRegistry()
    registry.register(InMemoryMarketDataProvider((_candle(),)))
    repository = InMemoryMarketDataRepository()
    event_bus = InMemoryEventBus()
    with tempfile.TemporaryDirectory(prefix="ten-resource-probe-") as directory:
        service = MarketDataService(
            ProviderManager(registry, preferred=ProviderName.MEMORY.value),
            repository=repository,
            cache=MarketDataCache(Path(directory), max_entries=100),
            event_bus=event_bus,
        )
        worker = MarketDataWorker(
            service,
            enabled=True,
            symbols=("XAUUSD",),
            timeframes=(Timeframe.M1,),
            bootstrap_enabled=False,
            bootstrap_candles=500,
            poll_seconds=10,
        )
        log_events = 0
        log_bytes = 0

        class CountingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                nonlocal log_events, log_bytes
                log_events += 1
                log_bytes += len(self.format(record).encode())

        handler = CountingHandler(level=logging.INFO)
        worker_logger = logging.getLogger("backend.app.engines.market_data_engine.worker")
        original_level, original_propagate = worker_logger.level, worker_logger.propagate
        worker_logger.setLevel(logging.INFO)
        worker_logger.propagate = False
        worker_logger.addHandler(handler)
        gc.collect()
        tracemalloc.start()
        rss_before = _rss_bytes()
        heap_before, _ = tracemalloc.get_traced_memory()
        cpu_before = time.process_time()
        started = time.perf_counter()
        try:
            for _ in range(iterations):
                await worker._poll("XAUUSD", Timeframe.M1)
        finally:
            worker_logger.removeHandler(handler)
            worker_logger.setLevel(original_level)
            worker_logger.propagate = original_propagate
        elapsed = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_before
        heap_after, heap_peak = tracemalloc.get_traced_memory()
        rss_after = _rss_bytes()
        lag = await _event_loop_lag()
        tracemalloc.stop()
        provider = service.manager.statistics[ProviderName.MEMORY.value]
        return {
            "scenario": "identical_closed_m1_poll",
            "iterations": iterations,
            "elapsed_seconds": elapsed,
            "cpu_seconds": cpu_seconds,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": None if rss_before is None or rss_after is None else rss_after - rss_before,
            "python_heap_before_bytes": heap_before,
            "python_heap_after_bytes": heap_after,
            "python_heap_peak_bytes": heap_peak,
            "active_asyncio_tasks": len(asyncio.all_tasks()),
            "thread_count": threading.active_count(),
            "event_loop_lag": lag,
            "provider_requests": provider.requests,
            "info_log_events": log_events,
            "info_log_bytes": log_bytes,
            "event_bus_published": event_bus.published_total,
            "event_bus_retained": len(event_bus.history()),
            "realtime_observations_retained": len(repository._realtime),
            "durable_candles": await repository.count("XAUUSD", Timeframe.M1),
            "memory_cache_entries": len(service.cache._memory),
        }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(probe()), indent=2, sort_keys=True))
