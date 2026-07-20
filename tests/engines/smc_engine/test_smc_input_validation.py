"""Regression tests for the exact SMC failure reported live: `smc_analysis` failing with every
downstream stage correctly skipped, while Market Intelligence kept showing a stale-but-successful
SMC snapshot as if the pipeline were healthy.

Root-cause trace (see PART 1 investigation notes / final report for the full function-by-function
walk): `SMCService.analyze_candles()` -> `BaselineSMCAnalyzer.analyze_snapshot()` ->
`CandleContext.build()` is the first and only point in the SMC pipeline that raises on malformed
input *before* any analysis logic runs — it is the "normalization" step referenced in
`_run()`'s SMC-analysis stage. Every downstream computation (displacement, swing detection,
BOS/CHOCH, FVG, order block, dealing range) was traced and found to guard every division with an
epsilon floor (`max(x, 1e-12)`), so a bare `ZeroDivisionError` from those stages is not plausible;
a malformed/duplicate/out-of-order candle history reaching `CandleContext.build()` is the
concrete, reproducible failure mode matching the reported symptom (fails during analysis/
normalization, before snapshot construction, persistence, or commit are ever reached).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.smc_engine.config import SMCConfig
from backend.app.engines.smc_engine.context import CandleContext
from backend.app.engines.smc_engine.exceptions import InvalidSMCInput


_BASE = datetime(2026, 7, 20, 15, 0, tzinfo=UTC)


def _candle(offset_minutes: int, *, open_: float = 3350.0, high: float = 3352.0, low: float = 3348.0, close: float = 3351.0) -> Candle:
    return Candle(
        timestamp=_BASE + timedelta(minutes=offset_minutes),
        symbol="XAUUSD",
        timeframe=Timeframe.M15,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1200.0,
        provider="twelve_data",
    )


def _history(count: int = 10) -> list[Candle]:
    return [_candle(offset) for offset in range(0, count * 15, 15)]


def test_duplicate_candle_timestamp_raises_invalid_smc_input_during_normalization() -> None:
    """The exact input shape that produces this: `market_data.history()` returning two candles
    that share a timestamp (e.g. a forming-candle re-fetch racing a just-closed candle write)."""
    candles = _history(6)
    duplicate = candles[3].model_copy(update={"close": candles[3].close + 1})
    malformed = [*candles[:4], duplicate, *candles[4:]]

    with pytest.raises(InvalidSMCInput, match="duplicate candle"):
        CandleContext.build(malformed, SMCConfig())


def test_non_chronological_candle_history_raises_invalid_smc_input_during_normalization() -> None:
    """The exact input shape that produces this: `market_data.history()` returning candles out of
    strict timestamp order (e.g. a repository read racing a concurrent write, or a provider
    returning an unordered page)."""
    candles = _history(6)
    out_of_order = [candles[0], candles[1], candles[3], candles[2], candles[4], candles[5]]

    with pytest.raises(InvalidSMCInput, match="strictly chronological"):
        CandleContext.build(out_of_order, SMCConfig())


def test_mixed_symbol_in_history_raises_invalid_smc_input_during_normalization() -> None:
    candles = _history(6)
    candles[2] = candles[2].model_copy(update={"symbol": "EURUSD"})

    with pytest.raises(InvalidSMCInput, match="symbol and timeframe"):
        CandleContext.build(candles, SMCConfig())


def test_well_formed_history_builds_successfully_confirming_the_failure_is_input_specific() -> None:
    """Control case: the same volume of history, strictly ordered and deduplicated, must not
    raise — proving the failure modes above are genuinely about malformed input, not a general
    regression in `CandleContext.build()` itself."""
    context = CandleContext.build(_history(10), SMCConfig())
    assert context.symbol == "XAUUSD"
    assert len(context.candles) == 10


def test_flat_identical_candles_produce_zero_atr_but_do_not_crash_downstream() -> None:
    """An illiquid/flat period (identical OHLC across many candles) drives ATR to exactly 0.0 —
    `structure.displacement()`, `swing._point()`, and `advanced._add_zone()` all floor their
    denominators at 1e-12, so this must not raise even though it is an extreme, data-dependent
    edge case worth covering explicitly."""
    from backend.app.engines.smc_engine.analyzer import BaselineSMCAnalyzer
    from backend.app.engines.smc_engine.models import ProcessingMode

    flat = [_candle(minute, open_=3350.0, high=3350.0, low=3350.0, close=3350.0) for minute in range(0, 150, 15)]
    analyzer = BaselineSMCAnalyzer(SMCConfig())
    snapshot = analyzer.analyze_snapshot(flat, ProcessingMode.HISTORICAL)
    assert snapshot is not None
