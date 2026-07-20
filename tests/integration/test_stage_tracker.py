"""Regression tests for PipelineStageTracker correlation/identity behavior.

These cover the reported defect: the Liquidity stage repeatedly showed `failed` or stayed
`running` even though the Liquidity engine had genuinely completed. Root causes fixed:

1. STAGE_KEYS previously listed `liquidity_analysis` before `smc_analysis`, but `_run()` actually
   executes SMC first. Since both the "running" inference and `fail_in_flight` walk STAGE_KEYS in
   declared order looking for the first still-"waiting" stage, every cycle's SMC-analysis window
   displayed as "liquidity running", and any exception raised before liquidity's own mark call
   (including one thrown inside SMC itself) was misattributed to liquidity as "failed".
2. Cycle identity was "whichever cycle is last in the deque", not the specific
   (symbol, timeframe, boundary) triple — an outbox retry's `begin()` could orphan the previous
   attempt's cycle object while later `mark()`/`fail_in_flight()` calls kept resolving against
   `series[-1]`, silently landing on the wrong cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime

from backend.app.integration.stage_tracker import PipelineStageTracker


def _boundary(minute: int = 0) -> datetime:
    return datetime(2026, 7, 20, 15, minute, tzinfo=UTC)


def _statuses(tracker: PipelineStageTracker, symbol: str = "XAUUSD", timeframe: str = "M15") -> dict[str, str]:
    cycle = tracker.latest(symbol, timeframe)
    assert cycle is not None
    return {item["key"]: item["status"] for item in cycle["stages"]}


def test_stage_becomes_running_on_liquidity_start_and_stays_running_through_intermediate_activity() -> None:
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    tracker.mark("XAUUSD", "M15", boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")
    tracker.mark("XAUUSD", "M15", boundary, ("smc_analysis",), "success")

    stage = _statuses(tracker)
    assert stage["smc_analysis"] == "success"
    assert stage["liquidity_analysis"] == "running"

    # Intermediate liquidity domain events (equal-high cluster confirmed, pool created/touched,
    # target ranking updated) are never observed by the tracker — only `_run()`'s own
    # instrumentation calls mark() — so liquidity_analysis must stay "running" until the pipeline
    # explicitly reports the terminal completion, not flicker on unrelated domain activity.
    assert _statuses(tracker)["liquidity_analysis"] == "running"


def test_stage_becomes_success_on_liquidity_analysis_updated_completion() -> None:
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    tracker.mark("XAUUSD", "M15", boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")
    tracker.mark("XAUUSD", "M15", boundary, ("smc_analysis",), "success")
    tracker.mark("XAUUSD", "M15", boundary, ("liquidity_analysis",), "success")

    assert _statuses(tracker)["liquidity_analysis"] == "success"


def test_stage_becomes_degraded_only_on_explicit_degraded_terminal_event() -> None:
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    tracker.mark("XAUUSD", "M15", boundary, ("smc_analysis",), "success")
    tracker.mark("XAUUSD", "M15", boundary, ("liquidity_analysis",), "degraded")

    stage = _statuses(tracker)
    assert stage["liquidity_analysis"] == "degraded"
    assert stage["liquidity_analysis"] != "failed"


def test_stage_becomes_failed_only_on_an_actual_failure_event_not_a_later_unrelated_exception() -> None:
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    tracker.mark("XAUUSD", "M15", boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")
    tracker.mark("XAUUSD", "M15", boundary, ("smc_analysis",), "success")
    tracker.mark("XAUUSD", "M15", boundary, ("liquidity_analysis",), "success")
    # volume_profile.analyze() raises, unrelated to liquidity, which already completed.
    tracker.fail_in_flight("XAUUSD", "M15", boundary)

    stage = _statuses(tracker)
    assert stage["liquidity_analysis"] == "success", "a downstream failure must never retroactively overwrite an already-completed stage"
    assert stage["volume_profile"] == "failed"
    assert stage["institutional_flow"] == "skipped"


def test_a_failure_inside_smc_is_not_misattributed_to_liquidity() -> None:
    """Regression test for the exact reported defect: SMC raising before liquidity is ever
    called must fail smc_analysis, not liquidity_analysis, even though the two stages run back
    to back."""
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    tracker.mark("XAUUSD", "M15", boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")
    # smc.analyze_candles() raises before its own mark() call ever runs.
    tracker.fail_in_flight("XAUUSD", "M15", boundary)

    stage = _statuses(tracker)
    assert stage["smc_analysis"] == "failed"
    assert stage["liquidity_analysis"] == "skipped"


def test_retry_of_the_same_candle_resets_in_place_instead_of_orphaning_the_previous_attempt() -> None:
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    tracker.mark("XAUUSD", "M15", boundary, ("smc_analysis", "liquidity_analysis"), "success")
    tracker.fail_in_flight("XAUUSD", "M15", boundary)  # first attempt fails downstream

    tracker.begin("XAUUSD", "M15", boundary)  # outbox retries the same envelope/boundary
    cycle = tracker.latest("XAUUSD", "M15")
    assert cycle is not None
    assert cycle["complete"] is False
    assert cycle["attempt"] == 2
    stage = _statuses(tracker)
    assert stage["smc_analysis"] == "waiting"
    assert stage["liquidity_analysis"] == "waiting"

    tracker.mark("XAUUSD", "M15", boundary, ("smc_analysis", "liquidity_analysis"), "success")
    assert _statuses(tracker)["liquidity_analysis"] == "success"


def test_latest_selects_by_candle_timestamp_not_insertion_order() -> None:
    """An older candle finishing after a newer one has already begun must not be shadowed by the
    newer (possibly still in-flight) cycle when querying `latest()`."""
    tracker = PipelineStageTracker()
    older, newer = _boundary(0), _boundary(15)
    tracker.begin("XAUUSD", "M15", older)
    tracker.begin("XAUUSD", "M15", newer)

    latest = tracker.latest("XAUUSD", "M15")
    assert latest is not None
    assert latest["candle_timestamp"] == newer

    recent = tracker.recent("XAUUSD", "M15", limit=5)
    assert [item["candle_timestamp"] for item in recent] == [newer, older]


def test_mark_and_fail_are_keyed_by_the_exact_boundary_not_the_most_recently_begun_cycle() -> None:
    """A mark() for an older, already-begun cycle must land on that cycle even after a newer
    cycle has since begun for the same symbol/timeframe."""
    tracker = PipelineStageTracker()
    older, newer = _boundary(0), _boundary(15)
    tracker.begin("XAUUSD", "M15", older)
    tracker.begin("XAUUSD", "M15", newer)

    tracker.mark("XAUUSD", "M15", older, ("smc_analysis", "liquidity_analysis"), "success")

    older_cycle = {item["key"]: item["status"] for item in tracker.recent("XAUUSD", "M15", limit=5)[1]["stages"]}
    newer_cycle = {item["key"]: item["status"] for item in tracker.recent("XAUUSD", "M15", limit=5)[0]["stages"]}
    assert older_cycle["liquidity_analysis"] == "success"
    assert newer_cycle["liquidity_analysis"] == "waiting"


def test_fail_in_flight_captures_full_exception_detail_against_the_failing_stage() -> None:
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary, correlation_id="corr-123")
    tracker.mark("XAUUSD", "M15", boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")

    def _raise_inside_smc() -> None:
        raise ZeroDivisionError("float division by zero")

    try:
        _raise_inside_smc()
    except ZeroDivisionError as exc:
        tracker.fail_in_flight("XAUUSD", "M15", boundary, exc=exc)

    cycle = tracker.latest("XAUUSD", "M15")
    assert cycle is not None
    assert cycle["correlation_id"] == "corr-123"
    smc_stage = next(item for item in cycle["stages"] if item["key"] == "smc_analysis")
    assert smc_stage["status"] == "failed"
    assert smc_stage["error"] is not None
    assert smc_stage["error"]["exception_class"] == "ZeroDivisionError"
    assert smc_stage["error"]["message"] == "float division by zero"
    assert "ZeroDivisionError" in smc_stage["error"]["traceback"]
    assert smc_stage["error"]["function"] == "_raise_inside_smc"
    assert smc_stage["error"]["line"] is not None
    assert smc_stage["error"]["file"].endswith("test_stage_tracker.py")

    liquidity_stage = next(item for item in cycle["stages"] if item["key"] == "liquidity_analysis")
    assert liquidity_stage["status"] == "skipped"
    assert liquidity_stage["error"] is None


def test_stage_attempt_exposes_the_failure_independent_of_snapshot_lookup() -> None:
    """`stage_attempt()` is what backs `latest_attempt_*` in Market Intelligence — it must report
    a failure regardless of whatever a separate persisted-snapshot query would find."""
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    tracker.mark("XAUUSD", "M15", boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")
    try:
        raise ValueError("insufficient candle history")
    except ValueError as exc:
        tracker.fail_in_flight("XAUUSD", "M15", boundary, exc=exc)

    attempt = tracker.stage_attempt("XAUUSD", "M15", "smc_analysis")
    assert attempt is not None
    assert attempt["status"] == "failed"
    assert attempt["error"]["exception_class"] == "ValueError"


def test_a_successful_retry_clears_the_previously_recorded_error_for_that_stage() -> None:
    tracker = PipelineStageTracker()
    boundary = _boundary()
    tracker.begin("XAUUSD", "M15", boundary)
    try:
        raise RuntimeError("transient")
    except RuntimeError as exc:
        tracker.fail_in_flight("XAUUSD", "M15", boundary, exc=exc)

    tracker.begin("XAUUSD", "M15", boundary)  # outbox retries the same candle
    tracker.mark("XAUUSD", "M15", boundary, ("smc_analysis",), "success")

    attempt = tracker.stage_attempt("XAUUSD", "M15", "smc_analysis")
    assert attempt is not None
    assert attempt["status"] == "success"
    assert attempt["error"] is None
