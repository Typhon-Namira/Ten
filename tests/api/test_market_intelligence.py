"""Regression tests for the Market Intelligence aggregation/retrieval path.

Covers the reported defect: most engine fields showed "unavailable" and spread showed 0.00 even
though snapshots existed. These tests exercise the pure `build_market_intelligence` function
directly (no app/DB needed) with lightweight stand-ins for each engine's snapshot shape, proving:

- a persisted snapshot from every source (liquidity, market_regime, institutional_flow,
  volume_profile, ai_scoring, signal_decision) is reflected in the response, not silently dropped;
- a genuinely missing snapshot is reported as "unavailable" (not indistinguishable from an error);
- a source-level exception is reported as "error" with the exception type, distinct from
  "unavailable";
- an old-but-present snapshot is still returned, labeled "stale" rather than hidden;
- a decision whose validity window has lapsed is still shown, flagged `decision_active: False`,
  instead of disappearing the moment it expires;
- spread is `null` (never a fabricated 0.00) whenever the active provider doesn't support real
  bid/ask spread data, and is passed through when it does.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from backend.app.api.market_intelligence import build_market_intelligence


def _now() -> datetime:
    return datetime(2026, 7, 20, 18, 0, 0, tzinfo=UTC)


class _Model:
    """Minimal pydantic-like stand-in exposing attributes and a `.model_dump(mode=...)`."""

    def __init__(self, **kwargs: object) -> None:
        self._data = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        return dict(self._data)


def _candle(spread: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(timestamp=_now() - timedelta(minutes=15), open=3350.0, high=3360.0, low=3345.0, close=3355.0, volume=1200.0, spread=spread, provider="twelve_data")


def _liquidity(as_of: datetime) -> SimpleNamespace:
    return SimpleNamespace(analysis_timestamp=as_of, state=_Model(active_pool_ids=["p1"], bias="bullish"))


def _market_regime(as_of: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        analysis_timestamp=as_of,
        dominant_regime=SimpleNamespace(value="trending"),
        trend_regime=SimpleNamespace(value="uptrend"),
        directional_bias=SimpleNamespace(value="bullish"),
        trend_strength=0.72,
        volatility_score=0.4,
        confidence=0.81,
    )


def _institutional_flow(as_of: datetime) -> SimpleNamespace:
    return SimpleNamespace(analysis_timestamp=as_of, state=_Model(net_flow="buying"), quality=_Model(score=0.8))


def _volume_profile(as_of: datetime) -> SimpleNamespace:
    return SimpleNamespace(analysis_timestamp=as_of, volume_data_quality=_Model(completeness=0.95))


def _ai_score(as_of: datetime) -> SimpleNamespace:
    return SimpleNamespace(as_of=as_of, confidence_score=64.0, directional_label=SimpleNamespace(value="bullish"), composite_score=42.0, missing_sources=[], degraded_sources=[])


def _smc_snapshot(as_of: datetime) -> SimpleNamespace:
    structure = SimpleNamespace(current_direction=SimpleNamespace(value="bullish"), active_dealing_range_id=None)
    return SimpleNamespace(
        status=SimpleNamespace(value="ready"),
        analysis_timestamp=as_of,
        structure_state=structure,
        structure_events=(),
        zones=(),
        dealing_ranges=(),
        multi_timeframe_context=None,
    )


def _decision(as_of: datetime, valid_until: datetime) -> SimpleNamespace:
    return SimpleNamespace(as_of=as_of, state=SimpleNamespace(value="eligible"), direction=SimpleNamespace(value="bullish"), eligibility_score=78.0, valid_until=valid_until)


def _build(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = dict(
        instrument="XAUUSD",
        timeframe="M15",
        now=_now(),
        session=None,
        candle=None,
        spread_supported=None,
        smc=None,
        liquidity=None,
        volume_profile=None,
        institutional_flow=None,
        market_regime=None,
        economic_context=None,
        ai_score=None,
        decision=None,
        errors={},
    )
    defaults.update(overrides)
    return build_market_intelligence(**defaults)  # type: ignore[arg-type]


def test_persisted_snapshots_from_every_engine_appear_in_market_intelligence() -> None:
    fresh = _now() - timedelta(minutes=2)
    result = _build(
        candle=_candle(spread=0.35),
        spread_supported=True,
        liquidity=_liquidity(fresh),
        volume_profile=_volume_profile(fresh),
        institutional_flow=_institutional_flow(fresh),
        market_regime=_market_regime(fresh),
        ai_score=_ai_score(fresh),
        decision=_decision(fresh, _now() + timedelta(minutes=10)),
    )

    assert result["liquidity"]["available"] is True
    assert result["liquidity"]["state"] == {"active_pool_ids": ["p1"], "bias": "bullish"}
    assert result["market_regime"]["available"] is True
    assert result["market_regime"]["dominant_regime"] == "trending"
    assert result["institutional_flow"]["available"] is True
    assert result["institutional_flow"]["state"] == {"net_flow": "buying"}
    assert result["volume_profile"]["available"] is True
    assert result["confidence_percent"] == 64.0
    assert result["decision_status"] == "eligible"
    assert result["decision_active"] is True
    assert result["scenario_readiness_percent"] == 78.0

    diagnostics = {item["source"]: item for item in result["diagnostics"]}
    for source in ("liquidity", "market_regime", "institutional_flow", "volume_profile", "ai_scoring", "signal_decision"):
        assert diagnostics[source]["status"] == "ok", source
        assert diagnostics[source]["snapshot_found"] is True, source
        assert diagnostics[source]["freshness"] == "fresh", source


def test_missing_snapshot_reports_unavailable_not_silently_dropped() -> None:
    result = _build()
    diagnostics = {item["source"]: item for item in result["diagnostics"]}
    assert diagnostics["liquidity"]["status"] == "unavailable"
    assert diagnostics["liquidity"]["snapshot_found"] is False
    assert result["liquidity"]["available"] is False


def test_source_error_is_surfaced_distinctly_from_unavailable() -> None:
    result = _build(errors={"liquidity": "OperationalError"})
    diagnostics = {item["source"]: item for item in result["diagnostics"]}
    assert diagnostics["liquidity"]["status"] == "error"
    assert diagnostics["liquidity"]["error"] == "OperationalError"


def test_stale_snapshot_is_still_returned_but_labeled_stale_not_hidden() -> None:
    old = _now() - timedelta(hours=3)
    result = _build(liquidity=_liquidity(old))
    assert result["liquidity"]["available"] is True
    diagnostic = {item["source"]: item for item in result["diagnostics"]}["liquidity"]
    assert diagnostic["freshness"] == "stale"
    assert diagnostic["age_seconds"] == pytest.approx(3 * 3600, rel=0.01)


def test_expired_decision_still_shown_but_flagged_inactive() -> None:
    result = _build(decision=_decision(_now() - timedelta(minutes=30), _now() - timedelta(minutes=15)))
    assert result["decision_status"] == "eligible"
    assert result["decision_active"] is False


def test_spread_is_null_when_provider_does_not_support_it_even_if_field_is_zero() -> None:
    result = _build(candle=_candle(spread=0.0), spread_supported=False)
    assert result["spread"] is None
    assert result["current_candle"]["spread"] is None


def test_spread_is_returned_when_provider_supports_it() -> None:
    result = _build(candle=_candle(spread=0.42), spread_supported=True)
    assert result["spread"] == 0.42


def test_latest_attempt_failure_is_visible_even_with_a_stale_successful_snapshot() -> None:
    """Regression test for the exact reported gap: Market Intelligence showed SMC as `ok` from a
    stale 16:00 snapshot while the live 18:30 cycle had actually failed. `status`/`snapshot_found`
    may legitimately still describe the old snapshot, but `latest_attempt_status` must show the
    real, current failure — the two must never be conflated into one misleading "healthy" signal.
    """
    stale = _now() - timedelta(hours=2, minutes=30)
    failure_at = _now() - timedelta(seconds=5)
    stage_attempts = {
        "smc": {
            "status": "failed",
            "updated_at": failure_at,
            "error": {"exception_class": "ZeroDivisionError", "message": "float division by zero", "file": "backend/app/engines/smc_engine/analyzer.py", "line": 214, "function": "_displacement_score"},
        }
    }
    result = _build(smc=_smc_snapshot(stale), stage_attempts=stage_attempts)

    diagnostic = {item["source"]: item for item in result["diagnostics"]}["smc"]
    assert diagnostic["status"] == "ok"  # the stale snapshot is genuinely still there
    assert diagnostic["snapshot_found"] is True
    assert diagnostic["freshness"] == "stale"
    assert diagnostic["latest_attempt_status"] == "failed"  # but the latest run did not succeed
    assert diagnostic["latest_attempt_at"] == failure_at
    assert diagnostic["latest_attempt_error"]["exception_class"] == "ZeroDivisionError"
    assert diagnostic["latest_attempt_error"]["file"] == "backend/app/engines/smc_engine/analyzer.py"


def test_source_without_a_stage_mapping_reports_no_latest_attempt() -> None:
    """market_data and economic_calendar aren't tracked pipeline stages (see STAGE_KEYS), so they
    must never claim a latest_attempt_status rather than fabricating one."""
    result = _build(candle=_candle(), spread_supported=True)
    diagnostic = {item["source"]: item for item in result["diagnostics"]}["market_data"]
    assert diagnostic["latest_attempt_status"] is None
    assert diagnostic["latest_attempt_error"] is None


def test_economic_calendar_degraded_data_reflected_in_diagnostics_not_just_ok() -> None:
    """Regression test: diagnostics previously said `ok`/`fresh` for economic_calendar purely
    because the call didn't raise, while the main panel separately said `degraded` based on the
    actual data (`unavailable_context`) — the two must agree."""
    context = SimpleNamespace(
        analysis_timestamp=_now(),
        unavailable_context=("no relevant event mapping",),
        risk_window_phase=SimpleNamespace(value="outside"),
        risk_score=0.0,
        next_relevant_event=None,
    )
    result = _build(economic_context=context)

    assert result["economic_status"]["degraded"] is True
    diagnostic = {item["source"]: item for item in result["diagnostics"]}["economic_calendar"]
    assert diagnostic["status"] == "degraded"
    assert diagnostic["status"] != "ok"


def test_economic_calendar_with_relevant_events_reports_ok_consistently() -> None:
    context = SimpleNamespace(
        analysis_timestamp=_now(),
        unavailable_context=(),
        risk_window_phase=SimpleNamespace(value="pre_event"),
        risk_score=0.4,
        next_relevant_event=SimpleNamespace(display_name="Non-Farm Payrolls"),
    )
    result = _build(economic_context=context)

    assert result["economic_status"]["degraded"] is False
    diagnostic = {item["source"]: item for item in result["diagnostics"]}["economic_calendar"]
    assert diagnostic["status"] == "ok"


def test_generated_at_is_separate_from_latest_candle_and_snapshot_timestamps() -> None:
    fresh = _now() - timedelta(minutes=2)
    result = _build(candle=_candle(), spread_supported=True, liquidity=_liquidity(fresh))
    assert result["generated_at"] == _now()
    assert result["latest_candle_timestamp"] == _candle().timestamp
    diagnostic = {item["source"]: item for item in result["diagnostics"]}["liquidity"]
    assert diagnostic["last_successful_snapshot_at"] == fresh
    assert diagnostic["last_successful_snapshot_at"] != result["generated_at"]
