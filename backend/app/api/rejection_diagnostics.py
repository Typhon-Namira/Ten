"""Human-readable "why was this rejected" breakdown for a Signal Decision.

Pure presentation layer: reads already-computed `SignalDecision`/`AIScoreSnapshot`/session data
and maps it onto explicit categories. It never re-evaluates eligibility, adds no new hard-block
rules, and cannot change which decisions are eligible/blocked/observe-only — it only explains a
decision that has already been made deterministically by the Signal Decision Engine.
"""

from __future__ import annotations

from typing import Any

from backend.app.engines.ai_scoring_engine import AIScoreSnapshot
from backend.app.engines.market_data_engine.models import MarketSession
from backend.app.engines.signal_decision_engine import RuleCategory, SignalDecision

CATEGORY_LABELS: dict[str, str] = {
    "confidence_too_low": "Confidence too low",
    "liquidity_missing": "Liquidity missing",
    "htf_trend_conflict": "HTF trend conflict",
    "weak_displacement": "Weak displacement",
    "fvg_not_confirmed": "FVG not confirmed",
    "ob_invalid": "OB invalid",
    "news_filter": "News filter",
    "economic_filter": "Economic filter",
    "institutional_confirmation": "Insufficient institutional confirmation",
    "session_mismatch": "Session mismatch",
    "volume_confirmation": "Volume confirmation failed",
    "confidence_score": "Confidence score",
}

_LOW_LIQUIDITY_SESSIONS = {MarketSession.ASIA, MarketSession.CLOSED, MarketSession.WEEKEND, MarketSession.HOLIDAY}


def _entry(key: str, status: str, detail: str, *, observed: Any = None, threshold: Any = None) -> dict[str, Any]:
    return {"key": key, "label": CATEGORY_LABELS[key], "status": status, "detail": detail, "observed_value": observed, "threshold": threshold}


def _rule_entries(decision: SignalDecision, category: RuleCategory) -> list:
    return [item for item in decision.rules if item.category == category]


def _component(score: AIScoreSnapshot | None, source_engine: str):
    if score is None:
        return None
    return next((item for item in score.components if item.source_engine == source_engine), None)


def _component_entry(key: str, score: AIScoreSnapshot | None, source_engine: str, weak_markers: tuple[str, ...] = ()) -> dict[str, Any]:
    if score is None:
        return _entry(key, "not_evaluated", "AI scoring snapshot unavailable for this candle")
    if source_engine in score.missing_sources:
        return _entry(key, "failed", f"{source_engine} evidence was unavailable for this AI score")
    component = _component(score, source_engine)
    if component is None:
        return _entry(key, "not_evaluated", f"{source_engine} did not contribute to this AI score")
    flagged = source_engine in score.degraded_sources or any(marker in code for code in component.reason_codes for marker in weak_markers)
    status = "failed" if flagged else "passed"
    detail = ", ".join(component.reason_codes) if component.reason_codes else f"{source_engine} evidence within normal bounds"
    return _entry(key, status, detail, observed=round(component.quality_contribution, 2), threshold=None)


def build_rejection_diagnostics(decision: SignalDecision, score: AIScoreSnapshot | None, active_session: MarketSession | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    confidence_rules = [item for item in decision.rules if item.rule_id == "confidence.minimum"]
    if confidence_rules:
        rule = confidence_rules[0]
        status = "passed" if rule.outcome.value == "passed" else "failed"
        entries.append(_entry("confidence_too_low", status, rule.reason_code, observed=rule.observed_value, threshold=rule.threshold))
    else:
        entries.append(_entry("confidence_too_low", "not_evaluated", "confidence rule was not evaluated"))
    entries.append(_entry("confidence_score", "informational", f"{decision.confidence_score:.1f}%", observed=decision.confidence_score))

    entries.append(_component_entry("liquidity_missing", score, "liquidity"))

    regime_rules = _rule_entries(decision, RuleCategory.MARKET_REGIME)
    if regime_rules:
        rule = regime_rules[0]
        status = "passed" if rule.outcome.value == "passed" else "failed"
        entries.append(_entry("htf_trend_conflict", status, rule.reason_code, observed=rule.observed_value))
    else:
        entries.append(_entry("htf_trend_conflict", "not_evaluated", "market regime context was not evaluated"))

    entries.append(_component_entry("weak_displacement", score, "smc", weak_markers=("displacement",)))
    entries.append(_component_entry("fvg_not_confirmed", score, "smc", weak_markers=("fvg", "imbalance", "gap")))
    entries.append(_component_entry("ob_invalid", score, "smc", weak_markers=("order_block", "ob_")))

    economic_rules = _rule_entries(decision, RuleCategory.ECONOMIC_EVENT)
    if economic_rules:
        rule = economic_rules[0]
        status = "passed" if rule.outcome.value == "passed" else "failed"
        entries.append(_entry("news_filter", status, rule.reason_code))
        entries.append(_entry("economic_filter", status, rule.reason_code))
    else:
        entries.append(_entry("news_filter", "not_evaluated", "economic event window was not evaluated"))
        entries.append(_entry("economic_filter", "not_evaluated", "economic event window was not evaluated"))

    entries.append(_component_entry("institutional_confirmation", score, "institutional_flow"))
    entries.append(_component_entry("volume_confirmation", score, "volume_profile"))

    if active_session is None:
        entries.append(_entry("session_mismatch", "not_evaluated", "session context unavailable"))
    elif active_session in _LOW_LIQUIDITY_SESSIONS:
        entries.append(_entry("session_mismatch", "failed", f"{active_session.value} session has reduced liquidity for this instrument"))
    else:
        entries.append(_entry("session_mismatch", "passed", f"{active_session.value} session is an active trading window"))

    return entries
