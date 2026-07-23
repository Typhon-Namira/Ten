"""Authoritative, same-cycle dashboard read model.

This endpoint never runs analytics. It only joins persisted records through the latest
``UnifiedMarketState`` boundary so the frontend cannot accidentally combine unrelated cycles.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request

from backend.app.core.feature_flags import FeatureFlag
from backend.app.engines.market_data_engine.models import canonical_symbol

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])
logger = logging.getLogger(__name__)


def _stage(
    *,
    status: str,
    reason: str,
    data: Any = None,
    record_id: object | None = None,
    timestamp: datetime | None = None,
    error_code: str | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "record_id": str(record_id) if record_id is not None else None,
        "timestamp": timestamp,
        "error_code": error_code,
        "retryable": retryable,
        "data": data.model_dump(mode="json") if hasattr(data, "model_dump") else data,
    }


def _record_status(value: Any, *, available_reason: str) -> tuple[str, str]:
    raw_status = str(getattr(getattr(value, "status", None), "value", getattr(value, "status", "")))
    if raw_status in {"failed", "invalid", "unavailable", "error"}:
        reasons = tuple(getattr(value, "reason_codes", ()) or ())
        reason = str(getattr(value, "failure_state", None) or (reasons[0] if reasons else raw_status))
        return "failed", reason
    if raw_status in {"degraded", "partial"}:
        return "degraded", raw_status
    return "available", available_reason


@router.get("/latest")
async def latest_dashboard(request: Request, instrument: str = "XAUUSD") -> dict[str, Any]:
    started = perf_counter()
    correlation_id = request.headers.get("x-correlation-id") or str(uuid4())
    symbol = canonical_symbol(instrument)
    flags = request.app.state.engine_registry.context.feature_flags
    shadow_enabled = flags.is_enabled(FeatureFlag.AI_CENTRIC_SHADOW_MODE)
    proposals_enabled = flags.is_enabled(FeatureFlag.AI_SIGNAL_PROPOSALS)
    monitoring_enabled = flags.is_enabled(FeatureFlag.AI_SIGNAL_MONITORING)
    publication_enabled = flags.is_enabled(FeatureFlag.AI_SIGNAL_PUBLICATION)
    now = datetime.now(UTC)
    logger.info(
        "dashboard_api.request",
        extra={"path": request.url.path, "instrument": symbol, "correlation_id": correlation_id},
    )

    usage_rows = await request.app.state.final_decision_repository.usage_for_date(now.date().isoformat())
    usage = {
        "request_count": sum(item.request_count for item in usage_rows),
        "total_tokens": (
            sum(item.total_tokens or 0 for item in usage_rows)
            if any(item.total_tokens is not None for item in usage_rows)
            else None
        ),
        "successful_requests": sum(item.success for item in usage_rows),
        "failed_requests": sum(not item.success for item in usage_rows),
    }
    calibration = await request.app.state.quant_forecast_repository.latest_calibration(
        request.app.state.quant_forecast_service.config.model_name
    )
    performance = await request.app.state.final_decision_repository.latest_performance_report()
    readiness = await request.app.state.final_decision_repository.latest_readiness_report()
    quant_health = request.app.state.quant_forecast_service.health()
    ai_health = request.app.state.ai_reasoning_service.health()
    guardrail_health = request.app.state.final_decision_service.health()
    runtime = {
        "operating_profile": (
            "analytical_live"
            if publication_enabled
            else "shadow"
            if shadow_enabled
            else "safe_test"
        ),
        "feature_flags": flags.snapshot(),
        "analytical_only": True,
        "broker_execution_available": False,
    }

    state = await request.app.state.unified_market_state_repository.latest_state(symbol)
    if state is None:
        reason = "ai_centric_shadow_mode_disabled" if not shadow_enabled else "awaiting_synchronized_m1_m5_m15_state"
        stages = {
            "market_state": _stage(status="not_available", reason=reason),
            "engine_outputs": _stage(status="not_available", reason=reason),
            "quant_forecast": _stage(status="not_available", reason="awaiting_unified_market_state"),
            "ai_reasoning": _stage(status="not_available", reason="awaiting_quant_forecast"),
            "ai_proposal": _stage(
                status="not_available",
                reason="ai_signal_proposals_disabled" if not proposals_enabled else "awaiting_ai_reasoning",
            ),
            "guardrails": _stage(status="not_evaluated", reason="awaiting_ai_proposal"),
            "final_action": _stage(status="not_available", reason="awaiting_guardrail_evaluation"),
            "publication": _stage(
                status="not_available",
                reason="ai_signal_publication_disabled" if not publication_enabled else "awaiting_final_action",
            ),
            "monitoring": _stage(
                status="not_available",
                reason="ai_signal_monitoring_disabled" if not monitoring_enabled else "awaiting_managed_signal",
            ),
            "outcome": _stage(status="not_evaluated", reason="awaiting_managed_signal"),
        }
        response = {
            "status": "pending",
            "instrument": symbol,
            "generated_at": now,
            "correlation_id": correlation_id,
            "cycle": None,
            "stages": stages,
            "calibration": _stage(status="not_evaluated", reason="awaiting_validated_forecast_sample"),
            "performance": _stage(status="not_evaluated", reason="insufficient_validated_sample"),
            "readiness": _stage(status="not_evaluated", reason="insufficient_validated_sample"),
            "reasoning": {
                "forecast": None,
                "proposal": None,
                "managed_signals": [],
                "signal_histories": {},
                "final_actions": {},
                "publications": {},
                "llm_usage": usage,
                "performance": performance.model_dump(mode="json") if performance else None,
                "production_readiness": readiness.model_dump(mode="json") if readiness else None,
                "runtime": runtime,
                "health": {**ai_health, "guardrails": guardrail_health},
            },
            "health": {
                "quant": quant_health,
                "ai": ai_health,
                "guardrails": guardrail_health,
                "feature_flags": flags.snapshot(),
            },
        }
        duration_ms = (perf_counter() - started) * 1000
        logger.info(
            "dashboard_api.empty",
            extra={
                "path": request.url.path,
                "instrument": symbol,
                "status_code": 200,
                "duration_ms": duration_ms,
                "data_status": "pending",
                "correlation_id": correlation_id,
            },
        )
        return response

    quant = await request.app.state.quant_forecast_repository.result_for_state(state.state_id)
    forecast = await request.app.state.ai_reasoning_repository.forecast_for_state(state.state_id)
    proposal = await request.app.state.ai_reasoning_repository.proposal_for_state(state.state_id)
    action = await request.app.state.final_decision_repository.action_for_state(state.state_id)
    active_signals = await request.app.state.ai_reasoning_repository.active_signals(symbol)
    signal = next(
        (item for item in active_signals if action is not None and item.signal_id == action.managed_signal_id),
        active_signals[0] if active_signals else None,
    )
    publication = (
        await request.app.state.final_decision_repository.publication_for_signal(action.managed_signal_id)
        if action is not None
        else None
    )
    outcome = (
        await request.app.state.final_decision_repository.outcome_for_signal(action.managed_signal_id)
        if action is not None
        else None
    )
    signal_history = (
        await request.app.state.ai_reasoning_repository.signal_history(signal.signal_id)
        if signal is not None
        else None
    )

    state_status = "degraded" if state.status.value == "degraded" else "available"
    quant_status, quant_reason = (
        _record_status(quant, available_reason="same_cycle_quant_forecast_persisted")
        if quant is not None
        else ("pending", "quant_forecast_not_yet_persisted_for_cycle")
    )
    forecast_status, forecast_reason = (
        _record_status(forecast, available_reason="same_cycle_ai_reasoning_persisted")
        if forecast is not None
        else ("pending", "ai_reasoning_not_yet_persisted_for_cycle")
    )
    proposal_reason = (
        "same_cycle_ai_proposal_persisted"
        if proposal is not None
        else "ai_signal_proposals_disabled"
        if not proposals_enabled
        else "ai_reasoning_produced_no_proposal"
    )
    action_reason = (
        "same_cycle_guardrail_result_persisted"
        if action is not None
        else "awaiting_ai_proposal"
        if proposal is None
        else "guardrail_result_not_yet_persisted"
    )
    publication_reason = (
        "analytical_signal_persisted"
        if publication is not None
        else "ai_signal_publication_disabled"
        if not publication_enabled
        else "final_action_not_publication_eligible"
    )
    monitoring_reason = (
        "managed_signal_active"
        if signal is not None
        else "ai_signal_monitoring_disabled"
        if not monitoring_enabled
        else "no_managed_signal_for_latest_cycle"
    )

    stages = {
        "market_state": _stage(
            status=state_status,
            reason="synchronized_m1_m5_m15_state_persisted",
            data=state,
            record_id=state.state_id,
            timestamp=state.market_data_boundary,
        ),
        "engine_outputs": _stage(
            status=state_status,
            reason="complete_structured_evidence_preserved",
            data=[item.model_dump(mode="json") for item in state.evidence],
            record_id=state.state_id,
            timestamp=state.market_data_boundary,
        ),
        "quant_forecast": _stage(
            status=quant_status,
            reason=quant_reason,
            data=quant,
            record_id=getattr(quant, "result_id", None),
            timestamp=getattr(quant, "generated_at", None),
            retryable=quant_status == "failed",
        ),
        "ai_reasoning": _stage(
            status=forecast_status,
            reason=forecast_reason,
            data=forecast,
            record_id=getattr(forecast, "forecast_id", None),
            timestamp=getattr(forecast, "generated_at", None),
            error_code=getattr(forecast, "failure_state", None),
            retryable=forecast_status == "failed",
        ),
        "ai_proposal": _stage(
            status="available" if proposal is not None else "not_available",
            reason=proposal_reason,
            data=proposal,
            record_id=getattr(proposal, "proposal_id", None),
            timestamp=getattr(proposal, "created_at", None),
        ),
        "guardrails": _stage(
            status="available" if action is not None else "not_evaluated",
            reason=action_reason,
            data=list(getattr(action, "gate_evaluations", ()) or ()),
            record_id=getattr(action, "final_action_id", None),
            timestamp=getattr(action, "created_at", None),
        ),
        "final_action": _stage(
            status="available" if action is not None else "not_available",
            reason=action_reason,
            data=action,
            record_id=getattr(action, "final_action_id", None),
            timestamp=getattr(action, "created_at", None),
        ),
        "publication": _stage(
            status="available" if publication is not None else "not_available",
            reason=publication_reason,
            data=publication,
            record_id=getattr(publication, "publication_id", None),
            timestamp=getattr(publication, "published_at", None),
        ),
        "monitoring": _stage(
            status="available" if signal is not None else "not_available",
            reason=monitoring_reason,
            data=signal,
            record_id=getattr(signal, "signal_id", None),
            timestamp=getattr(signal, "updated_at", None),
        ),
        "outcome": _stage(
            status="available" if outcome is not None else "not_evaluated",
            reason="signal_outcome_persisted" if outcome is not None else "evaluation_horizon_not_complete",
            data=outcome,
            record_id=getattr(outcome, "outcome_id", None),
            timestamp=getattr(outcome, "evaluated_at", None),
        ),
    }
    substantive = [stages[name]["status"] for name in ("market_state", "quant_forecast", "ai_reasoning")]
    overall_status = (
        "failed"
        if "failed" in substantive
        else "partial"
        if any(value in {"pending", "not_available", "not_evaluated"} for value in substantive)
        else "complete"
    )
    response = {
        "status": overall_status,
        "instrument": symbol,
        "generated_at": now,
        "correlation_id": correlation_id,
        "cycle": {
            "event_id": str(state.cycle_id),
            "market_state_id": str(state.state_id),
            "analysis_timestamp": state.market_data_boundary,
            "knowledge_cutoff": state.knowledge_cutoff,
            "freshness": "fresh" if (now - state.market_data_boundary).total_seconds() <= 1200 else "stale",
        },
        "stages": stages,
        "calibration": _stage(
            status="available" if calibration is not None else "not_evaluated",
            reason="calibration_report_persisted" if calibration is not None else "insufficient_validated_sample",
            data=calibration,
            record_id=getattr(calibration, "report_id", None),
            timestamp=getattr(calibration, "generated_at", None),
        ),
        "performance": _stage(
            status="available" if performance is not None else "not_evaluated",
            reason="performance_report_persisted" if performance is not None else "insufficient_validated_sample",
            data=performance,
            record_id=getattr(performance, "report_id", None),
            timestamp=getattr(performance, "generated_at", None),
        ),
        "readiness": _stage(
            status="available" if readiness is not None else "not_evaluated",
            reason="readiness_report_persisted" if readiness is not None else "insufficient_validated_sample",
            data=readiness,
            record_id=getattr(readiness, "report_id", None),
            timestamp=getattr(readiness, "generated_at", None),
        ),
        "reasoning": {
            "forecast": forecast.model_dump(mode="json") if forecast else None,
            "proposal": proposal.model_dump(mode="json") if proposal else None,
            "managed_signals": [signal.model_dump(mode="json")] if signal else [],
            "signal_histories": (
                {
                    str(signal.signal_id): {
                        key: [item.model_dump(mode="json") for item in values]
                        for key, values in signal_history.items()
                    }
                }
                if signal is not None and signal_history is not None
                else {}
            ),
            "final_actions": (
                {str(signal.signal_id): [action.model_dump(mode="json")]}
                if signal is not None and action is not None
                else {}
            ),
            "publications": (
                {
                    str(signal.signal_id): (
                        publication.model_dump(mode="json") if publication is not None else None
                    )
                }
                if signal is not None
                else {}
            ),
            "llm_usage": usage,
            "performance": performance.model_dump(mode="json") if performance else None,
            "production_readiness": readiness.model_dump(mode="json") if readiness else None,
            "runtime": runtime,
            "health": {**ai_health, "guardrails": guardrail_health},
        },
        "health": {
            "quant": quant_health,
            "ai": ai_health,
            "guardrails": guardrail_health,
            "feature_flags": flags.snapshot(),
        },
    }
    duration_ms = (perf_counter() - started) * 1000
    logger.info(
        "dashboard_api.response",
        extra={
            "path": request.url.path,
            "instrument": symbol,
            "status_code": 200,
            "duration_ms": duration_ms,
            "record_id": str(state.state_id),
            "cycle_id": str(state.cycle_id),
            "data_status": overall_status,
            "correlation_id": correlation_id,
        },
    )
    return response
