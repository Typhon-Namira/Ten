"""Authoritative, same-cycle dashboard read model.

This endpoint never runs analytics. It only joins persisted records through the latest
``UnifiedMarketState`` boundary so the frontend cannot accidentally combine unrelated cycles.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from sqlalchemy import text

from backend.app.api.dashboard_status import (
    StageResult,
    derive_ai_proposal_stage,
    derive_ai_reasoning_stage,
    derive_final_action_stage,
    derive_guardrails_stage,
    derive_monitoring_stage,
    derive_outcome_stage,
    derive_publication_stage,
)
from backend.app.core.feature_flags import FeatureFlag
from backend.app.engines.market_data_engine.models import canonical_symbol

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])
system_status_router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
logger = logging.getLogger(__name__)

_PIPELINE_STAGES = (
    ("market_data", "Market Data"),
    ("smc", "Smart Money Concepts"),
    ("liquidity", "Liquidity"),
    ("volume_profile", "Volume Profile"),
    ("institutional_flow", "Institutional Flow"),
    ("market_regime", "Market Regime"),
    ("economic_calendar", "Economic Calendar"),
    ("unified_market_state", "Unified Market State"),
    ("quant_forecast", "Quant Forecast"),
    ("ai_reasoning", "AI Reasoning"),
    ("proposal", "Proposal"),
    ("guardrails", "Guardrails"),
    ("final_decision", "Final Decision"),
)
_VALID_STAGE_STATUSES = {
    "healthy", "running", "degraded", "failed", "disabled", "blocked", "stale", "no_data"
}


def _system_stage(
    stage_id: str,
    label: str,
    status: str,
    reason: str,
    *,
    timestamp: datetime | None = None,
    record_id: object | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert status in _VALID_STAGE_STATUSES
    return {
        "id": stage_id,
        "label": label,
        "status": status,
        "reason": reason,
        "timestamp": timestamp,
        "record_id": str(record_id) if record_id is not None else None,
        "details": details or {},
    }


def _stage_fingerprint(stage: dict[str, Any]) -> str:
    fingerprint_payload = {
        key: stage[key]
        for key in ("id", "status", "reason", "record_id", "details")
    }
    return hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


async def _storage_diagnostics(request: Request) -> dict[str, Any]:
    integration = getattr(request.app.state, "integration_service", None)
    exhausted_until = getattr(integration, "storage_exhausted_until", None)
    circuit_open = bool(exhausted_until and exhausted_until > datetime.now(UTC))
    factory = getattr(request.app.state, "database_session_factory", None)
    if factory is None:
        return {
            "status": "failed" if circuit_open else "disabled",
            "reason": "storage_exhausted" if circuit_open else "persistent_database_not_configured",
            "database_bytes": None,
            "growth_bytes_per_hour": None,
            "projected_gb_per_day": None,
            "largest_relations": [],
            "retention": {"status": "disabled", "policies": []},
            "circuit_retry_at": exhausted_until,
        }
    async with factory() as session:
        database_bytes = int(
            await session.scalar(text("SELECT pg_database_size(current_database())")) or 0
        )
        rows = (
            await session.execute(
                text(
                    """
                    SELECT relname,
                           pg_total_relation_size(relid) AS total_bytes,
                           pg_relation_size(relid) AS table_bytes,
                           pg_indexes_size(relid) AS index_bytes,
                           n_live_tup, n_dead_tup
                    FROM pg_stat_user_tables
                    ORDER BY pg_total_relation_size(relid) DESC
                    LIMIT 12
                    """
                )
            )
        ).mappings().all()
        try:
            policies = (
                await session.execute(
                    text(
                        """
                        SELECT relation_name, retention_days, cleanup_batch_size, protected
                        FROM storage_retention_policies
                        ORDER BY relation_name
                        """
                    )
                )
            ).mappings().all()
        except Exception:
            await session.rollback()
            policies = []
    measured_at = datetime.now(UTC)
    previous = getattr(request.app.state, "dashboard_storage_sample", None)
    growth_bytes_per_hour: int | None = None
    if previous is not None:
        previous_at, previous_bytes = previous
        elapsed = (measured_at - previous_at).total_seconds()
        if elapsed > 0:
            growth_bytes_per_hour = round(
                (database_bytes - int(previous_bytes)) * 3600 / elapsed
            )
    request.app.state.dashboard_storage_sample = (measured_at, database_bytes)
    return {
        "status": "failed" if circuit_open else "healthy",
        "reason": "storage_exhausted" if circuit_open else "database_size_measured",
        "database_bytes": database_bytes,
        "growth_bytes_per_hour": growth_bytes_per_hour,
        "projected_gb_per_day": (
            round(growth_bytes_per_hour * 24 / 1024**3, 3)
            if growth_bytes_per_hour is not None else None
        ),
        "largest_relations": [dict(item) for item in rows],
        "retention": {
            "status": "healthy" if policies else "no_data",
            "policies": [dict(item) for item in policies],
        },
        "circuit_retry_at": exhausted_until,
    }


async def _persist_stage_projection(
    request: Request,
    instrument: str,
    stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cache current status and append history only when its fingerprint changes."""

    factory = getattr(request.app.state, "database_session_factory", None)
    if factory is None:
        return []
    async with factory() as session:
        try:
            for stage in stages:
                fingerprint = _stage_fingerprint(stage)
                params = {
                    "instrument": instrument,
                    "stage": stage["id"],
                    "status": stage["status"],
                    "reason": stage["reason"],
                    "fingerprint": fingerprint,
                    "record_id": stage["record_id"],
                    "observed_at": stage["timestamp"] or datetime.now(UTC),
                    "details": json.dumps(stage["details"], default=str),
                    "updated_at": datetime.now(UTC),
                }
                await session.execute(
                    text(
                        """
                        INSERT INTO pipeline_stage_history
                            (instrument, stage, status, reason, fingerprint, record_id,
                             observed_at, details)
                        VALUES
                            (:instrument, :stage, :status, :reason, :fingerprint, :record_id,
                             :observed_at, CAST(:details AS jsonb))
                        ON CONFLICT (instrument, stage, fingerprint) DO NOTHING
                        """
                    ),
                    params,
                )
                await session.execute(
                    text(
                        """
                        INSERT INTO pipeline_stage_current
                            (instrument, stage, status, reason, fingerprint, record_id,
                             observed_at, updated_at, details)
                        VALUES
                            (:instrument, :stage, :status, :reason, :fingerprint, :record_id,
                             :observed_at, :updated_at, CAST(:details AS jsonb))
                        ON CONFLICT (instrument, stage) DO UPDATE SET
                            status = EXCLUDED.status,
                            reason = EXCLUDED.reason,
                            fingerprint = EXCLUDED.fingerprint,
                            record_id = EXCLUDED.record_id,
                            observed_at = EXCLUDED.observed_at,
                            updated_at = EXCLUDED.updated_at,
                            details = EXCLUDED.details
                        WHERE pipeline_stage_current.fingerprint <> EXCLUDED.fingerprint
                        """
                    ),
                    params,
                )
            await session.commit()
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT stage, status, reason, observed_at AS timestamp
                        FROM pipeline_stage_history
                        WHERE instrument = :instrument
                          AND status IN ('failed','degraded','blocked','stale')
                        ORDER BY observed_at DESC
                        LIMIT 50
                        """
                    ),
                    {"instrument": instrument},
                )
            ).mappings().all()
            return [dict(item) for item in rows]
        except Exception as exc:
            await session.rollback()
            # Rolling deploys may briefly serve with the previous schema. Status remains
            # available; the projection cache begins as soon as the migration completes.
            logger.warning(
                "dashboard_stage_projection.failed",
                extra={"exception_class": type(exc).__name__},
            )
            return []


@system_status_router.get("/system-status")
async def dashboard_system_status(request: Request, instrument: str = "XAUUSD") -> dict[str, Any]:
    """One backend-authoritative read model for pipeline, storage and failures."""

    now = datetime.now(UTC)
    symbol = canonical_symbol(instrument)
    flags = request.app.state.engine_registry.context.feature_flags
    state = await request.app.state.unified_market_state_repository.latest_state(symbol)
    stages: dict[str, dict[str, Any]] = {}
    evidence_by_engine = {
        item.source_engine: item for item in (state.evidence if state is not None else ())
    }
    market_timestamp = getattr(state, "market_data_boundary", None)
    stale = bool(market_timestamp and (now - market_timestamp).total_seconds() > 1200)
    stages["market_data"] = _system_stage(
        "market_data",
        "Market Data",
        "no_data" if state is None else "stale" if stale else "healthy",
        "awaiting_first_synchronized_candle" if state is None else "market_boundary_stale" if stale else "closed_candles_available",
        timestamp=market_timestamp,
    )
    for stage_id, label in _PIPELINE_STAGES[1:7]:
        evidence = evidence_by_engine.get(stage_id)
        availability = getattr(getattr(evidence, "availability", None), "value", None)
        status = {
            "available": "healthy",
            "degraded": "degraded",
            "stale": "stale",
            "unavailable": "no_data",
        }.get(str(availability), "no_data")
        reason_codes = tuple(getattr(evidence, "reason_codes", ()) or ())
        stages[stage_id] = _system_stage(
            stage_id,
            label,
            status,
            reason_codes[0] if reason_codes else f"{stage_id}_{availability or 'not_persisted'}",
            timestamp=getattr(evidence, "available_at", None),
            record_id=getattr(evidence, "evidence_id", None),
        )
    stages["unified_market_state"] = _system_stage(
        "unified_market_state",
        "Unified Market State",
        "no_data" if state is None else "degraded" if state.status.value == "degraded" else "healthy",
        "awaiting_synchronized_m1_m5_m15_state" if state is None else "point_in_time_state_persisted",
        timestamp=market_timestamp,
        record_id=getattr(state, "state_id", None),
        details={"evidence_completeness": getattr(state, "evidence_completeness", None)},
    )
    quant = (
        await request.app.state.quant_forecast_repository.result_for_state(state.state_id)
        if state is not None else None
    )
    forecast = (
        await request.app.state.ai_reasoning_repository.forecast_for_state(state.state_id)
        if state is not None else None
    )
    proposal = (
        await request.app.state.ai_reasoning_repository.proposal_for_state(state.state_id)
        if state is not None else None
    )
    action = (
        await request.app.state.final_decision_repository.action_for_state(state.state_id)
        if state is not None else None
    )
    stages["quant_forecast"] = _system_stage(
        "quant_forecast", "Quant Forecast",
        "blocked" if state is None else "running" if quant is None else "failed" if str(getattr(quant, "status", "")).endswith("failed") else "healthy",
        "awaiting_unified_market_state" if state is None else "forecast_in_progress" if quant is None else "quant_forecast_persisted",
        timestamp=getattr(quant, "generated_at", None), record_id=getattr(quant, "result_id", None),
    )
    reasoning_enabled = flags.is_enabled(FeatureFlag.AI_CENTRIC_SHADOW_MODE)
    stages["ai_reasoning"] = _system_stage(
        "ai_reasoning", "AI Reasoning",
        "disabled" if not reasoning_enabled else "blocked" if quant is None else "running" if forecast is None else "failed" if getattr(forecast, "failure_state", None) else "degraded" if getattr(forecast, "validation_degraded", False) else "healthy",
        "ai_centric_shadow_mode_disabled" if not reasoning_enabled else "awaiting_quant_forecast" if quant is None else "reasoning_in_progress" if forecast is None else getattr(forecast, "failure_state", None) or "ai_reasoning_persisted",
        timestamp=getattr(forecast, "generated_at", None), record_id=getattr(forecast, "forecast_id", None),
    )
    proposals_enabled = flags.is_enabled(FeatureFlag.AI_SIGNAL_PROPOSALS)
    stages["proposal"] = _system_stage(
        "proposal", "Proposal",
        "disabled" if not proposals_enabled else "blocked" if forecast is None else "running" if proposal is None else "healthy",
        "ai_signal_proposals_disabled" if not proposals_enabled else "awaiting_ai_reasoning" if forecast is None else "proposal_in_progress" if proposal is None else "proposal_persisted",
        timestamp=getattr(proposal, "created_at", None), record_id=getattr(proposal, "proposal_id", None),
    )
    stages["guardrails"] = _system_stage(
        "guardrails", "Guardrails",
        "blocked" if proposal is None else "running" if action is None else "healthy",
        "awaiting_proposal" if proposal is None else "guardrails_in_progress" if action is None else "deterministic_guardrails_completed",
        timestamp=getattr(action, "created_at", None), record_id=getattr(action, "final_action_id", None),
    )
    stages["final_decision"] = _system_stage(
        "final_decision", "Final Decision",
        "blocked" if action is None else "healthy",
        "awaiting_guardrails" if action is None else "final_decision_persisted",
        timestamp=getattr(action, "created_at", None), record_id=getattr(action, "final_action_id", None),
    )
    storage = await _storage_diagnostics(request)
    stage_list = [stages[item[0]] for item in _PIPELINE_STAGES]
    persisted_failures = await _persist_stage_projection(request, symbol, stage_list)
    overall = (
        "failed" if storage["status"] == "failed" or any(item["status"] == "failed" for item in stage_list)
        else "degraded" if any(item["status"] in {"degraded", "stale"} for item in stage_list)
        else "running" if any(item["status"] == "running" for item in stage_list)
        else "healthy"
    )
    return {
        "status": overall,
        "instrument": symbol,
        "generated_at": now,
        "cycle_id": str(state.cycle_id) if state is not None else None,
        "stages": stage_list,
        "current_decision": action.model_dump(mode="json") if action is not None else None,
        "storage": storage,
        "failure_history": persisted_failures or [
            {
                "stage": item["id"],
                "status": item["status"],
                "reason": item["reason"],
                "timestamp": item["timestamp"],
            }
            for item in stage_list if item["status"] in {"failed", "degraded", "blocked", "stale"}
        ],
    }


def _stage(
    *,
    status: str,
    reason: str,
    data: Any = None,
    record_id: object | None = None,
    timestamp: datetime | None = None,
    error_code: str | None = None,
    retryable: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "record_id": str(record_id) if record_id is not None else None,
        "timestamp": timestamp,
        "error_code": error_code,
        "retryable": retryable,
        "data": data.model_dump(mode="json") if hasattr(data, "model_dump") else data,
        **(extra or {}),
    }


def _stage_from_result(
    result: StageResult,
    *,
    data: Any = None,
    record_id: object | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    return _stage(
        status=result.status,
        reason=result.reason,
        data=data,
        record_id=record_id,
        timestamp=timestamp,
        error_code=result.error_code,
        retryable=result.retryable,
        extra=result.extra,
    )


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
    ai_request = await request.app.state.ai_reasoning_repository.request_for_state(
        state.state_id
    )
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
    ai_reasoning_result = derive_ai_reasoning_stage(
        forecast=forecast,
        request=ai_request,
        ai_health=ai_health,
        now=now,
        cycle_available_at=state.knowledge_cutoff,
    )
    ai_proposal_result = derive_ai_proposal_stage(
        forecast=forecast,
        proposal=proposal,
        proposals_enabled=proposals_enabled,
    )
    guardrails_result = derive_guardrails_stage(
        forecast=forecast,
        proposal=proposal,
        action=action,
    )
    final_action_result = derive_final_action_stage(
        forecast=forecast,
        proposal=proposal,
        action=action,
    )
    publication_config_source = (
        "environment variable TEN_AI_SIGNAL_PUBLICATION"
        if request.app.state.settings.ai_signal_publication is not None
        else "configs/feature_flags.yaml (ai_signal_publication)"
    )
    publication_result = derive_publication_stage(
        publication=publication,
        publication_enabled=publication_enabled,
        publication_config_source=publication_config_source,
    )
    monitoring_result = derive_monitoring_stage(
        signal=signal,
        final_action_status=final_action_result.status,
        action=action,
        publication=publication,
        publication_enabled=publication_enabled,
        monitoring_enabled=monitoring_enabled,
    )
    outcome_result = derive_outcome_stage(
        outcome=outcome,
        final_action_status=final_action_result.status,
        action=action,
        publication=publication,
        publication_enabled=publication_enabled,
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
        "ai_reasoning": _stage_from_result(
            ai_reasoning_result,
            data=forecast,
            record_id=getattr(forecast, "forecast_id", None),
            timestamp=getattr(forecast, "generated_at", None),
        ),
        "ai_proposal": _stage_from_result(
            ai_proposal_result,
            data=proposal,
            record_id=getattr(proposal, "proposal_id", None),
            timestamp=getattr(proposal, "created_at", None),
        ),
        "guardrails": _stage_from_result(
            guardrails_result,
            data=list(getattr(action, "gate_evaluations", ()) or ()),
            record_id=getattr(action, "final_action_id", None),
            timestamp=getattr(action, "created_at", None),
        ),
        "final_action": _stage_from_result(
            final_action_result,
            data=action,
            record_id=getattr(action, "final_action_id", None),
            timestamp=getattr(action, "created_at", None),
        ),
        "publication": _stage_from_result(
            publication_result,
            data=publication,
            record_id=getattr(publication, "publication_id", None),
            timestamp=getattr(publication, "published_at", None),
        ),
        "monitoring": _stage_from_result(
            monitoring_result,
            data=signal,
            record_id=getattr(signal, "signal_id", None),
            timestamp=getattr(signal, "updated_at", None),
        ),
        "outcome": _stage_from_result(
            outcome_result,
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
        if any(
            value
            in {
                "pending",
                "not_available",
                "not_evaluated",
                "blocked",
                "disabled",
                "running",
                "degraded",
            }
            for value in substantive
        )
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
