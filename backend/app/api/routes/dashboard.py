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
    derive_ai_reasoning_stage,
    derive_final_action_stage,
    derive_guardrails_stage,
    derive_monitoring_stage,
    derive_outcome_stage,
    derive_publication_stage,
)
from backend.app.ai_reasoning.telemetry import (
    current_operational_usage,
    provider_attempts,
    usage_parameter as scoped_usage_parameter,
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
    analysis = (
        await request.app.state.ai_reasoning_repository.analysis_for_state(state.state_id)
        if state is not None else None
    )
    analysis_signal = (
        await request.app.state.ai_reasoning_repository.signal_for_analysis(
            analysis.analysis_id
        )
        if analysis is not None
        else None
    )
    signal_decision = await request.app.state.signal_decision_service.repository.get_latest_decision(
        symbol,
        getattr(state, "trigger_timeframe", None)
        or request.app.state.settings.market_data_timeframes[0],
    )
    if (
        state is not None
        and signal_decision is not None
        and (
            signal_decision.source_lineage is None
            or signal_decision.source_lineage.market_snapshot_id != state.state_id
            or (
                analysis_signal is not None
                and signal_decision.source_lineage.current_ai_signal_id
                != analysis_signal.signal_id
            )
        )
    ):
        signal_decision = None
    stages["quant_forecast"] = _system_stage(
        "quant_forecast", "Quant Forecast",
        "blocked" if state is None else "running" if quant is None else "failed" if str(getattr(quant, "status", "")).endswith("failed") else "healthy",
        "awaiting_unified_market_state" if state is None else "forecast_in_progress" if quant is None else "quant_forecast_persisted",
        timestamp=getattr(quant, "generated_at", None), record_id=getattr(quant, "result_id", None),
    )
    reasoning_enabled = flags.is_enabled(FeatureFlag.AI_CENTRIC_SHADOW_MODE)
    stages["ai_reasoning"] = _system_stage(
        "ai_reasoning", "AI Market Analysis",
        "disabled" if not reasoning_enabled else "blocked" if quant is None else "running" if analysis is None else "failed" if not analysis.validation_passed or analysis_signal is None else "healthy",
        "ai_centric_shadow_mode_disabled" if not reasoning_enabled else "awaiting_quant_forecast" if quant is None else "analysis_in_progress" if analysis is None else "structured_output_invalid" if not analysis.validation_passed else "analysis_signal_persistence_missing" if analysis_signal is None else "analysis_and_signal_persisted",
        timestamp=getattr(analysis, "analysis_timestamp", None), record_id=getattr(analysis, "analysis_id", None),
        details={
            "signal_id": getattr(analysis_signal, "signal_id", None),
            "signal": getattr(getattr(analysis_signal, "signal", None), "value", None),
            "confidence": getattr(analysis_signal, "confidence", None),
            "strength": getattr(getattr(analysis_signal, "strength", None), "value", None),
        },
    )
    stages["proposal"] = _system_stage(
        "proposal", "AI Proposal (retired)",
        "disabled",
        "signal_engine_is_only_decision_authority",
    )
    stages["guardrails"] = _system_stage(
        "guardrails", "Deterministic Risk Rules",
        "blocked" if signal_decision is None else "healthy",
        "awaiting_signal_engine" if signal_decision is None else "deterministic_risk_rules_completed",
        timestamp=getattr(signal_decision, "decided_at", None), record_id=getattr(signal_decision, "decision_id", None),
    )
    stages["final_decision"] = _system_stage(
        "final_decision", "Final Decision",
        "blocked" if signal_decision is None else "healthy",
        "awaiting_signal_engine" if signal_decision is None else "signal_engine_decision_persisted",
        timestamp=getattr(signal_decision, "decided_at", None), record_id=getattr(signal_decision, "decision_id", None),
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
        "current_decision": signal_decision.model_dump(mode="json") if signal_decision is not None else None,
        "current_analysis_signal": (
            analysis_signal.model_dump(mode="json")
            if analysis_signal is not None
            else None
        ),
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
    ai_service = request.app.state.ai_reasoning_service
    historical_policy_usage_rows = tuple(
        item
        for item in usage_rows
        if item.generation_parameters.get("telemetry_policy") == "five_minute_v1"
    )
    policy_usage_rows = current_operational_usage(
        usage_rows,
        deployment_id=ai_service.deployment_id,
        prompt_version=ai_service.config.prompt_version_new_market,
        now=now,
    )
    legacy_usage_rows = tuple(
        item
        for item in usage_rows
        if item not in historical_policy_usage_rows
    )

    def usage_summary(rows: tuple[Any, ...]) -> dict[str, int | None]:
        return {
            "provider_http_calls": sum(item.request_count for item in rows),
            "total_tokens": (
                sum(item.total_tokens or 0 for item in rows)
                if any(item.total_tokens is not None for item in rows)
                else None
            ),
            "successful_requests": sum(item.success for item in rows),
            "failed_requests": sum(not item.success for item in rows),
        }

    def usage_parameter(
        name: str,
        rows: tuple[Any, ...] = policy_usage_rows,
    ) -> int:
        return scoped_usage_parameter(rows, name)

    recent_provider_attempts = sorted(
        provider_attempts(policy_usage_rows),
        key=lambda item: str(item.get("recorded_at") or ""),
        reverse=True,
    )[:20]
    all_provider_attempts = provider_attempts(policy_usage_rows)
    output_token_samples = sorted(
        int(value)
        for attempt in all_provider_attempts
        if isinstance((value := attempt.get("output_tokens")), int)
    )
    completed_analyses = sum(item.success for item in policy_usage_rows)
    policy_tokens = sum(item.total_tokens or 0 for item in policy_usage_rows)
    policy_calls = sum(item.request_count for item in policy_usage_rows)
    truncated_outputs = usage_parameter("truncated_outputs")
    analysis_requests = usage_parameter("analysis_requests")
    usage = {
        "request_count": sum(
            item.request_count for item in policy_usage_rows
        ),
        "provider_http_calls": sum(
            item.request_count for item in policy_usage_rows
        ),
        "groq_calls": usage_parameter("groq_calls"),
        "retries": usage_parameter("retry_attempts"),
        "schema_corrections": usage_parameter("schema_corrections"),
        "initial_analysis_requests": usage_parameter("analysis_requests"),
        "initial_parse_failures": usage_parameter("initial_parse_failures"),
        "initial_schema_validation_failures": usage_parameter(
            "initial_schema_validation_failures"
        ),
        "schema_corrections_succeeded": usage_parameter(
            "schema_corrections_succeeded"
        ),
        "schema_corrections_failed": usage_parameter(
            "schema_corrections_failed"
        ),
        "http_429_responses": usage_parameter("http_429_responses"),
        "provider_http_successes": usage_parameter("provider_http_successes"),
        "schema_valid_analyses": usage_parameter("schema_valid_analyses"),
        "truncated_outputs": truncated_outputs,
        "compact_retries": usage_parameter("compact_retries"),
        "request_policy_failures": usage_parameter("request_policy_failures"),
        "tokens_per_completed_analysis": (
            round(policy_tokens / completed_analyses, 2)
            if completed_analyses
            else None
        ),
        "provider_calls_per_completed_analysis": (
            round(policy_calls / completed_analyses, 2)
            if completed_analyses
            else None
        ),
        "truncation_rate": (
            round(truncated_outputs / analysis_requests, 4)
            if analysis_requests
            else None
        ),
        "average_input_tokens": (
            round(
                sum(
                    int(value)
                    for attempt in all_provider_attempts
                    if isinstance((value := attempt.get("input_tokens")), int)
                )
                / max(
                    1,
                    sum(
                        isinstance(attempt.get("input_tokens"), int)
                        for attempt in all_provider_attempts
                    ),
                ),
                2,
            )
            if any(
                isinstance(attempt.get("input_tokens"), int)
                for attempt in all_provider_attempts
            )
            else None
        ),
        "average_output_tokens": (
            round(sum(output_token_samples) / len(output_token_samples), 2)
            if output_token_samples
            else None
        ),
        "p95_output_tokens": (
            output_token_samples[
                min(
                    len(output_token_samples) - 1,
                    int(len(output_token_samples) * 0.95),
                )
            ]
            if output_token_samples
            else None
        ),
        "completion_rate": (
            round(completed_analyses / analysis_requests, 4)
            if analysis_requests
            else None
        ),
        "recent_provider_attempts": recent_provider_attempts,
        "provider_failures": usage_parameter("provider_failure"),
        "validation_failures": usage_parameter("validation_failure"),
        "total_tokens": (
            sum(item.total_tokens or 0 for item in policy_usage_rows)
            if any(
                item.total_tokens is not None
                for item in policy_usage_rows
            )
            else None
        ),
        "successful_requests": sum(
            item.success for item in policy_usage_rows
        ),
        "failed_requests": sum(
            not item.success for item in policy_usage_rows
        ),
        "legacy_cumulative_daily": usage_summary(legacy_usage_rows),
        "five_minute_policy": usage_summary(policy_usage_rows),
        "telemetry_scope": {
            "deployment_id": ai_service.deployment_id,
            "prompt_version": ai_service.config.prompt_version_new_market,
            "schema_versions": ["compact-1.1", "compact-retry-1.1"],
            "output_profiles": ["compact", "compact_retry"],
            "window": "last_24_hours",
        },
        "historical_total": usage_summary(tuple(usage_rows)),
        "historical_five_minute_policy": usage_summary(
            historical_policy_usage_rows
        ),
    }
    calibration = await request.app.state.quant_forecast_repository.latest_calibration(
        request.app.state.quant_forecast_service.config.model_name
    )
    performance = await request.app.state.final_decision_repository.latest_performance_report()
    readiness = await request.app.state.final_decision_repository.latest_readiness_report()
    quant_health = request.app.state.quant_forecast_service.health()
    ai_health = request.app.state.ai_reasoning_service.health()
    provider_states = ai_health.get("providers")
    if isinstance(provider_states, dict):
        for account_id, provider_state in provider_states.items():
            if not isinstance(account_id, str) or not isinstance(provider_state, dict):
                continue
            provider_state["calls_current_window"] = usage_parameter(
                f"{account_id}_calls"
            )
            provider_state["successful_analyses"] = sum(
                int(item.success)
                for item in policy_usage_rows
                if item.generation_parameters.get("provider") == account_id
            )
            provider_state["provider_failures"] = usage_parameter(
                f"{account_id}_provider_failures"
            )
            provider_state["rate_limit_failures"] = usage_parameter(
                f"{account_id}_rate_limit_failures"
            )
            provider_state["quota_failures"] = usage_parameter(
                f"{account_id}_quota_failures"
            )
            provider_state["analysis_requests"] = usage_parameter(
                f"{account_id}_analysis_requests"
            )
            provider_state["schema_correction_requests"] = usage_parameter(
                f"{account_id}_schema_correction_requests"
            )
            provider_state["http_429_responses"] = usage_parameter(
                f"{account_id}_http_429_responses"
            )
            provider_state["recent_429_count"] = provider_state[
                "http_429_responses"
            ]
            provider_state["historical_429_count"] = usage_parameter(
                f"{account_id}_http_429_responses",
                tuple(usage_rows),
            )
            provider_state["token_usage"] = {
                "input_tokens": usage_parameter(f"{account_id}_input_tokens"),
                "output_tokens": usage_parameter(f"{account_id}_output_tokens"),
                "total_tokens": usage_parameter(f"{account_id}_total_tokens"),
            }
    call_control = ai_health.get("call_control")
    if isinstance(call_control, dict):
        usage.update(
            {
                "eligible_five_minute_cycles": call_control.get(
                    "eligible_five_minute_cycles",
                    0,
                ),
                "analyses_successfully_completed": call_control.get(
                    "analyses_successfully_completed",
                    0,
                ),
                "skipped_before_provider_call": call_control.get(
                    "skipped_before_provider_call",
                    0,
                ),
                "deduplicated_before_provider_call": call_control.get(
                    "deduplicated_before_provider_call",
                    0,
                ),
            }
        )
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
                "analysis": None,
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
    analysis = await request.app.state.ai_reasoning_repository.analysis_for_state(state.state_id)
    analysis_signal = (
        await request.app.state.ai_reasoning_repository.signal_for_analysis(
            analysis.analysis_id
        )
        if analysis is not None
        else None
    )
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
    reasoning_artifact = analysis if analysis is not None else forecast
    ai_reasoning_result = derive_ai_reasoning_stage(
        forecast=reasoning_artifact,
        request=ai_request,
        ai_health=ai_health,
        now=now,
        cycle_available_at=state.knowledge_cutoff,
    )
    ai_proposal_result = StageResult(
        "not_required",
        "signal_engine_is_only_decision_authority",
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
            data=reasoning_artifact,
            record_id=(
                getattr(reasoning_artifact, "analysis_id", None)
                or getattr(reasoning_artifact, "forecast_id", None)
            ),
            timestamp=(
                getattr(reasoning_artifact, "analysis_timestamp", None)
                or getattr(reasoning_artifact, "generated_at", None)
            ),
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
            "analysis": analysis.model_dump(mode="json") if analysis else None,
            "analysis_signal": (
                analysis_signal.model_dump(mode="json")
                if analysis_signal
                else None
            ),
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
