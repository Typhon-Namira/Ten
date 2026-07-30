"""Authoritative, same-cycle dashboard read model.

This endpoint never runs analytics. It only joins persisted records through the latest
``UnifiedMarketState`` boundary so the frontend cannot accidentally combine unrelated cycles.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import json
import logging
from time import perf_counter
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
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
from backend.app.core.security import Role, require_role
from backend.app.engines.market_data_engine.freshness import (
    evaluate_market_data_freshness,
)
from backend.app.engines.market_data_engine.models import Timeframe, canonical_symbol
from backend.app.market_state.models import REQUIRED_TIMEFRAMES

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


async def _decision_for_analysis_signal(
    request: Request,
    signal: Any,
) -> Any | None:
    return await request.app.state.signal_decision_service.repository.find_by_analysis_lineage(
        signal.instrument,
        signal.timeframe,
        signal.snapshot_id,
        signal.analysis_id,
        signal.signal_id,
    )


async def _latest_complete_cycle_lineage(
    request: Request,
    instrument: str,
) -> tuple[Any, Any, Any, Any, Any, str | None] | None:
    """Select the newest cycle whose complete persisted lineage is readable.

    A newer analysis or signal that is still awaiting its deterministic decision must not
    temporarily replace the previous completed cycle. Conversely, HOLD and publication-
    ineligible decisions are complete and are deliberately not filtered out.
    """

    repository = request.app.state.ai_reasoning_repository
    completed: list[tuple[Any, Any, Any, Any, Any]] = []
    offset = 0
    batch_size = 100
    while len(completed) < 2:
        candidates = await repository.list_analysis_signals(
            instrument,
            None,
            None,
            None,
            None,
            None,
            None,
            offset,
            batch_size,
        )
        if not candidates:
            break
        for signal in candidates:
            analysis = await repository.get_analysis(signal.analysis_id)
            if (
                analysis is None
                or getattr(analysis.status, "value", analysis.status) != "available"
                or not analysis.validation_passed
                or analysis.cycle_id != signal.cycle_id
                or analysis.market_snapshot_id != signal.snapshot_id
            ):
                continue
            decision = await _decision_for_analysis_signal(request, signal)
            if decision is None:
                continue
            state = await request.app.state.unified_market_state_repository.get_state(
                signal.snapshot_id
            )
            quant = await request.app.state.quant_forecast_repository.result_for_state(
                signal.snapshot_id
            )
            if (
                state is None
                or quant is None
                or getattr(state, "state_id", None) != signal.snapshot_id
                or getattr(quant, "result_id", None)
                != analysis.quantitative_forecast_id
            ):
                continue
            completed.append((analysis, signal, decision, state, quant))
            if len(completed) == 2:
                break
        if len(candidates) < batch_size:
            break
        offset += len(candidates)
    if not completed:
        return None
    analysis, signal, decision, state, quant = completed[0]
    previous_cycle_id = (
        str(completed[1][1].cycle_id) if len(completed) > 1 else None
    )
    return analysis, signal, decision, state, quant, previous_cycle_id


def _publication_projection(decision: Any | None) -> dict[str, Any]:
    if decision is None:
        return {
            "status": "PENDING",
            "eligible": False,
            "reason": "deterministic_decision_not_yet_persisted",
        }
    blockers = [
        item.model_dump(mode="json")
        for item in tuple(getattr(decision, "blockers", ()) or ())
    ]
    eligible = bool(decision.publication_eligible)
    return {
        "status": "ELIGIBLE" if eligible else "INELIGIBLE",
        "eligible": eligible,
        "reason": (
            "deterministic_decision_publication_eligible"
            if eligible
            else decision.decision_reason
            or (blockers[0].get("message") if blockers else None)
            or "deterministic_publication_rules_not_satisfied"
        ),
        "blockers": blockers,
    }


def _authoritative_signal_projection(
    signal: Any,
    decision: Any,
    *,
    lifecycle_status: str,
    multi_timeframe_signal: Any | None = None,
) -> dict[str, Any]:
    """Preserve analysis while projecting later lifecycle stages independently."""
    serialized: dict[str, Any] = dict(signal.model_dump(mode="json"))
    combined = (
        multi_timeframe_signal.combined_signal
        if multi_timeframe_signal is not None
        else None
    )
    if combined is not None:
        serialized.update(
            {
                "signal": combined.analytical_direction.value,
                "signal_confidence": combined.confidence,
                "confidence": combined.confidence,
                "strength": combined.strength.value,
                "bullish_score": combined.bullish_score,
                "bearish_score": combined.bearish_score,
                "reasoning_summary": combined.directional_thesis,
                "execution_eligibility": combined.execution_eligibility.value,
                "execution_status": combined.execution_status.value,
                "blocking_reasons": list(combined.blocking_reasons),
                "entry": combined.geometry.entry if combined.geometry else None,
                "stop_loss": (
                    combined.geometry.stop_loss if combined.geometry else None
                ),
                "take_profit": (
                    combined.geometry.take_profit if combined.geometry else None
                ),
                "risk_reward_ratio": (
                    combined.geometry.risk_reward_ratio
                    if combined.geometry
                    else None
                ),
                "geometry_basis": (
                    list(combined.geometry.basis_fact_identifiers)
                    if combined.geometry
                    else []
                ),
                "synthesis_id": (
                    str(multi_timeframe_signal.synthesis_id)
                    if multi_timeframe_signal is not None
                    else None
                ),
            }
        )
    serialized.update(
        {
            "analytical_direction": serialized["signal"],
            "final_action": decision.final_action.value,
            "overall_confidence": decision.overall_confidence,
            "lifecycle_status": lifecycle_status,
        }
    )
    return serialized


def _contribution_projection(combined: Any | None) -> dict[str, Any]:
    families: dict[str, list[Any]] = {}
    if combined is not None:
        for item in combined.evidence_breakdown:
            families.setdefault(item.family, []).append(item)

    def project(name: str, aliases: tuple[str, ...]) -> dict[str, Any]:
        values = [
            item
            for family, items in families.items()
            if family in aliases
            for item in items
        ]
        if not values:
            return {
                "family": name,
                "status": "no_qualifying_contribution",
                "normalized_contribution": None,
                "weighted_contribution": None,
                "evidence_count": 0,
            }
        total_weight = sum(item.effective_weight for item in values)
        normalized = (
            sum(item.normalized_score * item.effective_weight for item in values)
            / total_weight
            if total_weight
            else None
        )
        return {
            "family": name,
            "status": "contributed",
            "normalized_contribution": normalized,
            "weighted_contribution": sum(item.weighted_score for item in values),
            "evidence_count": len(values),
        }

    return {
        "trend": project("trend", ("market_structure", "price_action")),
        "institutional": project(
            "institutional",
            ("institutional_flow", "order_blocks", "liquidity"),
        ),
        "volume": project("volume", ("volume", "volume_profile")),
        "evidence": project("evidence", tuple(families)),
    }


def _geometry_projection(
    multi_timeframe_signal: Any | None,
    signal: Any,
    *,
    minimum_risk_reward: float,
) -> dict[str, Any] | None:
    if multi_timeframe_signal is None:
        return None
    combined = multi_timeframe_signal.combined_signal
    if combined.geometry is None:
        return None
    owner = next(
        (
            item.timeframe
            for item in multi_timeframe_signal.timeframe_signals
            if item.geometry == combined.geometry
        ),
        "COMBINED",
    )
    return {
        "owner_timeframe": owner,
        "direction": combined.analytical_direction.value,
        **combined.geometry.model_dump(mode="json"),
        "required_minimum_risk_reward": minimum_risk_reward,
        "validation_status": (
            "VALID"
            if combined.geometry.risk_reward_ratio >= minimum_risk_reward
            else "INVALID"
        ),
        "created_at": multi_timeframe_signal.created_at,
        "expires_at": signal.valid_until,
    }


async def _signal_lifecycle_projection(
    request: Request,
    signal: Any,
    *,
    now: datetime,
) -> dict[str, Any]:
    outcome = await request.app.state.ai_reasoning_repository.analysis_signal_outcome(
        signal.signal_id
    )
    serialized = outcome.model_dump(mode="json") if outcome is not None else None
    status = (
        outcome.status.value
        if outcome is not None
        else getattr(getattr(signal, "lifecycle_status", None), "value", "ACTIVE")
    )
    valid_until = getattr(signal, "valid_until", None)
    if valid_until is None:
        legacy_validity_seconds = {
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
        }.get(str(getattr(signal, "timeframe", "")).upper(), 300)
        valid_until = signal.generated_at + timedelta(
            seconds=legacy_validity_seconds
        )
    if status in {"ACTIVE", "STALE"} and valid_until is not None and now >= valid_until:
        status = "EXPIRED"
    remaining = (
        max(0.0, (valid_until - now).total_seconds())
        if valid_until is not None
        else None
    )
    return {
        "status": status,
        "signal_age_seconds": max(
            0.0,
            (now - signal.generated_at).total_seconds(),
        ),
        "remaining_validity_seconds": remaining,
        "valid_from": getattr(signal, "valid_from", None),
        "valid_until": valid_until,
        "expected_holding_seconds": getattr(
            signal,
            "expected_holding_seconds",
            None,
        ),
        "outcome": serialized,
    }


async def _completed_cycle_projection(
    request: Request,
    instrument: str,
    timeframe: str | None,
) -> dict[str, Any] | None:
    logger.info(
        "dashboard.latest_cycle.selector",
        extra={
            "instrument": instrument,
            "requested_timeframe": timeframe,
            "joins": [
                "ai_market_analyses.analysis_id = ai_analysis_signals.analysis_id",
                "ai_analysis_signals.snapshot_id -> unified_market_states.state_id",
                "ai_market_analyses.quantitative_forecast_id -> quantitative_forecasts.result_id",
                "signal_decisions.source_lineage.current_ai_signal_id = ai_analysis_signals.signal_id",
            ],
            "predicates": [
                "ai_market_analyses.symbol = :instrument",
                "ai_market_analyses.status = 'available'",
                "ai_market_analyses.validation_passed IS TRUE",
                "unified_market_states.state_id = ai_analysis_signals.snapshot_id",
                "quantitative_forecasts.result_id = ai_market_analyses.quantitative_forecast_id",
                "signal_decisions source lineage matches snapshot, analysis, and signal",
                "no timeframe predicate (AI cadence is independent of chart timeframe)",
            ],
            "ordering": [
                "ai_analysis_signals.generated_at DESC",
                "ai_market_analyses.analysis_timestamp DESC",
                "ai_analysis_signals.signal_id DESC",
            ],
        },
    )
    selected = await _latest_complete_cycle_lineage(request, instrument)
    if selected is None:
        return None
    analysis, signal, decision, state, quant, previous_cycle_id = selected
    synthesis_repository = getattr(
        request.app.state, "multi_timeframe_signal_repository", None
    )
    multi_timeframe_signal = (
        await synthesis_repository.for_state(signal.snapshot_id)
        if synthesis_repository is not None
        else None
    )
    publication = _publication_projection(decision)
    generated_signal_count = (
        await request.app.state.ai_reasoning_repository.count_analysis_signals(
            instrument,
            timeframe,
        )
    )
    minimum_sample = int(
        getattr(
            getattr(request.app.state.final_decision_service, "config", None),
            "minimum_readiness_sample_size",
            30,
        )
    )
    now = datetime.now(UTC)
    lifecycle = await _signal_lifecycle_projection(
        request,
        signal,
        now=now,
    )
    authoritative_action = decision.final_action.value
    authoritative_lifecycle_status = (
        "HOLD"
        if authoritative_action == "HOLD"
        else "BLOCKED"
        if not decision.publication_eligible
        else "CURRENT"
        if lifecycle["status"] == "ACTIVE"
        else lifecycle["status"]
    )
    authoritative_lifecycle = {
        **lifecycle,
        "status": authoritative_lifecycle_status,
    }
    outcome_updated_at = (
        lifecycle["outcome"].get("evaluated_at")
        if lifecycle["outcome"] is not None
        else None
    )
    if isinstance(outcome_updated_at, str):
        outcome_updated_at = datetime.fromisoformat(
            outcome_updated_at.replace("Z", "+00:00")
        )
    updated_at = max(
        item
        for item in (
            analysis.created_at,
            signal.generated_at,
            decision.decided_at,
            outcome_updated_at,
        )
        if item is not None
    )
    cycle_version = hashlib.sha256(
        (
            f"{signal.cycle_id}:{analysis.analysis_id}:{signal.signal_id}:"
            f"{decision.decision_id}:{updated_at.isoformat()}"
        ).encode()
    ).hexdigest()
    outcome_count, completed_outcome_count = (
        await request.app.state.ai_reasoning_repository.count_analysis_signal_outcomes(
            instrument
        )
    )
    logger.info(
        "dashboard.latest_cycle.selected",
        extra={
            "requested_instrument": instrument,
            "requested_chart_timeframe": timeframe,
            "cycle_id": str(signal.cycle_id),
            "analysis_id": str(analysis.analysis_id),
            "signal_id": str(signal.signal_id),
            "decision_id": str(decision.decision_id),
            "action": authoritative_action,
            "publication_eligible": publication["eligible"],
            "lifecycle_status": authoritative_lifecycle_status,
            "analysis_timestamp": analysis.analysis_timestamp,
            "selected_at": now,
            "cycle_age_seconds": max(
                0.0,
                (now - analysis.analysis_timestamp).total_seconds(),
            ),
            "replaced_previous_cycle": previous_cycle_id is not None,
            "previous_cycle_id": previous_cycle_id,
            "unified_market_state_id": str(signal.snapshot_id),
            "quantitative_forecast_id": str(analysis.quantitative_forecast_id),
            "ai_market_analysis_id": str(analysis.analysis_id),
            "ai_analysis_signal_id": str(signal.signal_id),
            "signal_decision_id": str(decision.decision_id),
        },
    )
    market_time = (
        state.market_data_boundary if state is not None else analysis.knowledge_cutoff
    )
    stage_statuses = {
        "unified_market_state": {
            "status": "healthy",
            "reason": "same_cycle_market_state_persisted",
            "record_id": str(signal.snapshot_id),
        },
        "quant_forecast": {
            "status": "healthy",
            "reason": "same_cycle_quant_forecast_persisted",
            "record_id": str(analysis.quantitative_forecast_id),
        },
        "ai_reasoning": {
            "status": "healthy",
            "reason": "validated_analysis_persisted",
            "record_id": str(analysis.analysis_id),
        },
        "analytical_signal": {
            "status": "healthy" if multi_timeframe_signal is not None else "degraded",
            "reason": (
                "multi_timeframe_analysis_signal_persisted"
                if multi_timeframe_signal is not None
                else "legacy_single_timeframe_signal_only"
            ),
            "record_id": str(
                multi_timeframe_signal.synthesis_id
                if multi_timeframe_signal is not None
                else signal.signal_id
            ),
        },
        "guardrails": {
            "status": "healthy",
            "reason": "deterministic_guardrails_completed",
            "record_id": str(decision.decision_id),
        },
        "final_decision": {
            "status": "healthy",
            "reason": "deterministic_final_decision_persisted",
            "record_id": str(decision.decision_id),
        },
        "publication": {
            "status": "healthy" if publication["eligible"] else "blocked",
            "reason": publication["reason"],
            "record_id": str(decision.decision_id),
        },
    }
    serialized_state = state.model_dump(mode="json") if state is not None else None
    serialized_quant = quant.model_dump(mode="json") if quant is not None else None
    serialized_analysis = analysis.model_dump(mode="json")
    serialized_signal = _authoritative_signal_projection(
        signal,
        decision,
        lifecycle_status=authoritative_lifecycle_status,
        multi_timeframe_signal=multi_timeframe_signal,
    )
    if signal.schema_version == "1.0":
        serialized_signal["analysis_confidence"] = (
            analysis.output.analysis_confidence * 100
            if analysis.output is not None
            else 0
        )
    combined_signal = (
        multi_timeframe_signal.combined_signal
        if multi_timeframe_signal is not None
        else None
    )
    minimum_rr = float(
        getattr(
            getattr(
                request.app.state.multi_timeframe_signal_synthesizer,
                "config",
                None,
            ),
            "minimum_risk_reward",
            2.0,
        )
    )
    quant_predictions = tuple(
        item
        for item in getattr(quant, "predictions", ())
        if item.horizon.timeframe in {"M5", "M15"}
    )
    quant_buy = (
        sum(item.buy_probability for item in quant_predictions)
        / len(quant_predictions)
        if quant_predictions
        else None
    )
    quant_sell = (
        sum(item.sell_probability for item in quant_predictions)
        / len(quant_predictions)
        if quant_predictions
        else None
    )
    quant_direction = (
        "BUY"
        if quant_buy is not None and quant_sell is not None and quant_buy >= quant_sell
        else "SELL"
        if quant_sell is not None
        else None
    )
    analytical_direction = (
        combined_signal.analytical_direction.value
        if combined_signal is not None
        else signal.signal.value
    )
    geometry = _geometry_projection(
        multi_timeframe_signal,
        signal,
        minimum_risk_reward=minimum_rr,
    )
    guardrail_blockers = [
        item.model_dump(mode="json") for item in decision.blockers
    ]
    result = {
        "status": "completed",
        "symbol": signal.instrument,
        "instrument": signal.instrument,
        "timeframe": signal.timeframe,
        "cycle_id": str(signal.cycle_id),
        "snapshot_id": str(signal.snapshot_id),
        "analysis_id": str(analysis.analysis_id),
        "signal_id": str(signal.signal_id),
        "decision_id": str(decision.decision_id) if decision is not None else None,
        "analysis_timestamp": analysis.analysis_timestamp,
        "signal_generated_at": signal.generated_at,
        "decision_timestamp": decision.decided_at,
        "action": authoritative_action,
        "publication_eligible": publication["eligible"],
        "lifecycle_status": authoritative_lifecycle_status,
        "cycle_version": cycle_version,
        "market_time": market_time,
        "completed_at": decision.decided_at,
        "dashboard_refreshed_at": now,
        "updated_at": updated_at,
        "state": serialized_state,
        "market_state": serialized_state,
        "quant_forecast": serialized_quant,
        "analysis": serialized_analysis,
        "ai_analysis": serialized_analysis,
        "analytical_signal": serialized_signal,
        "multi_timeframe_signal": (
            multi_timeframe_signal.model_dump(mode="json")
            if multi_timeframe_signal is not None
            else None
        ),
        "timeframe_matrix": (
            [
                item.model_dump(mode="json")
                for item in (
                    *multi_timeframe_signal.timeframe_signals,
                    multi_timeframe_signal.combined_signal,
                )
            ]
            if multi_timeframe_signal is not None
            else []
        ),
        "signal_lifecycle": authoritative_lifecycle,
        "final_decision": (
            decision.model_dump(mode="json") if decision is not None else None
        ),
        "publication": publication,
        "analytical_direction": {
            "direction": analytical_direction,
            "confidence": (
                combined_signal.confidence
                if combined_signal is not None
                else signal.signal_confidence
            ),
            "strength": (
                combined_signal.strength.value
                if combined_signal is not None
                else signal.strength.value
            ),
            "bullish_score": (
                combined_signal.bullish_score
                if combined_signal is not None
                else signal.scoring_components.get("bullish_score")
            ),
            "bearish_score": (
                combined_signal.bearish_score
                if combined_signal is not None
                else signal.scoring_components.get("bearish_score")
            ),
        },
        "structural_trade_setup": geometry,
        "execution_eligibility": {
            "status": (
                combined_signal.execution_status.value
                if combined_signal is not None
                else signal.execution_status.value
            ),
            "blockers": (
                list(combined_signal.blocking_reasons)
                if combined_signal is not None
                else list(signal.blocking_reasons)
            ),
        },
        "confidence_semantics": {
            "analytical_confidence": (
                combined_signal.confidence if combined_signal is not None else None
            ),
            "ai_interpretation_confidence": (
                analysis.output.analysis_confidence * 100
                if analysis.output is not None
                else None
            ),
            "quant_direction": quant_direction,
            "quant_directional_probability": (
                max(quant_buy, quant_sell) * 100
                if quant_buy is not None and quant_sell is not None
                else None
            ),
            "quant_calibration_status": quant.calibration_status.value,
            "quant_ai_alignment": (
                "AGREEMENT" if quant_direction == analytical_direction else "DISAGREEMENT"
                if quant_direction is not None else "UNAVAILABLE"
            ),
            "evidence_completeness": (
                combined_signal.confidence_decomposition.evidence_completeness
                if combined_signal is not None
                else None
            ),
            "guardrail_confidence": decision.guardrail_confidence,
            "final_overall_confidence": decision.overall_confidence,
        },
        "evidence_contributions": _contribution_projection(combined_signal),
        "guardrail_decision": {
            "status": "APPROVED" if decision.publication_eligible else "REJECTED",
            "state": decision.state.value,
            "readiness": decision.readiness.value,
            "blockers": guardrail_blockers,
            "warnings": [
                item.model_dump(mode="json") for item in decision.warnings
            ],
        },
        "stages": stage_statuses,
        "lineage": {
            "cycle_id": str(signal.cycle_id),
            "market_snapshot_id": str(signal.snapshot_id),
            "quantitative_forecast_id": str(analysis.quantitative_forecast_id),
            "analysis_id": str(analysis.analysis_id),
            "signal_id": str(signal.signal_id),
            "multi_timeframe_synthesis_id": (
                str(multi_timeframe_signal.synthesis_id)
                if multi_timeframe_signal is not None
                else None
            ),
            "decision_id": (
                str(decision.decision_id) if decision is not None else None
            ),
        },
        "cycle": {
            "eligible_cycle_id": str(signal.cycle_id),
            "snapshot_id": str(signal.snapshot_id),
            "status": "COMPLETED",
        },
        "performance": {
            "signals_generated": generated_signal_count,
            "signals_awaiting_outcome": max(
                0,
                generated_signal_count - completed_outcome_count,
            ),
            "signals_evaluated": completed_outcome_count,
            "minimum_required_sample": minimum_sample,
            "calibration_sample_size": completed_outcome_count,
            "state": (
                "available"
                if completed_outcome_count >= minimum_sample
                else "insufficient_sample"
                if completed_outcome_count
                else "signals_exist_outcomes_pending"
                if outcome_count
                else "no_signals"
            ),
        },
    }
    return result


async def _latest_cycle_selection_diagnostics(
    request: Request,
    instrument: str,
    requested_timeframe: str | None,
) -> dict[str, Any]:
    repository = request.app.state.ai_reasoning_repository
    state = await request.app.state.unified_market_state_repository.latest_state(
        instrument
    )
    quant = await request.app.state.quant_forecast_repository.latest_result(instrument)
    analysis = await repository.latest_analysis(instrument)
    signal = await repository.latest_analysis_signal(instrument)
    decision = (
        await _decision_for_analysis_signal(request, signal)
        if signal is not None
        else None
    )
    conditions = {
        "analysis_exists_for_symbol": analysis is not None,
        "signal_exists_for_symbol": signal is not None,
        "analysis_signal_join_matches": bool(
            analysis is not None
            and signal is not None
            and signal.analysis_id == analysis.analysis_id
        ),
        "analysis_status_is_available": bool(
            analysis is not None and analysis.status.value == "available"
        ),
        "analysis_validation_passed": bool(
            analysis is not None and analysis.validation_passed
        ),
    }
    predicate_names = {
        "analysis_exists_for_symbol": "ai_market_analyses.symbol = :instrument",
        "signal_exists_for_symbol": "ai_analysis_signals.instrument = :instrument",
        "analysis_signal_join_matches": (
            "ai_market_analyses.analysis_id = ai_analysis_signals.analysis_id"
        ),
        "analysis_status_is_available": (
            "ai_market_analyses.status = 'available'"
        ),
        "analysis_validation_passed": (
            "ai_market_analyses.validation_passed IS TRUE"
        ),
    }
    eliminated_by = next(
        (
            predicate_names[name]
            for name, passed in conditions.items()
            if not passed
        ),
        None,
    )
    diagnostics = {
        "latest_unified_market_state_id": (
            str(state.state_id) if state is not None else None
        ),
        "latest_quantitative_forecast_id": (
            str(quant.result_id) if quant is not None else None
        ),
        "latest_ai_market_analysis_id": (
            str(analysis.analysis_id) if analysis is not None else None
        ),
        "latest_ai_analysis_signal_id": (
            str(signal.signal_id) if signal is not None else None
        ),
        "latest_signal_decision_id": (
            str(decision.decision_id) if decision is not None else None
        ),
        "latest_analysis_timeframe": (
            analysis.timeframe if analysis is not None else None
        ),
        "latest_signal_timeframe": (
            signal.timeframe if signal is not None else None
        ),
        "requested_timeframe": requested_timeframe,
        "requested_chart_timeframe_matches_signal": bool(
            requested_timeframe is None
            or (
                signal is not None
                and signal.timeframe.upper() == requested_timeframe.upper()
            )
        ),
        "requested_chart_timeframe_is_filter": False,
        "conditions": conditions,
        "eliminated_by": eliminated_by,
    }
    logger.warning(
        "dashboard.latest_cycle.no_match",
        extra=diagnostics,
    )
    return diagnostics


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


@system_status_router.get("/latest-cycle")
async def dashboard_latest_cycle(
    request: Request,
    response: Response,
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Return the latest fully persisted analytical cycle through one lineage."""

    response.headers["Cache-Control"] = (
        "private, no-store, no-cache, max-age=0, must-revalidate"
    )
    response.headers["CDN-Cache-Control"] = "no-store"
    response.headers["Surrogate-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    instrument = canonical_symbol(symbol)
    cycle = await _completed_cycle_projection(request, instrument, timeframe)
    if cycle is not None:
        return cycle
    diagnostics = await _latest_cycle_selection_diagnostics(
        request,
        instrument,
        timeframe,
    )
    return {
        "status": "no_data",
        "symbol": instrument,
        "instrument": instrument,
        "timeframe": timeframe,
        "cycle_id": None,
        "analysis_id": None,
        "signal_id": None,
        "decision_id": None,
        "analysis_timestamp": None,
        "signal_generated_at": None,
        "decision_timestamp": None,
        "action": None,
        "publication_eligible": False,
        "lifecycle_status": None,
        "cycle_version": None,
        "market_time": None,
        "completed_at": None,
        "dashboard_refreshed_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "cycle": None,
        "market_state": None,
        "quant_forecast": None,
        "ai_analysis": None,
        "analysis": None,
        "analytical_signal": None,
        "multi_timeframe_signal": None,
        "timeframe_matrix": [],
        "guardrail_decision": None,
        "final_decision": None,
        "publication": {
            "status": "PENDING",
            "eligible": False,
            "reason": "no_completed_analytical_cycle",
        },
        "stages": {},
        "lineage": {},
        "selection_diagnostics": diagnostics,
        "performance": {
            "signals_generated": 0,
            "signals_awaiting_outcome": 0,
            "signals_evaluated": 0,
            "minimum_required_sample": int(
                getattr(
                    getattr(
                        request.app.state.final_decision_service,
                        "config",
                        None,
                    ),
                    "minimum_readiness_sample_size",
                    30,
                )
            ),
            "calibration_sample_size": 0,
            "state": "no_signals",
        },
    }


@system_status_router.get("/signals")
async def dashboard_signal_history(
    request: Request,
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    direction: str | None = None,
    minimum_confidence: Annotated[int | None, Query(ge=0, le=100)] = None,
    strength: str | None = None,
    status: str | None = None,
    setup: str | None = None,
    cursor: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    for name, value in (("start", start), ("end", end)):
        if value is not None and value.tzinfo is None:
            raise HTTPException(422, f"{name} must include a timezone")
    if start is not None:
        start = start.astimezone(UTC)
    if end is not None:
        end = end.astimezone(UTC)
    if start is not None and end is not None and start > end:
        raise HTTPException(422, "start must precede end")
    normalized_direction = direction.upper() if direction else None
    if normalized_direction not in {None, "BUY", "SELL", "HOLD"}:
        raise HTTPException(422, "direction must be BUY, SELL, or HOLD")
    normalized_strength = strength.upper() if strength else None
    valid_strengths = {
        None,
        "VERY_WEAK",
        "WEAK",
        "MODERATE",
        "STRONG",
        "VERY_STRONG",
    }
    if normalized_strength not in valid_strengths:
        raise HTTPException(422, "invalid signal strength")
    instrument = canonical_symbol(symbol)
    candidates = await request.app.state.ai_reasoning_repository.list_analysis_signals(
        instrument,
        timeframe,
        start,
        end,
        normalized_direction,
        minimum_confidence,
        normalized_strength,
        0,
        10_000,
    )
    filtered: list[Any] = []
    normalized_status = status.lower() if status else None
    normalized_setup = setup.lower() if setup else None
    for item in candidates:
        if normalized_status is None and normalized_setup is None:
            filtered.append(item)
            continue
        decision = await _decision_for_analysis_signal(request, item)
        decision_statuses = (
            {
                decision.state.value.lower(),
                decision.final_action.value.lower(),
                "eligible" if decision.publication_eligible else "ineligible",
            }
            if decision is not None
            else {"pending"}
        )
        if normalized_status is not None and normalized_status not in decision_statuses:
            continue
        if normalized_setup is not None and (
            decision is None
            or decision.setup_family is None
            or decision.setup_family.lower() != normalized_setup
        ):
            continue
        filtered.append(item)
    visible = filtered[cursor : cursor + limit]
    has_more = cursor + limit < len(filtered)
    total = len(filtered)
    history_items: list[dict[str, Any]] = []
    for item in visible:
        decision = await _decision_for_analysis_signal(request, item)
        publication = _publication_projection(decision)
        lifecycle = await _signal_lifecycle_projection(
            request,
            item,
            now=datetime.now(UTC),
        )
        serialized_signal = (
            _authoritative_signal_projection(
                item,
                decision,
                lifecycle_status=lifecycle["status"],
            )
            if decision is not None
            else item.model_dump(mode="json")
        )
        history_items.append(
            {
                "analytical_signal": serialized_signal,
                "publication": publication,
                "guardrail_outcome": (
                    decision.state.value if decision is not None else "pending"
                ),
                "final_action": (
                    decision.final_action.value if decision is not None else "PENDING"
                ),
                "outcome_status": lifecycle["status"],
                "outcome": lifecycle["outcome"],
                "decision_id": (
                    str(decision.decision_id) if decision is not None else None
                ),
            }
        )
    return {
        "items": history_items,
        "cursor": cursor,
        "next_cursor": cursor + limit if has_more else None,
        "total": total,
        "filters_applied": {
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "direction": normalized_direction,
            "minimum_confidence": minimum_confidence,
            "strength": normalized_strength,
            "status": normalized_status,
            "setup": normalized_setup,
        },
        "result_cap_reached": len(candidates) == 10_000,
    }


@system_status_router.get("/signals/{signal_id}")
async def dashboard_signal_detail(
    signal_id: UUID,
    request: Request,
) -> dict[str, Any]:
    signal = await request.app.state.ai_reasoning_repository.get_analysis_signal(
        signal_id
    )
    if signal is None:
        raise HTTPException(404, "analytical signal not found")
    analysis = await request.app.state.ai_reasoning_repository.get_analysis(
        signal.analysis_id
    )
    state = await request.app.state.unified_market_state_repository.get_state(
        signal.snapshot_id
    )
    quant = await request.app.state.quant_forecast_repository.result_for_state(
        signal.snapshot_id
    )
    decision = await _decision_for_analysis_signal(request, signal)
    return {
        "analytical_signal": signal.model_dump(mode="json"),
        "analysis": analysis.model_dump(mode="json") if analysis else None,
        "state": state.model_dump(mode="json") if state else None,
        "quant_forecast": quant.model_dump(mode="json") if quant else None,
        "final_decision": decision.model_dump(mode="json") if decision else None,
        "publication": _publication_projection(decision),
        "lineage": {
            "cycle_id": str(signal.cycle_id),
            "market_snapshot_id": str(signal.snapshot_id),
            "analysis_id": str(signal.analysis_id),
            "signal_id": str(signal.signal_id),
            "decision_id": str(decision.decision_id) if decision else None,
        },
    }


@system_status_router.get("/analyses")
async def dashboard_analysis_history(
    request: Request,
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
    cursor: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    instrument = canonical_symbol(symbol)
    analyses = await request.app.state.ai_reasoning_repository.list_analyses(
        instrument,
        timeframe,
        None,
        None,
        None,
        None,
        cursor,
        limit + 1,
    )
    visible = analyses[:limit]
    items: list[dict[str, Any]] = []
    for analysis in visible:
        signal = await request.app.state.ai_reasoning_repository.signal_for_analysis(
            analysis.analysis_id
        )
        items.append(
            {
                "analysis": analysis.model_dump(mode="json"),
                "analytical_signal": (
                    signal.model_dump(mode="json") if signal else None
                ),
            }
        )
    return {
        "items": items,
        "cursor": cursor,
        "next_cursor": cursor + limit if len(analyses) > limit else None,
    }


@system_status_router.get("/analyses/{analysis_id}")
async def dashboard_analysis_detail(
    analysis_id: UUID,
    request: Request,
) -> dict[str, Any]:
    analysis = await request.app.state.ai_reasoning_repository.get_analysis(
        analysis_id
    )
    if analysis is None:
        raise HTTPException(404, "AI market analysis not found")
    signal = await request.app.state.ai_reasoning_repository.signal_for_analysis(
        analysis_id
    )
    state = await request.app.state.unified_market_state_repository.get_state(
        analysis.market_snapshot_id
    )
    quant = await request.app.state.quant_forecast_repository.result_for_state(
        analysis.market_snapshot_id
    )
    decision = (
        await _decision_for_analysis_signal(request, signal)
        if signal is not None
        else None
    )
    return {
        "analysis": analysis.model_dump(mode="json"),
        "analytical_signal": signal.model_dump(mode="json") if signal else None,
        "state": state.model_dump(mode="json") if state else None,
        "quant_forecast": quant.model_dump(mode="json") if quant else None,
        "final_decision": decision.model_dump(mode="json") if decision else None,
        "publication": _publication_projection(decision),
    }


@system_status_router.get(
    "/reconciliation",
    dependencies=[Depends(require_role(Role.OPERATOR))],
)
async def dashboard_reconciliation(
    request: Request,
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Read-only operator report; it never mutates or backfills production data."""

    instrument = canonical_symbol(symbol)
    repository = request.app.state.ai_reasoning_repository
    signals = await repository.count_analysis_signals(instrument, timeframe)
    signal_records = await repository.list_analysis_signals(
        instrument,
        timeframe,
        None,
        None,
        None,
        None,
        None,
        0,
        10_000,
    )
    analyses = await repository.list_analyses(
        instrument,
        timeframe,
        None,
        None,
        None,
        None,
        0,
        10_000,
    )
    linked = 0
    orphan_analysis_ids: list[str] = []
    for analysis in analyses:
        signal = await repository.signal_for_analysis(analysis.analysis_id)
        if signal is None:
            orphan_analysis_ids.append(str(analysis.analysis_id))
        else:
            linked += 1
    signal_without_analysis_ids: list[str] = []
    signal_without_decision_ids: list[str] = []
    latest_decision_id: str | None = None
    for signal in signal_records:
        if await repository.get_analysis(signal.analysis_id) is None:
            signal_without_analysis_ids.append(str(signal.signal_id))
        decision = await _decision_for_analysis_signal(request, signal)
        if decision is None:
            signal_without_decision_ids.append(str(signal.signal_id))
        elif latest_decision_id is None:
            latest_decision_id = str(decision.decision_id)
    coherent = await repository.latest_completed_analysis_cycle(
        instrument,
        timeframe,
    )
    latest_analysis = analyses[0] if analyses else None
    latest_signal = signal_records[0] if signal_records else None
    latest_incomplete = next(
        (
            item
            for item in analyses
            if str(item.analysis_id) in set(orphan_analysis_ids)
        ),
        None,
    )
    warnings: list[dict[str, Any]] = []
    if orphan_analysis_ids:
        warnings.append(
            {
                "code": "completed_analysis_without_signal",
                "record_ids": orphan_analysis_ids[:200],
            }
        )
    if signal_without_analysis_ids:
        warnings.append(
            {
                "code": "signal_without_analysis",
                "record_ids": signal_without_analysis_ids[:200],
            }
        )
    if signal_without_decision_ids:
        warnings.append(
            {
                "code": "signal_without_final_decision",
                "record_ids": signal_without_decision_ids[:200],
            }
        )
    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "analysis_count": len(analyses),
        "analytical_signal_count": signals,
        "linked_analysis_count": linked,
        "analysis_without_signal_count": len(orphan_analysis_ids),
        "analysis_without_signal_ids": orphan_analysis_ids[:200],
        "signal_without_analysis_count": len(signal_without_analysis_ids),
        "signal_without_analysis_ids": signal_without_analysis_ids[:200],
        "signal_without_final_decision_count": len(signal_without_decision_ids),
        "signal_without_final_decision_ids": signal_without_decision_ids[:200],
        "latest_completed_analysis_id": (
            str(latest_analysis.analysis_id) if latest_analysis else None
        ),
        "latest_signal_id": (
            str(latest_signal.signal_id) if latest_signal else None
        ),
        "latest_final_decision_id": latest_decision_id,
        "latest_coherent_cycle_id": (
            str(coherent[1].cycle_id) if coherent else None
        ),
        "latest_incomplete_cycle_id": (
            str(latest_incomplete.cycle_id) if latest_incomplete else None
        ),
        "legacy_records_preserved": True,
        "mutation_performed": False,
        "warnings": warnings
        + (
            [{"code": "analysis_count_capped_at_10000", "record_ids": []}]
            if len(analyses) == 10_000
            else []
        ),
    }


@system_status_router.get("/system-status")
async def dashboard_system_status(request: Request, instrument: str = "XAUUSD") -> dict[str, Any]:
    """One backend-authoritative read model for pipeline, storage and failures."""

    now = datetime.now(UTC)
    symbol = canonical_symbol(instrument)
    flags = request.app.state.engine_registry.context.feature_flags
    completed_pair = (
        await request.app.state.ai_reasoning_repository.latest_completed_analysis_cycle(
            symbol
        )
    )
    completed_analysis = completed_pair[0] if completed_pair else None
    completed_signal = completed_pair[1] if completed_pair else None
    state = (
        await request.app.state.unified_market_state_repository.get_state(
            completed_signal.snapshot_id
        )
        if completed_signal is not None
        else await request.app.state.unified_market_state_repository.latest_state(symbol)
    )
    latest_live_state = (
        await request.app.state.unified_market_state_repository.latest_state(symbol)
    )
    stages: dict[str, dict[str, Any]] = {}
    evidence_by_engine = {
        item.source_engine: item for item in (state.evidence if state is not None else ())
    }
    market_timestamp = getattr(state, "market_data_boundary", None)
    live_ums_market_timestamp = getattr(
        latest_live_state, "market_data_boundary", None
    )
    settings = request.app.state.settings
    freshness = await evaluate_market_data_freshness(
        request.app.state.market_data_service,
        symbol=symbol,
        timeframes=tuple(Timeframe(item) for item in REQUIRED_TIMEFRAMES),
        worker_utc_now=now,
        freshness_limit_seconds=settings.max_candle_staleness_seconds,
        ums_market_timestamp=live_ums_market_timestamp,
    )
    freshness.update(
        {
            "ums_state_id": getattr(latest_live_state, "state_id", None),
            "ums_cycle_id": getattr(latest_live_state, "cycle_id", None),
            "completed_ai_cycle_market_timestamp": market_timestamp,
        }
    )
    market_data_status = freshness["status"]
    stages["market_data"] = _system_stage(
        "market_data",
        "Market Data",
        (
            "healthy"
            if market_data_status in {"FRESH", "MARKET_CLOSED"}
            else "stale"
        ),
        (
            "MARKET_CLOSED"
            if market_data_status == "MARKET_CLOSED"
            else "market_boundary_stale"
            if market_data_status == "STALE"
            else "closed_candles_available"
        ),
        timestamp=freshness["latest_candle_timestamp"],
        details=freshness,
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
        "awaiting_synchronized_m5_m15_state" if state is None else "point_in_time_state_persisted",
        timestamp=market_timestamp,
        record_id=getattr(state, "state_id", None),
        details={"evidence_completeness": getattr(state, "evidence_completeness", None)},
    )
    quant = (
        await request.app.state.quant_forecast_repository.result_for_state(state.state_id)
        if state is not None else None
    )
    analysis = completed_analysis or (
        await request.app.state.ai_reasoning_repository.analysis_for_state(state.state_id)
        if state is not None
        else None
    )
    analysis_signal = completed_signal or (
        await request.app.state.ai_reasoning_repository.signal_for_analysis(
            analysis.analysis_id
        )
        if analysis is not None
        else None
    )
    signal_decision = (
        await _decision_for_analysis_signal(request, analysis_signal)
        if analysis_signal is not None
        else None
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
            "candidate_signal": getattr(getattr(analysis_signal, "signal", None), "value", None),
            "signal": getattr(getattr(signal_decision, "final_action", None), "value", None),
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
            _authoritative_signal_projection(
                analysis_signal,
                signal_decision,
                lifecycle_status=(
                    "HOLD"
                    if signal_decision.final_action.value == "HOLD"
                    else "CURRENT"
                ),
            )
            if analysis_signal is not None and signal_decision is not None
            else analysis_signal.model_dump(mode="json")
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
        reason = "ai_centric_shadow_mode_disabled" if not shadow_enabled else "awaiting_synchronized_m5_m15_state"
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
            reason="synchronized_m5_m15_state_persisted",
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
