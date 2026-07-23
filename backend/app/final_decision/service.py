"""Deterministic final-decision and analytical-publication orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import logging
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.ai_reasoning.models import (
    AIMarketForecast,
    AIResultStatus,
    AISignalProposal,
    ManagedSignal,
    ManagedSignalState,
    ProposalAction,
)
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from backend.app.market_state import EvidenceAvailability, UnifiedMarketState
from backend.app.quant_forecasting.models import ForecastStatus, QuantForecastResult

from .config import GuardrailPolicyConfig
from .models import (
    ApprovalState,
    ExecutionContext,
    FinalAction,
    FinalSystemAction,
    GateEvaluation,
    GateStatus,
    ProposalModification,
    PublicationState,
    PublishedAnalyticalSignal,
    RiskClassification,
)
from .registry import HardGateRegistry
from .repository import FinalDecisionRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalDecisionResult:
    action: FinalSystemAction
    publication: PublishedAnalyticalSignal | None


def _canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class FinalDecisionService:
    """Pure policy evaluation plus append-only persistence; never calls an LLM or broker."""

    _ACCOUNT_GATES = {
        "maximum_risk_per_signal",
        "maximum_simultaneous_exposure",
        "maximum_daily_loss",
        "maximum_aggregate_drawdown",
        "conflicting_active_exposure",
        "position_size_valid",
    }

    def __init__(
        self,
        repository: FinalDecisionRepository,
        registry: HardGateRegistry,
        setup_registry: SetupFamilyRegistry,
        config: GuardrailPolicyConfig,
        *,
        publication_enabled: bool,
        adjustments_enabled: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.registry = registry
        self.setup_registry = setup_registry
        self.config = config
        self.publication_enabled = publication_enabled
        self.adjustments_enabled = adjustments_enabled
        self.clock = clock or (lambda: datetime.now(UTC))
        self.actions_evaluated = 0
        self.publications_succeeded = 0
        self.publications_failed = 0
        self.last_failure: str | None = None

    async def evaluate(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        forecast: AIMarketForecast,
        proposal: AISignalProposal,
        signal: ManagedSignal,
        context: ExecutionContext,
    ) -> FinalDecisionResult:
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        quant = QuantForecastResult.model_validate(quant.model_dump(mode="python"))
        forecast = AIMarketForecast.model_validate(forecast.model_dump(mode="python"))
        proposal = AISignalProposal.model_validate(proposal.model_dump(mode="python"))
        signal = ManagedSignal.model_validate(signal.model_dump(mode="python"))
        context = ExecutionContext.model_validate(context.model_dump(mode="python"))
        original_hash = _canonical_hash(proposal.model_dump(mode="json"))
        action_id = uuid5(
            NAMESPACE_URL,
            f"ten:final-action:{proposal.proposal_id}:{state.state_hash}:{self.config.version}",
        )
        for definition in self.registry.all():
            await self.repository.save_gate_definition(definition, self.registry.version)
        evaluations = tuple(
            self._evaluate_gate(definition.gate_id, action_id, state, quant, forecast, proposal, signal, context)
            for definition in self.registry.all()
        )
        modifications = self._policy_modifications(proposal, context)
        final_expiry = proposal.expires_at
        for modification in modifications:
            if modification.field_name == "expires_at":
                final_expiry = datetime.fromisoformat(str(modification.final_value))
        blocking = [
            item
            for item in evaluations
            if item.status == GateStatus.FAILED
            or (
                item.status == GateStatus.UNAVAILABLE
                and item.gate_id
                in {
                    "market_open",
                    "spread_within_session_limit",
                    "publication_service_available",
                    "persistence_state_valid",
                    "economic_event_blackout",
                }
            )
        ]
        if blocking:
            if any(item.block_behavior == "postpone" for item in blocking):
                final_action, approval = FinalAction.POSTPONED, ApprovalState.POSTPONED
            else:
                final_action, approval = FinalAction.REJECTED, ApprovalState.REJECTED
            publication_state = PublicationState.DISABLED if not self.publication_enabled else PublicationState.NOT_REQUESTED
            risk_classification = RiskClassification.BLOCKED
        else:
            final_action = FinalAction.APPROVED_REDUCED_RISK if modifications else FinalAction.APPROVED
            approval = ApprovalState.MODIFIED if modifications else ApprovalState.APPROVED
            publication_state = PublicationState.PENDING if self.publication_enabled else PublicationState.DISABLED
            risk_classification = RiskClassification.REDUCED if modifications else RiskClassification.STANDARD
        risk_to_reward = self._risk_to_reward(proposal)
        base_action = FinalSystemAction(
            final_action_id=action_id,
            ai_proposal_id=proposal.proposal_id,
            managed_signal_id=signal.signal_id,
            market_state_id=state.state_id,
            quantitative_forecast_id=quant.result_id,
            ai_forecast_id=forecast.forecast_id,
            action=final_action,
            approval_state=approval,
            publication_state=publication_state,
            final_direction=proposal.direction,
            final_entry=proposal.entry_zone,
            final_stop_loss=proposal.stop_loss,
            final_take_profits=proposal.take_profit_levels,
            final_risk_to_reward=risk_to_reward,
            final_expiry=final_expiry,
            final_risk_classification=risk_classification,
            gate_evaluations=evaluations,
            modifications=modifications,
            modification_reasons=tuple(item.exact_reason for item in modifications),
            policy_versions=self.policy_versions(),
            original_proposal_hash=original_hash,
            created_at=self.clock(),
        )
        # Persist the action before child evaluation rows to honor relational integrity.
        await self.repository.save_action(base_action)
        for evaluation in evaluations:
            await self.repository.save_evaluation(evaluation)
        logger.info(
            "guardrails.completed",
            extra={
                "cycle_id": str(state.cycle_id),
                "market_state_id": str(state.state_id),
                "forecast_id": str(forecast.forecast_id),
                "proposal_id": str(proposal.proposal_id),
                "final_action_id": str(base_action.final_action_id),
                "gate_count": len(evaluations),
                "blocking_gate_count": len(blocking),
                "publication_enabled": self.publication_enabled,
            },
        )
        publication: PublishedAnalyticalSignal | None = None
        final = base_action
        if not blocking and self.publication_enabled and proposal.recommended_action in {ProposalAction.BUY, ProposalAction.SELL}:
            try:
                publication = await self._publish(base_action, forecast, proposal, signal)
                final = base_action.model_copy(
                    update={"action": FinalAction.PUBLISHED, "publication_state": PublicationState.PUBLISHED}
                )
                await self.repository.save_action(final)
                self.publications_succeeded += 1
            except Exception as exc:
                self.publications_failed += 1
                self.last_failure = f"publication_failure:{type(exc).__name__}"
                final = base_action.model_copy(update={"publication_state": PublicationState.FAILED})
                await self.repository.save_action(final)
        self.actions_evaluated += 1
        logger.info(
            "final_decision.completed",
            extra={
                "cycle_id": str(state.cycle_id),
                "market_state_id": str(state.state_id),
                "forecast_id": str(forecast.forecast_id),
                "proposal_id": str(proposal.proposal_id),
                "final_action_id": str(final.final_action_id),
                "action": final.action.value,
                "approval_state": final.approval_state.value,
                "publication_state": final.publication_state.value,
            },
        )
        return FinalDecisionResult(action=final, publication=publication)

    def _evaluate_gate(
        self,
        gate_id: str,
        action_id: object,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        forecast: AIMarketForecast,
        proposal: AISignalProposal,
        signal: ManagedSignal,
        context: ExecutionContext,
    ) -> GateEvaluation:
        definition = self.registry.get(gate_id)
        status, reasons, audit = self._gate_result(gate_id, state, quant, forecast, proposal, signal, context)
        return GateEvaluation(
            evaluation_id=uuid5(NAMESPACE_URL, f"ten:gate:{action_id}:{gate_id}:{definition.gate_version}"),
            final_action_id=action_id,
            gate_id=gate_id,
            gate_version=definition.gate_version,
            category=definition.category,
            status=status,
            severity=definition.severity,
            block_behavior=definition.block_behavior,
            reason_codes=reasons,
            audit_payload=audit,
            configuration_source=definition.configuration_source,
            evaluated_at=self.clock(),
        )

    def _gate_result(
        self,
        gate_id: str,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        forecast: AIMarketForecast,
        proposal: AISignalProposal,
        signal: ManagedSignal,
        context: ExecutionContext,
    ) -> tuple[GateStatus, tuple[str, ...], dict[str, Any]]:
        passed = GateStatus.PASSED
        if gate_id == "market_state_consistent":
            return passed, (), {"state_hash": state.state_hash, "status": state.status.value}
        if gate_id == "authoritative_data_fresh":
            stale = bool(state.stale_evidence) or any(item.stale for item in state.timeframes)
            age = max(0.0, (context.evaluated_at - state.market_data_boundary).total_seconds())
            ok = not stale and age <= self.config.maximum_signal_age_seconds
            return self._boolean(ok, "stale_authoritative_market_data", {"age_seconds": age, "stale_evidence": len(state.stale_evidence)})
        if gate_id in {"point_in_time_valid", "future_data_absent"}:
            ok = (
                state.market_data_boundary <= state.knowledge_cutoff
                and quant.market_state_id == state.state_id
                and quant.point_in_time <= state.knowledge_cutoff
                and forecast.market_state_id == state.state_id
                and forecast.quantitative_forecast_id == quant.result_id
            )
            return self._boolean(ok, gate_id, {"knowledge_cutoff": state.knowledge_cutoff.isoformat()})
        if gate_id == "market_open":
            return self._optional_boolean(context.market_open, "market_closed", "market_status_unavailable")
        if gate_id == "quantitative_forecast_available":
            return self._boolean(quant.status == ForecastStatus.AVAILABLE, "quantitative_forecast_unavailable", {"status": quant.status.value})
        if gate_id == "ai_forecast_valid":
            ok = forecast.status in {AIResultStatus.AVAILABLE, AIResultStatus.NON_ACTIONABLE} and forecast.validation_passed
            return self._boolean(ok, "ai_forecast_invalid", {"status": forecast.status.value, "validation_passed": forecast.validation_passed})
        if gate_id == "ai_proposal_valid":
            actionable = proposal.recommended_action in {ProposalAction.BUY, ProposalAction.SELL}
            return self._boolean(actionable, "ai_proposal_not_actionable", {"action": proposal.recommended_action.value})
        if gate_id == "mandatory_setup_evidence":
            kinds = {"market_data"}
            kinds.update({
                token
                for item in state.evidence
                if item.availability != EvidenceAvailability.UNAVAILABLE
                for token in (item.source_engine, item.evidence_type)
            })
            setup = forecast.selected_setup_family or signal.setup_family
            errors = self.setup_registry.validate_requirements(
                setup,
                kinds,
                forecast.evidence_completeness or 0.0,
                proposal.recommended_action.value,
            )
            return (GateStatus.FAILED if errors else passed), errors, {"setup_family": setup, "available_evidence": sorted(kinds)}
        if gate_id == "signal_lifecycle_valid":
            ok = signal.state not in {
                ManagedSignalState.CLOSED,
                ManagedSignalState.CANCELLED,
                ManagedSignalState.INVALIDATED,
                ManagedSignalState.EXPIRED,
                ManagedSignalState.STOPPED,
            }
            return self._boolean(ok, "invalid_signal_lifecycle_state", {"state": signal.state.value})
        if gate_id == "duplicate_structural_opportunity":
            duplicate = proposal.structural_opportunity_key in context.active_opportunity_keys and context.active_signal_id != signal.signal_id
            return self._boolean(not duplicate, "duplicate_active_structural_opportunity", {"opportunity_key": proposal.structural_opportunity_key})
        if gate_id == "publication_service_available":
            if not self.publication_enabled:
                return GateStatus.NOT_APPLICABLE, (), {"publication_enabled": False}
            return self._boolean(context.publication_service_available, "publication_service_unavailable", {})
        if gate_id == "persistence_state_valid":
            return self._boolean(context.persistence_available, "invalid_persistence_state", {})
        if gate_id in {"entry_geometry_valid", "stop_geometry_valid", "target_geometry_valid"}:
            return self._geometry_gate(gate_id, proposal)
        if gate_id == "signal_type_supported":
            entry_type = (proposal.entry_type or "zone").lower()
            return self._boolean(entry_type in self.config.supported_entry_types, "unsupported_signal_type", {"entry_type": entry_type})
        if gate_id == "spread_within_session_limit":
            if context.spread is None:
                return GateStatus.UNAVAILABLE, ("spread_unavailable",), {}
            limit = getattr(self.config.maximum_spread, context.session.lower().replace(" ", "_"), self.config.maximum_spread.unknown)
            return self._boolean(context.spread <= limit, "extreme_spread", {"spread": context.spread, "limit": limit, "session": context.session})
        if gate_id == "price_precision_valid":
            values = [proposal.stop_loss, *proposal.take_profit_levels]
            if proposal.entry_zone:
                values.extend((proposal.entry_zone.low, proposal.entry_zone.high))
            ok = all(value is None or self._decimal_places(value) <= self.config.price_precision for value in values)
            return self._boolean(ok, "invalid_price_precision", {"precision": self.config.price_precision})
        if gate_id == "entry_distance_valid":
            if context.current_price is None or proposal.entry_zone is None:
                return GateStatus.UNAVAILABLE, ("entry_distance_unavailable",), {}
            midpoint = (proposal.entry_zone.low + proposal.entry_zone.high) / 2
            distance = abs(context.current_price - midpoint)
            return self._boolean(distance <= self.config.maximum_entry_distance, "market_too_far_from_entry", {"distance": distance, "limit": self.config.maximum_entry_distance})
        if gate_id == "setup_not_expired":
            if proposal.expires_at is None:
                return GateStatus.FAILED, ("setup_expiry_missing",), {}
            return self._boolean(context.evaluated_at < proposal.expires_at, "setup_expired", {"expires_at": proposal.expires_at.isoformat()})
        if gate_id == "execution_context_available":
            ok = context.analytical_only or context.broker_execution_available
            return self._boolean(ok, "execution_context_unavailable", {"analytical_only": context.analytical_only})
        if gate_id == "absolute_risk_to_reward":
            rr = self._risk_to_reward(proposal)
            return self._boolean(rr is not None and rr >= self.config.minimum_absolute_risk_to_reward, "risk_to_reward_below_absolute_minimum", {"risk_to_reward": rr, "minimum": self.config.minimum_absolute_risk_to_reward})
        if gate_id in self._ACCOUNT_GATES:
            if not context.authoritative_account_risk_available:
                return GateStatus.NOT_APPLICABLE, ("authoritative_account_risk_unavailable",), {"fabricated": False, "analytical_only": context.analytical_only}
            account_values = {
                "maximum_risk_per_signal": context.risk_per_signal,
                "maximum_simultaneous_exposure": context.simultaneous_exposure,
                "maximum_daily_loss": context.daily_loss,
                "maximum_aggregate_drawdown": context.aggregate_drawdown,
                "conflicting_active_exposure": context.conflicting_active_exposure,
                "position_size_valid": context.position_size_valid,
            }
            value = account_values[gate_id]
            if gate_id in {"conflicting_active_exposure"}:
                ok = value is False
            elif gate_id == "position_size_valid":
                ok = value is True
            else:
                # Limits are intentionally unavailable until authoritative account policy exists.
                return GateStatus.UNAVAILABLE, ("authoritative_risk_limit_not_configured",), {"value": value}
            return self._boolean(ok, gate_id, {"value": value})
        if gate_id == "economic_event_blackout":
            if not context.economic_context_available or context.prohibited_economic_event_window is None:
                if self.config.economic_event_blackout_required:
                    return GateStatus.UNAVAILABLE, ("economic_context_unavailable",), {}
                return GateStatus.NOT_APPLICABLE, (), {}
            return self._boolean(not context.prohibited_economic_event_window, "prohibited_economic_event_window", {})
        if gate_id == "stop_distance_safe":
            if proposal.entry_zone is None or proposal.stop_loss is None:
                return GateStatus.FAILED, ("stop_distance_unavailable",), {}
            midpoint = (proposal.entry_zone.low + proposal.entry_zone.high) / 2
            distance = abs(midpoint - proposal.stop_loss)
            ok = self.config.minimum_stop_distance <= distance <= self.config.maximum_stop_distance
            return self._boolean(ok, "unsafe_stop_loss_distance", {"distance": distance, "minimum": self.config.minimum_stop_distance, "maximum": self.config.maximum_stop_distance})
        raise KeyError(gate_id)

    @staticmethod
    def _boolean(ok: bool, reason: str, audit: dict[str, Any]) -> tuple[GateStatus, tuple[str, ...], dict[str, Any]]:
        return (GateStatus.PASSED, (), audit) if ok else (GateStatus.FAILED, (reason,), audit)

    @staticmethod
    def _optional_boolean(value: bool | None, failed: str, unavailable: str) -> tuple[GateStatus, tuple[str, ...], dict[str, Any]]:
        if value is None:
            return GateStatus.UNAVAILABLE, (unavailable,), {}
        return (GateStatus.PASSED, (), {}) if value else (GateStatus.FAILED, (failed,), {})

    def _geometry_gate(self, gate_id: str, proposal: AISignalProposal) -> tuple[GateStatus, tuple[str, ...], dict[str, Any]]:
        if proposal.entry_zone is None or proposal.stop_loss is None or not proposal.take_profit_levels:
            return GateStatus.FAILED, ("incomplete_price_geometry",), {}
        if proposal.direction.value == "BUY":
            entry_ok = proposal.entry_zone.low <= proposal.entry_zone.high
            stop_ok = proposal.stop_loss < proposal.entry_zone.low
            target_ok = all(item > proposal.entry_zone.high for item in proposal.take_profit_levels)
        else:
            entry_ok = proposal.entry_zone.low <= proposal.entry_zone.high
            stop_ok = proposal.stop_loss > proposal.entry_zone.high
            target_ok = all(item < proposal.entry_zone.low for item in proposal.take_profit_levels)
        mapping = {
            "entry_geometry_valid": (entry_ok, "invalid_entry_geometry"),
            "stop_geometry_valid": (stop_ok, "invalid_stop_loss_geometry"),
            "target_geometry_valid": (target_ok, "invalid_take_profit_geometry"),
        }
        ok, reason = mapping[gate_id]
        return self._boolean(ok, reason, proposal.model_dump(mode="json", include={"entry_zone", "stop_loss", "take_profit_levels", "direction"}))

    def _policy_modifications(self, proposal: AISignalProposal, context: ExecutionContext) -> tuple[ProposalModification, ...]:
        if proposal.expires_at is None:
            return ()
        maximum = context.evaluated_at + timedelta(seconds=self.config.maximum_setup_expiry_seconds)
        if proposal.expires_at <= maximum:
            return ()
        return (
            ProposalModification(
                field_name="expires_at",
                original_value=proposal.expires_at.isoformat(),
                final_value=maximum.isoformat(),
                modifying_gate_or_policy=self.config.execution_policy_version,
                exact_reason="setup expiry reduced to the configured maximum; the original AI proposal remains unchanged",
            ),
        )

    @staticmethod
    def _decimal_places(value: float) -> int:
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        return len(text.split(".", 1)[1]) if "." in text else 0

    @staticmethod
    def _risk_to_reward(proposal: AISignalProposal) -> float | None:
        if proposal.entry_zone is None or proposal.stop_loss is None or not proposal.take_profit_levels:
            return None
        entry = (proposal.entry_zone.low + proposal.entry_zone.high) / 2
        risk = abs(entry - proposal.stop_loss)
        return abs(proposal.take_profit_levels[0] - entry) / risk if risk > 0 else None

    async def _publish(
        self,
        action: FinalSystemAction,
        forecast: AIMarketForecast,
        proposal: AISignalProposal,
        signal: ManagedSignal,
    ) -> PublishedAnalyticalSignal:
        if proposal.entry_zone is None or proposal.stop_loss is None:
            raise ValueError("publication requires validated entry and stop")
        if any(value is None for value in (forecast.buy_probability, forecast.sell_probability, forecast.neutral_probability)):
            raise ValueError("publication requires available forecast probabilities")
        assert forecast.buy_probability is not None and forecast.sell_probability is not None and forecast.neutral_probability is not None
        publication = PublishedAnalyticalSignal(
            publication_id=uuid5(NAMESPACE_URL, f"ten:analytical-publication:{signal.signal_id}"),
            signal_id=signal.signal_id,
            final_action_id=action.final_action_id,
            proposal_id=proposal.proposal_id,
            instrument=signal.instrument,
            direction=proposal.direction,
            setup_family=signal.setup_family,
            entry_zone=proposal.entry_zone,
            stop_loss=proposal.stop_loss,
            take_profit_levels=proposal.take_profit_levels,
            invalidation_price=proposal.invalidation_price,
            invalidation_conditions=proposal.invalidation_conditions,
            expires_at=action.final_expiry,
            expected_horizon=forecast.expected_horizon or "unspecified",
            buy_probability=forecast.buy_probability,
            sell_probability=forecast.sell_probability,
            neutral_probability=forecast.neutral_probability,
            forecast_confidence=forecast.forecast_confidence,
            uncertainty=forecast.uncertainty,
            proposal_confidence=proposal.proposal_confidence,
            final_risk_classification=action.final_risk_classification,
            dominant_scenario=forecast.dominant_scenario or "unavailable",
            supporting_evidence_summary=tuple(str(item) for item in forecast.supporting_evidence_ids),
            contradicting_evidence_summary=tuple(str(item) for item in forecast.contradicting_evidence_ids),
            lifecycle_state=ManagedSignalState.CONFIRMED,
            model_versions={
                "llm": forecast.model_identifier,
                "quantitative": forecast.quantitative_model_version,
                "feature_schema": forecast.feature_schema_version,
            },
            policy_versions=action.policy_versions,
            published_at=self.clock(),
        )
        return await self.repository.save_publication(publication)

    def policy_versions(self) -> dict[str, str]:
        return {
            "guardrails": self.config.version,
            "hard_gates": self.registry.version,
            "risk": self.config.risk_policy_version,
            "execution": self.config.execution_policy_version,
            "publication": self.config.publication_schema_version,
        }

    def health(self) -> dict[str, Any]:
        total_publications = self.publications_succeeded + self.publications_failed
        return {
            "status": "degraded" if self.last_failure else "healthy",
            "publication_enabled": self.publication_enabled,
            "adjustments_enabled": self.adjustments_enabled,
            "analytical_only": True,
            "broker_execution_available": False,
            "actions_evaluated": self.actions_evaluated,
            "publications_succeeded": self.publications_succeeded,
            "publications_failed": self.publications_failed,
            "publication_failure_rate": self.publications_failed / total_publications if total_publications else 0.0,
            "last_failure": self.last_failure,
            "policy_versions": self.policy_versions(),
            "daily_request_allowance": self.config.maximum_daily_llm_requests,
            "daily_token_allowance": self.config.maximum_daily_llm_tokens,
            "llm_concurrency_limit": self.config.llm_concurrency_limit,
        }
