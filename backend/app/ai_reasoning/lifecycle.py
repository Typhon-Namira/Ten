"""Persistent signal identity, transitions, monitoring, and level revisions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from .models import (
    AIMarketForecast,
    AISignalProposal,
    Direction,
    ManagedSignal,
    ManagedSignalState,
    ProposalAction,
    SignalLevelRevision,
    SignalMonitoringEvaluation,
    SignalStateTransition,
)


class LifecycleRepository(Protocol):
    async def signal_by_opportunity(self, opportunity_key: str) -> ManagedSignal | None: ...
    async def save_signal(self, value: ManagedSignal) -> ManagedSignal: ...
    async def save_transition(self, value: SignalStateTransition) -> SignalStateTransition: ...
    async def save_revision(self, value: SignalLevelRevision) -> SignalLevelRevision: ...
    async def save_monitoring(self, value: SignalMonitoringEvaluation) -> SignalMonitoringEvaluation: ...


TERMINAL_STATES = {
    ManagedSignalState.CLOSED,
    ManagedSignalState.CANCELLED,
    ManagedSignalState.INVALIDATED,
    ManagedSignalState.EXPIRED,
    ManagedSignalState.STOPPED,
}

ALLOWED_TRANSITIONS = {
    ManagedSignalState.DETECTED: {ManagedSignalState.PROPOSED, ManagedSignalState.CANCELLED},
    ManagedSignalState.PROPOSED: {ManagedSignalState.CONFIRMED, ManagedSignalState.WAITING_FOR_ENTRY, ManagedSignalState.CANCELLED, ManagedSignalState.INVALIDATED, ManagedSignalState.EXPIRED},
    ManagedSignalState.CONFIRMED: {ManagedSignalState.WAITING_FOR_ENTRY, ManagedSignalState.CANCELLED, ManagedSignalState.INVALIDATED, ManagedSignalState.EXPIRED},
    ManagedSignalState.WAITING_FOR_ENTRY: {ManagedSignalState.ACTIVE, ManagedSignalState.CANCELLED, ManagedSignalState.INVALIDATED, ManagedSignalState.EXPIRED, ManagedSignalState.TEMPORARILY_BLOCKED},
    ManagedSignalState.TEMPORARILY_BLOCKED: {ManagedSignalState.WAITING_FOR_ENTRY, ManagedSignalState.CANCELLED, ManagedSignalState.INVALIDATED, ManagedSignalState.EXPIRED},
    ManagedSignalState.ACTIVE: {ManagedSignalState.PARTIALLY_REALIZED, ManagedSignalState.TP1_HIT, ManagedSignalState.TP2_HIT, ManagedSignalState.CLOSED, ManagedSignalState.INVALIDATED, ManagedSignalState.STOPPED},
    ManagedSignalState.PARTIALLY_REALIZED: {ManagedSignalState.TP1_HIT, ManagedSignalState.TP2_HIT, ManagedSignalState.CLOSED, ManagedSignalState.INVALIDATED, ManagedSignalState.STOPPED},
    ManagedSignalState.TP1_HIT: {ManagedSignalState.TP2_HIT, ManagedSignalState.CLOSED, ManagedSignalState.INVALIDATED, ManagedSignalState.STOPPED},
    ManagedSignalState.TP2_HIT: {ManagedSignalState.CLOSED},
}


class SignalLifecycleService:
    def __init__(
        self,
        repository: LifecycleRepository,
        *,
        policy_version: str,
        model_version: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.policy_version = policy_version
        self.model_version = model_version
        self.clock = clock or (lambda: datetime.now(UTC))

    async def apply_proposal(
        self,
        forecast: AIMarketForecast,
        proposal: AISignalProposal,
        *,
        setup_family: str,
    ) -> ManagedSignal | None:
        if proposal.recommended_action == ProposalAction.WAIT:
            return None
        now = self.clock()
        existing = await self.repository.signal_by_opportunity(proposal.structural_opportunity_key)
        if existing is None:
            signal_id = uuid5(NAMESPACE_URL, f"ten:managed-signal:{proposal.structural_opportunity_key}")
            signal = ManagedSignal(
                signal_id=signal_id,
                instrument="XAUUSD",
                structural_opportunity_key=proposal.structural_opportunity_key,
                setup_family=setup_family,
                direction=proposal.direction,
                state=ManagedSignalState.PROPOSED,
                current_proposal_id=proposal.proposal_id,
                originating_market_state_id=proposal.market_state_id,
                latest_market_state_id=proposal.market_state_id,
                entry_zone=proposal.entry_zone,
                stop_loss=proposal.stop_loss,
                take_profit_levels=proposal.take_profit_levels,
                invalidation_price=proposal.invalidation_price,
                expires_at=proposal.expires_at,
                created_at=now,
                updated_at=now,
            )
            await self.repository.save_signal(signal)
            await self._transition(
                signal,
                ManagedSignalState.DETECTED,
                ManagedSignalState.PROPOSED,
                "validated_ai_proposal",
                forecast,
                proposal,
            )
            return signal
        if existing.state in TERMINAL_STATES:
            return existing
        # Phase 3/4 stores AI recommendations but does not execute them. State/level mutation is
        # reserved for the later guardrail phase; updating the current proposal reference is
        # observational and preserves the managed signal's identity and complete proposal history.
        target = existing.state
        updated = existing.model_copy(
            update={
                "state": target,
                "current_proposal_id": proposal.proposal_id,
                "latest_market_state_id": proposal.market_state_id,
                "updated_at": now,
            }
        )
        await self.repository.save_signal(updated)
        if target != existing.state:
            await self._transition(existing, existing.state, target, proposal.recommended_action.value, forecast, proposal)
        return updated

    async def apply_guardrail_approved_transition(
        self,
        signal: ManagedSignal,
        target: ManagedSignalState,
        *,
        approval_rule: str,
        forecast: AIMarketForecast,
        proposal: AISignalProposal,
    ) -> ManagedSignal:
        if not approval_rule:
            raise ValueError("managed-signal transition requires an explicit guardrail approval rule")
        if target not in ALLOWED_TRANSITIONS.get(signal.state, set()):
            raise ValueError(f"illegal managed-signal transition: {signal.state.value} -> {target.value}")
        now = self.clock()
        updated = signal.model_copy(update={"state": target, "updated_at": now})
        await self.repository.save_signal(updated)
        await self._transition(signal, signal.state, target, approval_rule, forecast, proposal)
        return updated

    async def monitor(
        self,
        signal: ManagedSignal,
        forecast: AIMarketForecast,
        proposal: AISignalProposal | None,
        *,
        previous_probability: float | None,
    ) -> SignalMonitoringEvaluation:
        current_probability = forecast.dominant_scenario_probability
        change = (
            current_probability - previous_probability
            if current_probability is not None and previous_probability is not None
            else None
        )
        action = proposal.recommended_action if proposal else ProposalAction.WAIT
        thesis_valid = action not in {
            ProposalAction.CANCEL_SETUP,
            ProposalAction.INVALIDATE_SIGNAL,
            ProposalAction.CLOSE_SIGNAL,
        }
        evaluation = SignalMonitoringEvaluation(
            evaluation_id=uuid5(
                NAMESPACE_URL,
                f"ten:monitoring:{signal.signal_id}:{forecast.forecast_id}",
            ),
            signal_id=signal.signal_id,
            forecast_id=forecast.forecast_id,
            proposal_id=proposal.proposal_id if proposal else None,
            market_state_id=forecast.market_state_id,
            thesis_valid=thesis_valid,
            scenario_probability_change=change,
            changed_evidence_ids=forecast.supporting_evidence_ids + forecast.contradicting_evidence_ids,
            reason_codes=(action.value,),
            recommended_action=action,
            evaluated_at=self.clock(),
        )
        return await self.repository.save_monitoring(evaluation)

    async def revise_level(
        self,
        signal: ManagedSignal,
        *,
        level_type: str,
        new_value: Any,
        reason: str,
        evidence_ids: tuple[UUID, ...],
        approved_rule: str | None = None,
    ) -> tuple[ManagedSignal, SignalLevelRevision]:
        old_value: Any
        if level_type == "stop_loss":
            old_value = signal.stop_loss
            if old_value is not None and isinstance(new_value, (int, float)):
                widened = (
                    signal.direction == Direction.BUY and new_value < old_value
                ) or (
                    signal.direction == Direction.SELL and new_value > old_value
                )
                if widened and approved_rule is None:
                    raise ValueError("stop loss cannot be widened without an explicitly approved rule")
            update = {"stop_loss": new_value}
        elif level_type == "entry_zone":
            old_value = signal.entry_zone.model_dump(mode="json") if signal.entry_zone else None
            update = {"entry_zone": new_value}
        elif level_type == "take_profit_levels":
            old_value = signal.take_profit_levels
            update = {"take_profit_levels": tuple(new_value)}
        elif level_type == "invalidation_price":
            old_value = signal.invalidation_price
            update = {"invalidation_price": new_value}
        else:
            raise ValueError("unsupported signal level type")
        now = self.clock()
        revision = SignalLevelRevision(
            revision_id=uuid5(
                NAMESPACE_URL,
                f"ten:signal-revision:{signal.signal_id}:{level_type}:{old_value}:{new_value}:{now.isoformat()}",
            ),
            signal_id=signal.signal_id,
            level_type=level_type,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            evidence_ids=evidence_ids,
            model_version=self.model_version,
            policy_version=self.policy_version,
            approved_rule=approved_rule,
            created_at=now,
        )
        updated = signal.model_copy(update={**update, "updated_at": now})
        await self.repository.save_revision(revision)
        await self.repository.save_signal(updated)
        return updated, revision

    async def _transition(
        self,
        signal: ManagedSignal,
        previous: ManagedSignalState,
        new: ManagedSignalState,
        reason: str,
        forecast: AIMarketForecast,
        proposal: AISignalProposal,
    ) -> SignalStateTransition:
        transition = SignalStateTransition(
            transition_id=uuid5(
                NAMESPACE_URL,
                f"ten:signal-transition:{signal.signal_id}:{previous.value}:{new.value}:{proposal.proposal_id}",
            ),
            signal_id=signal.signal_id,
            previous_state=previous,
            new_state=new,
            reason=reason,
            supporting_evidence_ids=proposal.supporting_evidence_ids,
            ai_forecast_id=forecast.forecast_id,
            ai_proposal_id=proposal.proposal_id,
            market_state_id=proposal.market_state_id,
            policy_version=self.policy_version,
            model_version=self.model_version,
            created_at=self.clock(),
        )
        return await self.repository.save_transition(transition)
